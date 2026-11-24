"""FastAPI boundary over PotatoForge application workflows."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import re
from pathlib import Path
from typing import Annotated, Any, Callable, Literal
from urllib.parse import quote

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..audits.activation_audit import (
    ACTIVATION_AUDIT_METRICS,
    activation_audit_pair_paths,
    inspect_activation_audit,
    score_activation_audit,
)
from ..audits.activation_measurements import MEASUREMENT_METHODS, METHOD_ACTIONS
from ..application.commands import (
    activation_audit_model,
    audit_model,
    inspect_model,
    optimize_activation_profile,
    optimize_profile,
    quantize_model,
)
from ..audits.weight_audit import (
    DEFAULT_COMMAND_AUDIT_METHODS,
    QuantizationMethod,
    audit_method_options,
)
from ..lora.lora_extract import (
    extract_lora,
    resolve_output_paths,
    validate_options as validate_extraction_options,
)
from ..conversion_methods import get_conversion_method
from ..profiles import load_profile
from .model_library import (
    DEFAULT_CONFIG_PATH,
    FOLDER_KINDS,
    add_library_folder,
    discover_audit_reports,
    discover_activation_audit_caches,
    discover_activation_calibrations,
    discover_loras,
    discover_models,
    discover_profiles,
    is_in_library,
    load_library_folders,
    remove_library_folder,
)
from .activation_jobs import ActivationAuditJobManager
from .audit_jobs import AuditJobManager, audit_result_payload
from .jobs import ActiveJobError, InactiveJobError, UnknownJobError
from .extract_jobs import ExtractJobManager, build_extraction_options
from .quantize_jobs import QuantizeJobManager, quantize_result_payload
from .comfy_runner import ComfyClient, ComfyRunner, launch_comfy
from .comfy_workflow import inspect_dataset, inspect_workflow, load_json, validate_bindings


class CommandResult(BaseModel):
    result: dict[str, Any]


class InspectRequest(BaseModel):
    model_path: Path
    output: Path | None = None
    overwrite: bool = False


class ModelFolderRequest(BaseModel):
    path: str


class AuditRequest(BaseModel):
    source_path: Path
    output: Path
    device: Literal["cpu", "cuda"] = "cpu"
    overwrite: bool = False
    methods: list[QuantizationMethod] = Field(
        default_factory=lambda: list(DEFAULT_COMMAND_AUDIT_METHODS),
        min_length=1,
    )


class ActivationAuditRequest(BaseModel):
    source_path: Path
    activation_calibration: Path
    output: Path
    device: Literal["cpu", "cuda"] = "cpu"
    methods: list[str] = Field(
        default_factory=lambda: list(MEASUREMENT_METHODS),
        min_length=1,
    )
    overwrite: bool = False


class ActivationScoreRequest(BaseModel):
    audit_cache: Path
    activation_calibration: Path
    metric: str = ACTIVATION_AUDIT_METRICS[0]


class ActivationInspectRequest(BaseModel):
    audit_cache: Path
    activation_calibration: Path
    tensor_name: str | None = None
    top_n: int = Field(default=5, ge=1, le=100)


class ActivationOptimizeRequest(BaseModel):
    audit_cache: Path
    activation_calibration: Path
    output_path: Path
    summary_output: Path | None = None
    source_path: Path | None = None
    target_size_gib: float | None = Field(default=None, gt=0)
    promotion_budget_mib: float | None = Field(default=None, ge=0)
    metric: str = ACTIVATION_AUDIT_METRICS[0]
    methods: list[str] | None = None
    baseline_method: str = "bf16"
    exclude_prefixes: list[str] = Field(default_factory=list)
    exclude_suffixes: list[str] = Field(default_factory=list)
    profile_id: str = Field(min_length=1)
    overwrite: bool = False
    dry_run: bool = False


class OptimizeRequest(BaseModel):
    audit_path: Path
    output_path: Path
    profile_id: str
    target_size_gib: float
    overwrite: bool = False
    dry_run: bool = False


class QuantizeRequest(BaseModel):
    source_path: Path
    output_path: Path | None = None
    profile: Path
    device: Literal["cpu", "cuda"] = "cpu"
    adapter_path: list[Path] = Field(default_factory=list)
    adapter_strength: list[float] = Field(default_factory=list)
    dry_run: bool = False
    timing: bool = False


class ExtractLoraRequest(BaseModel):
    base_path: Path
    tuned_path: Path
    output_path: Path
    ranks: list[Annotated[int, Field(ge=1, le=4096)]] = Field(
        default_factory=lambda: [64],
        min_length=1,
        max_length=8,
    )
    energy: float | None = Field(default=None, gt=0, le=1)
    split_fused: list[str] = Field(default_factory=list)
    include_diffs: bool = False
    include: str | None = None
    exclude: str | None = None
    dtype: Literal["fp16", "bf16", "fp32"] = "fp16"
    device: Literal["cpu", "cuda"] = "cpu"


class ComfyConnectionRequest(BaseModel):
    base_url: str = "http://127.0.0.1:8188"


class ComfyFileRequest(BaseModel):
    path: Path


class ComfyLaunchRequest(ComfyConnectionRequest):
    command: Path
    working_directory: Path
    timeout_seconds: int = Field(default=120, ge=1, le=600)


class ComfyBinding(BaseModel):
    dataset_field: str
    node_id: str
    class_type: str
    input_name: str


class ComfyRunRequest(ComfyConnectionRequest):
    workflow_path: Path
    dataset_path: Path
    bindings: list[ComfyBinding]
    final_workflow_path: Path | None = None


async def _run_workflow(action: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return await asyncio.to_thread(action, *args, **kwargs)
    except FileExistsError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def _with_default_suffix(path: Path, suffix: str) -> Path:
    return path if path.suffix else path.with_suffix(suffix)


def _prepare_quantize_request(
    request: QuantizeRequest,
    config_path: Path,
    *,
    check_existing_output: bool = False,
) -> None:
    if request.output_path is not None:
        request.output_path = _with_default_suffix(request.output_path, ".safetensors")
    if not is_in_library("model", request.source_path, config_path):
        raise HTTPException(
            status_code=422,
            detail="Model path must be inside a configured model folder.",
        )
    if not is_in_library("profile", request.profile, config_path):
        raise HTTPException(
            status_code=422,
            detail="Profile path must be inside a configured profile folder.",
        )
    if len(request.adapter_path) != len(request.adapter_strength):
        raise HTTPException(
            status_code=422,
            detail="LoRA files and strengths must be paired.",
        )
    if any(
        not is_in_library("lora", adapter_path, config_path)
        for adapter_path in request.adapter_path
    ):
        raise HTTPException(
            status_code=422,
            detail="LoRA path must be inside a configured LoRA folder.",
        )
    if request.output_path is not None and not is_in_library("output", request.output_path, config_path):
        raise HTTPException(
            status_code=422,
            detail="Output path must be inside a configured output folder.",
        )
    if check_existing_output and not request.dry_run and request.output_path is not None:
        if request.source_path.resolve() == request.output_path.resolve():
            raise HTTPException(
                status_code=422,
                detail="Source and Output path cannot be the same",
            )
        if request.output_path.exists():
            raise HTTPException(
                status_code=409,
                detail=f"Refusing to overwrite existing output: {request.output_path}",
            )
        partial = request.output_path.with_name(
            f"{request.output_path.name}.partial"
        )
        if partial.exists():
            raise HTTPException(
                status_code=409,
                detail=f"Partial output already exists: {partial}",
            )


def _prepare_extract_request(request: ExtractLoraRequest, config_path: Path) -> None:
    request.output_path = _with_default_suffix(request.output_path, ".safetensors")
    if len(request.ranks) > 1 and "{rank}" not in str(request.output_path):
        request.output_path = request.output_path.with_name(
            f"{request.output_path.stem}_r{{rank}}{request.output_path.suffix}"
        )
    for label, path in (("Base", request.base_path), ("Tuned", request.tuned_path)):
        if not is_in_library("model", path, config_path):
            raise HTTPException(
                status_code=422,
                detail=f"{label} checkpoint must be inside a configured model folder.",
            )
    if request.base_path.resolve() == request.tuned_path.resolve():
        raise HTTPException(
            status_code=422,
            detail="Base and tuned checkpoints must be different files.",
        )
    try:
        options = build_extraction_options(request)
        validate_extraction_options(options)
        for pattern in (request.include, request.exclude):
            if pattern:
                re.compile(pattern)
        outputs = resolve_output_paths(request.output_path, options)
    except (ValueError, re.error) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    for output in outputs.values():
        if not (
            is_in_library("output", output, config_path)
            or is_in_library("lora", output, config_path)
        ):
            raise HTTPException(
                status_code=422,
                detail="Output path must be inside a configured output or LoRA folder.",
            )
        if output.resolve() in (request.base_path.resolve(), request.tuned_path.resolve()):
            raise HTTPException(
                status_code=422,
                detail="Output must differ from the input checkpoints.",
            )
        if output.exists():
            raise HTTPException(
                status_code=409,
                detail=f"Refusing to overwrite existing output: {output}",
            )
        partial = output.with_name(f"{output.name}.partial")
        if partial.exists():
            raise HTTPException(
                status_code=409,
                detail=f"Partial output already exists: {partial}",
            )


def _prepare_optimize_request(request: OptimizeRequest, config_path: Path) -> None:
    request.output_path = _with_default_suffix(request.output_path, ".json")
    if not is_in_library("output", request.audit_path, config_path):
        raise HTTPException(
            status_code=422,
            detail="Audit report path must be inside a configured output folder.",
        )
    if not is_in_library("profile", request.output_path, config_path):
        raise HTTPException(
            status_code=422,
            detail="Profile path must be inside a configured profile folder.",
        )


def _prepare_audit_request(request: AuditRequest, config_path: Path) -> None:
    request.output = _with_default_suffix(request.output, ".json")
    if not is_in_library("model", request.source_path, config_path):
        raise HTTPException(
            status_code=422,
            detail="Model path must be inside a configured model folder.",
        )
    if not is_in_library("output", request.output, config_path):
        raise HTTPException(
            status_code=422,
            detail="Output path must be inside a configured output folder.",
        )


def _validate_activation_path(
    path: Path,
    config_path: Path,
    detail: str,
) -> None:
    if not is_in_library("activation", path, config_path):
        raise HTTPException(status_code=422, detail=detail)


def _validate_activation_cache_pair_output(
    output: Path,
    config_path: Path,
) -> tuple[Path, Path]:
    metadata_path, tensors_path = activation_audit_pair_paths(output)
    if not (
        is_in_library("activation", metadata_path, config_path)
        and is_in_library("activation", tensors_path, config_path)
    ):
        raise HTTPException(
            status_code=422,
            detail="Activation cache output must be inside a configured activation folder.",
        )
    return metadata_path, tensors_path


def _validate_activation_audit_paths(
    request: ActivationAuditRequest,
    config_path: Path,
    *,
    check_existing_output: bool = True,
) -> None:
    if not is_in_library("model", request.source_path, config_path):
        raise HTTPException(
            status_code=422,
            detail="Model path must be inside a configured model folder.",
        )
    _validate_activation_path(
        request.activation_calibration,
        config_path,
        "Activation calibration path must be inside a configured activation folder.",
    )
    metadata_path, tensors_path = _validate_activation_cache_pair_output(
        request.output,
        config_path,
    )
    if len(request.methods) != len(set(request.methods)):
        raise HTTPException(
            status_code=422,
            detail="Activation audit methods must be unique.",
        )
    if any(method not in MEASUREMENT_METHODS for method in request.methods):
        raise HTTPException(
            status_code=422,
            detail="Activation audit methods must be supported candidate methods.",
        )
    if check_existing_output and not request.overwrite:
        existing = next(
            (path for path in (metadata_path, tensors_path) if path.exists()),
            None,
        )
        if existing is not None:
            raise HTTPException(
                status_code=409,
                detail=f"Refusing to overwrite existing activation cache: {existing}",
            )


def _validate_activation_review_paths(
    audit_cache: Path,
    activation_calibration: Path,
    metric: str | None,
    config_path: Path,
) -> None:
    _validate_activation_path(
        audit_cache,
        config_path,
        "Activation audit cache path must be inside a configured activation folder.",
    )
    _validate_activation_path(
        activation_calibration,
        config_path,
        "Activation calibration path must be inside a configured activation folder.",
    )
    if metric is not None and metric not in ACTIVATION_AUDIT_METRICS:
        raise HTTPException(status_code=422, detail=f"Unsupported activation audit metric: {metric!r}")


def _prepare_activation_optimize_request(
    request: ActivationOptimizeRequest,
    config_path: Path,
) -> None:
    _validate_activation_review_paths(
        request.audit_cache,
        request.activation_calibration,
        request.metric,
        config_path,
    )
    request.output_path = _with_default_suffix(request.output_path, ".json")
    if request.summary_output is None:
        request.summary_output = request.output_path.with_name(
            f"{request.output_path.stem}-summary.json"
        )
    else:
        request.summary_output = _with_default_suffix(request.summary_output, ".json")
    if not is_in_library("profile", request.output_path, config_path):
        raise HTTPException(
            status_code=422,
            detail="Profile path must be inside a configured profile folder.",
        )
    assert request.summary_output is not None
    if not is_in_library("profile", request.summary_output, config_path):
        raise HTTPException(
            status_code=422,
            detail="Summary path must be inside a configured profile folder.",
        )
    if request.output_path.resolve() == request.summary_output.resolve():
        raise HTTPException(
            status_code=422,
            detail="Profile and summary output paths must be different.",
        )
    if request.source_path is not None and not is_in_library("model", request.source_path, config_path):
        raise HTTPException(
            status_code=422,
            detail="Source path must be inside a configured model folder.",
        )
    if (request.target_size_gib is None) == (request.promotion_budget_mib is None):
        raise HTTPException(
            status_code=422,
            detail="Provide exactly one target-size or promotion-budget option.",
        )
    if request.methods is not None:
        request.methods = [method.strip() for method in request.methods]
        if not request.methods or len(request.methods) != len(set(request.methods)):
            raise HTTPException(
                status_code=422,
                detail="Activation optimizer methods must be non-empty and unique.",
            )
    request.exclude_prefixes = [
        value.strip() for value in request.exclude_prefixes if value.strip()
    ]
    request.exclude_suffixes = [
        value.strip() for value in request.exclude_suffixes if value.strip()
    ]
    request.profile_id = request.profile_id.strip()
    if not request.profile_id:
        raise HTTPException(status_code=422, detail="profile_id must be non-empty.")
    if not request.overwrite:
        existing = next(
            (
                path
                for path in (request.output_path, request.summary_output)
                if path.exists()
            ),
            None,
        )
        if existing is not None:
            raise HTTPException(
                status_code=409,
                detail=f"Refusing to overwrite existing output: {existing}",
            )


def _activation_method_options(methods: list[str] | tuple[str, ...]) -> list[dict[str, str]]:
    return [
        {"value": method, "label": get_conversion_method(METHOD_ACTIONS[method]).label}
        for method in methods
    ]


def _register_folder_routes(app: FastAPI, kind: str, config_path: Path) -> None:
    route = f"/api/{kind}-folders"

    def folder_list() -> dict[str, list[str]]:
        return {"folders": [str(folder) for folder in load_library_folders(kind, config_path)]}

    def change(update: Any, request: ModelFolderRequest) -> dict[str, list[str]]:
        try:
            folders = update(kind, request.path, config_path)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {"folders": [str(folder) for folder in folders]}

    def add(request: ModelFolderRequest) -> dict[str, list[str]]:
        return change(add_library_folder, request)

    def remove(request: ModelFolderRequest) -> dict[str, list[str]]:
        return change(remove_library_folder, request)

    app.get(route, name=f"{kind}_folders")(folder_list)
    app.post(route, status_code=201, name=f"add_{kind}_folder")(add)
    app.delete(route, name=f"remove_{kind}_folder")(remove)


def create_app(
    config_path: Path = DEFAULT_CONFIG_PATH,
    static_dir: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="PotatoForge")
    audit_jobs = AuditJobManager()
    activation_jobs = ActivationAuditJobManager()
    quantize_jobs = QuantizeJobManager()
    extract_jobs = ExtractJobManager()
    comfy_jobs = ComfyRunner(config_path.with_name("comfy_runner_state.json"))
    app.state.audit_jobs = audit_jobs
    app.state.activation_jobs = activation_jobs
    app.state.quantize_jobs = quantize_jobs
    app.state.extract_jobs = extract_jobs
    app.state.comfy_jobs = comfy_jobs

    async def comfy_preflight_check(request: ComfyRunRequest) -> dict[str, Any]:
        workflow, _ = load_json(request.workflow_path)
        dataset, _ = load_json(request.dataset_path)
        bindings = [binding.model_dump() for binding in request.bindings]
        validate_bindings(workflow, dataset, bindings)
        async with ComfyClient(request.base_url) as client:
            await client.get("/system_stats")
            definitions: dict[str, Any] = {}
            for class_type in {binding.class_type for binding in request.bindings}:
                response = await client.http.get(f"/object_info/{quote(class_type, safe='')}")
                if response.status_code == 200:
                    definitions.update(response.json())
            validate_bindings(workflow, dataset, bindings, definitions)
        if request.final_workflow_path is not None:
            final_workflow, _ = load_json(request.final_workflow_path)
            inspect_workflow(final_workflow)
        return {"valid": True, "run_count": len(dataset["runs"]),
                "checked_classes": list(definitions)}

    @app.post("/api/comfy/connect")
    async def comfy_connect(request: ComfyConnectionRequest) -> dict[str, Any]:
        try:
            async with ComfyClient(request.base_url) as client:
                system = await client.get("/system_stats")
                jobs = await client.http.get("/api/jobs", params={"limit": 1})
                return {"connected": True, "base_url": client.base_url, "system": system,
                        "capabilities": {"jobs_api": jobs.status_code == 200,
                                         "targeted_cancel": jobs.status_code == 200}}
        except (ValueError, httpx.HTTPError) as error:
            raise HTTPException(status_code=422, detail=f"Could not connect to ComfyUI: {error}") from error

    @app.post("/api/comfy/launch")
    async def comfy_launch(request: ComfyLaunchRequest, client_request: Request) -> dict[str, Any]:
        try:
            local = client_request.client is not None and ipaddress.ip_address(client_request.client.host).is_loopback
        except ValueError:
            local = False
        if not local:
            raise HTTPException(status_code=403, detail="Comfy launch is available only to local clients.")
        try:
            return await launch_comfy(request.base_url, request.command,
                                      request.working_directory, request.timeout_seconds)
        except (ValueError, RuntimeError, TimeoutError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/comfy/workflows/inspect")
    def comfy_workflow_inspect(request: ComfyFileRequest) -> dict[str, Any]:
        try:
            workflow, _ = load_json(request.path)
            return {"path": str(request.path), "nodes": inspect_workflow(workflow)}
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/comfy/json-files")
    def comfy_json_files(folder: Path) -> dict[str, Any]:
        try:
            if not folder.is_dir():
                raise ValueError(f"JSON folder does not exist: {folder}")
            files = sorted((path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".json"),
                           key=lambda path: path.name.casefold())
            return {"files": [{"name": path.name, "path": str(path)} for path in files[:500]]}
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/comfy/datasets/inspect")
    def comfy_dataset_inspect(request: ComfyFileRequest) -> dict[str, Any]:
        try:
            dataset, _ = load_json(request.path)
            return {"path": str(request.path), **inspect_dataset(dataset)}
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/comfy/runs", status_code=202)
    async def comfy_start_run(request: ComfyRunRequest) -> dict[str, Any]:
        try:
            await comfy_preflight_check(request)
            return {"job": comfy_jobs.start_run(request.base_url, request.workflow_path,
                request.dataset_path, [binding.model_dump() for binding in request.bindings],
                request.final_workflow_path)}
        except ActiveJobError as error:
            raise HTTPException(status_code=409, detail="A Comfy Runner job is already active.") from error
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except httpx.HTTPError as error:
            raise HTTPException(status_code=502, detail=f"ComfyUI preflight failed: {error}") from error

    @app.post("/api/comfy/preflight")
    async def comfy_preflight(request: ComfyRunRequest) -> dict[str, Any]:
        try:
            return await comfy_preflight_check(request)
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except httpx.HTTPError as error:
            raise HTTPException(status_code=502, detail=f"ComfyUI preflight failed: {error}") from error

    @app.get("/api/comfy/runs/current")
    def comfy_current_run() -> dict[str, Any]:
        return {"job": comfy_jobs.current()}

    @app.delete("/api/comfy/runs/{job_id}")
    def comfy_cancel_run(job_id: str) -> dict[str, Any]:
        try:
            return {"job": comfy_jobs.cancel(job_id)}
        except UnknownJobError as error:
            raise HTTPException(status_code=404, detail="Comfy Runner job not found.") from error
        except InactiveJobError as error:
            raise HTTPException(status_code=409, detail="Comfy Runner job is no longer active.") from error

    @app.get("/api/comfy/runs/{job_id}/events")
    async def comfy_run_events(job_id: str) -> StreamingResponse:
        try:
            subscriber = comfy_jobs.subscribe(job_id)
        except UnknownJobError as error:
            raise HTTPException(status_code=404, detail="Comfy Runner job not found.") from error
        return StreamingResponse(comfy_jobs.stream(subscriber), media_type="text/event-stream")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    for kind in FOLDER_KINDS:
        _register_folder_routes(app, kind, config_path)

    @app.get("/api/models")
    def models() -> dict[str, list[dict[str, str | bool]]]:
        return {"models": discover_models(config_path)}

    @app.get("/api/loras")
    def loras() -> dict[str, list[dict[str, str]]]:
        return {"loras": discover_loras(config_path)}

    @app.get("/api/profiles")
    def profiles() -> dict[str, list[dict[str, str]]]:
        return {"profiles": discover_profiles(config_path)}

    @app.get("/api/profile-details")
    def profile_details(path: Path) -> dict[str, Any]:
        if not is_in_library("profile", path, config_path):
            raise HTTPException(
                status_code=422,
                detail="Profile path must be inside a configured profile folder.",
            )
        profile = load_profile(path)
        return {
            "profile_id": profile["profile_id"],
            "default": profile["default"],
            "rules": profile["rules"],
        }

    @app.get("/api/audit-reports")
    def audit_reports() -> dict[str, list[dict[str, str]]]:
        return {"reports": discover_audit_reports(config_path)}

    @app.get("/api/audit-methods")
    def audit_methods() -> dict[str, list[dict[str, str]] | list[str]]:
        return {
            "methods": audit_method_options(),
            "default_methods": list(DEFAULT_COMMAND_AUDIT_METHODS),
        }

    @app.get("/api/activation-calibrations")
    def activation_calibrations() -> dict[str, list[dict[str, object]]]:
        return {"calibrations": discover_activation_calibrations(config_path)}

    @app.get("/api/activation-audit-caches")
    def activation_audit_caches() -> dict[str, list[dict[str, str]]]:
        return {"caches": discover_activation_audit_caches(config_path)}

    @app.get("/api/activation-methods")
    def activation_methods() -> dict[str, list[dict[str, str]] | list[str]]:
        return {
            "methods": _activation_method_options(MEASUREMENT_METHODS),
            "default_methods": list(MEASUREMENT_METHODS),
        }

    @app.get("/api/activation-metrics")
    def activation_metrics() -> dict[str, object]:
        return {
            "metrics": [
                {"value": metric, "label": metric.replace("_", " ")}
                for metric in ACTIVATION_AUDIT_METRICS
            ],
            "default_metric": ACTIVATION_AUDIT_METRICS[0],
        }

    @app.get("/api/activation-cache-details")
    def activation_cache_details(path: Path) -> dict[str, object]:
        _validate_activation_path(
            path,
            config_path,
            "Activation audit cache path must be inside a configured activation folder.",
        )
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise HTTPException(status_code=422, detail="Invalid activation audit cache.") from error
        methods = document.get("requested_methods") if isinstance(document, dict) else None
        if not isinstance(methods, list) or any(
            not isinstance(method, str) or method not in MEASUREMENT_METHODS
            for method in methods
        ):
            raise HTTPException(status_code=422, detail="Invalid activation audit cache methods.")
        return {
            "methods": _activation_method_options(tuple(methods)),
            "default_methods": methods,
            "baseline_methods": ["bf16", *methods],
        }

    @app.post("/api/inspect")
    async def inspect(request: InspectRequest) -> CommandResult:
        if request.output is not None:
            request.output = _with_default_suffix(request.output, ".json")
        if not is_in_library("model", request.model_path, config_path):
            raise HTTPException(
                status_code=422,
                detail="Model path must be inside a configured model folder.",
            )
        if request.output is not None and not is_in_library("output", request.output, config_path
        ):
            raise HTTPException(
                status_code=422,
                detail="Report output path must be inside a configured output folder.",
            )
        result = await _run_workflow(
            inspect_model,
            request.model_path,
            output=request.output,
            overwrite=request.overwrite,
        )
        return CommandResult(result=result.report)

    @app.post("/api/audit")
    async def audit(request: AuditRequest) -> CommandResult:
        _prepare_audit_request(request, config_path)
        result = await _run_workflow(
            audit_model,
            request.source_path,
            request.output,
            device=request.device,
            overwrite=request.overwrite,
            methods=request.methods,
        )
        return CommandResult(result=audit_result_payload(request, result))

    @app.post("/api/audit/jobs", status_code=202)
    async def start_audit_job(request: AuditRequest) -> dict[str, Any]:
        _prepare_audit_request(request, config_path)
        try:
            job = audit_jobs.start(request, audit_model)
        except ActiveJobError as error:
            raise HTTPException(
                status_code=409,
                detail="An Audit job is already active.",
            ) from error
        return {"job": job}

    @app.get("/api/audit/jobs/current")
    def current_audit_job() -> dict[str, Any]:
        return {"job": audit_jobs.current()}

    @app.get("/api/audit/jobs/{job_id}/events")
    async def audit_job_events(job_id: str) -> StreamingResponse:
        try:
            subscriber = audit_jobs.subscribe(job_id)
        except UnknownJobError as error:
            raise HTTPException(status_code=404, detail="Audit job not found.") from error
        return StreamingResponse(
            audit_jobs.stream(subscriber),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.delete("/api/audit/jobs/{job_id}")
    def cancel_audit_job(job_id: str) -> dict[str, Any]:
        try:
            return {"job": audit_jobs.cancel(job_id)}
        except UnknownJobError as error:
            raise HTTPException(status_code=404, detail="Audit job not found.") from error
        except InactiveJobError as error:
            raise HTTPException(status_code=409, detail="Audit job is no longer active.") from error

    @app.post("/api/activation/audit/jobs", status_code=202)
    async def start_activation_audit_job(
        request: ActivationAuditRequest,
    ) -> dict[str, Any]:
        _validate_activation_audit_paths(request, config_path)
        try:
            job = activation_jobs.start(request, activation_audit_model)
        except ActiveJobError as error:
            raise HTTPException(
                status_code=409,
                detail="An activation audit job is already active.",
            ) from error
        return {"job": job}

    @app.get("/api/activation/audit/jobs/current")
    def current_activation_audit_job() -> dict[str, Any]:
        return {"job": activation_jobs.current()}

    @app.get("/api/activation/audit/jobs/{job_id}/events")
    async def activation_audit_job_events(job_id: str) -> StreamingResponse:
        try:
            subscriber = activation_jobs.subscribe(job_id)
        except UnknownJobError as error:
            raise HTTPException(
                status_code=404,
                detail="Activation audit job not found.",
            ) from error
        return StreamingResponse(
            activation_jobs.stream(subscriber),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.delete("/api/activation/audit/jobs/{job_id}")
    def cancel_activation_audit_job(job_id: str) -> dict[str, Any]:
        try:
            return {"job": activation_jobs.cancel(job_id)}
        except UnknownJobError as error:
            raise HTTPException(
                status_code=404,
                detail="Activation audit job not found.",
            ) from error
        except InactiveJobError as error:
            raise HTTPException(
                status_code=409,
                detail="Activation audit job is no longer active.",
            ) from error

    @app.post("/api/activation/score")
    async def score_activation(request: ActivationScoreRequest) -> CommandResult:
        _validate_activation_review_paths(
            request.audit_cache,
            request.activation_calibration,
            request.metric,
            config_path,
        )
        result = await _run_workflow(
            score_activation_audit,
            request.audit_cache,
            request.activation_calibration,
            metric=request.metric,
        )
        return CommandResult(result=result)

    @app.post("/api/activation/inspect")
    async def inspect_activation(request: ActivationInspectRequest) -> CommandResult:
        _validate_activation_review_paths(
            request.audit_cache,
            request.activation_calibration,
            None,
            config_path,
        )
        result = await _run_workflow(
            inspect_activation_audit,
            request.audit_cache,
            request.activation_calibration,
            tensor_name=request.tensor_name,
            top_n=request.top_n,
        )
        return CommandResult(result=result)

    @app.post("/api/activation/optimize")
    async def optimize_activation(
        request: ActivationOptimizeRequest,
    ) -> CommandResult:
        _prepare_activation_optimize_request(request, config_path)
        result = await _run_workflow(
            optimize_activation_profile,
            request.audit_cache,
            request.activation_calibration,
            request.output_path,
            summary_output=request.summary_output,
            source_path=request.source_path,
            target_size_gib=request.target_size_gib,
            promotion_budget_mib=request.promotion_budget_mib,
            metric=request.metric,
            methods=request.methods,
            baseline_method=request.baseline_method,
            exclude_prefixes=request.exclude_prefixes,
            exclude_suffixes=request.exclude_suffixes,
            profile_id=request.profile_id,
            overwrite=request.overwrite,
            dry_run=request.dry_run,
        )
        return CommandResult(
            result={
                "audit_cache": str(result.audit_cache),
                "profile_path": None if result.dry_run else str(result.profile_path),
                "summary_path": None if result.dry_run else str(result.summary_path),
                "profile_id": result.profile_id,
                "metric": result.metric,
                "target_bytes": result.target_bytes,
                "promotion_budget_bytes": result.promotion_budget_bytes,
                "estimated_output_bytes": result.estimated_output_bytes,
                "dry_run": result.dry_run,
                "warning": (
                    "Activation measurements guide profile selection; generated "
                    "artifacts and runtime output still need validation."
                ),
            }
        )

    @app.post("/api/optimize")
    async def optimize(request: OptimizeRequest) -> CommandResult:
        _prepare_optimize_request(request, config_path)
        result = await _run_workflow(
            optimize_profile,
            request.audit_path,
            request.output_path,
            profile_id=request.profile_id,
            target_size_gib=request.target_size_gib,
            overwrite=request.overwrite,
            dry_run=request.dry_run,
        )
        return CommandResult(
            result={
                "audit_path": str(result.audit_path),
                "profile_path": None if result.dry_run else str(result.output_path),
                "profile_id": result.profile_id,
                "target_bytes": result.optimized.target_bytes,
                "estimated_output_bytes": result.optimized.output_bytes,
                "dry_run": result.dry_run,
                "warning": "Generated profiles still require artifact and runtime validation.",
            }
        )

    @app.post("/api/quantize")
    async def quantize(request: QuantizeRequest) -> CommandResult:
        _prepare_quantize_request(request, config_path)
        result = await _run_workflow(
            quantize_model,
            request.source_path,
            request.output_path,
            profile=request.profile,
            device=request.device,
            adapter_path=request.adapter_path,
            adapter_strength=request.adapter_strength,
            dry_run=request.dry_run,
            timing=request.timing,
        )
        return CommandResult(result=quantize_result_payload(result, timing=request.timing))

    @app.post("/api/quantize/jobs", status_code=202)
    async def start_quantize_job(request: QuantizeRequest) -> dict[str, Any]:
        _prepare_quantize_request(request, config_path, check_existing_output=True)
        try:
            job = quantize_jobs.start(request, quantize_model)
        except ActiveJobError as error:
            raise HTTPException(
                status_code=409,
                detail="A Quantize job is already active.",
            ) from error
        return {"job": job}

    @app.get("/api/quantize/jobs/current")
    def current_quantize_job() -> dict[str, Any]:
        return {"job": quantize_jobs.current()}

    @app.delete("/api/quantize/jobs/{job_id}")
    def cancel_quantize_job(job_id: str) -> dict[str, Any]:
        try:
            return {"job": quantize_jobs.cancel(job_id)}
        except UnknownJobError as error:
            raise HTTPException(status_code=404, detail="Quantize job not found.") from error
        except InactiveJobError as error:
            raise HTTPException(status_code=409, detail="Quantize job is no longer active.") from error

    @app.get("/api/quantize/jobs/{job_id}/events")
    async def quantize_job_events(job_id: str) -> StreamingResponse:
        try:
            subscriber = quantize_jobs.subscribe(job_id)
        except UnknownJobError as error:
            raise HTTPException(status_code=404, detail="Quantize job not found.") from error
        return StreamingResponse(
            quantize_jobs.stream(subscriber),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/lora/extract/jobs", status_code=202)
    async def start_extract_job(request: ExtractLoraRequest) -> dict[str, Any]:
        _prepare_extract_request(request, config_path)
        try:
            job = extract_jobs.start(request, extract_lora)
        except ActiveJobError as error:
            raise HTTPException(
                status_code=409,
                detail="A LoRA extraction job is already active.",
            ) from error
        return {"job": job}

    @app.get("/api/lora/extract/jobs/current")
    def current_extract_job() -> dict[str, Any]:
        return {"job": extract_jobs.current()}

    @app.delete("/api/lora/extract/jobs/{job_id}")
    def cancel_extract_job(job_id: str) -> dict[str, Any]:
        try:
            return {"job": extract_jobs.cancel(job_id)}
        except UnknownJobError as error:
            raise HTTPException(status_code=404, detail="Extract job not found.") from error
        except InactiveJobError as error:
            raise HTTPException(status_code=409, detail="Extract job is no longer active.") from error

    @app.get("/api/lora/extract/jobs/{job_id}/events")
    async def extract_job_events(job_id: str) -> StreamingResponse:
        try:
            subscriber = extract_jobs.subscribe(job_id)
        except UnknownJobError as error:
            raise HTTPException(status_code=404, detail="Extract job not found.") from error
        return StreamingResponse(
            extract_jobs.stream(subscriber),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if static_dir is not None:
        web_root = static_dir.resolve()
        index_path = web_root / "index.html"

        @app.get("/{path:path}", include_in_schema=False)
        def frontend(path: str) -> FileResponse:
            if path == "api" or path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not found")
            requested_path = (web_root / path).resolve()
            if requested_path.is_file() and requested_path.is_relative_to(web_root):
                return FileResponse(requested_path)
            return FileResponse(index_path)

    return app


app = create_app()
