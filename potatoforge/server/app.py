"""FastAPI boundary over PotatoForge application workflows."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..audits.activation_audit import (
    ACTIVATION_AUDIT_METRICS,
    activation_audit_pair_paths,
)
from ..audits.activation_measurements import MEASUREMENT_METHODS, METHOD_ACTIONS
from ..application.commands import (
    activation_audit_model,
    audit_model,
    inspect_activation_cache,
    inspect_model,
    optimize_activation_profile,
    optimize_profile,
    quantize_model,
    score_activation_cache,
)
from ..audits.weight_audit import (
    DEFAULT_COMMAND_AUDIT_METHODS,
    QuantizationMethod,
    audit_method_options,
)
from ..conversion_methods import get_conversion_method
from ..profiles import load_profile
from .model_library import (
    DEFAULT_CONFIG_PATH,
    add_activation_folder,
    add_folder,
    add_lora_folder,
    add_output_folder,
    add_profile_folder,
    discover_audit_reports,
    discover_activation_audit_caches,
    discover_activation_calibrations,
    discover_loras,
    discover_models,
    discover_profiles,
    is_configured_activation,
    is_configured_model,
    is_configured_lora,
    is_configured_output,
    is_configured_profile,
    load_activation_folders,
    load_folders,
    load_lora_folders,
    load_output_folders,
    load_profile_folders,
    remove_activation_folder,
    remove_folder,
    remove_lora_folder,
    remove_output_folder,
    remove_profile_folder,
)
from .activation_jobs import ActivationAuditJobManager
from .audit_jobs import AuditJobManager
from .jobs import ActiveJobError, InactiveJobError, UnknownJobError
from .quantize_jobs import QuantizeJobManager


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


async def _run_workflow(action: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return await asyncio.to_thread(action, *args, **kwargs)
    except FileExistsError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def _with_default_suffix(path: Path, suffix: str) -> Path:
    return path if path.suffix else path.with_suffix(suffix)


def _validate_quantize_paths(
    request: QuantizeRequest,
    config_path: Path,
    *,
    check_existing_output: bool = False,
) -> None:
    if request.output_path is not None:
        request.output_path = _with_default_suffix(request.output_path, ".safetensors")
    if not is_configured_model(request.source_path, config_path):
        raise HTTPException(
            status_code=422,
            detail="Model path must be inside a configured model folder.",
        )
    if not is_configured_profile(request.profile, config_path):
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
        not is_configured_lora(adapter_path, config_path)
        for adapter_path in request.adapter_path
    ):
        raise HTTPException(
            status_code=422,
            detail="LoRA path must be inside a configured LoRA folder.",
        )
    if request.output_path is not None and not is_configured_output(
        request.output_path,
        config_path,
    ):
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


def _validate_optimize_paths(request: OptimizeRequest, config_path: Path) -> None:
    request.output_path = _with_default_suffix(request.output_path, ".json")
    if not is_configured_output(request.audit_path, config_path):
        raise HTTPException(
            status_code=422,
            detail="Audit report path must be inside a configured output folder.",
        )
    if not is_configured_profile(request.output_path, config_path):
        raise HTTPException(
            status_code=422,
            detail="Profile path must be inside a configured profile folder.",
        )


def _validate_audit_paths(request: AuditRequest, config_path: Path) -> None:
    request.output = _with_default_suffix(request.output, ".json")
    if not is_configured_model(request.source_path, config_path):
        raise HTTPException(
            status_code=422,
            detail="Model path must be inside a configured model folder.",
        )
    if not is_configured_output(request.output, config_path):
        raise HTTPException(
            status_code=422,
            detail="Output path must be inside a configured output folder.",
        )


def _validate_activation_path(
    path: Path,
    config_path: Path,
    detail: str,
) -> None:
    if not is_configured_activation(path, config_path):
        raise HTTPException(status_code=422, detail=detail)


def _validate_activation_cache_pair_output(
    output: Path,
    config_path: Path,
) -> tuple[Path, Path]:
    metadata_path, tensors_path = activation_audit_pair_paths(output)
    if not (
        is_configured_activation(metadata_path, config_path)
        and is_configured_activation(tensors_path, config_path)
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
    if not is_configured_model(request.source_path, config_path):
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


def _validate_activation_optimize_paths(
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
    if not is_configured_profile(request.output_path, config_path):
        raise HTTPException(
            status_code=422,
            detail="Profile path must be inside a configured profile folder.",
        )
    assert request.summary_output is not None
    if not is_configured_profile(request.summary_output, config_path):
        raise HTTPException(
            status_code=422,
            detail="Summary path must be inside a configured profile folder.",
        )
    if request.output_path.resolve() == request.summary_output.resolve():
        raise HTTPException(
            status_code=422,
            detail="Profile and summary output paths must be different.",
        )
    if request.source_path is not None and not is_configured_model(
        request.source_path,
        config_path,
    ):
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


def create_app(
    config_path: Path = DEFAULT_CONFIG_PATH,
    static_dir: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="PotatoForge")
    audit_jobs = AuditJobManager()
    activation_jobs = ActivationAuditJobManager()
    quantize_jobs = QuantizeJobManager()
    app.state.audit_jobs = audit_jobs
    app.state.activation_jobs = activation_jobs
    app.state.quantize_jobs = quantize_jobs

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/model-folders")
    def model_folders() -> dict[str, list[str]]:
        return {"folders": [str(folder) for folder in load_folders(config_path)]}

    @app.post("/api/model-folders", status_code=201)
    def add_model_folder(request: ModelFolderRequest) -> dict[str, list[str]]:
        folders = add_folder(request.path, config_path)
        return {"folders": [str(folder) for folder in folders]}

    @app.delete("/api/model-folders")
    def remove_model_folder(request: ModelFolderRequest) -> dict[str, list[str]]:
        folders = remove_folder(request.path, config_path)
        return {"folders": [str(folder) for folder in folders]}

    @app.get("/api/activation-folders")
    def activation_folder_list() -> dict[str, list[str]]:
        return {
            "folders": [str(folder) for folder in load_activation_folders(config_path)]
        }

    @app.post("/api/activation-folders", status_code=201)
    def add_configured_activation_folder(
        request: ModelFolderRequest,
    ) -> dict[str, list[str]]:
        try:
            folders = add_activation_folder(request.path, config_path)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {"folders": [str(folder) for folder in folders]}

    @app.delete("/api/activation-folders")
    def remove_configured_activation_folder(
        request: ModelFolderRequest,
    ) -> dict[str, list[str]]:
        try:
            folders = remove_activation_folder(request.path, config_path)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {"folders": [str(folder) for folder in folders]}

    @app.get("/api/models")
    def models() -> dict[str, list[dict[str, str | bool]]]:
        return {"models": discover_models(config_path)}

    @app.get("/api/lora-folders")
    def lora_folders() -> dict[str, list[str]]:
        return {"folders": [str(folder) for folder in load_lora_folders(config_path)]}

    @app.post("/api/lora-folders", status_code=201)
    def add_configured_lora_folder(
        request: ModelFolderRequest,
    ) -> dict[str, list[str]]:
        folders = add_lora_folder(request.path, config_path)
        return {"folders": [str(folder) for folder in folders]}

    @app.delete("/api/lora-folders")
    def remove_configured_lora_folder(
        request: ModelFolderRequest,
    ) -> dict[str, list[str]]:
        folders = remove_lora_folder(request.path, config_path)
        return {"folders": [str(folder) for folder in folders]}

    @app.get("/api/loras")
    def loras() -> dict[str, list[dict[str, str]]]:
        return {"loras": discover_loras(config_path)}

    @app.get("/api/profile-folders")
    def profile_folders() -> dict[str, list[str]]:
        return {
            "folders": [str(folder) for folder in load_profile_folders(config_path)]
        }

    @app.post("/api/profile-folders", status_code=201)
    def add_configured_profile_folder(
        request: ModelFolderRequest,
    ) -> dict[str, list[str]]:
        folders = add_profile_folder(request.path, config_path)
        return {"folders": [str(folder) for folder in folders]}

    @app.delete("/api/profile-folders")
    def remove_configured_profile_folder(
        request: ModelFolderRequest,
    ) -> dict[str, list[str]]:
        folders = remove_profile_folder(request.path, config_path)
        return {"folders": [str(folder) for folder in folders]}

    @app.get("/api/profiles")
    def profiles() -> dict[str, list[dict[str, str]]]:
        return {"profiles": discover_profiles(config_path)}

    @app.get("/api/profile-details")
    def profile_details(path: Path) -> dict[str, Any]:
        if not is_configured_profile(path, config_path):
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

    @app.get("/api/output-folders")
    def output_folders() -> dict[str, list[str]]:
        return {
            "folders": [str(folder) for folder in load_output_folders(config_path)]
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
    def activation_calibrations() -> dict[str, list[dict[str, str]]]:
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

    @app.post("/api/output-folders", status_code=201)
    def add_configured_output_folder(
        request: ModelFolderRequest,
    ) -> dict[str, list[str]]:
        folders = add_output_folder(request.path, config_path)
        return {"folders": [str(folder) for folder in folders]}

    @app.delete("/api/output-folders")
    def remove_configured_output_folder(
        request: ModelFolderRequest,
    ) -> dict[str, list[str]]:
        folders = remove_output_folder(request.path, config_path)
        return {"folders": [str(folder) for folder in folders]}

    @app.post("/api/inspect")
    async def inspect(request: InspectRequest) -> CommandResult:
        if request.output is not None:
            request.output = _with_default_suffix(request.output, ".json")
        if not is_configured_model(request.model_path, config_path):
            raise HTTPException(
                status_code=422,
                detail="Model path must be inside a configured model folder.",
            )
        if request.output is not None and not is_configured_output(
            request.output, config_path
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
        _validate_audit_paths(request, config_path)
        result = await _run_workflow(
            audit_model,
            request.source_path,
            request.output,
            device=request.device,
            overwrite=request.overwrite,
            methods=request.methods,
        )
        summary = result.document["summary"]
        return CommandResult(
            result={
                "source_path": str(request.source_path),
                "audit_path": str(request.output),
                "audited_layer_count": summary["audited_layer_count"],
                "skipped_tensor_count": summary["skipped_tensor_count"],
                "warning": "Weight reconstruction only; not runtime speed or image quality.",
            }
        )

    @app.post("/api/audit/jobs", status_code=202)
    async def start_audit_job(request: AuditRequest) -> dict[str, Any]:
        _validate_audit_paths(request, config_path)
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
            score_activation_cache,
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
            inspect_activation_cache,
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
        _validate_activation_optimize_paths(request, config_path)
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
        _validate_optimize_paths(request, config_path)
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
        _validate_quantize_paths(request, config_path)
        result = await _run_workflow(
            quantize_model,
            request.source_path,
            request.output_path,
            profile=request.profile,
            device=request.device,
            adapter_path=request.adapter_path,
            adapter_strength=request.adapter_strength,
            dry_run=request.dry_run,
        )
        if result.estimated_bytes is not None:
            return CommandResult(
                result={
                    "estimated_output_bytes": result.estimated_bytes,
                    "dry_run": True,
                }
            )
        assert result.output_path is not None
        return CommandResult(
            result={
                "output_path": str(result.output_path),
                "output_bytes": result.output_path.stat().st_size,
                "adapter_count": result.adapter_count,
            }
        )

    @app.post("/api/quantize/jobs", status_code=202)
    async def start_quantize_job(request: QuantizeRequest) -> dict[str, Any]:
        _validate_quantize_paths(request, config_path, check_existing_output=True)
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
