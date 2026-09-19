"""FastAPI boundary over PotatoForge application workflows."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..application.commands import (
    audit_model,
    inspect_model,
    optimize_profile,
    quantize_model,
)
from ..audits.weight_audit import (
    DEFAULT_COMMAND_AUDIT_METHODS,
    QuantizationMethod,
    audit_method_options,
)
from ..profiles import load_profile
from .model_library import (
    DEFAULT_CONFIG_PATH,
    add_folder,
    add_lora_folder,
    add_output_folder,
    add_profile_folder,
    discover_audit_reports,
    discover_loras,
    discover_models,
    discover_profiles,
    is_configured_model,
    is_configured_lora,
    is_configured_output,
    is_configured_profile,
    load_folders,
    load_lora_folders,
    load_output_folders,
    load_profile_folders,
    remove_folder,
    remove_lora_folder,
    remove_output_folder,
    remove_profile_folder,
)
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


def create_app(
    config_path: Path = DEFAULT_CONFIG_PATH,
    static_dir: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="PotatoForge")
    audit_jobs = AuditJobManager()
    quantize_jobs = QuantizeJobManager()
    app.state.audit_jobs = audit_jobs
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
