"""Application workflows for the commands exposed by the local UI."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from ..audits.activation_audit import (
    ACTIVATION_AUDIT_METRICS,
    activation_audit_pair_paths,
    inspect_activation_audit,
    run_activation_audit,
    score_activation_audit,
)
from ..audits.activation_measurements import MEASUREMENT_METHODS
from ..audits.activation_profiles import generate_activation_cache_profile
from ..audits.profile_optimizer import (
    SUPPORTED_METHODS,
    audited_global_relative_l2,
    load_weight_audit,
    optimize_target_size,
    write_profile,
)
from ..audits.weight_audit import audit_bf16_source
from ..calibration import ActivationCalibration
from ..config import load_optimize_config, load_quantize_config
from ..converter import convert_model_from_profile, estimate_output_bytes
from ..headers.header_reader import read_header_from_safetensors
from ..lora.lora_merge import AdapterMergeInput
from ..planning import (
    QUANTIZATION_LAYERS_METADATA_KEY,
    QUANTIZATION_METADATA_KEY,
    PlanEntry,
    parse_quantization_layers,
)
from ..timing import TimingCollector


def write_json(path: Path, document: object, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8") as output:
        json.dump(document, output, indent=2)
        output.write("\n")


def preflight_output(path: Path, overwrite: bool) -> None:
    if not overwrite and path.exists():
        raise FileExistsError(f"Output already exists: {path}")


def gib_to_bytes(value: float) -> int:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("must be a finite positive number")
    byte_count = int(value * (1024**3))
    if byte_count <= 0:
        raise ValueError("must represent at least one byte")
    return byte_count


def mib_to_bytes(value: float, *, allow_zero: bool = False) -> int:
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError("must be a finite positive number")
    byte_count = int(value * (1024**2))
    if byte_count <= 0 and not allow_zero:
        raise ValueError("must represent at least one byte")
    return byte_count


def split_comma_separated(value: str | None) -> tuple[str, ...]:
    return () if value is None else tuple(part.strip() for part in value.split(","))


def build_adapter_inputs(
    paths: Iterable[Path], strengths: Iterable[float]
) -> tuple[AdapterMergeInput, ...]:
    adapter_paths = tuple(paths)
    adapter_strengths = tuple(strengths)
    if len(adapter_paths) != len(adapter_strengths):
        raise ValueError(
            "--adapter-path and --adapter-strength must be repeated the same number of times."
        )
    return tuple(
        AdapterMergeInput(path=path, strength=strength)
        for path, strength in zip(adapter_paths, adapter_strengths)
    )


def parse_quantization_metadata(metadata: object) -> tuple[str, dict[str, str]]:
    if not isinstance(metadata, dict):
        return "unavailable", {}
    summary = metadata.get(QUANTIZATION_METADATA_KEY)
    if not isinstance(summary, str):
        summary = "unavailable"
    raw_layers = metadata.get(QUANTIZATION_LAYERS_METADATA_KEY)
    if raw_layers is None:
        return summary, {}
    if not isinstance(raw_layers, str):
        return "invalid", {}
    try:
        return summary, parse_quantization_layers(raw_layers)
    except ValueError:
        return "invalid", {}


@dataclass(frozen=True)
class InspectionResult:
    report: dict[str, Any]
    quantization_summary: str
    quantization_layers: dict[str, str]


def inspect_model(
    model_path: Path,
    *,
    output: Path | None = None,
    overwrite: bool = False,
    read_header: Callable[[Path], dict[str, Any]] = read_header_from_safetensors,
    write_document: Callable[[Path, object, bool], None] = write_json,
) -> InspectionResult:
    header = read_header(model_path)
    metadata = header.get("__metadata__", {})
    tensors = {name: descriptor for name, descriptor in header.items() if name != "__metadata__"}
    total_bytes = sum(
        descriptor["data_offsets"][1] - descriptor["data_offsets"][0]
        for descriptor in tensors.values()
    )
    quantization_summary, quantization_layers = parse_quantization_metadata(metadata)
    report = {
        "source_path": str(model_path),
        "metadata": metadata,
        "model_info": tensors,
        "quantization": {"summary": quantization_summary, "layers": quantization_layers},
        "summary": {
            "metadata_count": len(metadata) if isinstance(metadata, dict) else 0,
            "tensor_count": len(tensors),
            "group_count": len(Counter(name.split(".", 1)[0] for name in tensors)),
            "total_raw_payload_bytes": total_bytes,
        },
    }
    if output is not None:
        if output.resolve() == model_path.resolve():
            raise ValueError("Report output path cannot be the source checkpoint.")
        write_document(output, report, overwrite)
    return InspectionResult(report, quantization_summary, quantization_layers)


@dataclass(frozen=True)
class AuditResult:
    document: dict[str, Any]
    source_path: Path
    output: Path


@dataclass(frozen=True)
class ActivationAuditResult:
    source_path: Path
    activation_calibration: Path
    cache_metadata_path: Path
    cache_tensors_path: Path
    method_count: int
    layer_count: int


def activation_audit_model(
    source_path: Path,
    activation_calibration: Path,
    output: Path,
    *,
    device: str = "cpu",
    methods: Iterable[str] | None = None,
    overwrite: bool = False,
    on_tensor_started: Callable[[int, int, str], None] | None = None,
    check_cancelled: Callable[[], None] | None = None,
) -> ActivationAuditResult:
    requested_methods = tuple(MEASUREMENT_METHODS if methods is None else methods)
    metadata_path, tensors_path = activation_audit_pair_paths(output)
    if not overwrite:
        preflight_output(metadata_path, False)
        preflight_output(tensors_path, False)
    if check_cancelled is not None:
        check_cancelled()
    calibration = ActivationCalibration.load(
        activation_calibration,
        check_cancelled=check_cancelled,
    )
    actual_metadata_path, actual_tensors_path = run_activation_audit(
        source_path,
        calibration,
        output,
        calibration_metadata_path=activation_calibration,
        calibration_stats_path=None,
        requested_methods=requested_methods,
        on_tensor_started=on_tensor_started,
        check_cancelled=check_cancelled,
        device=device,
        overwrite=overwrite,
    )
    return ActivationAuditResult(
        source_path=source_path,
        activation_calibration=activation_calibration,
        cache_metadata_path=actual_metadata_path,
        cache_tensors_path=actual_tensors_path,
        method_count=len(requested_methods),
        layer_count=len(calibration.tensor_names()),
    )


def score_activation_cache(
    audit_cache: Path,
    activation_calibration: Path,
    *,
    metric: str = ACTIVATION_AUDIT_METRICS[0],
) -> dict[str, object]:
    return score_activation_audit(
        audit_cache,
        activation_calibration,
        metric=metric,
    )


def inspect_activation_cache(
    audit_cache: Path,
    activation_calibration: Path,
    *,
    tensor_name: str | None = None,
    top_n: int = 5,
) -> dict[str, object]:
    return inspect_activation_audit(
        audit_cache,
        activation_calibration,
        tensor_name=tensor_name,
        top_n=top_n,
    )


@dataclass(frozen=True)
class ActivationOptimizeResult:
    audit_cache: Path
    profile_path: Path | None
    summary_path: Path | None
    profile_id: str
    metric: str
    target_bytes: int | None
    promotion_budget_bytes: int | None
    estimated_output_bytes: int
    dry_run: bool


def optimize_activation_profile(
    audit_cache: Path,
    activation_calibration: Path,
    output_path: Path,
    *,
    summary_output: Path | None = None,
    source_path: Path | None = None,
    target_size_gib: float | None = None,
    promotion_budget_mib: float | None = None,
    metric: str = ACTIVATION_AUDIT_METRICS[0],
    methods: Iterable[str] | None = None,
    baseline_method: str = "bf16",
    exclude_prefixes: Iterable[str] = (),
    exclude_suffixes: Iterable[str] = (),
    profile_id: str = "activation-audit",
    overwrite: bool = False,
    dry_run: bool = False,
) -> ActivationOptimizeResult:
    if (target_size_gib is None) == (promotion_budget_mib is None):
        raise ValueError("Provide exactly one target-size or promotion-budget option.")
    target_bytes = (
        gib_to_bytes(target_size_gib) if target_size_gib is not None else None
    )
    promotion_budget_bytes = (
        mib_to_bytes(promotion_budget_mib, allow_zero=True)
        if promotion_budget_mib is not None
        else None
    )
    effective_summary_output = summary_output or output_path.with_name(
        f"{output_path.stem}-summary.json"
    )
    if output_path.resolve() == effective_summary_output.resolve():
        raise ValueError("Profile and summary output paths must be different.")
    if not overwrite:
        preflight_output(output_path, False)
        preflight_output(effective_summary_output, False)

    generated = generate_activation_cache_profile(
        audit_cache,
        activation_calibration,
        source_path=source_path,
        target_bytes=target_bytes,
        promotion_budget_bytes=promotion_budget_bytes,
        profile_id=profile_id,
        metric=metric,
        allowed_methods=(
            None if methods is None else frozenset(methods)
        ),
        baseline_method=baseline_method,
        excluded_prefixes=tuple(
            value.strip() for value in exclude_prefixes if value.strip()
        ),
        excluded_suffixes=tuple(
            value.strip() for value in exclude_suffixes if value.strip()
        ),
    )
    if not dry_run:
        write_profile(output_path, generated.optimized.profile, overwrite=overwrite)
        write_json(effective_summary_output, generated.summary, overwrite)
    return ActivationOptimizeResult(
        audit_cache=audit_cache,
        profile_path=None if dry_run else output_path,
        summary_path=None if dry_run else effective_summary_output,
        profile_id=profile_id,
        metric=metric,
        target_bytes=target_bytes,
        promotion_budget_bytes=promotion_budget_bytes,
        estimated_output_bytes=generated.optimized.output_bytes,
        dry_run=dry_run,
    )


def audit_model(
    source_path: Path,
    output: Path,
    *,
    device: str = "cpu",
    overwrite: bool = False,
    methods: Iterable[str] | None = None,
    on_entry_started: Callable[[int, int, PlanEntry], None] | None = None,
    preflight: Callable[[Path, bool], None] = preflight_output,
    audit_source: Callable[..., dict[str, Any]] = audit_bf16_source,
    write_document: Callable[[Path, object, bool], None] = write_json,
) -> AuditResult:
    preflight(output, overwrite)
    options = {
        "on_entry_started": on_entry_started,
        "device": device,
        "include_plain_methods": methods is not None,
    }
    if methods is not None:
        options["methods"] = tuple(methods)
    document = audit_source(source_path, **options)
    write_document(output, document, overwrite)
    return AuditResult(document, source_path, output)


@dataclass(frozen=True)
class OptimizeResult:
    audit_path: Path
    output_path: Path
    profile_id: str
    optimized: Any
    audit_document: dict[str, Any]
    dry_run: bool


def optimize_profile(
    audit_path: Path | None,
    output_path: Path | None,
    *,
    profile_id: str | None,
    target_size_gib: float | None,
    method: str | None = None,
    max_relative_l2_error: float | None = None,
    exclude_prefix: str | None = None,
    exclude_suffix: str | None = None,
    overwrite: bool | None = None,
    config: Path | None = None,
    output: Path | None = None,
    dry_run: bool = False,
    to_bytes: Callable[[float], int] = gib_to_bytes,
    load_config: Callable[[Path], Any] = load_optimize_config,
    load_audit: Callable[[Path], dict[str, Any]] = load_weight_audit,
    optimize: Callable[..., Any] = optimize_target_size,
    write: Callable[..., None] = write_profile,
) -> OptimizeResult:
    if config is not None:
        if audit_path is not None or output_path is not None:
            raise ValueError("Positional paths cannot be combined with --config.")
        command_config = load_config(config)
        audit_path = command_config.paths.audit_report
        output_path = output or command_config.paths.profile
        profile_id = (
            profile_id if profile_id is not None else command_config.profile_id
        )
        target_size_gib = (
            target_size_gib
            if target_size_gib is not None
            else command_config.target_size_gib
        )
        methods = split_comma_separated(method) if method is not None else tuple(command_config.methods)
        max_relative_l2_error = max_relative_l2_error if max_relative_l2_error is not None else command_config.max_relative_l2_error
        prefixes = split_comma_separated(exclude_prefix) if exclude_prefix is not None else command_config.exclude_prefixes
        suffixes = split_comma_separated(exclude_suffix) if exclude_suffix is not None else command_config.exclude_suffixes
        overwrite = command_config.overwrite if overwrite is None else overwrite
    else:
        if None in (audit_path, output_path, profile_id, target_size_gib):
            raise ValueError("Direct optimize requires AUDIT_PATH, OUTPUT_PATH, --profile-id, and --target-size-gib.")
        if output is not None:
            raise ValueError("--output requires --config.")
        methods = split_comma_separated(method) if method is not None else tuple(SUPPORTED_METHODS)
        prefixes = split_comma_separated(exclude_prefix)
        suffixes = split_comma_separated(exclude_suffix)
        overwrite = bool(overwrite)

    assert audit_path is not None and output_path is not None
    assert profile_id is not None and target_size_gib is not None
    if audit_path.resolve() == output_path.resolve():
        raise ValueError("Profile output path cannot be the audit report.")
    audit_document = load_audit(audit_path)
    optimized = optimize(
        audit_document,
        profile_id,
        to_bytes(target_size_gib),
        frozenset(methods),
        max_relative_l2_error,
        prefixes,
        suffixes,
    )
    if not dry_run:
        write(output_path, optimized.profile, overwrite=overwrite)
    return OptimizeResult(audit_path, output_path, profile_id, optimized, audit_document, dry_run)


@dataclass(frozen=True)
class QuantizeResult:
    source_path: Path
    output_path: Path | None
    profile: Path
    adapter_count: int
    estimated_bytes: int | None = None
    timing: TimingCollector | None = None


def quantize_model(
    source_path: Path | None,
    output_path: Path | None,
    *,
    profile: Path | None,
    device: str = "cpu",
    gptq_calibration: Path | None = None,
    adapter_path: Iterable[Path] = (),
    adapter_strength: Iterable[float] = (),
    config: Path | None = None,
    dry_run: bool = False,
    timing: bool = False,
    on_entry_started: Callable[[int, int, str], None] | None = None,
    build_adapters: Callable[[Iterable[Path], Iterable[float]], tuple[AdapterMergeInput, ...]] = build_adapter_inputs,
    load_config: Callable[[Path], Any] = load_quantize_config,
    estimate: Callable[[Path, Path], int] = estimate_output_bytes,
    convert: Callable[..., None] = convert_model_from_profile,
) -> QuantizeResult:
    adapters = build_adapters(adapter_path, adapter_strength)
    if config is not None:
        if source_path is not None or output_path is not None:
            raise ValueError("Source and output positional paths cannot be combined with --config.")
        command_config = load_config(config)
        source_path = command_config.paths.source
        output_path = command_config.paths.quantized_output
        profile = profile or command_config.paths.profile
        gptq_calibration = gptq_calibration or command_config.paths.gptq_calibration
        adapters = adapters or command_config.adapters
    elif source_path is None or (output_path is None and not dry_run) or profile is None:
        raise ValueError(
            "Direct quantize requires SOURCE_PATH, " + ("OUTPUT_PATH, " if not dry_run else "") + "and --profile."
        )

    if source_path is None or profile is None:
        raise ValueError("Quantize config requires paths.source and paths.profile.")
    if dry_run:
        return QuantizeResult(source_path, output_path, profile, len(adapters), estimate(source_path, profile))
    if output_path is None:
        raise ValueError("Quantize config requires paths.quantized_output.")
    timing_collector = TimingCollector() if timing else None
    convert(
        source_path,
        output_path,
        profile,
        on_entry_started=on_entry_started,
        adapters=adapters,
        timing=timing_collector,
        device=device,
        **({"gptq_calibration": gptq_calibration} if gptq_calibration is not None else {}),
    )
    return QuantizeResult(source_path, output_path, profile, len(adapters), timing=timing_collector)
