"""Typer front door for the existing PotatoForge operations."""

from __future__ import annotations

import json
import math
import unittest
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import typer

from .audits.analysis import (
    print_activation_ranking,
    print_tensor_analysis,
    write_analysis_workbook,
)
from .audits.activation_audit import (
    activation_audit_pair_paths,
    inspect_activation_audit,
    run_activation_audit,
    score_activation_audit,
)
from .audits.activation_comparison import (
    V2_ACTIVATION_METRICS,
    generate_activation_comparison_workbook,
)
from .audits.activation_measurements import MEASUREMENT_METHODS
from .audits.activation_profiles import (
    generate_activation_cache_profile,
    generate_activation_profile,
)
from .audits.profile_optimizer import (
    SUPPORTED_METHODS,
    audited_global_relative_l2,
    generate_profile_sweep,
    load_weight_audit,
    optimize_target_size,
    validate_weight_audit_against_source,
    write_profile,
)
from .audits.weight_audit import (
    audit_bf16_source,
    print_weight_audit_table,
)
from .converter import (
    convert_model_from_profile,
    estimate_output_bytes,
    print_conversion_progress,
)
from .extraction import extract_tensors
from .headers.header_reader import read_header_from_safetensors
from .headers.source_header import read_source_model_header
from .lora.lora_discovery import inspect_adapter_header
from .lora.lora_merge import AdapterMergeInput, merge_bf16_adapters
from .patch_planning import build_patch_metadata, build_patch_plan
from .patching import execute_patch_plan
from .planning import (
    QUANTIZATION_LAYERS_METADATA_KEY,
    QUANTIZATION_METADATA_KEY,
    parse_quantization_layers,
)
from .profiles import validate_quantization_action
from .config import load_optimize_config, load_quantize_config
from .calibration import (
    ActivationCalibration,
    merge_activation_calibrations,
    score_activation_probe,
)
from .calibration.activation_probe import activation_pair_paths


app = typer.Typer(
    no_args_is_help=True,
    invoke_without_command=True,
    help=(
        "Streaming safetensors inspection, LoRA, audit, profile, "
        "and quantization commands."
    ),
)
VERSION = "0.1.0"


def _finish(summary: dict[str, Any]) -> None:
    for key, value in summary.items():
        typer.echo(f"{key}: {value}")


def _fail(command: str, error: Exception, code: int) -> None:
    typer.echo(f"{command} failed: {error}", err=True)
    raise typer.Exit(code)


def _run(
    command: str,
    action: Callable[[], None],
) -> None:
    try:
        action()
    except KeyboardInterrupt as error:
        _fail(command, error, 130)
    except FileExistsError as error:
        _fail(command, error, 4)
    except (FileNotFoundError, ValueError) as error:
        _fail(command, error, 3)
    except OSError as error:
        _fail(command, error, 5)


def _write_json(path: Path, document: object, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8") as output_file:
        json.dump(document, output_file, indent=2)
        output_file.write("\n")


def _preflight_audit_outputs(
    output: Path,
    activation_probe_output: Path | None,
    overwrite: bool,
) -> None:
    paths = [output]
    if activation_probe_output is not None:
        paths.extend(activation_pair_paths(activation_probe_output))
    resolved_paths = [path.resolve() for path in paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("Audit and activation probe output paths must be different.")
    if not overwrite:
        for path in paths:
            if path.exists():
                raise FileExistsError(f"Output already exists: {path}")


def _parse_quantization_metadata(
    metadata: object,
) -> tuple[str, dict[str, str]]:
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
        layers = parse_quantization_layers(raw_layers)
    except ValueError:
        return "invalid", {}

    return summary, layers


def _print_quantization(
    summary: str,
    layers: dict[str, str],
) -> None:
    typer.echo(f"quantization: {summary}")
    typer.echo(f"quantized_layer_count: {len(layers)}")
    if not layers:
        return

    typer.echo("formats:")
    for action, count in sorted(Counter(layers.values()).items()):
        label = "layer" if count == 1 else "layers"
        typer.echo(f"  {action}: {count} {label}")

    typer.echo("layers:")
    for name, action in layers.items():
        typer.echo(f"  {name}: {action}")


def _gib_to_bytes(value: float) -> int:
    if not math.isfinite(value) or value <= 0:
        raise typer.BadParameter("must be a finite positive number")
    byte_count = int(value * (1024**3))
    if byte_count <= 0:
        raise typer.BadParameter("must represent at least one byte")
    return byte_count


def _mib_to_bytes(value: float, *, allow_zero: bool = False) -> int:
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise typer.BadParameter("must be a finite positive number")
    byte_count = int(value * (1024**2))
    if byte_count <= 0 and not allow_zero:
        raise typer.BadParameter("must represent at least one byte")
    return byte_count


def _resolve_profile_budget(
    target_size_mib: float | None,
    target_size_gib: float | None,
    promotion_budget_mib: float | None,
    promotion_budget_bytes_option: int | None,
    *,
    option_error: str = "Provide exactly one target-size or promotion-budget option.",
) -> tuple[int | None, int | None]:
    budget_options = sum(
        value is not None
        for value in (
            target_size_mib,
            target_size_gib,
            promotion_budget_mib,
            promotion_budget_bytes_option,
        )
    )
    if budget_options != 1:
        raise ValueError(option_error)
    if target_size_mib is not None:
        return _mib_to_bytes(target_size_mib), None
    if target_size_gib is not None:
        return _gib_to_bytes(target_size_gib), None
    if promotion_budget_bytes_option is not None:
        if promotion_budget_bytes_option < 0:
            raise ValueError("--promotion-budget-bytes must be non-negative.")
        return None, promotion_budget_bytes_option
    assert promotion_budget_mib is not None
    return None, _mib_to_bytes(promotion_budget_mib, allow_zero=True)


def _split_comma_separated(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(part.strip() for part in value.split(","))


def _print_progress(index: int, count: int, name: str) -> None:
    typer.echo(f"[{index}/{count}] {name}", err=True)


def _print_profile_sweep_progress(
    index: int,
    count: int,
    target_bytes: int,
) -> None:
    typer.echo(
        f"[profile {index}/{count}] target={target_bytes / (1024**3):.2f} GiB",
        err=True,
    )


def _print_patch_progress(message: str) -> None:
    typer.echo(message, err=True)


def _build_adapter_inputs(
    adapter_paths: list[Path],
    adapter_strengths: list[float],
) -> tuple[AdapterMergeInput, ...]:
    if len(adapter_paths) != len(adapter_strengths):
        raise ValueError(
            "--adapter-path and --adapter-strength must be repeated "
            "the same number of times."
        )
    return tuple(
        AdapterMergeInput(path=path, strength=strength)
        for path, strength in zip(adapter_paths, adapter_strengths)
    )


@app.callback()
def main(version: bool = typer.Option(False, "--version")) -> None:
    """Show the PotatoForge command surface."""
    if version:
        typer.echo(VERSION)
        raise typer.Exit()


@app.command("inspect-header")
def inspect_header(
    model_path: Path = typer.Argument(..., help="Source safetensors checkpoint."),
    output: Path | None = typer.Option(None, help="Optional JSON report path."),
    overwrite: bool = typer.Option(False, help="Replace an existing JSON report."),
    show_quantization: bool = typer.Option(
        False,
        "--quantization",
        help="Show per-layer quantization metadata.",
    ),
) -> None:
    """Inspect a safetensors header without reading tensor payloads."""
    def action() -> None:
        header = read_header_from_safetensors(model_path)
        metadata = header.get("__metadata__", {})
        tensors = {
            name: descriptor
            for name, descriptor in header.items()
            if name != "__metadata__"
        }
        total_bytes = sum(
            descriptor["data_offsets"][1] - descriptor["data_offsets"][0]
            for descriptor in tensors.values()
        )
        prefix_counts = Counter(name.split(".", 1)[0] for name in tensors)
        quantization_summary, quantization_layers = (
            _parse_quantization_metadata(metadata)
        )
        report = {
            "source_path": str(model_path),
            "metadata": metadata,
            "model_info": tensors,
            "quantization": {
                "summary": quantization_summary,
                "layers": quantization_layers,
            },
            "summary": {
                "metadata_count": len(metadata) if isinstance(metadata, dict) else 0,
                "tensor_count": len(tensors),
                "group_count": len(prefix_counts),
                "total_raw_payload_bytes": total_bytes,
            },
        }
        if output is not None:
            _write_json(output, report, overwrite)
        if show_quantization:
            _print_quantization(
                quantization_summary,
                quantization_layers,
            )
        else:
            _finish(report["summary"])

    _run("inspect-header", action)


@app.command("inspect-lora")
def inspect_lora(
    adapter_path: Path = typer.Argument(..., help="LoRA adapter checkpoint."),
    output: Path | None = typer.Option(None, help="Optional JSON report path."),
    overwrite: bool = typer.Option(False, help="Replace an existing JSON report."),
) -> None:
    """Inspect a LoRA adapter header without reading tensor payloads."""
    def action() -> None:
        header = read_source_model_header(adapter_path)
        inspection = inspect_adapter_header(header)
        kind_counts = Counter(record["kind"] for record in inspection["tensors"])
        contract_counts = Counter(
            record["contract"] for record in inspection["tensors"]
        )
        rank_counts = Counter(pair["rank"] for pair in inspection["pairs"])
        summary = {
            "adapter_path": str(adapter_path),
            "tensor_count": len(inspection["tensors"]),
            "pair_count": len(inspection["pairs"]),
            "additive_delta_count": len(inspection["additive_deltas"]),
            "lokr_group_count": len(inspection["lokr_groups"]),
            "kind_counts": dict(sorted(kind_counts.items())),
            "contract_counts": dict(sorted(contract_counts.items())),
            "rank_counts": {
                str(rank): count for rank, count in sorted(rank_counts.items())
            },
        }
        if output is not None:
            _write_json(
                output,
                {"metadata": header.metadata, **summary, "inspection": inspection},
                overwrite,
            )
        _finish(summary)

    _run("inspect-lora", action)


@app.command("merge-lora")
def merge_lora(
    source_path: Path = typer.Argument(...),
    output_path: Path = typer.Argument(...),
    adapter_path: list[Path] = typer.Option(
        ..., "--adapter-path", help="Adapter path; repeat for multiple adapters."
    ),
    adapter_strength: list[float] = typer.Option(
        ..., "--adapter-strength", help="Adapter strength; pair by option order."
    ),
) -> None:
    """Merge one or more LoRA adapters into a new checkpoint."""
    def action() -> None:
        adapters = _build_adapter_inputs(adapter_path, adapter_strength)
        merge_bf16_adapters(
            source_path,
            output_path,
            adapters,
            on_tensor_started=_print_progress,
        )
        _finish(
            {
                "source_path": str(source_path),
                "output_path": str(output_path),
                "adapter_count": len(adapters),
                "output_bytes": output_path.stat().st_size,
            },
        )

    _run("merge-lora", action)


@app.command("audit")
def audit(
    source_path: Path = typer.Argument(...),
    output: Path = typer.Option(..., help="JSON audit report path."),
    activation_calibration: Path | None = typer.Option(
        None,
        "--activation-calibration",
        help="Activation calibration metadata JSON.",
    ),
    method: str | None = typer.Option(
        None,
        "--method",
        help="Audit one candidate method; only convrot_w4a4 is supported.",
    ),
    activation_probe_output: Path | None = typer.Option(
        None,
        "--activation-probe-output",
        help=(
            "Reusable INT8 ConvRot to W4A4 probe cache path; requires "
            "--activation-calibration with an INT8 ConvRot baseline."
        ),
    ),
    overwrite: bool = typer.Option(False, help="Replace an existing JSON report."),
) -> None:
    """Measure supported weight reconstruction formats."""
    def action() -> None:
        _preflight_audit_outputs(output, activation_probe_output, overwrite)
        calibration = (
            None
            if activation_calibration is None
            else ActivationCalibration.load(activation_calibration)
        )
        audit_kwargs: dict[str, Any] = {
            "on_entry_started": _print_progress,
        }
        if calibration is not None:
            audit_kwargs["activation_calibration"] = calibration
        if method is not None:
            audit_kwargs["audit_method"] = method
        if activation_probe_output is not None:
            audit_kwargs["activation_probe_output"] = activation_probe_output
            audit_kwargs["activation_probe_overwrite"] = overwrite
        document = audit_bf16_source(source_path, **audit_kwargs)
        _write_json(output, document, overwrite)
        summary = document["summary"]
        print_weight_audit_table(
            document["results"],
            source_dtype=str(document["selection"]["dtype"]),
        )
        _finish(
            {
                "source_path": str(source_path),
                "audited_layer_count": summary["audited_layer_count"],
                "skipped_tensor_count": summary["skipped_tensor_count"],
                "audit_report": str(output),
                "warning": "Weight reconstruction only; not runtime speed or image quality.",
            },
        )

    _run("audit", action)


@app.command("activation-audit")
def activation_audit(
    source_path: Path = typer.Argument(...),
    activation_calibration: Path = typer.Option(
        ...,
        "--activation-calibration",
        help="V2 activation calibration metadata JSON.",
    ),
    output: Path = typer.Option(..., "--output", help="Activation audit cache."),
    calibration_stats: Path | None = typer.Option(
        None,
        "--calibration-stats",
        help="Optional calibration Safetensors path.",
    ),
    method: str | None = typer.Option(
        None,
        "--method",
        help=(
            "Comma-separated candidate methods; defaults to: "
            + ", ".join(MEASUREMENT_METHODS)
        ),
    ),
    tensor: str | None = typer.Option(
        None,
        "--tensor",
        help="Optional comma-separated calibration tensor names.",
    ),
    overwrite: bool = typer.Option(False, help="Replace an existing cache pair."),
) -> None:
    """Build a reusable activation-aware candidate measurement cache."""
    def action() -> None:
        requested_methods = _split_comma_separated(method) or None
        tensor_names = _split_comma_separated(tensor) or None
        calibration = ActivationCalibration.load(
            activation_calibration,
            calibration_stats,
        )
        pair_paths = activation_audit_pair_paths(output)
        if not overwrite:
            for output_path in pair_paths:
                if output_path.exists():
                    raise FileExistsError(f"Output already exists: {output_path}")
        metadata_path, tensors_path = run_activation_audit(
            source_path,
            calibration,
            output,
            calibration_metadata_path=activation_calibration,
            calibration_stats_path=calibration_stats,
            requested_methods=requested_methods,
            tensor_names=tensor_names,
            on_tensor_started=_print_progress,
            overwrite=overwrite,
        )
        _finish(
            {
                "source_path": str(source_path),
                "calibration_session_id": calibration.session_id,
                "metadata_path": str(metadata_path),
                "tensors_path": str(tensors_path),
                "method_count": len(
                    MEASUREMENT_METHODS
                    if requested_methods is None
                    else requested_methods
                ),
                "layer_count": len(calibration.tensor_names()),
            }
        )

    _run("activation-audit", action)


@app.command("activation-score")
def activation_score(
    probe_cache: Path | None = typer.Option(
        None,
        "--probe-cache",
        help="Reusable activation probe metadata JSON.",
    ),
    audit_cache: Path | None = typer.Option(
        None,
        "--audit-cache",
        help="Reusable V2 activation-audit metadata JSON.",
    ),
    activation_calibration: Path = typer.Option(
        ...,
        "--activation-calibration",
        help="Activation calibration metadata JSON.",
    ),
    output: Path = typer.Option(..., "--output", help="Activation score JSON."),
    metric: str = typer.Option(
        "aggregate_observed_relative_sse",
        "--metric",
        help="V2 cache metric when --audit-cache is used.",
    ),
    overwrite: bool = typer.Option(False, help="Replace an existing JSON report."),
) -> None:
    """Score a reusable activation probe or V2 activation-audit cache."""
    def action() -> None:
        if (probe_cache is None) == (audit_cache is None):
            raise ValueError(
                "Provide exactly one of --probe-cache or --audit-cache."
            )
        if probe_cache is not None:
            report = score_activation_probe(probe_cache, activation_calibration)
            scored_count = report["summary"]["scored_tensor_count"]
        else:
            assert audit_cache is not None
            report = score_activation_audit(
                audit_cache,
                activation_calibration,
                metric=metric,
            )
            scored_count = report["summary"]["available_candidate_count"]
        _write_json(output, report, overwrite)
        _finish(
            {
                "probe_cache": None if probe_cache is None else str(probe_cache),
                "audit_cache": None if audit_cache is None else str(audit_cache),
                "activation_calibration": str(activation_calibration),
                "score_report": str(output),
                "scored_tensor_count": scored_count,
            }
        )

    _run("activation-score", action)


@app.command("activation-inspect")
def activation_inspect(
    audit_cache: Path = typer.Option(
        ...,
        "--audit-cache",
        help="V2 activation-audit metadata JSON.",
    ),
    activation_calibration: Path = typer.Option(
        ...,
        "--activation-calibration",
        help="Activation calibration metadata JSON.",
    ),
    tensor: str | None = typer.Option(
        None,
        "--tensor",
        help="Optional tensor name; inspect all cached layers by default.",
    ),
    top_n: int = typer.Option(5, "--top-n", help="Worst evaluations to report."),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Optional inspection report JSON path.",
    ),
    overwrite: bool = typer.Option(False, help="Replace an existing JSON report."),
) -> None:
    """Inspect cached activation errors without requantizing weights."""
    def action() -> None:
        report = inspect_activation_audit(
            audit_cache,
            activation_calibration,
            tensor_name=tensor,
            top_n=top_n,
        )
        if output is None:
            typer.echo(json.dumps(report, indent=2))
            return
        _write_json(output, report, overwrite)
        _finish(
            {
                "audit_cache": str(audit_cache),
                "inspection_report": str(output),
                "layer_count": report["summary"]["layer_count"],
            }
        )

    _run("activation-inspect", action)


@app.command("activation-optimize")
def activation_optimize(
    audit_cache: Path = typer.Option(
        ...,
        "--audit-cache",
        help="V2 activation-audit metadata JSON.",
    ),
    activation_calibration: Path = typer.Option(
        ...,
        "--activation-calibration",
        help="V2 activation calibration metadata JSON.",
    ),
    output: Path = typer.Option(..., "--output", help="Output profile JSON."),
    summary_output: Path | None = typer.Option(
        None,
        "--summary-output",
        help="Optional optimization summary JSON.",
    ),
    source_path: Path | None = typer.Option(
        None,
        "--source",
        help="Optional source checkpoint override.",
    ),
    target_size_mib: float | None = typer.Option(None, "--target-size-mib"),
    target_size_gib: float | None = typer.Option(None, "--target-size-gib"),
    promotion_budget_mib: float | None = typer.Option(
        None,
        "--promotion-budget-mib",
    ),
    promotion_budget_bytes_option: int | None = typer.Option(
        None,
        "--promotion-budget-bytes",
    ),
    metric: str = typer.Option(
        "aggregate_observed_relative_sse",
        "--metric",
    ),
    method: str | None = typer.Option(
        None,
        "--method",
        help="Comma-separated methods allowed in the optimizer.",
    ),
    baseline_method: str = typer.Option(
        "bf16",
        "--baseline-method",
        help="Profile baseline method; use convrot_w4a4 for W4A4 baseline.",
    ),
    exclude_prefix: str | None = typer.Option(None, "--exclude-prefix"),
    exclude_suffix: str | None = typer.Option(None, "--exclude-suffix"),
    profile_id: str = typer.Option("activation-audit", "--profile-id"),
    overwrite: bool = typer.Option(False, help="Replace existing outputs."),
) -> None:
    """Generate a runtime profile from cache-only activation costs."""
    def action() -> None:
        target_bytes, promotion_budget_bytes = _resolve_profile_budget(
            target_size_mib,
            target_size_gib,
            promotion_budget_mib,
            promotion_budget_bytes_option,
        )
        effective_summary_output = summary_output or output.with_name(
            f"{output.stem}-summary.json"
        )
        if output.resolve() == effective_summary_output.resolve():
            raise ValueError("Profile and summary output paths must be different.")
        if not overwrite:
            for output_path in (output, effective_summary_output):
                if output_path.exists():
                    raise FileExistsError(f"Output already exists: {output_path}")

        generated = generate_activation_cache_profile(
            audit_cache,
            activation_calibration,
            source_path=source_path,
            target_bytes=target_bytes,
            promotion_budget_bytes=promotion_budget_bytes,
            profile_id=profile_id,
            metric=metric,
            allowed_methods=(
                None
                if method is None
                else frozenset(_split_comma_separated(method))
            ),
            baseline_method=baseline_method,
            excluded_prefixes=_split_comma_separated(exclude_prefix),
            excluded_suffixes=_split_comma_separated(exclude_suffix),
        )
        write_profile(output, generated.optimized.profile, overwrite=overwrite)
        _write_json(effective_summary_output, generated.summary, overwrite)
        _finish(
            {
                "audit_cache": str(audit_cache),
                "profile_path": str(output),
                "summary_path": str(effective_summary_output),
                "metric": metric,
                "target_bytes": generated.optimized.target_bytes,
                "estimated_output_bytes": generated.optimized.output_bytes,
            }
        )

    _run("activation-optimize", action)


@app.command("activation-compare")
def activation_compare(
    audit_cache: Path = typer.Option(
        ...,
        "--audit-cache",
        help="V2 activation-audit metadata JSON.",
    ),
    activation_calibration: Path = typer.Option(
        ...,
        "--activation-calibration",
        help="V2 activation calibration metadata JSON.",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        help="Excel comparison workbook.",
    ),
    source_path: Path | None = typer.Option(
        None,
        "--source",
        help="Optional source checkpoint override.",
    ),
    target_size_mib: float | None = typer.Option(None, "--target-size-mib"),
    target_size_gib: float | None = typer.Option(None, "--target-size-gib"),
    promotion_budget_mib: float | None = typer.Option(
        None,
        "--promotion-budget-mib",
    ),
    promotion_budget_bytes_option: int | None = typer.Option(
        None,
        "--promotion-budget-bytes",
    ),
    method: str | None = typer.Option(
        None,
        "--method",
        help="Comma-separated methods allowed in every comparison profile.",
    ),
    baseline_method: str = typer.Option(
        "bf16",
        "--baseline-method",
        help="Profile baseline method; use convrot_w4a4 for W4A4 baseline.",
    ),
    exclude_prefix: str | None = typer.Option(None, "--exclude-prefix"),
    exclude_suffix: str | None = typer.Option(None, "--exclude-suffix"),
    metrics: str | None = typer.Option(
        None,
        "--metrics",
        help="Optional comma-separated metric list; defaults to all V2 metrics.",
    ),
    top_n: int = typer.Option(20, "--top-n", help="Ranked rows per metric/method."),
    overwrite: bool = typer.Option(False, help="Replace an existing workbook."),
) -> None:
    """Compare all V2 activation metrics in one workbook."""
    def action() -> None:
        target_bytes, promotion_budget_bytes = _resolve_profile_budget(
            target_size_mib,
            target_size_gib,
            promotion_budget_mib,
            promotion_budget_bytes_option,
        )
        requested_metrics = _split_comma_separated(metrics) or V2_ACTIVATION_METRICS
        summary = generate_activation_comparison_workbook(
            audit_cache,
            activation_calibration,
            output,
            source_path=source_path,
            target_bytes=target_bytes,
            promotion_budget_bytes=promotion_budget_bytes,
            allowed_methods=(
                None
                if method is None
                else frozenset(_split_comma_separated(method))
            ),
            baseline_method=baseline_method,
            excluded_prefixes=_split_comma_separated(exclude_prefix),
            excluded_suffixes=_split_comma_separated(exclude_suffix),
            metrics=requested_metrics,
            top_n=top_n,
            overwrite=overwrite,
        )
        _finish(summary)

    _run("activation-compare", action)


@app.command("calibration-merge")
def calibration_merge(
    calibration_paths: list[Path] = typer.Argument(
        ...,
        help="Activation calibration JSON files to merge.",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        help="Merged calibration pair base path or JSON path.",
    ),
    overwrite: bool = typer.Option(False, help="Replace an existing calibration pair."),
) -> None:
    """Merge additive activation calibration statistics."""
    def action() -> None:
        metadata_path, tensors_path = merge_activation_calibrations(
            calibration_paths,
            output,
            overwrite=overwrite,
        )
        _finish(
            {
                "input_calibration_count": len(calibration_paths),
                "metadata_path": str(metadata_path),
                "tensors_path": str(tensors_path),
            }
        )

    _run("calibration-merge", action)


@app.command("profile-from-activation")
def profile_from_activation(
    activation_score: Path = typer.Option(
        ...,
        "--activation-score",
        help="Completed activation-score JSON report.",
    ),
    weight_audit: Path = typer.Option(
        ...,
        "--weight-audit",
        help="Existing weight-audit JSON containing storage estimates.",
    ),
    output: Path = typer.Option(..., "--output", help="Output profile JSON."),
    summary_output: Path | None = typer.Option(
        None,
        "--summary-output",
        help="Optional summary JSON; defaults to <output>-summary.json.",
    ),
    target_size_mib: float | None = typer.Option(
        None,
        "--target-size-mib",
        help="Final model-size budget in MiB.",
    ),
    target_size_gib: float | None = typer.Option(
        None,
        "--target-size-gib",
        help="Final model-size budget in GiB.",
    ),
    promotion_budget_mib: float | None = typer.Option(
        None,
        "--promotion-budget-mib",
        help="Additional storage budget above the W4A4 baseline.",
    ),
    promotion_budget_bytes_option: int | None = typer.Option(
        None,
        "--promotion-budget-bytes",
        help="Additional storage budget above the W4A4 baseline in bytes.",
    ),
    metric: str = typer.Option(
        "relative_output_sse",
        "--metric",
        help="relative_output_sse, relative_output_error, or relative_l2_error.",
    ),
    include_regex: str = typer.Option(
        r"^blocks\.",
        "--include-regex",
        help="Regex selecting eligible weight tensors.",
    ),
    profile_id: str | None = typer.Option(
        None,
        "--profile-id",
        help="Generated profile identifier.",
    ),
    overwrite: bool = typer.Option(False, help="Replace existing outputs."),
) -> None:
    """Generate an offline equal-budget mixed-precision profile."""
    def action() -> None:
        target_bytes, promotion_budget_bytes = _resolve_profile_budget(
            target_size_mib,
            target_size_gib,
            promotion_budget_mib,
            promotion_budget_bytes_option,
            option_error=(
                "Provide exactly one of --target-size-mib, --target-size-gib, "
                "--promotion-budget-mib, or --promotion-budget-bytes."
            ),
        )

        effective_profile_id = profile_id or (
            "activation-relative"
            if metric in ("relative_output_sse", "relative_output_error")
            else "legacy-static"
        )
        effective_summary_output = summary_output or output.with_name(
            f"{output.stem}-summary.json"
        )
        resolved_outputs = {
            output.resolve(),
            effective_summary_output.resolve(),
        }
        if len(resolved_outputs) != 2:
            raise ValueError("Profile and summary output paths must be different.")
        if not overwrite:
            for output_path in (output, effective_summary_output):
                if output_path.exists():
                    raise FileExistsError(f"Output already exists: {output_path}")

        generated = generate_activation_profile(
            activation_score,
            weight_audit,
            target_bytes=target_bytes,
            promotion_budget_bytes=promotion_budget_bytes,
            profile_id=effective_profile_id,
            metric=metric,
            include_regex=include_regex,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        write_profile(output, generated.optimized.profile, overwrite=overwrite)
        _write_json(effective_summary_output, generated.summary, overwrite)
        _finish(
            {
                "activation_score": str(activation_score),
                "weight_audit": str(weight_audit),
                "profile_path": str(output),
                "summary_path": str(effective_summary_output),
                "metric": metric,
                "target_bytes": generated.optimized.target_bytes,
                "estimated_output_bytes": generated.optimized.output_bytes,
                "promoted_int8cr_tensor_count": generated.summary[
                    "promoted_int8cr_tensor_count"
                ],
            }
        )

    _run("profile-from-activation", action)


@app.command("analyze")
def analyze(
    audit_path: Path | None = typer.Argument(
        None,
        help="Legacy positional JSON weight-audit report.",
    ),
    audit: Path | None = typer.Option(
        None,
        "--audit",
        help="Existing JSON weight-audit report.",
    ),
    source: Path | None = typer.Option(
        None,
        "--source",
        help="Source safetensors model for selected tensors.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Excel analysis workbook path.",
    ),
    tensor: str | None = typer.Option(
        None,
        "--tensor",
        help="Print one exact tensor.",
    ),
    tensors: str | None = typer.Option(
        None,
        "--tensors",
        help="Print comma-separated exact tensors.",
    ),
    activation_calibration: Path | None = typer.Option(
        None,
        "--activation-calibration",
        help="Activation calibration metadata JSON for --source.",
    ),
    target_size_gib: float | None = typer.Option(
        None,
        "--target-size-gib",
        "--target-size",
        help="Optional selected target output size in GiB.",
    ),
    target_size_step_gib: float = typer.Option(
        0.5,
        "--target-size-step-gib",
        help="GiB interval between generated profile options.",
    ),
    method: str | None = typer.Option(
        None,
        "--method",
        help="Comma-separated recommendation methods.",
    ),
    exclude_prefix: str | None = typer.Option(
        None,
        "--exclude-prefix",
        help="Comma-separated prefixes to keep in BF16.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Replace an existing Excel workbook.",
    ),
) -> None:
    """Report existing weight-audit measurements."""
    def action() -> None:
        if tensor is not None and tensors is not None:
            raise ValueError("--tensor and --tensors cannot be combined.")
        tensor_names = (
            _split_comma_separated(tensors)
            if tensors is not None
            else () if tensor is None else (tensor,)
        )
        audit_input = audit if audit is not None else audit_path
        if audit_input is not None and source is not None:
            raise ValueError("--audit and --source cannot be combined.")
        if audit_input is None and source is None:
            raise ValueError("Provide --audit or --source.")
        if audit_input is not None and activation_calibration is not None:
            raise ValueError(
                "--activation-calibration requires --source; "
                "include it when generating the audit report."
            )
        if source is not None:
            if not tensor_names:
                raise ValueError("--source requires --tensor or --tensors.")
            if output is not None:
                raise ValueError("--source cannot be combined with --output.")
            if target_size_gib is not None:
                raise ValueError(
                    "--source cannot be combined with --target-size-gib."
                )
            if method is not None:
                raise ValueError("--source cannot be combined with --method.")
            if exclude_prefix is not None:
                raise ValueError(
                    "--source cannot be combined with --exclude-prefix."
                )
            source_header = read_source_model_header(source)
            calibration = (
                None
                if activation_calibration is None
                else ActivationCalibration.load(activation_calibration)
            )
            audit_kwargs: dict[str, Any] = {
                "on_entry_started": _print_progress,
            }
            if calibration is not None:
                audit_kwargs["activation_calibration"] = calibration
            if tensor is not None:
                audit_kwargs["tensor_name"] = tensor
            else:
                audit_kwargs["tensor_names"] = tensor_names
            audit_document = audit_bf16_source(source, **audit_kwargs)
            for tensor_name in tensor_names:
                print_tensor_analysis(audit_document, source_header, tensor_name)
            print_activation_ranking(audit_document)
            return

        exclude_prefixes = _split_comma_separated(exclude_prefix)
        if not tensor_names and output is None:
            raise ValueError("Full analyze mode requires --output.")
        audit_document = load_weight_audit(audit_input)
        source_header = validate_weight_audit_against_source(audit_document)
        optimized = None
        profile_sweep = None
        methods = (
            _split_comma_separated(method)
            if method is not None
            else SUPPORTED_METHODS
        )
        allowed_methods = frozenset(methods)
        if output is not None:
            selected_target_bytes = (
                None
                if target_size_gib is None
                else _gib_to_bytes(target_size_gib)
            )
            profile_sweep = generate_profile_sweep(
                audit_document,
                "analysis",
                _gib_to_bytes(target_size_step_gib),
                allowed_methods,
                excluded_prefixes=exclude_prefixes,
                selected_target_bytes=selected_target_bytes,
                on_profile_started=_print_profile_sweep_progress,
            )
            optimized = (
                next(
                    (
                        profile
                        for profile in profile_sweep
                        if target_size_gib is not None
                        and profile.target_bytes == selected_target_bytes
                    ),
                    profile_sweep[-1],
                )
                if profile_sweep
                else None
            )
        elif target_size_gib is not None:
            optimized = optimize_target_size(
                audit_document,
                "analysis",
                _gib_to_bytes(target_size_gib),
                allowed_methods,
                excluded_prefixes=exclude_prefixes,
            )

        for tensor_name in tensor_names:
            print_tensor_analysis(
                audit_document,
                source_header,
                tensor_name,
                optimized if target_size_gib is not None else None,
            )
        print_activation_ranking(audit_document)
        if output is not None:
            if output.resolve() == audit_input.resolve():
                raise ValueError("Audit and output paths must be different.")
            write_analysis_workbook(
                audit_document,
                source_header,
                output,
                optimized,
                overwrite=overwrite,
                profile_sweep=profile_sweep,
                allowed_methods=allowed_methods,
                excluded_prefixes=exclude_prefixes,
            )
            _finish(
                {
                    "audit_path": str(audit_input),
                    "analysis_path": str(output),
                    "target_bytes": (
                        None if optimized is None else optimized.target_bytes
                    ),
                    "profile_count": (
                        None if profile_sweep is None else len(profile_sweep)
                    ),
                }
            )

    _run("analyze", action)


@app.command("optimize")
def optimize(
    audit_path: Path | None = typer.Argument(None),
    output_path: Path | None = typer.Argument(None),
    profile_id: str | None = typer.Option(None, help="Generated profile identifier."),
    target_size_gib: float | None = typer.Option(None, help="Target output size in GiB."),
    method: str | None = typer.Option(
        None, "--method", help="Comma-separated allowed methods."
    ),
    max_relative_l2_error: float | None = typer.Option(None),
    exclude_prefix: str | None = typer.Option(
        None,
        "--exclude-prefix",
        help="Comma-separated prefixes to keep in BF16.",
    ),
    exclude_suffix: str | None = typer.Option(
        None,
        "--exclude-suffix",
        help="Comma-separated suffixes to keep in BF16.",
    ),
    overwrite: bool | None = typer.Option(
        None,
        "--overwrite/--no-overwrite",
        help="Replace an existing JSON profile.",
    ),
    config: Path | None = typer.Option(None, help="Version-one TOML config."),
    output: Path | None = typer.Option(
        None,
        help="Output profile override for --config mode.",
    ),
    dry_run: bool = typer.Option(False, help="Validate and report without writing."),
) -> None:
    """Generate a target-size profile from a weight audit."""
    def action() -> None:
        if config is not None:
            if audit_path is not None or output_path is not None:
                raise ValueError("Positional paths cannot be combined with --config.")
            optimize_config = load_optimize_config(config)
            effective_audit_path = optimize_config.paths.audit_report
            effective_output_path = output or optimize_config.paths.profile
            effective_profile_id = (
                profile_id
                if profile_id is not None
                else optimize_config.profile_id
            )
            effective_target_size_gib = (
                target_size_gib
                if target_size_gib is not None
                else optimize_config.target_size_gib
            )
            effective_methods = (
                _split_comma_separated(method)
                if method is not None
                else list(optimize_config.methods)
            )
            effective_max_error = (
                max_relative_l2_error
                if max_relative_l2_error is not None
                else optimize_config.max_relative_l2_error
            )
            effective_prefixes = (
                _split_comma_separated(exclude_prefix)
                if exclude_prefix is not None
                else optimize_config.exclude_prefixes
            )
            effective_suffixes = (
                _split_comma_separated(exclude_suffix)
                if exclude_suffix is not None
                else optimize_config.exclude_suffixes
            )
            effective_overwrite = (
                overwrite
                if overwrite is not None
                else optimize_config.overwrite
            )
        else:
            if (
                audit_path is None
                or output_path is None
                or profile_id is None
                or target_size_gib is None
            ):
                raise ValueError(
                    "Direct optimize requires AUDIT_PATH, OUTPUT_PATH, "
                    "--profile-id, and --target-size-gib."
                )
            if output is not None:
                raise ValueError("--output requires --config.")
            effective_audit_path = audit_path
            effective_output_path = output_path
            effective_profile_id = profile_id
            effective_target_size_gib = target_size_gib
            effective_methods = (
                _split_comma_separated(method)
                if method is not None
                else list(SUPPORTED_METHODS)
            )
            effective_max_error = max_relative_l2_error
            effective_prefixes = _split_comma_separated(exclude_prefix)
            effective_suffixes = _split_comma_separated(exclude_suffix)
            effective_overwrite = bool(overwrite)

        audit_document = load_weight_audit(effective_audit_path)
        optimized = optimize_target_size(
            audit_document,
            effective_profile_id,
            _gib_to_bytes(effective_target_size_gib),
            frozenset(effective_methods),
            effective_max_error,
            effective_prefixes,
            effective_suffixes,
        )
        if not dry_run:
            write_profile(
                effective_output_path,
                optimized.profile,
                overwrite=effective_overwrite,
            )
        _finish(
            {
                "audit_path": str(effective_audit_path),
                "profile_path": None if dry_run else str(effective_output_path),
                "profile_id": effective_profile_id,
                "target_bytes": optimized.target_bytes,
                "estimated_output_bytes": optimized.output_bytes,
                "reconstruction_sse": optimized.reconstruction_sse,
                "audited_global_relative_l2": audited_global_relative_l2(
                    audit_document,
                    optimized.reconstruction_sse,
                ),
                "dry_run": dry_run,
                "warning": "Generated profiles still require artifact and runtime validation.",
            },
        )

    _run("optimize", action)


@app.command("quantize")
def quantize(
    source_path: Path | None = typer.Argument(None),
    output_path: Path | None = typer.Argument(None),
    profile: Path | None = typer.Option(
        None,
        help="JSON quantization profile; required without --config.",
    ),
    ),
    adapter_path: list[Path] = typer.Option(
        [],
        "--adapter-path",
        help="LoRA adapter path; repeat for multiple adapters.",
    ),
    adapter_strength: list[float] = typer.Option(
        [],
        "--adapter-strength",
        help="LoRA strength; pair by option order.",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help="TOML quantize config.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Estimate output storage without writing a checkpoint.",
    ),
) -> None:
    """Convert or estimate a checkpoint with an explicit JSON profile."""
    def action() -> None:
        cli_adapters = _build_adapter_inputs(adapter_path, adapter_strength)

        if config is not None:
            if source_path is not None or output_path is not None:
                raise ValueError(
                    "Source and output positional paths cannot be combined "
                    "with --config."
                )
            quantize_config = load_quantize_config(config)
            effective_source_path = quantize_config.paths.source
            effective_output_path = quantize_config.paths.quantized_output
            effective_profile = profile or quantize_config.paths.profile
            effective_adapters = (
                cli_adapters if cli_adapters else quantize_config.adapters
            )
        else:
            if (
                source_path is None
                or (output_path is None and not dry_run)
                or profile is None
            ):
                raise ValueError(
                    "Direct quantize requires SOURCE_PATH, "
                    + ("OUTPUT_PATH, " if not dry_run else "")
                    + "and --profile."
                )
            effective_source_path = source_path
            effective_output_path = output_path
            effective_profile = profile
            effective_adapters = cli_adapters

        if effective_source_path is None or effective_profile is None:
            raise ValueError(
                "Quantize config requires paths.source and paths.profile."
            )

        if dry_run:
            estimated_bytes = estimate_output_bytes(
                effective_source_path,
                effective_profile,
            )
            _finish(
                {
                    "source_path": str(effective_source_path),
                    "profile_path": str(effective_profile),
                    "estimated_output_bytes": estimated_bytes,
                    "estimated_output_mib": round(
                        estimated_bytes / 1024**2,
                        2,
                    ),
                    "estimated_output_gib": round(
                        estimated_bytes / 1024**3,
                        3,
                    ),
                    "dry_run": True,
                }
            )
            return

        if effective_output_path is None:
            raise ValueError(
                "Quantize config requires paths.quantized_output."
            )

        convert_model_from_profile(
            effective_source_path,
            effective_output_path,
            effective_profile,
            on_entry_started=print_conversion_progress,
            adapters=effective_adapters,
        )
        _finish(
            {
                "source_path": str(effective_source_path),
                "profile_path": str(effective_profile),
                "output_path": str(effective_output_path),
                "output_bytes": effective_output_path.stat().st_size,
                "adapter_count": len(effective_adapters),
            },
        )

    _run("quantize", action)


@app.command("patch")
def patch(
    source_path: Path = typer.Argument(...),
    output_path: Path = typer.Argument(...),
    tensor: str = typer.Option(
        ...,
        "--tensor",
        help="Exact tensor name or a prefix ending in .*.",
    ),
    action_name: str = typer.Option(
        ...,
        "--action",
        help="Quantization action for the selected tensor(s).",
    ),
) -> None:
    """Write one quantization patch for an exact tensor or prefix."""
    def run() -> None:
        action = validate_quantization_action(
            action_name,
            "--action",
            allow_keep=False,
        )
        plan = build_patch_plan(
            read_source_model_header(source_path).tensors,
            tensor,
            action,
            output_path.stem,
        )
        execute_patch_plan(
            source_path,
            output_path,
            plan,
            metadata=build_patch_metadata(plan, source_path),
            on_progress=_print_patch_progress,
        )
        _finish(
            {
                "source_path": str(source_path),
                "tensor": tensor,
                "action": action,
                "selected_tensor_count": plan.selected_tensor_count,
                "output_path": str(output_path),
                "output_bytes": output_path.stat().st_size,
            }
        )

    _run("patch", run)


@app.command("extract")
def extract(
    source_path: Path = typer.Argument(...),
    output_path: Path = typer.Argument(...),
    prefix: str = typer.Option(..., help="Source tensor prefix."),
    output_prefix: str = typer.Option("", help="Prefix for extracted tensor names."),
) -> None:
    """Extract tensors matching a source prefix."""
    def action() -> None:
        count, byte_count = extract_tensors(
            source_path,
            output_path,
            prefix,
            output_prefix,
        )
        _finish(
            {
                "source_prefix": prefix,
                "output_prefix": output_prefix,
                "extracted_tensor_count": count,
                "extracted_bytes": byte_count,
                "output_path": str(output_path),
            },
        )

    _run("extract", action)


@app.command("test")
def test(
    verbosity: int = typer.Option(1, min=0, max=2),
) -> None:
    """Run the standard-library test suite."""
    result = unittest.TextTestRunner(verbosity=verbosity).run(
        unittest.defaultTestLoader.discover("tests")
    )
    if not result.wasSuccessful():
        raise typer.Exit(5)
    typer.echo("status: success")


if __name__ == "__main__":
    app()
