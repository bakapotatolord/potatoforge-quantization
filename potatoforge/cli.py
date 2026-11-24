"""Typer front door for the existing PotatoForge operations."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Literal

import typer

from .application.commands import (
    build_adapter_inputs as _build_adapter_inputs,
    gib_to_bytes,
    mib_to_bytes,
    preflight_output as _preflight_audit_output,
    split_comma_separated as _split_comma_separated,
    write_json as _write_json,
    audit_model,
    inspect_model,
    optimize_profile,
    quantize_model,
)
from .audits.analysis import (
    print_tensor_analysis,
    write_analysis_workbook,
)
from .audits.activation_audit import (
    activation_audit_pair_paths,
    inspect_activation_audit,
    render_activation_audit_timing,
    run_activation_audit,
    score_activation_audit,
)
from .audits.activation_comparison import (
    ACTIVATION_METRICS,
    generate_activation_comparison_workbook,
)
from .audits.activation_measurements import MEASUREMENT_METHODS
from .audits.activation_profiles import (
    generate_activation_cache_profile,
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
from .lora.lora_extract import (
    ExtractionOptions,
    extract_lora,
    parse_split_rule,
    summarize as summarize_lora_extraction,
)
from .lora.lora_merge import merge_bf16_adapters
from .krea2_mlp_pruning import (
    Krea2MLPPrunePlan,
    Krea2MLPSelectionResult,
    apply_krea2_mlp_plan,
    build_krea2_mlp_weight_stats,
    krea2_mlp_plan_sha256,
    load_krea2_mlp_cascade_audit,
    prune_krea2_mlp_checkpoint,
    render_krea2_mlp_cascade_audit,
    select_krea2_mlp_plan,
)
from .krea2_combined_pruning import apply_krea2_combined_prune
from .krea2_block_pruning import (
    add_krea2_block_to_plan,
    apply_krea2_block_plan,
    rank_krea2_blocks,
    remap_krea2_block_profile,
    summarize_krea2_block_audits,
)
from .patch_planning import build_patch_metadata, build_patch_plan
from .patching import execute_patch_plan
from .profiles import validate_quantization_action
from .config import load_optimize_config, load_quantize_config
from .calibration import (
    ActivationCalibration,
)
from .server.runner import run_ui


app = typer.Typer(
    no_args_is_help=True,
    invoke_without_command=True,
    help=(
        "Profile-driven safetensors tools for inspection, profiling, "
        "quantization, patching, and extraction."
    ),
)
inspect_app = typer.Typer(
    no_args_is_help=True,
    help=(
        "Consume safetensors or LoRA headers and optionally produce JSON "
        "inspection reports."
    ),
)
lora_app = typer.Typer(
    no_args_is_help=True,
    help=(
        "Consume a source checkpoint and LoRA files and produce an "
        "unquantized merged checkpoint, or extract a LoRA from a fine-tune."
    ),
)
profile_app = typer.Typer(
    no_args_is_help=True,
    help=(
        "Consume source checkpoints or weight-audit reports and produce "
        "audit reports, analysis workbooks, or profiles."
    ),
)
activation_app = typer.Typer(
    no_args_is_help=True,
    help=(
        "Consume source/calibration files or activation caches and produce "
        "caches, reports, workbooks, or profiles."
    ),
)
prune_app = typer.Typer(
    no_args_is_help=True,
    help="Physically prune supported model structures before quantization.",
)
krea2_mlp_app = typer.Typer(
    invoke_without_command=True,
    help="Calibrate, select, audit, and apply Krea2/Kroma MLP pruning plans.",
)
krea2_block_app = typer.Typer(
    invoke_without_command=True,
    help="Rank Krea2 block calibration and manage cumulative bypass plans.",
)
krea2_app = typer.Typer(
    invoke_without_command=True,
    help="Apply coordinated Krea2/Kroma structural pruning.",
)
app.add_typer(
    inspect_app,
    name="inspect",
    short_help="Inspect checkpoints and LoRA files.",
)
app.add_typer(lora_app, name="lora", short_help="Merge or extract LoRA files.")
app.add_typer(
    profile_app,
    name="profile",
    short_help="Create or inspect profiles.",
)
app.add_typer(
    activation_app,
    name="activation",
    short_help="Analyze activation calibration.",
)
app.add_typer(
    prune_app,
    name="prune",
    short_help="Physically prune model structures.",
)
prune_app.add_typer(krea2_mlp_app, name="krea2-mlp")
prune_app.add_typer(krea2_block_app, name="krea2-block")
prune_app.add_typer(krea2_app, name="krea2")
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
        return mib_to_bytes(target_size_mib), None
    if target_size_gib is not None:
        return gib_to_bytes(target_size_gib), None
    if promotion_budget_bytes_option is not None:
        if promotion_budget_bytes_option < 0:
            raise ValueError("--promotion-budget-bytes must be non-negative.")
        return None, promotion_budget_bytes_option
    assert promotion_budget_mib is not None
    return None, mib_to_bytes(promotion_budget_mib, allow_zero=True)


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


def _print_krea2_prune_plan(
    plan: Krea2MLPPrunePlan,
    output_path: Path,
) -> None:
    typer.echo("Krea2 main MLP pruning")
    typer.echo("----------------------")
    typer.echo(f"Blocks:                  {len(plan.block_indices)}")
    typer.echo(f"Hidden width:            {plan.hidden_width}")
    typer.echo(f"Original MLP width:      {plan.original_mlp_width}")
    typer.echo(
        f"Removed range:           "
        f"[{plan.start_channel}:{plan.end_channel})"
    )
    typer.echo(f"Removed channels:        {plan.remove_channels}")
    typer.echo(f"New MLP width:            {plan.new_mlp_width}")
    typer.echo(
        "ConvRot 256 eligible:    "
        f"{'yes' if plan.new_mlp_width % plan.convrot_group_size == 0 else 'no'}"
    )
    typer.echo(
        "Estimated source saving: "
        f"~{plan.estimated_saving_bytes / 1_000_000_000:.3f} GB"
    )
    typer.echo(f"Output:                  {output_path}")


def _print_krea2_selection(plan: dict[str, object], output_path: Path) -> None:
    selection = plan.get("selection")
    selection_document = selection if isinstance(selection, dict) else {}
    parent = selection_document.get("parent_plan_sha256", "none")
    add_count = selection_document.get("added_per_block_this_stage", "unknown")
    parent_removed = (
        0
        if parent == "none"
        else int(plan["removed_per_block"]) - int(add_count)
    )
    typer.echo("Krea2 MLP selection stage")
    typer.echo("-------------------------")
    typer.echo(f"Blocks:                      {plan['block_count']}")
    typer.echo(f"Original MLP width:          {plan['original_mlp_width']}")
    typer.echo(f"Parent removed/block:        {parent_removed}")
    typer.echo(f"Added/block this stage:      {add_count}")
    typer.echo(f"New cumulative removed:      {plan['removed_per_block']}")
    typer.echo(f"Runtime masked width:        {plan['new_mlp_width']}")
    typer.echo(
        "256-aligned width:           "
        f"{'yes' if int(plan['new_mlp_width']) % 256 == 0 else 'no'}"
    )
    typer.echo("Metric:                      mean_sq_activation × down_col_norm_sq")
    typer.echo(f"Parent plan:                 {parent}")
    typer.echo(f"Output plan:                 {output_path}")


def _print_krea2_activation_aggregation(
    input_path: Path,
    result: Krea2MLPSelectionResult,
) -> None:
    artifact_paths = result.activation_artifacts
    total_observations = result.activation_total_observations
    formatted_observations = (
        str(int(total_observations))
        if total_observations.is_integer()
        else str(total_observations)
    )
    plan = result.plan
    typer.echo("Krea2 MLP activation aggregation")
    typer.echo("---------------------------------")
    typer.echo(f"Input directory:             {input_path}")
    typer.echo(f"Calibration artifacts:       {len(artifact_paths)}")
    typer.echo(f"Blocks:                      {plan['block_count']}")
    typer.echo(f"Original MLP width:          {plan['original_mlp_width']}")
    typer.echo(f"Total observations:          {formatted_observations}")
    typer.echo(
        "Aggregation:                 raw sums/counts summed first; "
        "per-file means are never averaged"
    )


@app.callback()
def main(version: bool = typer.Option(False, "--version")) -> None:
    """Show the PotatoForge command surface."""
    if version:
        typer.echo(VERSION)
        raise typer.Exit()


@app.command("ui")
def ui() -> None:
    """Start the local browser UI and FastAPI server."""
    static_dir = Path(__file__).resolve().parents[1] / "web" / "dist"
    if not (static_dir / "index.html").is_file():
        typer.echo(
            "Command UI files are missing. Run: pnpm --dir web install; "
            "pnpm --dir web build",
            err=True,
        )
        raise typer.Exit(3)

    try:
        run_ui(static_dir)
    except (OSError, RuntimeError) as error:
        _fail("ui", error, 5)


@app.command("inspect-header", hidden=True)
@inspect_app.command("model")
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
        result = inspect_model(
            model_path,
            output=output,
            overwrite=overwrite,
            read_header=read_header_from_safetensors,
            write_document=_write_json,
        )
        if show_quantization:
            _print_quantization(
                result.quantization_summary,
                result.quantization_layers,
            )
        else:
            _finish(result.report["summary"])

    _run("inspect-header", action)


@app.command("inspect-lora", hidden=True)
@inspect_app.command("lora")
def inspect_lora(
    adapter_path: Path = typer.Argument(..., help="LoRA checkpoint."),
    output: Path | None = typer.Option(None, help="Optional JSON report path."),
    overwrite: bool = typer.Option(False, help="Replace an existing JSON report."),
) -> None:
    """Inspect a LoRA header without reading tensor payloads."""
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


@app.command("merge-lora", hidden=True)
@lora_app.command("merge")
def merge_lora(
    source_path: Path = typer.Argument(...),
    output_path: Path = typer.Argument(...),
    adapter_path: list[Path] = typer.Option(
        ..., "--adapter-path", help="LoRA path; repeat for multiple LoRA files."
    ),
    adapter_strength: list[float] = typer.Option(
        ..., "--adapter-strength", help="LoRA strength; pair by option order."
    ),
) -> None:
    """Merge one or more LoRA files into a new checkpoint."""
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


@lora_app.command("extract")
def extract_lora_command(
    base_path: Path = typer.Argument(..., help="Base checkpoint."),
    tuned_path: Path = typer.Argument(..., help="Fine-tuned checkpoint."),
    output_path: Path = typer.Argument(
        ..., help="Output LoRA; include {rank} when using several --rank values."
    ),
    rank: list[int] = typer.Option(
        [64], "--rank", help="LoRA rank; repeat to write several ranks in one pass."
    ),
    energy: float | None = typer.Option(
        None,
        help=(
            "Adaptive rank: per tensor, the smallest rank capturing this "
            "share (0-1) of the delta energy. --rank is then the per-tensor cap."
        ),
    ),
    split_fused: list[str] = typer.Option(
        [],
        "--split-fused",
        help="Factor a fused tensor as separate LoRAs, e.g. qkv:q,k,v.",
    ),
    key_prefix: str = typer.Option(
        "diffusion_model.", help="Prefix written onto LoRA keys."
    ),
    base_prefix: str | None = typer.Option(
        None, help="Strip this prefix from base keys (auto-detected by default)."
    ),
    tuned_prefix: str | None = typer.Option(
        None, help="Strip this prefix from tuned keys (auto-detected by default)."
    ),
    include: str | None = typer.Option(None, help="Regex; only matching tensors."),
    exclude: str | None = typer.Option(None, help="Regex; skip matching tensors."),
    min_relative_delta: float = typer.Option(
        1e-4, help="Skip tensors whose delta/base norm ratio is at or below this."
    ),
    include_diffs: bool = typer.Option(
        False, help="Also store changed non-matrix tensors (norms, biases) as .diff."
    ),
    dtype: Literal["fp16", "bf16", "fp32"] = typer.Option(
        "fp16", help="Stored LoRA dtype."
    ),
    device: str | None = typer.Option(
        None, help="Compute device; defaults to cuda when available."
    ),
    report: Path | None = typer.Option(None, help="Optional JSON report path."),
    overwrite_report: bool = typer.Option(
        False, help="Replace an existing JSON report."
    ),
) -> None:
    """Extract a LoRA from a fine-tune relative to its base checkpoint."""
    def action() -> None:
        if report is not None:
            _preflight_audit_output(report, overwrite_report)
        options = ExtractionOptions(
            ranks=tuple(rank),
            energy=energy,
            key_prefix=key_prefix,
            base_prefix=base_prefix,
            tuned_prefix=tuned_prefix,
            include=include,
            exclude=exclude,
            split=tuple(parse_split_rule(rule) for rule in split_fused),
            min_relative_delta=min_relative_delta,
            include_diffs=include_diffs,
            dtype=dtype,
            device=device,
        )
        result = extract_lora(
            base_path,
            tuned_path,
            output_path,
            options,
            on_tensor_started=_print_progress,
        )
        summary = summarize_lora_extraction(result)
        if report is not None:
            _write_json(
                report,
                {
                    "summary": summary,
                    "skipped": result.skipped,
                    "unmatched_base": result.unmatched_base,
                    "unmatched_tuned": result.unmatched_tuned,
                    "tensors": [vars(tensor) for tensor in result.tensors],
                },
                overwrite_report,
            )
        _finish(
            {
                "outputs": json.dumps(summary, indent=2),
                "skipped_tensors": len(result.skipped),
                "unmatched_base": len(result.unmatched_base),
                "unmatched_tuned": len(result.unmatched_tuned),
            },
        )

    _run("extract-lora", action)


@app.command("audit", hidden=True)
@profile_app.command("audit")
def audit(
    source_path: Path = typer.Argument(...),
    output: Path = typer.Option(..., help="JSON audit report path."),
    device: Literal["cpu", "cuda"] = typer.Option(
        "cpu",
        "--device",
        help=(
            "Audit reconstruction device; CUDA is used for ConvRot INT8, "
            "W4A4, and packed ConvRot INT6/INT7; plain INT8/INT6/INT7 "
            "stay on CPU."
        ),
    ),
    overwrite: bool = typer.Option(False, help="Replace an existing JSON report."),
) -> None:
    """Measure supported weight reconstruction formats."""
    def action() -> None:
        result = audit_model(
            source_path,
            output,
            device=device,
            overwrite=overwrite,
            on_entry_started=_print_progress,
            preflight=_preflight_audit_output,
            audit_source=audit_bf16_source,
            write_document=_write_json,
        )
        summary = result.document["summary"]
        print_weight_audit_table(
            result.document["results"],
            source_dtype=str(result.document["selection"]["dtype"]),
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


@app.command("activation-audit", hidden=True)
@activation_app.command("audit")
def activation_audit(
    source_path: Path = typer.Argument(...),
    device: Literal["cpu", "cuda"] = typer.Option(
        "cpu",
        "--device",
        help=(
            "Candidate quantization device; CUDA accelerates ConvRot INT8, "
            "W4A4, and packed ConvRot INT6/INT7; plain INT8/INT6/INT7 "
            "stay on CPU."
        ),
    ),
    activation_calibration: Path = typer.Option(
        ...,
        "--activation-calibration",
        help=(
            "V1 or V2 calibration JSON; cvar stores compact observed metrics, "
            "full also stores sampled and bias-free statistics."
        ),
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
    timing: bool = typer.Option(
        False,
        "--timing",
        help="Report full activation-audit timing by candidate method.",
    ),
    overwrite: bool = typer.Option(False, help="Replace an existing cache pair."),
) -> None:
    """Build a reusable activation-aware candidate measurement cache."""
    def action() -> None:
        timing_started = perf_counter() if timing else None
        method_timings = {} if timing else None
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
            device=device,
            method_timings=method_timings,
            overwrite=overwrite,
        )
        _finish(
            {
                "source_path": str(source_path),
                "device": device,
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
        if timing:
            assert timing_started is not None
            assert method_timings is not None
            typer.echo(
                render_activation_audit_timing(
                    perf_counter() - timing_started,
                    method_timings,
                    device=device,
                    tensor_count=len(calibration.tensor_names()),
                )
            )

    _run("activation-audit", action)


@app.command("activation-score", hidden=True)
@activation_app.command("score")
def activation_score(
    audit_cache: Path = typer.Option(
        ...,
        "--audit-cache",
        help="Reusable activation-audit metadata JSON.",
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
        help=(
            "Cache metric. V2 cvar supports observed-relative metrics; "
            "sampled and bias-free metrics require full calibration."
        ),
    ),
    overwrite: bool = typer.Option(False, help="Replace an existing JSON report."),
) -> None:
    """Score a reusable activation-audit cache."""
    def action() -> None:
        report = score_activation_audit(
            audit_cache,
            activation_calibration,
            metric=metric,
        )
        scored_count = report["summary"]["available_candidate_count"]
        _write_json(output, report, overwrite)
        _finish(
            {
                "audit_cache": str(audit_cache),
                "activation_calibration": str(activation_calibration),
                "score_report": str(output),
                "scored_tensor_count": scored_count,
            }
        )

    _run("activation-score", action)


@app.command("activation-inspect", hidden=True)
@activation_app.command("inspect")
def activation_inspect(
    audit_cache: Path = typer.Option(
        ...,
        "--audit-cache",
        help="Activation-audit metadata JSON.",
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


@app.command("activation-optimize", hidden=True)
@activation_app.command("optimize")
def activation_optimize(
    audit_cache: Path = typer.Option(
        ...,
        "--audit-cache",
        help="Activation-audit metadata JSON.",
    ),
    activation_calibration: Path = typer.Option(
        ...,
        "--activation-calibration",
        help="Activation calibration metadata JSON.",
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
        help=(
            "Optimization metric. V2 cvar supports observed-relative metrics; "
            "sampled and bias-free metrics require full calibration."
        ),
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


@app.command("activation-compare", hidden=True)
@activation_app.command("compare")
def activation_compare(
    audit_cache: Path = typer.Option(
        ...,
        "--audit-cache",
        help="Activation-audit metadata JSON.",
    ),
    activation_calibration: Path = typer.Option(
        ...,
        "--activation-calibration",
        help="Activation calibration metadata JSON.",
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
        help=(
            "Comma-separated metrics; defaults to all, with unavailable full-only "
            "metrics reported for cvar calibration."
        ),
    ),
    top_n: int = typer.Option(20, "--top-n", help="Ranked rows per metric/method."),
    overwrite: bool = typer.Option(False, help="Replace an existing workbook."),
) -> None:
    """Compare all activation metrics in one workbook."""
    def action() -> None:
        target_bytes, promotion_budget_bytes = _resolve_profile_budget(
            target_size_mib,
            target_size_gib,
            promotion_budget_mib,
            promotion_budget_bytes_option,
        )
        requested_metrics = _split_comma_separated(metrics) or ACTIVATION_METRICS
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


@app.command("analyze", hidden=True)
@profile_app.command("analyze")
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
            audit_kwargs: dict[str, Any] = {
                "on_entry_started": _print_progress,
            }
            if tensor is not None:
                audit_kwargs["tensor_name"] = tensor
            else:
                audit_kwargs["tensor_names"] = tensor_names
            audit_document = audit_bf16_source(source, **audit_kwargs)
            for tensor_name in tensor_names:
                print_tensor_analysis(audit_document, source_header, tensor_name)
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
                else gib_to_bytes(target_size_gib)
            )
            profile_sweep = generate_profile_sweep(
                audit_document,
                "analysis",
                gib_to_bytes(target_size_step_gib),
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
                gib_to_bytes(target_size_gib),
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


@app.command("optimize", hidden=True)
@profile_app.command("optimize")
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
        result = optimize_profile(
            audit_path,
            output_path,
            profile_id=profile_id,
            target_size_gib=target_size_gib,
            method=method,
            max_relative_l2_error=max_relative_l2_error,
            exclude_prefix=exclude_prefix,
            exclude_suffix=exclude_suffix,
            overwrite=overwrite,
            config=config,
            output=output,
            dry_run=dry_run,
            load_config=load_optimize_config,
            load_audit=load_weight_audit,
            optimize=optimize_target_size,
            write=write_profile,
        )
        _finish(
            {
                "audit_path": str(result.audit_path),
                "profile_path": None if result.dry_run else str(result.output_path),
                "profile_id": result.profile_id,
                "target_bytes": result.optimized.target_bytes,
                "estimated_output_bytes": result.optimized.output_bytes,
                "reconstruction_sse": result.optimized.reconstruction_sse,
                "audited_global_relative_l2": audited_global_relative_l2(
                    result.audit_document,
                    result.optimized.reconstruction_sse,
                ),
                "dry_run": result.dry_run,
                "warning": "Generated profiles still require artifact and runtime validation.",
            },
        )

    _run("optimize", action)


@krea2_mlp_app.callback()
def prune_krea2_mlp(
    ctx: typer.Context,
    model_path: Path | None = typer.Option(
        None,
        "--model",
        help="Unquantized floating-point Krea2/Kroma checkpoint.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        help="Output checkpoint path.",
    ),
    start_channel: int | None = typer.Option(
        None,
        "--start-channel",
        help="First intermediate MLP channel to remove.",
    ),
    remove_channels: int | None = typer.Option(
        None,
        "--remove-channels",
        help="Number of contiguous intermediate MLP channels to remove.",
    ),
) -> None:
    """Physically prune contiguous channels when no subcommand is selected."""
    if ctx.invoked_subcommand is not None:
        if any(
            value is not None
            for value in (model_path, output_path, start_channel, remove_channels)
        ):
            raise typer.BadParameter(
                "contiguous pruning options cannot be combined with a subcommand"
            )
        return
    if (
        model_path is None
        or output_path is None
        or start_channel is None
        or remove_channels is None
    ):
        raise typer.BadParameter(
            "direct pruning requires --model, --output, --start-channel, "
            "and --remove-channels"
        )

    def action() -> None:
        result = prune_krea2_mlp_checkpoint(
            model_path,
            output_path,
            start_channel,
            remove_channels,
            on_plan=lambda plan: _print_krea2_prune_plan(plan, output_path),
        )
        typer.echo("Pruning completed.")
        typer.echo(f"Target tensors modified: {result.target_tensors_modified}")
        typer.echo(f"Other tensors copied:    {result.other_tensors_copied}")
        typer.echo(f"Output size:              {result.output_bytes} bytes")

    _run("prune krea2-mlp", action)


@krea2_mlp_app.command("weight-stats")
def krea2_mlp_weight_stats(
    model_path: Path = typer.Option(
        ...,
        "--model",
        help="Original unquantized floating-point Krea2/Kroma checkpoint.",
    ),
    output_path: Path = typer.Option(
        ...,
        "--output",
        help="FP32 down-column norm Safetensors artifact.",
    ),
) -> None:
    """Build static per-block down.weight column norms."""
    def action() -> None:
        result = build_krea2_mlp_weight_stats(model_path, output_path)
        typer.echo("Krea2 MLP weight stats")
        typer.echo("---------------------")
        typer.echo(f"Blocks:              {len(result.stats.block_indices)}")
        typer.echo(f"Hidden width:        {result.stats.hidden_width}")
        typer.echo(f"Original MLP width:  {result.stats.original_mlp_width}")
        typer.echo(f"Metric:              down_column_norm_sq")
        typer.echo(f"Output:              {result.output_path}")
        typer.echo(f"Output size:         {result.output_bytes} bytes")

    _run("prune krea2-mlp weight-stats", action)


@krea2_mlp_app.command("select")
def krea2_mlp_select(
    activation_stats_path: Path = typer.Option(
        ...,
        "--activation-stats",
        help="Activation-stats artifact or direct directory of same-stem pairs.",
    ),
    weight_stats_path: Path = typer.Option(
        ...,
        "--weight-stats",
        help="Static down-column norm artifact.",
    ),
    output_plan_path: Path = typer.Option(
        ...,
        "--output-plan",
        "--output",
        help="Cumulative JSON plan output.",
    ),
    base_plan_path: Path | None = typer.Option(
        None,
        "--base-plan",
        help="Optional prior cumulative plan.",
    ),
    add_channels_per_block: int = typer.Option(
        256,
        "--add-channels-per-block",
        help="New original-coordinate channels to add per block.",
    ),
    importance_output_path: Path | None = typer.Option(
        None,
        "--importance-output",
        help="Optional FP32 importance Safetensors artifact.",
    ),
) -> None:
    """Select the next cumulative pruning stage."""
    def action() -> None:
        result = select_krea2_mlp_plan(
            activation_stats_path,
            weight_stats_path,
            output_plan_path,
            base_plan_path=base_plan_path,
            add_channels_per_block=add_channels_per_block,
            importance_output_path=importance_output_path,
        )
        if activation_stats_path.is_dir():
            _print_krea2_activation_aggregation(activation_stats_path, result)
        _print_krea2_selection(result.plan, result.output_path)
        if result.importance_path is not None:
            typer.echo(f"Importance output:           {result.importance_path}")

    _run("prune krea2-mlp select", action)


@krea2_mlp_app.command("apply-plan")
def krea2_mlp_apply_plan(
    model_path: Path = typer.Option(
        ...,
        "--model",
        help="Original unquantized floating-point Krea2/Kroma checkpoint.",
    ),
    plan_path: Path = typer.Option(..., "--plan", help="Accepted cumulative JSON plan."),
    output_path: Path = typer.Option(..., "--output", help="Pruned BF16/F16/F32 output."),
) -> None:
    """Apply arbitrary per-block plan indices once to the original source."""
    def action() -> None:
        result = apply_krea2_mlp_plan(
            model_path,
            plan_path,
            output_path,
            on_plan=lambda plan: _print_krea2_prune_plan(plan, output_path),
        )
        typer.echo("Pruning completed.")
        typer.echo(f"Target tensors modified: {result.target_tensors_modified}")
        typer.echo(f"Other tensors copied:    {result.other_tensors_copied}")
        typer.echo(f"Output size:              {result.output_bytes} bytes")

    _run("prune krea2-mlp apply-plan", action)


@krea2_mlp_app.command("audit-summary")
def krea2_mlp_audit_summary(
    audit_path: Path = typer.Option(..., "--audit", help="Cascade-audit JSON artifact."),
    plan_path: Path | None = typer.Option(
        None,
        "--plan",
        help="Optional plan whose hash must match the audit candidate.",
    ),
) -> None:
    """Print final-output cascade-audit metrics without a threshold decision."""
    def action() -> None:
        audit = load_krea2_mlp_cascade_audit(audit_path)
        if plan_path is not None and audit.candidate_plan_hash != krea2_mlp_plan_sha256(plan_path):
            raise ValueError("Cascade audit candidate_plan_hash does not match the plan.")
        typer.echo(render_krea2_mlp_cascade_audit(audit))

    _run("prune krea2-mlp audit-summary", action)


@krea2_block_app.command("rank")
def krea2_block_rank(
    calibration_path: Path = typer.Option(..., "--calibration", help="Block-calibration artifact or direct directory."),
    output_path: Path = typer.Option(..., "--output", help="Candidate JSON output."),
    top_k: int = typer.Option(5, "--top-k", help="Lowest-impact candidates to shortlist."),
) -> None:
    """Rank active blocks from raw calibration sums."""
    def action() -> None:
        result = rank_krea2_blocks(calibration_path, output_path, top_k=top_k)
        typer.echo(f"Artifacts: {result['artifact_count']}; top-K: {result['top_k']}; output: {output_path}")
    _run("prune krea2-block rank", action)


@krea2_block_app.command("audit-summary")
def krea2_block_audit_summary(
    audit_path: Path = typer.Option(..., "--audit", help="Block cascade-audit artifact or direct directory."),
    output_path: Path = typer.Option(..., "--output", help="Audit-summary JSON output."),
) -> None:
    """Group candidate audits without accepting a block."""
    def action() -> None:
        result = summarize_krea2_block_audits(audit_path, output_path)
        typer.echo(f"Artifacts: {result['artifact_count']}; candidates: {len(result['ranking'])}; output: {output_path}")
    _run("prune krea2-block audit-summary", action)


@krea2_block_app.command("plan-add")
def krea2_block_plan_add(
    block: int = typer.Option(..., "--block", help="Original block ID to accept."),
    output_path: Path = typer.Option(..., "--output", help="Next cumulative plan JSON output."),
    block_count: int | None = typer.Option(None, "--block-count", help="Required without --base-plan."),
    base_plan_path: Path | None = typer.Option(None, "--base-plan", help="Optional prior cumulative plan."),
) -> None:
    """Explicitly add one original block ID to a cumulative plan."""
    def action() -> None:
        plan = add_krea2_block_to_plan(output_path, block=block, block_count=block_count, base_plan_path=base_plan_path)
        typer.echo(f"Removed blocks: {plan['removed_blocks']}; output: {output_path}")
    _run("prune krea2-block plan-add", action)


@krea2_block_app.command("apply-plan")
def krea2_block_apply_plan(
    model_path: Path = typer.Option(..., "--model", help="Original-topology Krea2/Kroma safetensors model."),
    plan_path: Path = typer.Option(..., "--plan", help="Cumulative original-ID block-prune plan."),
    output_path: Path = typer.Option(..., "--output", help="Physically pruned safetensors output."),
    manifest_path: Path | None = typer.Option(None, "--manifest", help="Optional mapping-manifest JSON output."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate and report without writing files."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace existing output and manifest files."),
) -> None:
    """Remove planned blocks; plans always use original block IDs."""
    def action() -> None:
        result = apply_krea2_block_plan(
            model_path, plan_path, output_path, manifest_path=manifest_path,
            dry_run=dry_run, overwrite=overwrite,
        )
        typer.echo(
            f"Original main blocks: {result.original_block_count}; removed: {list(result.removed_blocks)}; "
            f"remaining: {result.original_block_count - len(result.removed_blocks)}; "
            f"renamed tensors: {result.renamed_tensor_count}; removed tensors: {result.removed_tensor_count}; "
            f"removed payload bytes: {result.removed_payload_bytes}; dry run: {result.dry_run}"
        )
        if dry_run:
            typer.echo(f"Old-to-new mapping: {json.dumps(result.old_to_new, sort_keys=True)}")
        if not dry_run:
            typer.echo(f"Output: {result.output_path}; manifest: {result.manifest_path}")
    _run("prune krea2-block apply-plan", action)


@krea2_block_app.command("remap-profile")
def krea2_block_remap_profile(
    profile_path: Path = typer.Option(..., "--profile", help="Original-topology quantization profile JSON."),
    plan_path: Path = typer.Option(..., "--plan", help="Cumulative original-ID block-prune plan."),
    output_path: Path = typer.Option(..., "--output", help="Remapped quantization profile JSON."),
    source_model_path: Path | None = typer.Option(None, "--source-model", "--model", help="Optional original-topology model for assignment verification."),
    pruned_model_path: Path | None = typer.Option(None, "--pruned-model", help="Optional physical-prune output for full assignment verification."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate and report without writing a profile."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace an existing profile output."),
) -> None:
    """Move exact block-prefix quantization rules with surviving tensors."""
    def action() -> None:
        result = remap_krea2_block_profile(
            profile_path, plan_path, output_path, source_model_path=source_model_path,
            pruned_model_path=pruned_model_path, dry_run=dry_run, overwrite=overwrite,
        )
        typer.echo(
            f"Removed rules: {result.removed_rules}; rewritten rules: {result.rewritten_rules}; "
            f"unchanged rules: {result.unchanged_rules}; verified assignments: {result.verified_assignments}; "
            f"dry run: {result.dry_run}"
        )
        if dry_run:
            typer.echo(f"Old-to-new mapping: {json.dumps(result.old_to_new, sort_keys=True)}")
        if not dry_run:
            typer.echo(f"Output: {result.output_path}")
    _run("prune krea2-block remap-profile", action)


@krea2_app.command("apply-combined")
def krea2_apply_combined(
    model_path: Path = typer.Option(..., "--model", help="Original-topology, original-width Krea2/Kroma checkpoint."),
    block_plan_path: Path = typer.Option(..., "--block-plan", help="Cumulative original-ID block-prune plan."),
    mlp_plan_path: Path = typer.Option(..., "--mlp-plan", help="Original-topology Krea2 MLP plan."),
    quant_profile_path: Path = typer.Option(..., "--quant-profile", help="Original-topology quantization profile."),
    output_path: Path = typer.Option(..., "--output", help="Final block- and MLP-pruned checkpoint."),
    remapped_mlp_plan_path: Path | None = typer.Option(None, "--remapped-mlp-plan", help="Optional remapped MLP-plan sidecar path."),
    output_profile_path: Path | None = typer.Option(None, "--output-profile", help="Optional remapped quant-profile sidecar path."),
    manifest_path: Path | None = typer.Option(None, "--manifest", help="Optional combined-manifest sidecar path."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate plans and profile mapping without writing outputs."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace existing output artifacts."),
) -> None:
    """Apply coordinated block pruning, MLP pruning, and profile remapping."""
    def action() -> None:
        result = apply_krea2_combined_prune(
            model_path, block_plan_path, mlp_plan_path, quant_profile_path, output_path,
            remapped_mlp_plan_path=remapped_mlp_plan_path,
            output_profile_path=output_profile_path,
            manifest_path=manifest_path,
            dry_run=dry_run,
            overwrite=overwrite,
        )
        typer.echo(
            f"Blocks: {result.original_block_count} -> {result.original_block_count - len(result.removed_blocks)}; "
            f"removed originals: {list(result.removed_blocks)}; MLP width: "
            f"{result.original_mlp_width} -> {result.new_mlp_width}; dry run: {result.dry_run}"
        )
        if result.dry_run:
            typer.echo(f"Old-to-new mapping: {json.dumps(result.old_to_new, sort_keys=True)}")
            typer.echo(
                f"MLP entries: {result.original_block_count} -> "
                f"{result.original_block_count - len(result.removed_blocks)}; profile rules removed: "
                f"{result.removed_profile_rules}; rewritten: {result.rewritten_profile_rules}"
            )
        if not result.dry_run:
            typer.echo(f"Output: {result.output_path}")
            typer.echo(f"MLP plan: {result.remapped_mlp_plan_path}")
            typer.echo(f"Quant profile: {result.output_profile_path}")
            typer.echo(f"Manifest: {result.manifest_path}; verified assignments: {result.verified_assignments}")
    _run("prune krea2 apply-combined", action)


@app.command("quantize")
def quantize(
    source_path: Path | None = typer.Argument(None),
    output_path: Path | None = typer.Argument(None),
    profile: Path | None = typer.Option(
        None,
        help="JSON quantization profile; required without --config.",
    ),
    gptq_calibration: Path | None = typer.Option(
        None,
        "--gptq-calibration",
        help="Versioned GPTQ calibration JSON; matching Safetensors must be beside it.",
    ),
    device: Literal["cpu", "cuda"] = typer.Option(
        "cpu",
        "--device",
        help=(
            "Quantization device; CUDA is used for ConvRot INT8, W4A4, "
            "and packed INT5/INT6/INT7."
        ),
    ),
    adapter_path: list[Path] = typer.Option(
        [],
        "--adapter-path",
        help="LoRA path; repeat for multiple LoRA files.",
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
    timing: bool = typer.Option(
        False,
        "--timing",
        help="Report conversion timing by stage, action, and tensor size.",
    ),
) -> None:
    """Convert or estimate a checkpoint with an explicit JSON profile."""
    def action() -> None:
        result = quantize_model(
            source_path,
            output_path,
            profile=profile,
            gptq_calibration=gptq_calibration,
            device=device,
            adapter_path=adapter_path,
            adapter_strength=adapter_strength,
            config=config,
            dry_run=dry_run,
            timing=timing,
            on_entry_started=print_conversion_progress,
            build_adapters=_build_adapter_inputs,
            load_config=load_quantize_config,
            estimate=estimate_output_bytes,
            convert=convert_model_from_profile,
        )
        if result.estimated_bytes is not None:
            _finish(
                {
                    "source_path": str(result.source_path),
                    "profile_path": str(result.profile),
                    "estimated_output_bytes": result.estimated_bytes,
                    "estimated_output_mib": round(
                        result.estimated_bytes / 1024**2,
                        2,
                    ),
                    "estimated_output_gib": round(
                        result.estimated_bytes / 1024**3,
                        3,
                    ),
                    "dry_run": True,
                }
            )
            return
        _finish(
            {
                "source_path": str(result.source_path),
                "profile_path": str(result.profile),
                "output_path": str(result.output_path),
                "output_bytes": result.output_path.stat().st_size,
                "adapter_count": result.adapter_count,
            },
        )
        if result.timing is not None:
            typer.echo(result.timing.render())

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


if __name__ == "__main__":
    app()
