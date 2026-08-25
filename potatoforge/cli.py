"""Typer front door for the existing PotatoForge operations."""

from __future__ import annotations

import json
import math
import unittest
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import typer

from .audits.profile_optimizer import (
    SUPPORTED_METHODS,
    load_weight_audit,
    optimize_target_size,
    write_profile,
)
from .audits.weight_audit import (
    audit_bf16_source,
    print_weight_audit_table,
)
from .converter import convert_model_from_profile, print_conversion_progress
from .extraction import extract_tensors
from .headers.header_reader import read_header_from_safetensors
from .headers.source_header import read_source_model_header
from .lora.lora_discovery import inspect_adapter_header
from .lora.lora_merge import AdapterMergeInput, merge_bf16_adapters
from .planning import IOMode
from .config import load_optimize_config, load_quantize_config


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


def _gib_to_bytes(value: float) -> int:
    if not math.isfinite(value) or value <= 0:
        raise typer.BadParameter("must be a finite positive number")
    byte_count = int(value * (1024**3))
    if byte_count <= 0:
        raise typer.BadParameter("must represent at least one byte")
    return byte_count


def _print_progress(index: int, count: int, name: str) -> None:
    typer.echo(f"[{index}/{count}] {name}", err=True)


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
        report = {
            "source_path": str(model_path),
            "metadata": metadata,
            "model_info": tensors,
            "summary": {
                "metadata_count": len(metadata) if isinstance(metadata, dict) else 0,
                "tensor_count": len(tensors),
                "group_count": len(prefix_counts),
                "total_raw_payload_bytes": total_bytes,
            },
        }
        if output is not None:
            _write_json(output, report, overwrite)
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
    overwrite: bool = typer.Option(False, help="Replace an existing JSON report."),
) -> None:
    """Measure supported weight reconstruction formats."""
    def action() -> None:
        document = audit_bf16_source(
            source_path,
            on_entry_started=_print_progress,
        )
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


@app.command("optimize")
def optimize(
    audit_path: Path | None = typer.Argument(None),
    output_path: Path | None = typer.Argument(None),
    profile_id: str | None = typer.Option(None, help="Generated profile identifier."),
    target_size_gib: float | None = typer.Option(None, help="Target output size in GiB."),
    method: list[str] | None = typer.Option(
        None, "--method", help="Allowed method; repeat to select several."
    ),
    enable_potatoforge_int6_runtime: bool | None = typer.Option(
        None,
        "--enable-potatoforge-int6-runtime/--no-enable-potatoforge-int6-runtime",
    ),
    max_relative_l2_error: float | None = typer.Option(None),
    exclude_prefix: list[str] | None = typer.Option(None, "--exclude-prefix"),
    exclude_suffix: list[str] | None = typer.Option(None, "--exclude-suffix"),
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
            effective_methods = method or list(optimize_config.methods)
            effective_enable_int6 = (
                enable_potatoforge_int6_runtime
                if enable_potatoforge_int6_runtime is not None
                else optimize_config.enable_potatoforge_int6_runtime
            )
            effective_max_error = (
                max_relative_l2_error
                if max_relative_l2_error is not None
                else optimize_config.max_relative_l2_error
            )
            effective_prefixes = (
                tuple(exclude_prefix)
                if exclude_prefix is not None
                else optimize_config.exclude_prefixes
            )
            effective_suffixes = (
                tuple(exclude_suffix)
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
            effective_methods = method or list(SUPPORTED_METHODS)
            effective_enable_int6 = bool(enable_potatoforge_int6_runtime)
            effective_max_error = max_relative_l2_error
            effective_prefixes = tuple(exclude_prefix or ())
            effective_suffixes = tuple(exclude_suffix or ())
            effective_overwrite = bool(overwrite)

        optimized = optimize_target_size(
            load_weight_audit(effective_audit_path),
            effective_profile_id,
            _gib_to_bytes(effective_target_size_gib),
            frozenset(effective_methods),
            effective_max_error,
            effective_enable_int6,
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
                "proxy_distortion": optimized.proxy_distortion,
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
    io_mode: IOMode | None = typer.Option(
        None,
        "--io-mode",
        help="Source I/O mode for this conversion.",
    ),
    input_buffer_gib: float | None = typer.Option(
        None,
        "--input-buffer-gib",
        help="Whole-tensor input staging target for batched mode.",
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
) -> None:
    """Convert a checkpoint with an explicit JSON profile."""
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
            effective_io_mode = (
                io_mode
                if io_mode is not None
                else quantize_config.io_mode
            )
            effective_input_buffer_gib = (
                input_buffer_gib
                if input_buffer_gib is not None
                else quantize_config.input_buffer_gib
            )
            effective_adapters = (
                cli_adapters if cli_adapters else quantize_config.adapters
            )
        else:
            if source_path is None or output_path is None or profile is None:
                raise ValueError(
                    "Direct quantize requires SOURCE_PATH, OUTPUT_PATH, "
                    "and --profile."
                )
            effective_source_path = source_path
            effective_output_path = output_path
            effective_profile = profile
            effective_io_mode = io_mode or "batched"
            effective_input_buffer_gib = input_buffer_gib
            effective_adapters = cli_adapters

        if (
            effective_source_path is None
            or effective_output_path is None
            or effective_profile is None
        ):
            raise ValueError(
                "Quantize config requires paths.source, paths.profile, "
                "and paths.quantized_output."
            )

        input_buffer_bytes = (
            None
            if effective_input_buffer_gib is None
            else _gib_to_bytes(effective_input_buffer_gib)
        )
        resolved_io_mode = convert_model_from_profile(
            effective_source_path,
            effective_output_path,
            effective_profile,
            on_entry_started=print_conversion_progress,
            adapters=effective_adapters,
            io_mode=effective_io_mode,
            input_buffer_bytes=input_buffer_bytes,
        )
        _finish(
            {
                "source_path": str(effective_source_path),
                "profile_path": str(effective_profile),
                "output_path": str(effective_output_path),
                "output_bytes": effective_output_path.stat().st_size,
                "io_mode": resolved_io_mode.mode,
                "io_mode_requested": effective_io_mode,
                "io_mode_reason": resolved_io_mode.reason,
                "input_buffer_bytes": input_buffer_bytes,
                "adapter_count": len(effective_adapters),
            },
        )

    _run("quantize", action)


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
