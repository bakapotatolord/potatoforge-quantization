from collections.abc import Sequence
from pathlib import Path

from .planning import (
    IOMode,
    PlanEntry,
    ResolvedIOMode,
    build_output_layout,
    build_plan,
    plan_input_batches,
    resolve_io_mode,
)
from .profiles import QuantizationProfile, load_profile
from .safetensors_writer import write_safetensors_file
from .headers.source_header import read_source_model_header
from .lora.lora_merge import (
    AdapterMergeInput,
    build_adapter_merger,
)
from .source_payloads import (
    ProgressReporter,
    stream_batched_output_payloads,
    stream_output_payloads,
)


def print_conversion_progress(entry_index: int, entry_count: int, entry: PlanEntry) -> None:
    print(
        f"[{entry_index}/{entry_count}] "
        f"{entry['action']}: {entry['tensor_name']}",
        flush=True,
    )


def convert_model(
    source_path: str | Path,
    output_path: str | Path,
    profile: QuantizationProfile,
    on_entry_started: ProgressReporter | None = print_conversion_progress,
    *,
    adapters: Sequence[AdapterMergeInput] = (),
    io_mode: IOMode = "batched",
    input_buffer_bytes: int | None = None,
) -> ResolvedIOMode:
    source = Path(source_path)
    output = Path(output_path)
    partial = output.with_name(f"{output.name}.partial")

    if source.resolve() == output.resolve():
        raise ValueError("Source and Output path cannot be the same")

    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output}"
        )

    if partial.exists():
        raise FileExistsError(
            f"Partial output already exists: {partial}"
        )

    resolved_io_mode = resolve_io_mode(
        io_mode,
        input_buffer_bytes,
    )
    if on_entry_started is not None:
        print(
            f"I/O mode: {resolved_io_mode.mode} "
            f"({resolved_io_mode.reason})",
            flush=True,
        )

    model = read_source_model_header(source)

    plan_entries = build_plan(model.tensors, profile)
    layout = build_output_layout(plan_entries)

    source_payload_transform = build_adapter_merger(model, adapters)
    if resolved_io_mode.mode == "serial":
        payloads = stream_output_payloads(
            source,
            plan_entries,
            on_entry_started,
            source_payload_transform=source_payload_transform,
        )
    elif resolved_io_mode.mode == "batched":
        if input_buffer_bytes is None:
            raise ValueError(
                "Buffered conversion requires input_buffer_bytes."
            )
        batches = plan_input_batches(plan_entries, input_buffer_bytes)
        payloads = stream_batched_output_payloads(
            source,
            plan_entries,
            batches,
            on_entry_started,
            source_payload_transform=source_payload_transform,
        )
    else:
        raise ValueError(f"Unknown I/O mode: {io_mode}")

    write_safetensors_file(
        partial,
        layout,
        payloads,
        metadata=model.metadata or None,
    )

    partial.rename(output)
    return resolved_io_mode


def convert_model_from_profile(
    source_path: str | Path,
    output_path: str | Path,
    profile_path: str | Path,
    on_entry_started: ProgressReporter | None = print_conversion_progress,
    *,
    adapters: Sequence[AdapterMergeInput] = (),
    io_mode: IOMode = "batched",
    input_buffer_bytes: int | None = None,
) -> ResolvedIOMode:
    profile = load_profile(profile_path)
    return convert_model(
        source_path,
        output_path,
        profile,
        on_entry_started,
        adapters=adapters,
        io_mode=io_mode,
        input_buffer_bytes=input_buffer_bytes,
    )
