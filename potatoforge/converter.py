from collections.abc import Sequence
from pathlib import Path

import torch

from .planning import (
    PlanEntry,
    build_output_layout,
    build_plan,
    build_quantization_metadata,
)
from .profiles import QuantizationProfile, load_profile
from .safetensors_writer import encode_safetensors_header, write_safetensors_file
from .headers.source_header import read_source_model_header
from .lora.lora_merge import (
    AdapterMergeInput,
    build_adapter_merger,
)
from .source_payloads import (
    ProgressReporter,
    stream_output_payloads,
)
from .timing import TimingCollector


def _validate_device(device: str) -> None:
    if device not in ("cpu", "cuda"):
        raise ValueError("Quantization device must be cpu or cuda.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA device requested but CUDA is unavailable."
        )


def print_conversion_progress(entry_index: int, entry_count: int, entry: PlanEntry) -> None:
    print(
        f"[{entry_index}/{entry_count}] "
        f"{entry['action']}: {entry['tensor_name']}",
        flush=True,
    )


def estimate_output_bytes(
    source_path: str | Path,
    profile_path: str | Path,
) -> int:
    source = read_source_model_header(source_path)
    profile = load_profile(profile_path)
    entries = build_plan(source.tensors, profile)
    layout = build_output_layout(entries)
    metadata = {
        **source.metadata,
        **build_quantization_metadata(entries),
    }
    return (
        8
        + len(encode_safetensors_header(layout, metadata))
        + layout.raw_data_bytes
    )


def convert_model(
    source_path: str | Path,
    output_path: str | Path,
    profile: QuantizationProfile,
    on_entry_started: ProgressReporter | None = print_conversion_progress,
    *,
    adapters: Sequence[AdapterMergeInput] = (),
    timing: TimingCollector | None = None,
    device: str = "cpu",
) -> None:
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

    _validate_device(device)

    model = read_source_model_header(source)

    plan_entries = build_plan(model.tensors, profile)
    layout = build_output_layout(plan_entries)

    source_payload_transform = build_adapter_merger(model, adapters)
    payloads = stream_output_payloads(
        source,
        plan_entries,
        on_entry_started,
        source_payload_transform=source_payload_transform,
        timing=timing,
        device=device,
    )

    write_safetensors_file(
        partial,
        layout,
        payloads,
        metadata={
            **model.metadata,
            **build_quantization_metadata(plan_entries),
        },
        timing=timing,
    )

    partial.rename(output)
    if timing is not None:
        timing.finish()


def convert_model_from_profile(
    source_path: str | Path,
    output_path: str | Path,
    profile_path: str | Path,
    on_entry_started: ProgressReporter | None = print_conversion_progress,
    *,
    adapters: Sequence[AdapterMergeInput] = (),
    timing: TimingCollector | None = None,
    device: str = "cpu",
) -> None:
    profile = load_profile(profile_path)
    return convert_model(
        source_path,
        output_path,
        profile,
        on_entry_started,
        adapters=adapters,
        timing=timing,
        device=device,
    )
