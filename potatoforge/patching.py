from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Callable, TypeAlias

from .headers.header_reader import read_raw_data_start
from .headers.source_header import read_source_model_header
from .patch_planning import PatchPlan
from .safetensors_writer import TensorPayload, write_safetensors_file
from .source_payloads import (
    read_source_tensor_bytes,
    stream_quantized_payloads,
)


PatchProgressReporter: TypeAlias = Callable[[str], None]


def _validate_patch_paths(source_path: Path, output_path: Path) -> Path:
    source = source_path.resolve()
    output = output_path.resolve()
    if source == output:
        raise ValueError("Source and output paths cannot be the same.")
    if not source.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {source_path}")
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output_path}"
        )

    partial = output_path.with_name(f"{output_path.name}.partial")
    if partial.exists():
        raise FileExistsError(f"Partial output already exists: {partial}")
    return partial


def _validate_file_ranges(
    file_path: Path,
    ranges: Iterable[tuple[str, tuple[int, int]]],
    file_label: str,
) -> None:
    with file_path.open("rb") as file:
        raw_data_start = read_raw_data_start(file, file_label=file_label)
        file.seek(0, 2)
        file_size = file.tell()

    if raw_data_start > file_size:
        raise ValueError(f"{file_label} header extends beyond the file.")
    for tensor_name, (start, end) in ranges:
        if start < 0 or end < start:
            raise ValueError(f"Invalid offsets for {tensor_name}.")
        if raw_data_start + end > file_size:
            raise ValueError(
                f"{file_label} payload range for {tensor_name} extends beyond "
                "the file."
            )


def _stream_patch_payloads(
    source_path: Path,
    plan: PatchPlan,
    on_progress: PatchProgressReporter | None,
) -> Iterator[TensorPayload]:
    with source_path.open("rb") as source_file:
        raw_data_start = read_raw_data_start(
            source_file,
            file_label="Source file",
        )
        for index, entry in enumerate(plan.entries, start=1):
            if on_progress is not None:
                on_progress(
                    f"[{index}/{len(plan.entries)}] "
                    f"{entry.action}: {entry.source_tensor_name}"
                )
            source_bytes = read_source_tensor_bytes(
                source_file,
                raw_data_start,
                entry.source_data_offsets,
                entry.source_input_bytes,
            )
            yield from stream_quantized_payloads(
                source_bytes,
                tensor_name=entry.source_tensor_name,
                source_dtype=entry.source_dtype,
                shape=entry.source_shape,
                action=entry.action,
                output_tensors=entry.output_tensors,
            )


def execute_patch_plan(
    source_path: str | Path,
    output_path: str | Path,
    plan: PatchPlan,
    metadata: Mapping[str, str] | None = None,
    *,
    on_progress: PatchProgressReporter | None = None,
) -> None:
    source = Path(source_path)
    output = Path(output_path)
    partial = _validate_patch_paths(source, output)
    _validate_file_ranges(
        source,
        (
            (entry.source_tensor_name, entry.source_data_offsets)
            for entry in plan.entries
        ),
        "Source file",
    )

    try:
        if on_progress is not None:
            on_progress("Writing quantization patch")
        write_safetensors_file(
            partial,
            plan.layout,
            _stream_patch_payloads(source, plan, on_progress),
            metadata=metadata,
        )
        if on_progress is not None:
            on_progress("Validating output")
        output_header = read_source_model_header(partial)
        _validate_file_ranges(
            partial,
            (
                (tensor_name, tuple(descriptor["data_offsets"]))
                for tensor_name, descriptor in output_header.tensors.items()
            ),
            "Output file",
        )
        partial.rename(output)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
