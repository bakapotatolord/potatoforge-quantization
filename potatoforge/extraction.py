from collections.abc import Iterator
from pathlib import Path

from .headers.header_reader import read_raw_data_start
from .planning import (
    OutputTensorSpec,
    build_layout_from_specs,
    source_bytes,
)
from .safetensors_writer import TensorPayload, write_safetensors_file
from .headers.source_header import read_source_model_header
from .source_payloads import read_source_tensor_bytes


def extract_tensors(
    source_path: str | Path,
    output_path: str | Path,
    source_prefix: str,
    output_prefix: str = "",
) -> tuple[int, int]:
    source = Path(source_path)
    output = Path(output_path)

    if source.resolve() == output.resolve():
        raise ValueError("Source and output paths cannot be the same.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    model = read_source_model_header(source)
    selected = [
        (name, descriptor, output_prefix + name[len(source_prefix):])
        for name, descriptor in model.tensors.items()
        if name.startswith(source_prefix)
    ]

    if not selected:
        raise ValueError(f"No tensors matched prefix: {source_prefix}")

    layout = build_layout_from_specs(
        OutputTensorSpec(
            name=output_name,
            dtype=descriptor["dtype"],
            shape=tuple(descriptor["shape"]),
            byte_count=source_bytes(descriptor),
        )
        for _, descriptor, output_name in selected
    )

    def payloads() -> Iterator[TensorPayload]:
        with source.open("rb") as file:
            raw_data_start = read_raw_data_start(
                file,
                file_label="Source file",
            )
            for _, descriptor, output_name in selected:
                source_start, source_end = descriptor["data_offsets"]
                byte_count = source_end - source_start
                yield output_name, read_source_tensor_bytes(
                    file,
                    raw_data_start,
                    (source_start, source_end),
                    byte_count,
                )

    partial = output.with_name(f"{output.name}.partial")
    if partial.exists():
        raise FileExistsError(f"Partial output already exists: {partial}")

    write_safetensors_file(
        partial,
        layout,
        payloads(),
        metadata=model.metadata or None,
    )
    partial.rename(output)
    return len(selected), layout.raw_data_bytes
