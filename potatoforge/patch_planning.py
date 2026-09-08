import json
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

from .headers.source_header import (
    validate_data_offsets,
    validate_dtype,
    validate_shape,
)
from .planning import (
    OutputTensorSpec,
    SafetensorsLayout,
    TensorDescriptor,
    build_layout_from_specs,
    build_quantized_tensor_plan,
    source_bytes,
)
from .profiles import QuantizationAction


class PatchEntry(NamedTuple):
    source_tensor_name: str
    logical_layer_name: str
    source_data_offsets: tuple[int, int]
    source_dtype: str
    source_shape: tuple[int, ...]
    source_input_bytes: int
    action: QuantizationAction
    output_tensors: tuple[OutputTensorSpec, ...]
    estimated_bytes: int


class PatchPlan(NamedTuple):
    patch_id: str
    entries: tuple[PatchEntry, ...]
    layout: SafetensorsLayout
    selected_tensor_count: int
    generated_tensor_count: int
    source_bytes_to_read: int
    replacement_bytes: int


def _validated_offsets(
    descriptor: TensorDescriptor,
    tensor_name: str,
) -> tuple[int, int]:
    raw_offsets = descriptor.get("data_offsets")
    if isinstance(raw_offsets, tuple):
        raw_offsets = list(raw_offsets)
    try:
        start, end = validate_data_offsets(raw_offsets)
        return start, end
    except ValueError as error:
        raise ValueError(
            f"Invalid source offsets for {tensor_name}: {error}"
        ) from error


def _validated_shape(
    descriptor: TensorDescriptor,
    tensor_name: str,
) -> tuple[int, ...]:
    raw_shape = descriptor.get("shape")
    if isinstance(raw_shape, tuple):
        raw_shape = list(raw_shape)
    try:
        shape = validate_shape(raw_shape)
    except ValueError as error:
        raise ValueError(
            f"Invalid source shape for {tensor_name}: {error}"
        ) from error
    if any(dimension < 0 for dimension in shape):
        raise ValueError(
            f"Invalid source shape for {tensor_name}: "
            "dimensions cannot be negative."
        )
    return tuple(shape)


def _validated_dtype(
    descriptor: TensorDescriptor,
    tensor_name: str,
) -> str:
    try:
        dtype = validate_dtype(descriptor.get("dtype"))
    except ValueError as error:
        raise ValueError(
            f"Invalid source dtype for {tensor_name}: {error}"
        ) from error
    if not dtype:
        raise ValueError(
            f"Invalid source dtype for {tensor_name}: dtype cannot be empty."
        )
    return dtype


def build_patch_plan(
    source_header: Mapping[str, TensorDescriptor],
    tensor_name: str,
    action: QuantizationAction,
    patch_id: str,
) -> PatchPlan:
    if tensor_name not in source_header:
        raise ValueError(
            f"Patch layer does not exist in source checkpoint: {tensor_name}"
        )
    if not tensor_name.endswith(".weight"):
        raise ValueError(
            f"Patch layer must be a canonical .weight tensor: {tensor_name}"
        )

    descriptor = source_header[tensor_name]
    try:
        quantized_plan = build_quantized_tensor_plan(
            action,
            tensor_name,
            descriptor,
        )
    except ValueError as error:
        raise ValueError(
            f"Cannot patch {tensor_name} as {action}: {error}"
        ) from error

    entry = PatchEntry(
        source_tensor_name=tensor_name,
        logical_layer_name=tensor_name.removesuffix(".weight"),
        source_data_offsets=_validated_offsets(descriptor, tensor_name),
        source_dtype=_validated_dtype(descriptor, tensor_name),
        source_shape=_validated_shape(descriptor, tensor_name),
        source_input_bytes=source_bytes(descriptor),
        action=action,
        output_tensors=quantized_plan.output_tensors,
        estimated_bytes=quantized_plan.estimated_bytes,
    )
    output_tensors = quantized_plan.output_tensors
    return PatchPlan(
        patch_id=patch_id,
        entries=(entry,),
        layout=build_layout_from_specs(output_tensors),
        selected_tensor_count=1,
        generated_tensor_count=len(output_tensors),
        source_bytes_to_read=entry.source_input_bytes,
        replacement_bytes=entry.estimated_bytes,
    )


def build_patch_metadata(
    plan: PatchPlan,
    source_path: Path,
) -> dict[str, str]:
    return {
        "potatoforge_file_type": "quant_patch",
        "potatoforge_patch_format": "1",
        "potatoforge_patch_id": plan.patch_id,
        "potatoforge_patch_replaces": json.dumps(
            [entry.logical_layer_name for entry in plan.entries],
            separators=(",", ":"),
        ),
        "potatoforge_patch_source": source_path.name,
    }
