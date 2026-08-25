from collections.abc import Iterator, Sequence
from math import prod
from pathlib import Path
from typing import BinaryIO, Callable, TypeAlias

from .planning import (
    CONVROT_W4A4_MARKER_PAYLOAD,
    InputBatchPlan,
    INT6_CONVROT_MARKER_PAYLOAD,
    INT6_ROWWISE_MARKER_PAYLOAD,
    INT8_CONVROT_MARKER_PAYLOAD,
    INT8_MARKER_PAYLOAD,
    PlanEntry,
    TensorDescriptor,
)
from .headers.header_reader import read_raw_data_start
from .safetensors_writer import TensorPayload
from .quantization.int6_rowwise import (
    quantize_int6_convrot,
    quantize_int6_rowwise,
)
from .quantization.int6_packing import pack_int6_row_major
from .quantization import quantize_int8_tensorwise, quantize_int8_convrot
from .quantization.convrot_w4a4 import quantize_convrot_w4a4

import torch

ProgressReporter: TypeAlias = Callable[[int, int, PlanEntry], None]
SourcePayloadTransform: TypeAlias = Callable[[PlanEntry, bytes], bytes]
Bf16TensorDescriptor: TypeAlias = tuple[str, TensorDescriptor]
Bf16TensorProgressReporter: TypeAlias = Callable[[int, int, str], None]

_TORCH_DTYPES: dict[str, torch.dtype] = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
}

_QUANTIZATION_SOURCE_DTYPES = frozenset(("BF16", "F16", "F32"))


def tensor_to_raw_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.device.type != "cpu":
        raise ValueError("Safetensors payload tensors must be on the CPU.")

    return tensor.contiguous().view(torch.uint8).flatten().numpy().tobytes()


def read_source_tensor_bytes(
    file: BinaryIO,
    raw_data_start: int,
    data_offsets: tuple[int, int],
    input_bytes: int,
) -> bytes:
    source_start, source_end = data_offsets

    if source_end - source_start != input_bytes:
        raise ValueError("Plan source offsets do not match input byte count.")

    file.seek(raw_data_start + source_start)
    raw_bytes = file.read(input_bytes)

    if len(raw_bytes) != input_bytes:
        raise ValueError("Source tensor payload was truncated.")

    return raw_bytes


def _read_source_batch(
    file: BinaryIO,
    raw_data_start: int,
    batch: InputBatchPlan,
) -> dict[str, bytes]:
    staged: dict[str, bytes] = {}

    for tensor in batch.tensors:
        if tensor.name in staged:
            raise ValueError(
                f"Batch contains duplicate source tensor: {tensor.name}."
            )
        staged[tensor.name] = read_source_tensor_bytes(
            file,
            raw_data_start,
            (tensor.data_start, tensor.data_end),
            tensor.nbytes,
        )

    return staged


def tensor_from_raw_bytes(
    raw_bytes: bytes,
    shape: Sequence[int],
    dtype: str,
    *,
    tensor_name: str = "tensor",
) -> torch.Tensor:
    torch_dtype = _TORCH_DTYPES.get(dtype)
    if torch_dtype is None:
        raise ValueError(
            f"Cannot load {tensor_name}: unsupported floating-point "
            f"dtype {dtype}."
        )

    expected_bytes = (
        prod(shape)
        * torch.empty((), dtype=torch_dtype).element_size()
    )

    if len(raw_bytes) != expected_bytes:
        raise ValueError(
            f"Payload size for {tensor_name} does not match its "
            f"{dtype} shape: expected {expected_bytes}, "
            f"got {len(raw_bytes)}."
        )

    return torch.frombuffer(
        bytearray(raw_bytes),
        dtype=torch_dtype,
    ).reshape(tuple(shape))


def stream_bf16_source_tensors(
    source_path: str | Path,
    descriptors: Sequence[Bf16TensorDescriptor],
    on_tensor_started: Bf16TensorProgressReporter | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    with open(str(source_path), "rb") as source_file:
        raw_data_start = read_raw_data_start(
            source_file,
            file_label="Source file",
        )
        tensor_count = len(descriptors)

        for tensor_index, (tensor_name, descriptor) in enumerate(
            descriptors,
            start=1,
        ):
            if descriptor["dtype"] not in _QUANTIZATION_SOURCE_DTYPES:
                raise ValueError(
                    f"Profile-free audit received unsupported dtype "
                    f"{descriptor['dtype']} "
                    f"for {tensor_name}."
                )

            if on_tensor_started is not None:
                on_tensor_started(tensor_index, tensor_count, tensor_name)

            source_start, source_end = descriptor["data_offsets"]
            raw_bytes = read_source_tensor_bytes(
                source_file,
                raw_data_start,
                (source_start, source_end),
                source_end - source_start,
            )
            yield tensor_name, tensor_from_raw_bytes(
                raw_bytes,
                descriptor["shape"],
                descriptor["dtype"],
                tensor_name=tensor_name,
            )


def _stream_entry_payloads(
    entry: PlanEntry,
    source_bytes: bytes,
) -> Iterator[TensorPayload]:
    action = entry["action"]

    if action == "keep":
        (output_spec,) = entry["output_tensors"]
        if output_spec.dtype == entry["source_dtype"]:
            yield output_spec.name, source_bytes
        elif output_spec.dtype == "BF16":
            weights = tensor_from_raw_bytes(
                source_bytes,
                entry["shape"],
                entry["source_dtype"],
                tensor_name=entry["tensor_name"],
            )
            yield output_spec.name, tensor_to_raw_bytes(
                weights.to(torch.bfloat16),
            )
        else:
            raise ValueError(
                "Unsupported kept-tensor dtype conversion: "
                f'{entry["source_dtype"]} -> {output_spec.dtype}'
            )
    elif action == "int8":
        weights = tensor_from_raw_bytes(
            source_bytes,
            entry["shape"],
            entry["source_dtype"],
            tensor_name=entry["tensor_name"],
        )
        result = quantize_int8_tensorwise(weights)

        (weight_spec, scale_spec, marker_spec) = entry["output_tensors"]

        yield weight_spec.name, tensor_to_raw_bytes(result.codes)
        yield scale_spec.name, tensor_to_raw_bytes(result.scales)
        yield marker_spec.name, INT8_MARKER_PAYLOAD
    elif action == "int6_rowwise":
        weights = tensor_from_raw_bytes(
            source_bytes,
            entry["shape"],
            entry["source_dtype"],
            tensor_name=entry["tensor_name"],
        )
        result = quantize_int6_rowwise(weights)

        (weight_spec, scale_spec, marker_spec) = entry["output_tensors"]

        packed = pack_int6_row_major(result.codes)
        yield weight_spec.name, tensor_to_raw_bytes(packed.packed_codes)
        yield scale_spec.name, tensor_to_raw_bytes(result.scales)
        yield marker_spec.name, INT6_ROWWISE_MARKER_PAYLOAD
    elif action == "int6_convrot":
        weights = tensor_from_raw_bytes(
            source_bytes,
            entry["shape"],
            entry["source_dtype"],
            tensor_name=entry["tensor_name"],
        )
        result = quantize_int6_convrot(weights)

        (weight_spec, scale_spec, marker_spec) = entry["output_tensors"]

        packed = pack_int6_row_major(result.codes)
        yield weight_spec.name, tensor_to_raw_bytes(packed.packed_codes)
        yield scale_spec.name, tensor_to_raw_bytes(result.scales)
        yield marker_spec.name, INT6_CONVROT_MARKER_PAYLOAD
    elif action == "int8_convrot":
        weights = tensor_from_raw_bytes(
            source_bytes,
            entry["shape"],
            entry["source_dtype"],
            tensor_name=entry["tensor_name"],
        )
        result = quantize_int8_convrot(weights)

        (
            weight_spec,
            scale_spec,
            marker_spec,
        ) = entry["output_tensors"]

        yield weight_spec.name, tensor_to_raw_bytes(result.codes)
        yield scale_spec.name, tensor_to_raw_bytes(result.scales)
        yield marker_spec.name, INT8_CONVROT_MARKER_PAYLOAD
    elif action == "convrot_w4a4":
        weights = tensor_from_raw_bytes(
            source_bytes,
            entry["shape"],
            entry["source_dtype"],
            tensor_name=entry["tensor_name"],
        )
        result = quantize_convrot_w4a4(weights)

        (
            weight_spec,
            scale_spec,
            marker_spec,
        ) = entry["output_tensors"]

        yield (
            weight_spec.name,
            tensor_to_raw_bytes(result.packed_codes),
        )
        yield (
            scale_spec.name,
            tensor_to_raw_bytes(result.scales),
        )
        yield marker_spec.name, CONVROT_W4A4_MARKER_PAYLOAD
    else:
        raise ValueError(f"Unknown payload action: {action}")


def stream_output_payloads(
    source_path: str | Path,
    entries: Sequence[PlanEntry],
    on_entry_started: ProgressReporter | None = None,
    *,
    source_payload_transform: SourcePayloadTransform | None = None,
) -> Iterator[TensorPayload]:
    with open(str(source_path), "rb") as f:
        raw_data_start = read_raw_data_start(
            f,
            file_label="Source file",
        )

        entry_count = len(entries)

        for entry_index, entry in enumerate(entries, start=1):
            if on_entry_started is not None:
                on_entry_started(entry_index, entry_count, entry)

            source_bytes = read_source_tensor_bytes(
                f,
                raw_data_start,
                entry["source_data_offsets"],
                entry["input_bytes"],
            )
            if source_payload_transform is not None:
                source_bytes = source_payload_transform(entry, source_bytes)

            yield from _stream_entry_payloads(entry, source_bytes)


def _stream_staged_batch_payloads(
    batch: InputBatchPlan,
    staged: dict[str, bytes],
    entry_lookup: dict[str, tuple[int, PlanEntry]],
    entry_count: int,
    on_entry_started: ProgressReporter | None,
    source_payload_transform: SourcePayloadTransform | None,
) -> Iterator[TensorPayload]:
    batch_entries: list[tuple[int, PlanEntry]] = []
    for tensor in batch.tensors:
        indexed_entry = entry_lookup.get(tensor.name)
        if indexed_entry is None:
            raise ValueError(
                f"Batch references unknown source tensor: {tensor.name}."
            )
        batch_entries.append(indexed_entry)

    batch_entries.sort(
        key=lambda indexed_entry: indexed_entry[0]
    )

    for entry_index, entry in batch_entries:
        if on_entry_started is not None:
            on_entry_started(
                entry_index + 1,
                entry_count,
                entry,
            )
        source_bytes = staged[entry["tensor_name"]]
        if source_payload_transform is not None:
            source_bytes = source_payload_transform(entry, source_bytes)
        yield from _stream_entry_payloads(entry, source_bytes)


def stream_batched_output_payloads(
    source_path: str | Path,
    entries: Sequence[PlanEntry],
    batches: Sequence[InputBatchPlan],
    on_entry_started: ProgressReporter | None = None,
    *,
    source_payload_transform: SourcePayloadTransform | None = None,
) -> Iterator[TensorPayload]:
    entry_count = len(entries)
    entry_lookup = {
        entry["tensor_name"]: (entry_index, entry)
        for entry_index, entry in enumerate(entries)
    }

    with open(str(source_path), "rb") as source_file:
        raw_data_start = read_raw_data_start(
            source_file,
            file_label="Source file",
        )

        for batch in batches:
            staged = _read_source_batch(
                source_file,
                raw_data_start,
                batch,
            )
            try:
                yield from _stream_staged_batch_payloads(
                    batch,
                    staged,
                    entry_lookup,
                    entry_count,
                    on_entry_started,
                    source_payload_transform,
                )
            finally:
                staged.clear()
