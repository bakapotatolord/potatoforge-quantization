from collections.abc import Iterator, Sequence
from math import prod
from pathlib import Path
from typing import BinaryIO, Callable, TypeAlias

from .planning import (
    CONVROT_W4A4_MARKER_PAYLOAD,
    INT6_CONVROT_MARKER_PAYLOAD,
    INT6_ROWWISE_MARKER_PAYLOAD,
    INT8_CONVROT_MARKER_PAYLOAD,
    INT8_MARKER_PAYLOAD,
    OutputTensorSpec,
    PlanEntry,
    QUANTIZATION_SOURCE_DTYPES,
    TensorDescriptor,
)
from .headers.header_reader import read_raw_data_start
from .safetensors_writer import TensorPayload
from .quantization.int6_rowwise import (
    quantize_int6_convrot,
    quantize_int6_rowwise,
)
from .quantization.int6_packing import pack_int6_row_major
from .quantization import (
    quantize_int8_convrot,
    quantize_int8_tensorwise,
)
from .quantization.convrot_w4a4 import (
    quantize_convrot_w4a4,
    quantize_convrot_w4a4_mse,
)
from .profiles import QuantizationAction

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
            if descriptor["dtype"] not in QUANTIZATION_SOURCE_DTYPES:
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
        return

    weights = tensor_from_raw_bytes(
        source_bytes,
        entry["shape"],
        entry["source_dtype"],
        tensor_name=entry["tensor_name"],
    )
    if action == "int8":
        result = quantize_int8_tensorwise(weights)
        code_bytes = tensor_to_raw_bytes(result.codes)
        marker_payload = INT8_MARKER_PAYLOAD
    elif action == "int6_rowwise":
        result = quantize_int6_rowwise(weights)
        code_bytes = tensor_to_raw_bytes(
            pack_int6_row_major(result.codes).packed_codes
        )
        marker_payload = INT6_ROWWISE_MARKER_PAYLOAD
    elif action == "int6_convrot":
        result = quantize_int6_convrot(weights)
        code_bytes = tensor_to_raw_bytes(
            pack_int6_row_major(result.codes).packed_codes
        )
        marker_payload = INT6_CONVROT_MARKER_PAYLOAD
    elif action == "int8_convrot":
        result = quantize_int8_convrot(weights)
        code_bytes = tensor_to_raw_bytes(result.codes)
        marker_payload = INT8_CONVROT_MARKER_PAYLOAD
    elif action in (
        "convrot_w4a4",
        "convrot_w4a4_mse",
    ):
        if action == "convrot_w4a4_mse":
            result = quantize_convrot_w4a4_mse(weights)
        else:
            result = quantize_convrot_w4a4(weights)
        code_bytes = tensor_to_raw_bytes(result.packed_codes)
        marker_payload = CONVROT_W4A4_MARKER_PAYLOAD
    else:
        raise ValueError(f"Unknown payload action: {action}")

    weight_spec, scale_spec, marker_spec = entry["output_tensors"]
    yield weight_spec.name, code_bytes
    yield scale_spec.name, tensor_to_raw_bytes(result.scales)
    yield marker_spec.name, marker_payload


def stream_quantized_payloads(
    source_bytes: bytes,
    *,
    tensor_name: str,
    source_dtype: str,
    shape: Sequence[int],
    action: QuantizationAction,
    output_tensors: tuple[OutputTensorSpec, ...],
) -> Iterator[TensorPayload]:
    if action == "keep":
        raise ValueError("Patch replacements must use a quantizing action.")

    entry: PlanEntry = {
        "tensor_name": tensor_name,
        "source_dtype": source_dtype,
        "shape": tuple(shape),
        "input_bytes": len(source_bytes),
        "action": action,
        "estimated_bytes": sum(
            tensor.byte_count for tensor in output_tensors
        ),
        "output_tensors": output_tensors,
        "source_data_offsets": (0, len(source_bytes)),
    }
    yield from _stream_entry_payloads(entry, source_bytes)

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
