"""CPU reference quantization for rowwise INT5 ConvRot weights."""

from __future__ import annotations

from typing import Final, NamedTuple

import torch

from .hadamard import (
    CONVROT_GROUP_SIZE,
    apply_hadamard_rotation,
    cached_cuda_hadamard_matrix,
)
from .int5_packing import (
    Int5PackedResult,
    pack_int5_row_major,
    unpack_int5_row_major,
)
from .int8_tensorwise import quantize_int8_rows
from ..timing import timed_internal_stage


INT5_QMAX: Final[int] = 15
INT5_QMIN: Final[int] = -15


class Int5RowwiseResult(NamedTuple):
    """Logical INT5 values stored in int8, plus one scale per output row."""

    codes: torch.Tensor
    scales: torch.Tensor


class Int5ConvRotPackedResult(NamedTuple):
    """Packed ConvRot INT5 values and row scales for serialization."""

    packed_codes: torch.Tensor
    scales: torch.Tensor
    original_shape: tuple[int, int]


def _validate_weight_metadata(
    weights: torch.Tensor,
    *,
    internal_timings: dict[str, float] | None = None,
) -> None:
    with timed_internal_stage(internal_timings, "validate_metadata", None):
        if weights.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("INT5 weights must use float16, bfloat16, or float32.")
        if weights.ndim != 2:
            raise ValueError("INT5 weights must be a rank-2 matrix.")


def _validate_finite(
    weights: torch.Tensor,
    *,
    internal_timings: dict[str, float] | None = None,
) -> None:
    with timed_internal_stage(internal_timings, "validate_finite", None):
        if not bool(torch.isfinite(weights).all()):
            raise ValueError("INT5 weights must contain only finite values.")


def _validate_weights(weights: torch.Tensor) -> None:
    _validate_weight_metadata(weights)
    _validate_finite(weights)


def quantize_int5_rowwise(weights: torch.Tensor) -> Int5RowwiseResult:
    """Quantize weights to logical signed INT5 codes held in an int8 tensor."""

    _validate_weights(weights)
    result = quantize_int8_rows(
        weights,
        qmin=INT5_QMIN,
        qmax=INT5_QMAX,
        use_float32_math=True,
        zero_scale=1.0,
    )
    return Int5RowwiseResult(result.codes, result.scales)


def dequantize_int5_rowwise(result: Int5RowwiseResult) -> torch.Tensor:
    """Reconstruct float32 weights from an INT5 rowwise result."""

    if result.codes.dtype != torch.int8 or result.codes.ndim != 2:
        raise ValueError("INT5 codes must be a rank-2 int8 tensor.")
    if result.scales.dtype != torch.float32:
        raise ValueError("INT5 scales must be float32.")
    if not bool(torch.isfinite(result.scales).all()) or bool(
        (result.scales <= 0).any()
    ):
        raise ValueError("INT5 scales must be finite and positive.")
    if result.scales.shape != (result.codes.shape[0], 1):
        raise ValueError("INT5 scales must have shape [out_features, 1].")
    if bool((result.codes < -16).any()) or bool(
        (result.codes > INT5_QMAX).any()
    ):
        raise ValueError("INT5 codes must be in the signed range [-16, 15].")
    return result.codes.float() * result.scales


def quantize_int5_convrot(weights: torch.Tensor) -> Int5RowwiseResult:
    """Rotate then quantize W5 weights with Comfy's fixed ConvRot grouping."""

    _validate_weights(weights)
    if weights.shape[1] % CONVROT_GROUP_SIZE != 0:
        raise ValueError(
            "ConvRot INT5 input features must be divisible by "
            f"{CONVROT_GROUP_SIZE}."
        )
    rotated_weights = apply_hadamard_rotation(
        weights,
        group_size=CONVROT_GROUP_SIZE,
        output_dtype=torch.float32,
    )
    return quantize_int5_rowwise(rotated_weights)


def quantize_int5_convrot_packed(
    weights: torch.Tensor,
    *,
    device: str | torch.device = "cpu",
    internal_timings: dict[str, float] | None = None,
) -> Int5ConvRotPackedResult:
    """Quantize ConvRot INT5 and return only its serialization payload."""

    with timed_internal_stage(internal_timings, "resolve_device", None):
        target_device = _resolve_device(device)
    if target_device.type == "cuda":
        return _quantize_int5_convrot_cuda(
            weights,
            target_device,
            internal_timings,
        )
    rowwise_result = quantize_int5_convrot(weights)
    packed_result = pack_int5_row_major(rowwise_result.codes)
    return Int5ConvRotPackedResult(
        packed_codes=packed_result.packed_codes,
        scales=rowwise_result.scales,
        original_shape=packed_result.original_shape,
    )


def _resolve_device(device: str | torch.device) -> torch.device:
    try:
        target_device = torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise ValueError("INT5 ConvRot device must be cpu or cuda.") from error
    if target_device.type not in ("cpu", "cuda"):
        raise ValueError("INT5 ConvRot device must be cpu or cuda.")
    if target_device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is unavailable.")
        if target_device.index is None:
            target_device = torch.device("cuda", torch.cuda.current_device())
    return target_device


def _quantize_int5_convrot_cuda(
    weights: torch.Tensor,
    device: torch.device,
    internal_timings: dict[str, float] | None,
) -> Int5ConvRotPackedResult:
    _validate_weight_metadata(weights, internal_timings=internal_timings)
    with timed_internal_stage(internal_timings, "prepare", device):
        device_weights = weights.to(device=device)
    with timed_internal_stage(internal_timings, "validate_finite", device):
        _validate_finite(device_weights)
    with timed_internal_stage(internal_timings, "rotation", device):
        rotated_weights = apply_hadamard_rotation(
            device_weights,
            group_size=CONVROT_GROUP_SIZE,
            output_dtype=torch.float32,
            hadamard=cached_cuda_hadamard_matrix(
                CONVROT_GROUP_SIZE,
                device,
                torch.float32,
            ),
        )
    rowwise_result = quantize_int8_rows(
        rotated_weights,
        qmin=INT5_QMIN,
        qmax=INT5_QMAX,
        use_float32_math=True,
        zero_scale=1.0,
        internal_timings=internal_timings,
        _timing_device=device if internal_timings is not None else None,
    )
    with timed_internal_stage(internal_timings, "pack", device):
        packed_result = pack_int5_row_major(rowwise_result.codes)
    with timed_internal_stage(internal_timings, "finalize", device):
        return Int5ConvRotPackedResult(
            packed_codes=packed_result.packed_codes.cpu(),
            scales=rowwise_result.scales.cpu(),
            original_shape=packed_result.original_shape,
        )


def dequantize_int5_convrot(result: Int5RowwiseResult) -> torch.Tensor:
    """Reconstruct ConvRot W5 weights in their original Linear space."""

    return apply_hadamard_rotation(
        dequantize_int5_rowwise(result),
        group_size=CONVROT_GROUP_SIZE,
        output_dtype=torch.float32,
    )


def dequantize_int5_convrot_packed(
    result: Int5ConvRotPackedResult,
) -> torch.Tensor:
    """Reconstruct ConvRot W5 weights from a packed serialization result."""

    codes = unpack_int5_row_major(
        Int5PackedResult(result.packed_codes, result.original_shape),
    )
    return dequantize_int5_convrot(Int5RowwiseResult(codes, result.scales))
