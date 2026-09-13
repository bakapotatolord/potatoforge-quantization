"""Small, unpacked INT6 rowwise reference for measurement experiments."""

from __future__ import annotations

from typing import Final, NamedTuple

import torch

from .hadamard import (
    CONVROT_GROUP_SIZE,
    apply_hadamard_rotation,
    cached_cuda_hadamard_matrix,
)
from .int6_packing import (
    Int6PackedResult,
    pack_int6_row_major,
    unpack_int6_row_major,
)
from ..timing import timed_internal_stage
from .int8_tensorwise import quantize_int8_rows


INT6_QMAX: Final[int] = 31
INT6_QMIN: Final[int] = -31


class Int6RowwiseResult(NamedTuple):
    """Logical INT6 values stored in int8, plus one scale per output row."""

    codes: torch.Tensor
    scales: torch.Tensor


class Int6ConvRotPackedResult(NamedTuple):
    """Packed ConvRot INT6 values and row scales for serialization."""

    packed_codes: torch.Tensor
    scales: torch.Tensor
    original_shape: tuple[int, int]


def _validate_weight_metadata(
    weights: torch.Tensor,
    *,
    internal_timings: dict[str, float] | None = None,
) -> None:
    with timed_internal_stage(
        internal_timings,
        "validate_metadata",
        None,
    ):
        if weights.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(
                "INT6 weights must use float16, bfloat16, or float32."
            )
        if weights.ndim != 2:
            raise ValueError("INT6 weights must be a rank-2 matrix.")


def _validate_finite(
    weights: torch.Tensor,
    *,
    internal_timings: dict[str, float] | None = None,
) -> None:
    with timed_internal_stage(
        internal_timings,
        "validate_finite",
        None,
    ):
        finite = torch.isfinite(weights).all()
        if not bool(finite):
            raise ValueError("INT6 weights must contain only finite values.")


def _validate_weights(
    weights: torch.Tensor,
    *,
    internal_timings: dict[str, float] | None = None,
) -> None:
    _validate_weight_metadata(
        weights,
        internal_timings=internal_timings,
    )
    _validate_finite(weights, internal_timings=internal_timings)


def quantize_int6_rowwise(weights: torch.Tensor) -> Int6RowwiseResult:
    """Quantize weights to logical signed INT6 codes held in an int8 tensor."""

    _validate_weights(weights)
    result = quantize_int8_rows(
        weights,
        qmin=INT6_QMIN,
        qmax=INT6_QMAX,
        use_float32_math=True,
        zero_scale=1.0,
    )
    return Int6RowwiseResult(result.codes, result.scales)


def dequantize_int6_rowwise(result: Int6RowwiseResult) -> torch.Tensor:
    """Reconstruct float32 weights from an INT6 rowwise result."""

    if result.codes.dtype != torch.int8 or result.codes.ndim != 2:
        raise ValueError("INT6 codes must be a rank-2 int8 tensor.")
    if result.scales.dtype != torch.float32:
        raise ValueError("INT6 scales must be float32.")
    if not bool(torch.isfinite(result.scales).all()) or bool(
        (result.scales <= 0).any()
    ):
        raise ValueError("INT6 scales must be finite and positive.")
    if result.scales.shape != (result.codes.shape[0], 1):
        raise ValueError("INT6 scales must have shape [out_features, 1].")
    if bool((result.codes < INT6_QMIN).any()) or bool(
        (result.codes > INT6_QMAX).any()
    ):
        raise ValueError("INT6 codes must be in the signed range [-31, 31].")
    return result.codes.float() * result.scales


def quantize_int6_convrot(weights: torch.Tensor) -> Int6RowwiseResult:
    """Rotate then quantize W6 weights with Comfy's fixed ConvRot grouping."""

    rotated_weights = apply_hadamard_rotation(
        weights,
        group_size=CONVROT_GROUP_SIZE,
        output_dtype=torch.float32,
    )
    return quantize_int6_rowwise(rotated_weights)


def _resolve_device(device: str | torch.device) -> torch.device:
    try:
        target_device = torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise ValueError("INT6 ConvRot device must be cpu or cuda.") from error

    if target_device.type not in ("cpu", "cuda"):
        raise ValueError("INT6 ConvRot device must be cpu or cuda.")
    if target_device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is unavailable.")
        if target_device.index is None:
            target_device = torch.device(
                "cuda",
                torch.cuda.current_device(),
            )
    return target_device


def _quantize_int6_convrot_cuda(
    weights: torch.Tensor,
    device: torch.device,
    internal_timings: dict[str, float] | None,
) -> Int6ConvRotPackedResult:
    _validate_weight_metadata(
        weights,
        internal_timings=internal_timings,
    )
    with timed_internal_stage(
        internal_timings,
        "prepare",
        device,
    ):
        device_weights = weights.to(device=device)

    with timed_internal_stage(
        internal_timings,
        "validate_finite",
        device,
    ):
        _validate_finite(device_weights)

    with timed_internal_stage(
        internal_timings,
        "rotation",
        device,
    ):
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
        qmin=INT6_QMIN,
        qmax=INT6_QMAX,
        use_float32_math=True,
        zero_scale=1.0,
        internal_timings=internal_timings,
        _timing_device=device if internal_timings is not None else None,
    )
    with timed_internal_stage(
        internal_timings,
        "pack",
        device,
    ):
        packed_result = pack_int6_row_major(rowwise_result.codes)

    with timed_internal_stage(
        internal_timings,
        "finalize",
        device,
    ):
        return Int6ConvRotPackedResult(
            packed_codes=packed_result.packed_codes.cpu(),
            scales=rowwise_result.scales.cpu(),
            original_shape=packed_result.original_shape,
        )


def quantize_int6_convrot_packed(
    weights: torch.Tensor,
    *,
    device: str | torch.device = "cpu",
    internal_timings: dict[str, float] | None = None,
) -> Int6ConvRotPackedResult:
    """Quantize ConvRot INT6 and return only its serialization payload."""

    with timed_internal_stage(
        internal_timings,
        "resolve_device",
        None,
    ):
        target_device = _resolve_device(device)
    if target_device.type == "cuda":
        return _quantize_int6_convrot_cuda(
            weights,
            target_device,
            internal_timings,
        )

    rowwise_result = quantize_int6_convrot(weights)
    packed_result = pack_int6_row_major(rowwise_result.codes)
    return Int6ConvRotPackedResult(
        packed_codes=packed_result.packed_codes,
        scales=rowwise_result.scales,
        original_shape=packed_result.original_shape,
    )


def dequantize_int6_convrot(result: Int6RowwiseResult) -> torch.Tensor:
    """Reconstruct ConvRot W6 weights in their original Linear space."""

    return apply_hadamard_rotation(
        dequantize_int6_rowwise(result),
        group_size=CONVROT_GROUP_SIZE,
        output_dtype=torch.float32,
    )


def dequantize_int6_convrot_packed(
    result: Int6ConvRotPackedResult,
    *,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Reconstruct ConvRot W6 weights from a packed serialization result."""

    packed_codes = result.packed_codes
    scales = result.scales
    if device is not None:
        packed_codes = packed_codes.to(device=device)
        scales = scales.to(device=device)
    codes = unpack_int6_row_major(
        Int6PackedResult(packed_codes, result.original_shape),
    )
    return dequantize_int6_convrot(Int6RowwiseResult(codes, scales))
