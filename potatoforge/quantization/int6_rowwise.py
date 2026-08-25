"""Small, unpacked INT6 rowwise reference for measurement experiments."""

from __future__ import annotations

from typing import Final, NamedTuple

import torch

from .hadamard import CONVROT_GROUP_SIZE, apply_hadamard_rotation
from .int8_tensorwise import quantize_int8_rows


INT6_QMAX: Final[int] = 31
INT6_QMIN: Final[int] = -31


class Int6RowwiseResult(NamedTuple):
    """Logical INT6 values stored in int8, plus one scale per output row."""

    codes: torch.Tensor
    scales: torch.Tensor


def _validate_weights(weights: torch.Tensor) -> None:
    if weights.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("INT6 weights must use float16, bfloat16, or float32.")
    if weights.ndim != 2:
        raise ValueError("INT6 weights must be a rank-2 matrix.")
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("INT6 weights must contain only finite values.")


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


def dequantize_int6_convrot(result: Int6RowwiseResult) -> torch.Tensor:
    """Reconstruct ConvRot W6 weights in their original Linear space."""

    return apply_hadamard_rotation(
        dequantize_int6_rowwise(result),
        group_size=CONVROT_GROUP_SIZE,
        output_dtype=torch.float32,
    )
