from __future__ import annotations

from typing import NamedTuple

import torch

from .hadamard import (
    CONVROT_GROUP_SIZE,
    apply_hadamard_rotation,
)

class Int8TensorwiseResult(NamedTuple):
    codes: torch.Tensor
    scales: torch.Tensor


def quantize_int8_rows(
    values: torch.Tensor,
    *,
    qmin: int = -128,
    qmax: int = 127,
    use_float32_math: bool = False,
    zero_scale: float | None = None,
) -> Int8TensorwiseResult:
    """Quantize rows while allowing callers to state their INT8 contract."""

    if qmin >= qmax:
        raise ValueError("qmin must be smaller than qmax.")

    if use_float32_math:
        math_values = values.float()
        max_abs = math_values.abs().amax(dim=1, keepdim=True)
        scales = max_abs / qmax
        if zero_scale is None:
            scales = scales.clamp_min(1e-30)
        else:
            scales = torch.where(
                max_abs == 0,
                torch.full_like(max_abs, zero_scale),
                scales,
            )
        codes = torch.round(math_values / scales).clamp(qmin, qmax)
    else:
        math_values = (
            values.float()
            if values.dtype == torch.float16
            else values
        )
        scales = (
            math_values.abs()
            .amax(dim=1, keepdim=True)
            .float()
            .div(qmax)
            .clamp_min(1e-30)
        )
        scale_math = scales.to(math_values.dtype)
        tiny = torch.finfo(math_values.dtype).tiny
        scale_math = torch.where(
            scale_math == 0,
            torch.full_like(scale_math, tiny),
            scale_math,
        )
        codes = torch.round(math_values / scale_math).clamp(qmin, qmax)

    return Int8TensorwiseResult(
        codes=codes.to(torch.int8),
        scales=scales.float(),
    )


def _validate_int8_tensorwise_weights(weights: torch.Tensor) -> None:
    if weights.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            "INT8 tensorwise expects BF16 or F16 weights, "
            f"got {weights.dtype}."
        )

    if weights.ndim != 2:
        raise ValueError(
            "INT8 tensorwise expects a 2D weight matrix, "
            f"got {weights.ndim} dimensions."
        )


def quantize_int8_tensorwise(weights: torch.Tensor) -> Int8TensorwiseResult:
    _validate_int8_tensorwise_weights(weights)

    return quantize_int8_rows(weights)


def dequantize_int8_tensorwise(result: Int8TensorwiseResult) -> torch.Tensor:
    return result.codes.float() * result.scales

def quantize_int8_convrot(weights: torch.Tensor) -> Int8TensorwiseResult:
    _validate_int8_tensorwise_weights(weights)

    if weights.shape[1] % CONVROT_GROUP_SIZE != 0:
        raise ValueError(
            "ConvRot INT8 input features must be divisible by "
            f"{CONVROT_GROUP_SIZE}."
        )

    rotated_weights = apply_hadamard_rotation(
        weights,
        group_size=CONVROT_GROUP_SIZE,
        output_dtype=(
            torch.float32
            if weights.dtype == torch.float16
            else torch.bfloat16
        ),
    )

    return quantize_int8_rows(rotated_weights)

def dequantize_int8_convrot(result: Int8TensorwiseResult) -> torch.Tensor:
    rotated_weights = dequantize_int8_tensorwise(result)

    return apply_hadamard_rotation(
        rotated_weights,
        group_size=CONVROT_GROUP_SIZE,
        output_dtype=torch.float32,
    )
