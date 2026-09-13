from __future__ import annotations

from typing import NamedTuple

import torch

from .hadamard import (
    CONVROT_GROUP_SIZE,
    apply_hadamard_rotation,
    cached_cuda_hadamard_matrix,
)
from ..timing import timed_internal_stage


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
    internal_timings: dict[str, float] | None = None,
    _timing_device: torch.device | None = None,
) -> Int8TensorwiseResult:
    """Quantize rows while allowing callers to state their INT8 contract."""

    if qmin >= qmax:
        raise ValueError("qmin must be smaller than qmax.")

    if use_float32_math:
        with timed_internal_stage(
            internal_timings,
            "prepare",
            _timing_device,
        ):
            math_values = values.float()

        with timed_internal_stage(
            internal_timings,
            "scale",
            _timing_device,
        ):
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

        with timed_internal_stage(
            internal_timings,
            "quantize_values",
            _timing_device,
        ):
            codes = torch.round(math_values / scales).clamp(qmin, qmax)
    else:
        if values.dtype == torch.float16:
            with timed_internal_stage(
                internal_timings,
                "prepare",
                _timing_device,
            ):
                math_values = values.float()
        else:
            math_values = values

        with timed_internal_stage(
            internal_timings,
            "scale",
            _timing_device,
        ):
            scales = (
                math_values.abs()
                .amax(dim=1, keepdim=True)
                .float()
                .div(qmax)
                .clamp_min(1e-30)
            )

        with timed_internal_stage(
            internal_timings,
            "quantize_values",
            _timing_device,
        ):
            scale_math = scales.to(math_values.dtype)
            tiny = torch.finfo(math_values.dtype).tiny
            scale_math = torch.where(
                scale_math == 0,
                torch.full_like(scale_math, tiny),
                scale_math,
            )
            codes = torch.round(math_values / scale_math).clamp(qmin, qmax)

    with timed_internal_stage(
        internal_timings,
        "finalize",
        _timing_device,
    ):
        return Int8TensorwiseResult(
            codes=codes.to(torch.int8),
            scales=scales.float(),
        )


def _validate_int8_tensorwise_weights(weights: torch.Tensor) -> None:
    if weights.dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ):
        raise ValueError(
            "INT8 tensorwise expects BF16, F16, or F32 weights, "
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

def quantize_int8_convrot(
    weights: torch.Tensor,
    *,
    internal_timings: dict[str, float] | None = None,
    device: str | torch.device = "cpu",
) -> Int8TensorwiseResult:
    _validate_int8_tensorwise_weights(weights)

    if weights.shape[1] % CONVROT_GROUP_SIZE != 0:
        raise ValueError(
            "ConvRot INT8 input features must be divisible by "
            f"{CONVROT_GROUP_SIZE}."
        )

    try:
        target_device = torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise ValueError("INT8 ConvRot device must be cpu or cuda.") from error

    if target_device.type not in ("cpu", "cuda"):
        raise ValueError("INT8 ConvRot device must be cpu or cuda.")
    if target_device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA device requested but CUDA is unavailable."
            )
        if target_device.index is None:
            target_device = torch.device(
                "cuda",
                torch.cuda.current_device(),
            )
        return _quantize_int8_convrot_cuda(
            weights,
            target_device,
            internal_timings,
        )

    with timed_internal_stage(
        internal_timings,
        "rotation",
        None,
    ):
        rotated_weights = apply_hadamard_rotation(
            weights,
            group_size=CONVROT_GROUP_SIZE,
            output_dtype=(
                torch.float32
                if weights.dtype in (torch.float16, torch.float32)
                else torch.bfloat16
            ),
        )
    return quantize_int8_rows(
        rotated_weights,
        internal_timings=internal_timings,
    )


def _quantize_int8_convrot_cuda(
    weights: torch.Tensor,
    device: torch.device,
    internal_timings: dict[str, float] | None,
) -> Int8TensorwiseResult:
    output_dtype = (
        torch.float32
        if weights.dtype in (torch.float16, torch.float32)
        else torch.bfloat16
    )
    with timed_internal_stage(
        internal_timings,
        "prepare",
        device,
    ):
        device_weights = weights.to(device=device)

    with timed_internal_stage(
        internal_timings,
        "rotation",
        device,
    ):
        hadamard = cached_cuda_hadamard_matrix(
            CONVROT_GROUP_SIZE,
            device,
            output_dtype,
        )
        rotated_weights = apply_hadamard_rotation(
            device_weights,
            group_size=CONVROT_GROUP_SIZE,
            output_dtype=output_dtype,
            hadamard=hadamard,
        )
    result = quantize_int8_rows(
        rotated_weights,
        internal_timings=internal_timings,
        _timing_device=device,
    )

    with timed_internal_stage(
        internal_timings,
        "finalize",
        device,
    ):
        return Int8TensorwiseResult(
            codes=result.codes.cpu(),
            scales=result.scales.cpu(),
        )

def dequantize_int8_convrot(result: Int8TensorwiseResult) -> torch.Tensor:
    rotated_weights = dequantize_int8_tensorwise(result)

    return apply_hadamard_rotation(
        rotated_weights,
        group_size=CONVROT_GROUP_SIZE,
        output_dtype=torch.float32,
    )
