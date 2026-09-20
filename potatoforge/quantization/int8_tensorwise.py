from __future__ import annotations

import torch

from .hadamard import (
    CONVROT_GROUP_SIZE,
    apply_hadamard_rotation,
    cached_cuda_hadamard_matrix,
)
from ..timing import timed_internal_stage
from .rowwise import RowwiseQuantizationResult, quantize_symmetric_rows


Int8TensorwiseResult = RowwiseQuantizationResult


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

    return quantize_symmetric_rows(weights)


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
    return quantize_symmetric_rows(
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
    result = quantize_symmetric_rows(
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
