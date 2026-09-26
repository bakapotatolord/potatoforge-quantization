"""ConvRot W4A4 MSE plus block-diagonal GPTQ, using the standard W4A4 payload."""

from __future__ import annotations

import torch

from .convrot_w4a4 import (
    ConvRotW4A4Result,
    _quantize_w4_codes,
    pack_signed_int4_row_major,
    select_w4a4_mse_scale,
    select_w4a4_mse_scale_cuda,
)
from .gptq import (
    DAMP_PERCENT,
    GPTQ_BLOCKSIZE,
    gptq_factor,
    gptq_quantize as _gptq_quantize,
    prepare_factors,
    rotate_gram,
    solve_blocks,
)
from .hadamard import CONVROT_GROUP_SIZE, apply_hadamard_rotation, cached_cuda_hadamard_matrix
from .rowwise import resolve_convrot_device
from ..timing import timed_internal_stage


def _w4_column_quantizer(scales: torch.Tensor):
    def quantize(current: torch.Tensor) -> torch.Tensor:
        codes, stored_scales = _quantize_w4_codes(current[:, None], scales)
        return codes[:, 0].float() * stored_scales[:, 0]

    return quantize


def gptq_quantize(
    rotated_weights: torch.Tensor,
    inverse_factor: torch.Tensor,
    scales: torch.Tensor,
    blocksize: int = GPTQ_BLOCKSIZE,
) -> torch.Tensor:
    """Reference GPTQ column order with the existing W4 stored-scale rule."""
    return _gptq_quantize(rotated_weights, inverse_factor, _w4_column_quantizer(scales), blocksize)


def gptq_quantize_blocks(
    rotated_weights: torch.Tensor,
    factors: tuple[torch.Tensor, ...],
    scales: torch.Tensor,
    blocksize: int = GPTQ_BLOCKSIZE,
) -> torch.Tensor:
    return solve_blocks(rotated_weights, factors, _w4_column_quantizer(scales), blocksize)


def quantize_convrot_w4a4_mse_gptq(
    weights: torch.Tensor,
    factors: tuple[torch.Tensor, ...],
    *,
    device: str | torch.device = "cpu",
    timings: dict[str, float] | None = None,
) -> ConvRotW4A4Result:
    if weights.ndim != 2 or weights.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("GPTQ W4A4 weights must be rank-2 BF16/F16/F32.")
    if sum(factor.shape[0] for factor in factors) != weights.shape[1]:
        raise ValueError("GPTQ Gram factors do not cover the weight input width.")
    target_device = resolve_convrot_device(device, format_name="W4A4 GPTQ")
    timing_device = target_device if target_device.type == "cuda" else None
    with torch.no_grad():
        with timed_internal_stage(timings, "prepare", timing_device):
            device_weights = weights.to(device=target_device)
        with timed_internal_stage(timings, "rotation", timing_device):
            rotated_weights = apply_hadamard_rotation(
                device_weights,
                CONVROT_GROUP_SIZE,
                hadamard=(
                    cached_cuda_hadamard_matrix(CONVROT_GROUP_SIZE, target_device, torch.float32)
                    if target_device.type == "cuda" else None
                ),
            )
        with timed_internal_stage(timings, "mse_scale", timing_device):
            scales = (
                select_w4a4_mse_scale_cuda(rotated_weights, target_device)
                if target_device.type == "cuda" else select_w4a4_mse_scale(rotated_weights)
            )
        with timed_internal_stage(timings, "gptq_solve", timing_device):
            result = gptq_quantize_blocks(
                rotated_weights, tuple(factor.to(target_device) for factor in factors), scales
            )
        with timed_internal_stage(timings, "pack", timing_device):
            codes, stored_scales = _quantize_w4_codes(result, scales)
            packed = pack_signed_int4_row_major(codes)
        with timed_internal_stage(timings, "finalize", timing_device):
            return ConvRotW4A4Result(packed.cpu(), stored_scales.squeeze(1).cpu())
