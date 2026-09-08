from __future__ import annotations

from typing import Final, NamedTuple

import torch

from .hadamard import CONVROT_GROUP_SIZE, apply_hadamard_rotation


_W4_QMIN: Final[int] = -7
_W4_QMAX: Final[int] = 7
_W4_SCALE_FLOOR: Final[float] = 1e-10
_W4A4_MSE_COARSE_MULTIPLIERS: Final[tuple[float, ...]] = tuple(
    index / 20 for index in range(2, 21)
)
_W4A4_MSE_FINE_STEPS: Final[int] = 16


class W4RowwiseResult(NamedTuple):
    packed_codes: torch.Tensor
    scales: torch.Tensor


class ConvRotW4A4Result(NamedTuple):
    packed_codes: torch.Tensor
    scales: torch.Tensor


def pack_signed_int4_row_major(codes: torch.Tensor) -> torch.Tensor:
    if codes.dtype != torch.int8:
        raise ValueError("INT4 codes must use torch.int8 storage")

    if codes.ndim != 2:
        raise ValueError("INT4 codes must be a 2D matrix")

    if codes.shape[1] % 2 != 0:
        raise ValueError("INT4 code rows must have an even width")

    if torch.any(codes < -8) or torch.any(codes > 7):
        raise ValueError("INT4 codes must be in the range [-8, 7]")

    low_nibbles = codes[:, 0::2].to(torch.int32) & 0x0F
    high_nibbles = codes[:, 1::2].to(torch.int32) & 0x0F

    return (low_nibbles | (high_nibbles << 4)).to(torch.int8)

def unpack_signed_int4_row_major(packed: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.int8:
        raise ValueError("packed INT4 data must use torch.int8 storage")

    if packed.ndim != 2:
        raise ValueError("packed INT4 data must be a 2D matrix")

    packed_i32 = packed.to(torch.int32)

    low_nibbles = packed_i32 & 0x0F
    high_nibbles = (packed_i32 >> 4) & 0x0F

    low_codes = torch.where(
        low_nibbles >= 8,
        low_nibbles - 16,
        low_nibbles,
    )
    high_codes = torch.where(
        high_nibbles >= 8,
        high_nibbles - 16,
        high_nibbles,
    )

    return torch.stack(
        (low_codes, high_codes),
        dim=2,
    ).flatten(start_dim=1).to(torch.int8)

def _validate_w4_weights(weights: torch.Tensor) -> None:
    if weights.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(
            "W4 weights must use torch.bfloat16, torch.float16, "
            "or torch.float32"
        )

    if weights.ndim != 2:
        raise ValueError("W4 weights must be a 2D matrix")


def _w4_math_weights(weights: torch.Tensor) -> torch.Tensor:
    return weights.float() if weights.dtype == torch.float16 else weights


def _select_w4_absmax_scale(weights: torch.Tensor) -> torch.Tensor:
    math_weights = _w4_math_weights(weights)
    return (
        math_weights.abs()
        .amax(dim=1, keepdim=True)
        .float()
        .div(_W4_QMAX)
        .clamp_min(_W4_SCALE_FLOOR)
    )


def _quantize_w4_rowwise_with_scale(
    weights: torch.Tensor,
    scales: torch.Tensor,
) -> W4RowwiseResult:
    codes, stored_scales = _quantize_w4_codes(weights, scales)
    return W4RowwiseResult(
        packed_codes=pack_signed_int4_row_major(codes),
        scales=stored_scales,
    )


def _quantize_w4_codes(
    weights: torch.Tensor,
    scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    math_weights = (
        weights.float()
        if weights.dtype == torch.float16
        else weights
    )

    stored_scales = scales.float().clamp_min(_W4_SCALE_FLOOR)
    scale_math = stored_scales.to(math_weights.dtype)

    codes = torch.round(math_weights / scale_math).clamp(
        _W4_QMIN,
        _W4_QMAX,
    ).to(torch.int8)

    return codes, stored_scales


def quantize_w4_rowwise(weights: torch.Tensor) -> W4RowwiseResult:
    _validate_w4_weights(weights)
    return _quantize_w4_rowwise_with_scale(
        weights,
        _select_w4_absmax_scale(weights),
    )


def _calculate_w4a4_candidate_mse(
    weights: torch.Tensor,
    weights_float: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    codes, stored_scales = _quantize_w4_codes(weights, scales)
    reconstructed = codes.float() * stored_scales
    return (weights_float - reconstructed).square().mean(
        dim=1,
        keepdim=True,
    )


def select_w4a4_mse_scale(weights: torch.Tensor) -> torch.Tensor:
    _validate_w4_weights(weights)
    base_scale = _select_w4_absmax_scale(weights)
    weights_float = weights.float()
    zero_rows = weights_float.abs().amax(dim=1, keepdim=True) == 0
    if bool(zero_rows.all()):
        return base_scale

    with torch.no_grad():
        best_mse = torch.full_like(base_scale, float("inf"))
        best_multiplier = torch.ones_like(base_scale)

        # ponytail: sequential grid search; batch candidates only if profiling
        # shows audit/conversion time warrants the extra memory.
        for multiplier in _W4A4_MSE_COARSE_MULTIPLIERS:
            candidate_multiplier = torch.full_like(
                best_multiplier,
                multiplier,
            )
            mse = _calculate_w4a4_candidate_mse(
                weights,
                weights_float,
                base_scale * multiplier,
            )
            better = mse < best_mse
            best_mse = torch.where(better, mse, best_mse)
            best_multiplier = torch.where(
                better,
                candidate_multiplier,
                best_multiplier,
            )

        lower = (best_multiplier - 0.05).clamp_min(0.10)
        upper = (best_multiplier + 0.05).clamp_max(1.0)
        for step in range(1, _W4A4_MSE_FINE_STEPS):
            fraction = step / _W4A4_MSE_FINE_STEPS
            candidate_multiplier = lower + (upper - lower) * fraction
            mse = _calculate_w4a4_candidate_mse(
                weights,
                weights_float,
                base_scale * candidate_multiplier,
            )
            better = mse < best_mse
            best_mse = torch.where(better, mse, best_mse)
            best_multiplier = torch.where(
                better,
                candidate_multiplier,
                best_multiplier,
            )

        return torch.where(
            zero_rows,
            base_scale,
            (base_scale * best_multiplier).clamp_min(_W4_SCALE_FLOOR),
        )


def dequantize_w4_rowwise(result: W4RowwiseResult) -> torch.Tensor:
    codes = unpack_signed_int4_row_major(result.packed_codes)

    return codes.float() * result.scales


def quantize_convrot_w4a4(weights: torch.Tensor) -> ConvRotW4A4Result:
    rotated_weights = apply_hadamard_rotation(
        weights,
        group_size=CONVROT_GROUP_SIZE,
    )
    rowwise_result = quantize_w4_rowwise(rotated_weights)

    return ConvRotW4A4Result(
        packed_codes=rowwise_result.packed_codes,
        scales=rowwise_result.scales.squeeze(dim=1),
    )


def quantize_convrot_w4a4_mse(weights: torch.Tensor) -> ConvRotW4A4Result:
    rotated_weights = apply_hadamard_rotation(
        weights,
        group_size=CONVROT_GROUP_SIZE,
    )
    rowwise_result = _quantize_w4_rowwise_with_scale(
        rotated_weights,
        select_w4a4_mse_scale(rotated_weights),
    )

    return ConvRotW4A4Result(
        packed_codes=rowwise_result.packed_codes,
        scales=rowwise_result.scales.squeeze(dim=1),
    )


def dequantize_convrot_w4a4(result: ConvRotW4A4Result) -> torch.Tensor:
    rotated_codes = unpack_signed_int4_row_major(
        result.packed_codes,
    )

    rotated_weights = (
        rotated_codes.float()
        * result.scales.unsqueeze(dim=1)
    )

    return apply_hadamard_rotation(
        rotated_weights,
        group_size=CONVROT_GROUP_SIZE,
    )
