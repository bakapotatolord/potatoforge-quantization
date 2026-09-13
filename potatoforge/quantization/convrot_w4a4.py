from __future__ import annotations

from collections.abc import Callable
from typing import Final, NamedTuple

import torch

from .hadamard import (
    CONVROT_GROUP_SIZE,
    apply_hadamard_rotation,
    cached_cuda_hadamard_matrix,
)
from ..timing import timed_internal_stage
from .w4a4_mse_native import load_w4a4_mse_candidate


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


_CandidateMSE = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]


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
    error = codes.float()
    error.mul_(stored_scales)
    error.sub_(weights_float)
    error.square_()
    return error.mean(
        dim=1,
        keepdim=True,
    )


def _w4a4_mse_zero_row_state(
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    weights_float = weights.float()
    zero_rows = weights_float.abs().amax(dim=1, keepdim=True) == 0
    return weights_float, zero_rows, bool(zero_rows.all())


def _run_w4a4_mse_coarse_search(
    weights: torch.Tensor,
    weights_float: torch.Tensor,
    base_scale: torch.Tensor,
    candidate_mse: _CandidateMSE | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        best_mse = torch.full_like(base_scale, float("inf"))
        best_multiplier = torch.ones_like(base_scale)
        evaluate_candidate = (
            _calculate_w4a4_candidate_mse
            if candidate_mse is None
            else candidate_mse
        )

        # ponytail: sequential grid search; batch candidates only if profiling
        # shows audit/conversion time warrants the extra memory.
        for multiplier in _W4A4_MSE_COARSE_MULTIPLIERS:
            candidate_multiplier = torch.full_like(
                best_multiplier,
                multiplier,
            )
            mse = evaluate_candidate(
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

        return best_mse, best_multiplier


def _run_w4a4_mse_fine_search(
    weights: torch.Tensor,
    weights_float: torch.Tensor,
    base_scale: torch.Tensor,
    best_mse: torch.Tensor,
    best_multiplier: torch.Tensor,
    candidate_mse: _CandidateMSE | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        lower = (best_multiplier - 0.05).clamp_min(0.10)
        upper = (best_multiplier + 0.05).clamp_max(1.0)
        evaluate_candidate = (
            _calculate_w4a4_candidate_mse
            if candidate_mse is None
            else candidate_mse
        )
        for step in range(1, _W4A4_MSE_FINE_STEPS):
            fraction = step / _W4A4_MSE_FINE_STEPS
            candidate_multiplier = lower + (upper - lower) * fraction
            mse = evaluate_candidate(
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

        return best_mse, best_multiplier


def _select_w4a4_mse_scale_prepared(
    weights: torch.Tensor,
    weights_float: torch.Tensor,
    base_scale: torch.Tensor,
    zero_rows: torch.Tensor,
    all_zero: bool,
) -> torch.Tensor:
    if all_zero:
        return base_scale

    best_mse, best_multiplier = _run_w4a4_mse_coarse_search(
        weights,
        weights_float,
        base_scale,
    )
    best_mse, best_multiplier = _run_w4a4_mse_fine_search(
        weights,
        weights_float,
        base_scale,
        best_mse,
        best_multiplier,
    )

    with torch.no_grad():
        return torch.where(
            zero_rows,
            base_scale,
            (base_scale * best_multiplier).clamp_min(_W4_SCALE_FLOOR),
        )


def select_w4a4_mse_scale(weights: torch.Tensor) -> torch.Tensor:
    _validate_w4_weights(weights)
    base_scale = _select_w4_absmax_scale(weights)
    weights_float, zero_rows, all_zero = _w4a4_mse_zero_row_state(weights)
    return _select_w4a4_mse_scale_prepared(
        weights,
        weights_float,
        base_scale,
        zero_rows,
        all_zero,
    )


def dequantize_w4_rowwise(result: W4RowwiseResult) -> torch.Tensor:
    codes = unpack_signed_int4_row_major(result.packed_codes)

    return codes.float() * result.scales


def _resolve_device(device: str | torch.device) -> torch.device:
    try:
        target_device = torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise ValueError("W4A4 ConvRot device must be cpu or cuda.") from error

    if target_device.type not in ("cpu", "cuda"):
        raise ValueError("W4A4 ConvRot device must be cpu or cuda.")
    if target_device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is unavailable.")
        if target_device.index is None:
            target_device = torch.device(
                "cuda",
                torch.cuda.current_device(),
            )
    return target_device


def _quantize_convrot_w4a4_cuda(
    weights: torch.Tensor,
    device: torch.device,
    internal_timings: dict[str, float] | None,
) -> ConvRotW4A4Result:
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

    with timed_internal_stage(
        internal_timings,
        "scale",
        device,
    ):
        scales = _select_w4_absmax_scale(rotated_weights)

    with timed_internal_stage(
        internal_timings,
        "quantize_values",
        device,
    ):
        codes_and_scales = _quantize_w4_codes(rotated_weights, scales)

    with timed_internal_stage(
        internal_timings,
        "pack",
        device,
    ):
        packed_codes = pack_signed_int4_row_major(codes_and_scales[0])

    with timed_internal_stage(
        internal_timings,
        "finalize",
        device,
    ):
        return ConvRotW4A4Result(
            packed_codes=packed_codes.cpu(),
            scales=codes_and_scales[1].squeeze(dim=1).cpu(),
        )


def _quantize_convrot_w4a4_mse_cuda(
    weights: torch.Tensor,
    device: torch.device,
    internal_timings: dict[str, float] | None,
) -> ConvRotW4A4Result:
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

    with timed_internal_stage(
        internal_timings,
        "scale_init",
        device,
    ):
        base_scale = _select_w4_absmax_scale(rotated_weights)

    with timed_internal_stage(
        internal_timings,
        "zero_row_check",
        device,
    ):
        weights_float, zero_rows, all_zero = _w4a4_mse_zero_row_state(
            rotated_weights,
        )

    best_mse = base_scale
    best_multiplier = base_scale
    if not all_zero:
        candidate_mse = load_w4a4_mse_candidate(
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        )
        with timed_internal_stage(
            internal_timings,
            "coarse_search",
            device,
        ):
            best_mse, best_multiplier = _run_w4a4_mse_coarse_search(
                rotated_weights,
                weights_float,
                base_scale,
                candidate_mse,
            )
        with timed_internal_stage(
            internal_timings,
            "fine_search",
            device,
        ):
            best_mse, best_multiplier = _run_w4a4_mse_fine_search(
                rotated_weights,
                weights_float,
                base_scale,
                best_mse,
                best_multiplier,
                candidate_mse,
            )

    def final_quantize() -> tuple[torch.Tensor, torch.Tensor]:
        if all_zero:
            scales = base_scale
        else:
            scales = torch.where(
                zero_rows,
                base_scale,
                (base_scale * best_multiplier).clamp_min(
                    _W4_SCALE_FLOOR,
                ),
            )
        return _quantize_w4_codes(rotated_weights, scales)

    with timed_internal_stage(
        internal_timings,
        "final_quantize",
        device,
    ):
        codes_and_scales = final_quantize()

    with timed_internal_stage(
        internal_timings,
        "pack",
        device,
    ):
        packed_codes = pack_signed_int4_row_major(codes_and_scales[0])

    with timed_internal_stage(
        internal_timings,
        "finalize",
        device,
    ):
        return ConvRotW4A4Result(
            packed_codes=packed_codes.cpu(),
            scales=codes_and_scales[1].squeeze(dim=1).cpu(),
        )


def quantize_convrot_w4a4(
    weights: torch.Tensor,
    *,
    device: str | torch.device = "cpu",
    internal_timings: dict[str, float] | None = None,
) -> ConvRotW4A4Result:
    target_device = _resolve_device(device)
    if target_device.type == "cuda":
        return _quantize_convrot_w4a4_cuda(
            weights,
            target_device,
            internal_timings,
        )

    rotated_weights = apply_hadamard_rotation(
        weights,
        group_size=CONVROT_GROUP_SIZE,
    )
    rowwise_result = quantize_w4_rowwise(rotated_weights)

    return ConvRotW4A4Result(
        packed_codes=rowwise_result.packed_codes,
        scales=rowwise_result.scales.squeeze(dim=1),
    )


def quantize_convrot_w4a4_mse(
    weights: torch.Tensor,
    *,
    device: str | torch.device = "cpu",
    internal_timings: dict[str, float] | None = None,
) -> ConvRotW4A4Result:
    target_device = _resolve_device(device)
    if target_device.type == "cuda":
        return _quantize_convrot_w4a4_mse_cuda(
            weights,
            target_device,
            internal_timings,
        )

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
