from __future__ import annotations

from collections.abc import Callable
from typing import Final, NamedTuple

import torch

from .hadamard import (
    CONVROT_GROUP_SIZE,
    apply_hadamard_rotation,
    cached_cuda_hadamard_matrix,
)
from .rowwise import divide_exactly, resolve_convrot_device
from ..timing import timed_internal_stage
from .mse_search_native import (
    fused_rowwise_mse_coarse_search,
    fused_rowwise_mse_fine_search,
)
from .w4a4_mse_native import load_w4a4_mse_candidate


_W4_QMIN: Final[int] = -7
_W4_QMAX: Final[int] = 7
_W4A4_FORMAT_NAME: Final[str] = "W4A4"
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

    low_nibbles = codes[:, 0::2] & 0x0F
    high_nibbles = codes[:, 1::2] & 0x0F

    # INT8 shifts preserve the packed byte pattern even when its sign bit is set.
    return low_nibbles | (high_nibbles << 4)

def unpack_signed_int4_row_major(packed: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.int8:
        raise ValueError("packed INT4 data must use torch.int8 storage")

    if packed.ndim != 2:
        raise ValueError("packed INT4 data must be a 2D matrix")

    unpacked = torch.empty(
        (packed.shape[0], packed.shape[1] * 2),
        dtype=torch.int8,
        device=packed.device,
    )
    # INT8 right shifts sign-extend each nibble directly in its output position.
    torch.bitwise_left_shift(packed, 4, out=unpacked[:, 0::2])
    unpacked[:, 0::2].bitwise_right_shift_(4)
    torch.bitwise_right_shift(packed, 4, out=unpacked[:, 1::2])
    return unpacked

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


def _w4_absmax_and_scale(
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    math_weights = _w4_math_weights(weights)
    max_abs = math_weights.abs().amax(dim=1, keepdim=True).float()
    return max_abs, divide_exactly(max_abs, _W4_QMAX).clamp_min(_W4_SCALE_FLOOR)


def _select_w4_absmax_scale(weights: torch.Tensor) -> torch.Tensor:
    _, scales = _w4_absmax_and_scale(weights)
    return scales


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

    codes = (math_weights / scale_math).round_().clamp_(
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
    math_weights = _w4_math_weights(weights)
    stored_scales = scales.float().clamp_min(_W4_SCALE_FLOOR)
    # Search needs rounded values, not INT8 storage for every candidate.
    codes = (math_weights / stored_scales.to(math_weights.dtype)).round_().clamp_(
        _W4_QMIN,
        _W4_QMAX,
    )
    error = codes.detach().float()
    error.mul_(stored_scales)
    error.sub_(weights_float)
    error.square_()
    return error.mean(
        dim=1,
        keepdim=True,
    )


def _w4a4_mse_zero_row_state(
    weights: torch.Tensor,
    max_abs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    weights_float = weights.float()
    if max_abs is None:
        max_abs = weights_float.abs().amax(dim=1, keepdim=True)
    zero_rows = max_abs == 0
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
    max_abs, base_scale = _w4_absmax_and_scale(weights)
    weights_float, zero_rows, all_zero = _w4a4_mse_zero_row_state(weights, max_abs)
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


def _quantize_convrot_w4a4_cuda(
    weights: torch.Tensor,
    device: torch.device,
    internal_timings: dict[str, float] | None,
    *,
    return_on_device: bool = False,
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

    del device_weights
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
        result_device = device if return_on_device else "cpu"
        return ConvRotW4A4Result(
            packed_codes=packed_codes.to(device=result_device),
            scales=codes_and_scales[1].squeeze(dim=1).to(device=result_device),
        )


def select_w4a4_mse_scale_cuda(
    rotated_weights: torch.Tensor,
    device: torch.device,
    internal_timings: dict[str, float] | None = None,
) -> torch.Tensor:
    with timed_internal_stage(
        internal_timings,
        "scale_init",
        device,
    ):
        max_abs, base_scale = _w4_absmax_and_scale(rotated_weights)

    with timed_internal_stage(
        internal_timings,
        "zero_row_check",
        device,
    ):
        weights_float, zero_rows, all_zero = _w4a4_mse_zero_row_state(
            rotated_weights,
            max_abs,
        )

    best_mse = base_scale
    best_multiplier = base_scale
    fused = None
    if not all_zero:
        with timed_internal_stage(internal_timings, "coarse_search", device):
            fused = fused_rowwise_mse_coarse_search(
                rotated_weights,
                base_scale,
                qmin=_W4_QMIN,
                qmax=_W4_QMAX,
                scale_floor=_W4_SCALE_FLOOR,
            )
        if fused is not None:
            with timed_internal_stage(internal_timings, "fine_search", device):
                best_mse, best_multiplier = fused_rowwise_mse_fine_search(
                    rotated_weights,
                    base_scale,
                    *fused,
                    qmin=_W4_QMIN,
                    qmax=_W4_QMAX,
                    scale_floor=_W4_SCALE_FLOOR,
                )
    if fused is None and not all_zero:
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

    if all_zero:
        return base_scale
    return torch.where(
        zero_rows,
        base_scale,
        (base_scale * best_multiplier).clamp_min(_W4_SCALE_FLOOR),
    )


def _quantize_convrot_w4a4_mse_cuda(
    weights: torch.Tensor,
    device: torch.device,
    internal_timings: dict[str, float] | None,
    *,
    return_on_device: bool = False,
) -> ConvRotW4A4Result:
    with timed_internal_stage(internal_timings, "prepare", device):
        device_weights = weights.to(device=device)
    with timed_internal_stage(internal_timings, "rotation", device):
        rotated_weights = apply_hadamard_rotation(
            device_weights,
            group_size=CONVROT_GROUP_SIZE,
            output_dtype=torch.float32,
            hadamard=cached_cuda_hadamard_matrix(CONVROT_GROUP_SIZE, device, torch.float32),
        )
    del device_weights
    scales = select_w4a4_mse_scale_cuda(rotated_weights, device, internal_timings)
    with timed_internal_stage(internal_timings, "final_quantize", device):
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
        result_device = device if return_on_device else "cpu"
        return ConvRotW4A4Result(
            packed_codes=packed_codes.to(device=result_device),
            scales=codes_and_scales[1].squeeze(dim=1).to(device=result_device),
        )


def quantize_convrot_w4a4(
    weights: torch.Tensor,
    *,
    device: str | torch.device = "cpu",
    internal_timings: dict[str, float] | None = None,
    return_on_device: bool = False,
) -> ConvRotW4A4Result:
    target_device = resolve_convrot_device(
        device,
        format_name=_W4A4_FORMAT_NAME,
    )
    if target_device.type == "cuda":
        return _quantize_convrot_w4a4_cuda(
            weights,
            target_device,
            internal_timings,
            return_on_device=return_on_device,
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
    return_on_device: bool = False,
) -> ConvRotW4A4Result:
    target_device = resolve_convrot_device(
        device,
        format_name=_W4A4_FORMAT_NAME,
    )
    if target_device.type == "cuda":
        return _quantize_convrot_w4a4_mse_cuda(
            weights,
            target_device,
            internal_timings,
            return_on_device=return_on_device,
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
