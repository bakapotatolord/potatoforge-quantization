from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch

from ..timing import timed_internal_stage


class RowwiseQuantizationResult(NamedTuple):
    codes: torch.Tensor
    scales: torch.Tensor


RowwiseCandidateMSE = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]

_MSE_COARSE_MULTIPLIERS: tuple[float, ...] = tuple(
    index / 20 for index in range(2, 21)
)
_MSE_FINE_STEPS = 16


def validate_rowwise_weight_metadata(
    weights: torch.Tensor,
    *,
    format_name: str,
    internal_timings: dict[str, float] | None = None,
) -> None:
    with timed_internal_stage(internal_timings, "validate_metadata", None):
        if weights.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(
                f"{format_name} weights must use float16, bfloat16, or float32."
            )
        if weights.ndim != 2:
            raise ValueError(f"{format_name} weights must be a rank-2 matrix.")


def validate_rowwise_finite_weights(
    weights: torch.Tensor,
    *,
    format_name: str,
    internal_timings: dict[str, float] | None = None,
) -> None:
    with timed_internal_stage(internal_timings, "validate_finite", None):
        if not bool(torch.isfinite(weights).all()):
            raise ValueError(f"{format_name} weights must contain only finite values.")


def validate_rowwise_weights(
    weights: torch.Tensor,
    *,
    format_name: str,
    internal_timings: dict[str, float] | None = None,
) -> None:
    validate_rowwise_weight_metadata(
        weights,
        format_name=format_name,
        internal_timings=internal_timings,
    )
    validate_rowwise_finite_weights(
        weights,
        format_name=format_name,
        internal_timings=internal_timings,
    )


def validate_convrot_input_width(
    weights: torch.Tensor,
    *,
    format_name: str,
    group_size: int,
) -> None:
    if weights.shape[1] % group_size != 0:
        raise ValueError(
            f"ConvRot {format_name} input features must be divisible by {group_size}."
        )


def dequantize_rowwise(
    codes: torch.Tensor,
    scales: torch.Tensor,
    *,
    format_name: str,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    if codes.dtype != torch.int8 or codes.ndim != 2:
        raise ValueError(f"{format_name} codes must be a rank-2 int8 tensor.")
    if scales.dtype != torch.float32:
        raise ValueError(f"{format_name} scales must be float32.")
    if not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
        raise ValueError(f"{format_name} scales must be finite and positive.")
    if scales.shape != (codes.shape[0], 1):
        raise ValueError(
            f"{format_name} scales must have shape [out_features, 1]."
        )
    if bool((codes < qmin).any()) or bool((codes > qmax).any()):
        raise ValueError(
            f"{format_name} codes must be in the signed range [{qmin}, {qmax}]."
        )
    return codes.float() * scales


def resolve_convrot_device(
    device: str | torch.device,
    *,
    format_name: str,
) -> torch.device:
    try:
        target_device = torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise ValueError(
            f"{format_name} ConvRot device must be cpu or cuda."
        ) from error
    if target_device.type not in ("cpu", "cuda"):
        raise ValueError(f"{format_name} ConvRot device must be cpu or cuda.")
    if target_device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is unavailable.")
        if target_device.index is None:
            target_device = torch.device("cuda", torch.cuda.current_device())
    return target_device


def select_rowwise_absmax_scale(
    weights: torch.Tensor,
    *,
    qmax: int,
) -> torch.Tensor:
    math_weights = weights.float()
    max_abs = math_weights.abs().amax(dim=1, keepdim=True)
    return torch.where(
        max_abs == 0,
        torch.ones_like(max_abs),
        (max_abs / qmax).clamp_min(1e-30),
    )


def quantize_rows_with_scale(
    weights: torch.Tensor,
    scales: torch.Tensor,
    *,
    qmin: int,
    qmax: int,
) -> RowwiseQuantizationResult:
    stored_scales = scales.float()
    codes = torch.round(weights.float() / stored_scales).clamp(qmin, qmax)
    return RowwiseQuantizationResult(codes.to(torch.int8), stored_scales)


def calculate_rowwise_candidate_mse(
    weights: torch.Tensor,
    weights_float: torch.Tensor,
    scales: torch.Tensor,
    *,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    result = quantize_rows_with_scale(
        weights,
        scales,
        qmin=qmin,
        qmax=qmax,
    )
    return (result.codes.float() * result.scales - weights_float).square().mean(
        dim=1,
        keepdim=True,
    )


def run_rowwise_mse_coarse_search(
    weights: torch.Tensor,
    weights_float: torch.Tensor,
    base_scale: torch.Tensor,
    *,
    qmin: int,
    qmax: int,
    candidate_mse: RowwiseCandidateMSE | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        best_mse = torch.full_like(base_scale, float("inf"))
        best_multiplier = torch.ones_like(base_scale)
        for multiplier in _MSE_COARSE_MULTIPLIERS:
            candidate_multiplier = torch.full_like(best_multiplier, multiplier)
            scales = (base_scale * multiplier).clamp_min(1e-30)
            mse = (
                calculate_rowwise_candidate_mse(
                    weights,
                    weights_float,
                    scales,
                    qmin=qmin,
                    qmax=qmax,
                )
                if candidate_mse is None
                else candidate_mse(weights, weights_float, scales)
            )
            better = mse < best_mse
            best_mse = torch.where(better, mse, best_mse)
            best_multiplier = torch.where(
                better,
                candidate_multiplier,
                best_multiplier,
            )
        return best_mse, best_multiplier


def run_rowwise_mse_fine_search(
    weights: torch.Tensor,
    weights_float: torch.Tensor,
    base_scale: torch.Tensor,
    best_mse: torch.Tensor,
    best_multiplier: torch.Tensor,
    *,
    qmin: int,
    qmax: int,
    candidate_mse: RowwiseCandidateMSE | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        lower = (best_multiplier - 0.05).clamp_min(0.10)
        upper = (best_multiplier + 0.05).clamp_max(1.0)
        for step in range(1, _MSE_FINE_STEPS):
            fraction = step / _MSE_FINE_STEPS
            candidate_multiplier = lower + (upper - lower) * fraction
            scales = (base_scale * candidate_multiplier).clamp_min(1e-30)
            mse = (
                calculate_rowwise_candidate_mse(
                    weights,
                    weights_float,
                    scales,
                    qmin=qmin,
                    qmax=qmax,
                )
                if candidate_mse is None
                else candidate_mse(weights, weights_float, scales)
            )
            better = mse < best_mse
            best_mse = torch.where(better, mse, best_mse)
            best_multiplier = torch.where(
                better,
                candidate_multiplier,
                best_multiplier,
            )
        return best_mse, best_multiplier


def select_rowwise_mse_scale(
    weights: torch.Tensor,
    *,
    qmin: int,
    qmax: int,
    candidate_mse: RowwiseCandidateMSE | None = None,
) -> torch.Tensor:
    base_scale = select_rowwise_absmax_scale(weights, qmax=qmax)
    weights_float = weights.float()
    zero_rows = weights_float.abs().amax(dim=1, keepdim=True) == 0
    if bool(zero_rows.all()):
        return base_scale
    best_mse, best_multiplier = run_rowwise_mse_coarse_search(
        weights,
        weights_float,
        base_scale,
        qmin=qmin,
        qmax=qmax,
        candidate_mse=candidate_mse,
    )
    best_mse, best_multiplier = run_rowwise_mse_fine_search(
        weights,
        weights_float,
        base_scale,
        best_mse,
        best_multiplier,
        qmin=qmin,
        qmax=qmax,
        candidate_mse=candidate_mse,
    )
    return torch.where(
        zero_rows,
        base_scale,
        base_scale * best_multiplier,
    ).clamp_min(1e-30)


def quantize_symmetric_rows(
    values: torch.Tensor,
    *,
    qmin: int = -128,
    qmax: int = 127,
    use_float32_math: bool = False,
    zero_scale: float | None = None,
    internal_timings: dict[str, float] | None = None,
    _timing_device: torch.device | None = None,
) -> RowwiseQuantizationResult:
    """Quantize each row symmetrically into signed codes stored in int8."""

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
                    scales.clamp_min(1e-30),
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
        return RowwiseQuantizationResult(
            codes=codes.to(torch.int8),
            scales=scales.float(),
        )
