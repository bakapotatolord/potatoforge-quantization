"""Pure reducers for cached activation-aware candidate measurements."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch


EvaluationReducer = Literal[
    "mean",
    "energy_weighted_mean",
    "global_ratio",
    "max",
    "p90",
    "p95",
    "cvar10",
    "cvar20",
]


def diagonal_reference_output_energy(
    reference_weights: torch.Tensor,
    eval_sum_x2: torch.Tensor,
) -> torch.Tensor:
    """Estimate reference output energy using diagonal input energy."""
    if reference_weights.ndim != 2 or eval_sum_x2.ndim != 2:
        raise ValueError("Reference weights and eval_sum_x2 must be rank 2.")
    if eval_sum_x2.shape[1] != reference_weights.shape[1]:
        raise ValueError(
            "Reference weights and eval_sum_x2 input features do not match."
        )
    weights = reference_weights.float()
    input_energy = eval_sum_x2.float()
    _require_finite(weights, "reference_weights")
    _require_nonnegative(input_energy, "eval_sum_x2")
    output_energy = torch.mm(input_energy, weights.square().transpose(0, 1))
    _require_finite(output_energy, "diagonal reference output energy")
    _require_nonnegative(output_energy, "diagonal reference output energy")
    return output_energy


def evaluation_relative_sse(
    error_by_eval_output: torch.Tensor,
    eval_sum_y2: torch.Tensor,
    *,
    absolute_floor: float = 0.0,
) -> torch.Tensor:
    """Return per-evaluation observed-output relative SSE."""
    error, output_energy = _validate_measurement_pair(
        error_by_eval_output,
        eval_sum_y2,
    )
    numerator = error.sum(dim=1)
    denominator = output_energy.sum(dim=1)
    return _safe_ratio(numerator, denominator, absolute_floor=absolute_floor)


def aggregate_observed_relative_sse(
    error_by_eval_output: torch.Tensor,
    eval_sum_y2: torch.Tensor,
    *,
    absolute_floor: float = 0.0,
) -> float:
    """Return the global ratio using observed calibration output energy."""
    error, output_energy = _validate_measurement_pair(
        error_by_eval_output,
        eval_sum_y2,
    )
    numerator = error.sum()
    denominator = output_energy.sum()
    return float(
        _safe_ratio(
            numerator,
            denominator,
            absolute_floor=absolute_floor,
        ).item()
    )


def bias_free_output_energy(
    eval_sum_y: torch.Tensor,
    eval_sum_y2: torch.Tensor,
    eval_sample_count: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Return observed output energy after removing a matched constant bias."""
    sum_y = eval_sum_y.float()
    sum_y2 = eval_sum_y2.float()
    counts = eval_sample_count.float()
    offset = bias.float()
    if sum_y.ndim != 2 or sum_y2.ndim != 2:
        raise ValueError("Output sums must be rank 2.")
    if tuple(sum_y.shape) != tuple(sum_y2.shape):
        raise ValueError("Output sum shapes do not match.")
    if counts.ndim != 1 or counts.shape[0] != sum_y.shape[0]:
        raise ValueError("Evaluation sample counts do not match output sums.")
    if offset.ndim != 1 or offset.shape[0] != sum_y.shape[1]:
        raise ValueError("Bias shape does not match output features.")
    for value, name in (
        (sum_y, "eval_sum_y"),
        (sum_y2, "eval_sum_y2"),
        (counts, "eval_sample_count"),
        (offset, "bias"),
    ):
        _require_finite(value, name)
    _require_nonnegative(sum_y2, "eval_sum_y2")
    _require_nonnegative(counts, "eval_sample_count")
    centered = sum_y2 - 2.0 * sum_y * offset + counts[:, None] * offset.square()
    _require_finite(centered, "bias-free output energy")
    scale = max(1.0, float(torch.abs(centered).max().item()))
    tolerance = 1e-5 * scale
    if bool((centered < -tolerance).any().item()):
        raise ValueError("Bias-free output energy contains meaningful negatives.")
    return centered.clamp_min(0.0).contiguous()


def aggregate_diagonal_relative_sse(
    error_by_eval_output: torch.Tensor,
    reference_weights: torch.Tensor,
    eval_sum_x2: torch.Tensor,
    *,
    absolute_floor: float = 0.0,
) -> float:
    """Return the global ratio using diagonal reference output energy."""
    error = _validate_error_matrix(error_by_eval_output)
    reference_energy = diagonal_reference_output_energy(
        reference_weights,
        eval_sum_x2,
    )
    if tuple(error.shape) != tuple(reference_energy.shape):
        raise ValueError(
            "Measurement and diagonal reference energy shapes do not match."
        )
    return float(
        _safe_ratio(
            error.sum(),
            reference_energy.sum(),
            absolute_floor=absolute_floor,
        ).item()
    )


def global_channel_contribution(
    error_by_eval_output: torch.Tensor,
    eval_sum_y2: torch.Tensor,
    *,
    absolute_floor: float = 0.0,
) -> torch.Tensor:
    """Return each channel's contribution normalized by total output energy."""
    error, output_energy = _validate_measurement_pair(
        error_by_eval_output,
        eval_sum_y2,
    )
    numerator = error.sum(dim=0)
    denominator = output_energy.sum()
    return _safe_ratio(
        numerator,
        denominator,
        absolute_floor=absolute_floor,
    )


def row_relative_channel_sse(
    error_by_eval_output: torch.Tensor,
    eval_sum_y2: torch.Tensor,
    *,
    floor_fraction: float = 0.0,
) -> torch.Tensor:
    """Return channel error ratios with a relative energy floor."""
    error, output_energy = _validate_measurement_pair(
        error_by_eval_output,
        eval_sum_y2,
    )
    if not math.isfinite(floor_fraction) or floor_fraction < 0:
        raise ValueError("floor_fraction must be finite and non-negative.")
    numerator = error.sum(dim=0)
    denominator = output_energy.sum(dim=0)
    positive = denominator[denominator > 0]
    typical_energy = (
        positive.mean()
        if positive.numel()
        else torch.zeros((), dtype=denominator.dtype, device=denominator.device)
    )
    denominator_floor = typical_energy * floor_fraction
    return _safe_ratio(
        numerator,
        denominator,
        absolute_floor=float(denominator_floor.item()),
    )


def sampled_output_error(
    sample_x: torch.Tensor,
    reference_weights: torch.Tensor,
    candidate_weights: torch.Tensor,
) -> torch.Tensor:
    """Return exact sampled ``X @ (Wq - W).T`` output errors."""
    _, error_output = _sampled_output_pair(
        sample_x,
        reference_weights,
        candidate_weights,
    )
    return error_output


def sampled_relative_sse(
    sample_x: torch.Tensor,
    reference_weights: torch.Tensor,
    candidate_weights: torch.Tensor,
    *,
    absolute_floor: float = 0.0,
) -> float:
    """Return exact sampled output SSE divided by sampled reference energy."""
    reference_output, error_output = _sampled_output_pair(
        sample_x,
        reference_weights,
        candidate_weights,
    )
    return float(
        _safe_ratio(
            error_output.square().sum(),
            reference_output.square().sum(),
            absolute_floor=absolute_floor,
        ).item()
    )


def sampled_per_row_relative_sse(
    sample_x: torch.Tensor,
    reference_weights: torch.Tensor,
    candidate_weights: torch.Tensor,
    *,
    absolute_floor: float = 0.0,
) -> torch.Tensor:
    """Return exact sampled relative SSE for each sampled input row."""
    reference_output, error_output = _sampled_output_pair(
        sample_x,
        reference_weights,
        candidate_weights,
    )
    return _safe_ratio(
        error_output.square().sum(dim=1),
        reference_output.square().sum(dim=1),
        absolute_floor=absolute_floor,
    )


def sampled_direction_error(
    sample_x: torch.Tensor,
    reference_weights: torch.Tensor,
    candidate_weights: torch.Tensor,
) -> torch.Tensor:
    """Return ``1 - cosine(reference_output, candidate_output)`` per row."""
    reference_output, candidate_error = _sampled_output_pair(
        sample_x,
        reference_weights,
        candidate_weights,
    )
    candidate_output = reference_output + candidate_error
    reference_norm = torch.linalg.vector_norm(reference_output, dim=1)
    candidate_norm = torch.linalg.vector_norm(candidate_output, dim=1)
    both_zero = (reference_norm == 0) & (candidate_norm == 0)
    one_zero = (reference_norm == 0) != (candidate_norm == 0)
    denominator = reference_norm * candidate_norm
    cosine = torch.zeros_like(denominator)
    nonzero = denominator > 0
    cosine[nonzero] = (
        (reference_output[nonzero] * candidate_output[nonzero]).sum(dim=1)
        / denominator[nonzero]
    ).clamp(-1.0, 1.0)
    direction_error = torch.where(
        both_zero,
        torch.zeros_like(cosine),
        torch.where(one_zero, torch.ones_like(cosine), 1.0 - cosine),
    )
    _require_finite(direction_error, "sampled direction error")
    _require_nonnegative(direction_error, "sampled direction error")
    return direction_error


def sampled_diagonal_sse(
    sample_x: torch.Tensor,
    reference_weights: torch.Tensor,
    candidate_weights: torch.Tensor,
) -> float:
    """Return diagonal SSE estimated from the same sampled input rows."""
    sample, reference, candidate = _validate_sampled_inputs(
        sample_x,
        reference_weights,
        candidate_weights,
    )
    delta_sq = (candidate - reference).square()
    sample_input_energy = sample.square().sum(dim=0)
    value = torch.sum(sample_input_energy * delta_sq.sum(dim=0))
    _require_finite(value, "sampled diagonal SSE")
    _require_nonnegative(value, "sampled diagonal SSE")
    return float(value.item())


def sampled_diagonal_approximation_ratio(
    sample_x: torch.Tensor,
    reference_weights: torch.Tensor,
    candidate_weights: torch.Tensor,
    *,
    absolute_floor: float = 0.0,
) -> float:
    """Compare exact sampled SSE with its same-sample diagonal approximation."""
    _, error_output = _sampled_output_pair(
        sample_x,
        reference_weights,
        candidate_weights,
    )
    exact_sse = error_output.square().sum()
    diagonal_sse = sampled_diagonal_sse(
        sample_x,
        reference_weights,
        candidate_weights,
    )
    diagonal_sse_tensor = torch.tensor(
        diagonal_sse,
        dtype=exact_sse.dtype,
        device=exact_sse.device,
    )
    return float(
        _safe_ratio(
            exact_sse,
            diagonal_sse_tensor,
            absolute_floor=absolute_floor,
        ).item()
    )


def reduce_evaluation_scores(
    scores: torch.Tensor | Sequence[float],
    reducer: EvaluationReducer | str,
    *,
    weights: torch.Tensor | Sequence[float] | None = None,
) -> float:
    """Reduce non-negative per-evaluation scores with explicit tail rules.

    Percentiles use Torch's linear interpolation. CVaR uses the arithmetic
    mean of the largest ``max(1, ceil(E * fraction))`` scores.
    """
    values = _validate_scores(scores)
    if reducer in ("energy_weighted_mean", "global_ratio"):
        if weights is None:
            raise ValueError(
                f"Reducer {reducer!r} requires per-evaluation weights."
            )
        weight_values = _validate_weights(weights, values)
        return float(
            _safe_ratio(
                torch.sum(values * weight_values),
                torch.sum(weight_values),
            ).item()
        )
    if weights is not None:
        raise ValueError(f"Reducer {reducer!r} does not use weights.")
    if reducer == "mean":
        return float(values.mean().item())
    if reducer == "max":
        return float(values.max().item())
    if reducer in ("p90", "p95"):
        percentile = 0.90 if reducer == "p90" else 0.95
        return float(
            torch.quantile(values, percentile, interpolation="linear").item()
        )
    if reducer in ("cvar10", "cvar20"):
        fraction = 0.10 if reducer == "cvar10" else 0.20
        tail_count = max(1, math.ceil(values.shape[0] * fraction))
        return float(torch.topk(values, tail_count).values.mean().item())
    raise ValueError(f"Unsupported evaluation reducer: {reducer!r}")


def _validate_measurement_pair(
    error_by_eval_output: torch.Tensor,
    eval_sum_y2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    error = _validate_error_matrix(error_by_eval_output)
    output_energy = eval_sum_y2.float()
    if output_energy.ndim != 2 or tuple(output_energy.shape) != tuple(error.shape):
        raise ValueError("Measurement and eval_sum_y2 shapes do not match.")
    _require_finite(output_energy, "eval_sum_y2")
    _require_nonnegative(output_energy, "eval_sum_y2")
    return error, output_energy


def _sampled_output_pair(
    sample_x: torch.Tensor,
    reference_weights: torch.Tensor,
    candidate_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sample, reference, candidate = _validate_sampled_inputs(
        sample_x,
        reference_weights,
        candidate_weights,
    )
    reference_output = torch.mm(sample, reference.transpose(0, 1))
    error_output = torch.mm(
        sample,
        (candidate - reference).transpose(0, 1),
    )
    _require_finite(reference_output, "sampled reference output")
    _require_finite(error_output, "sampled output error")
    return reference_output, error_output


def _validate_sampled_inputs(
    sample_x: torch.Tensor,
    reference_weights: torch.Tensor,
    candidate_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if sample_x.ndim != 2 or reference_weights.ndim != 2 or candidate_weights.ndim != 2:
        raise ValueError("Sampled inputs and weights must be rank 2.")
    if tuple(reference_weights.shape) != tuple(candidate_weights.shape):
        raise ValueError("Reference and candidate weight shapes do not match.")
    if sample_x.shape[1] != reference_weights.shape[1]:
        raise ValueError("Sampled inputs and weights input features do not match.")
    sample = sample_x.float()
    reference = reference_weights.float()
    candidate = candidate_weights.float()
    _require_finite(sample, "sample_x")
    _require_finite(reference, "reference_weights")
    _require_finite(candidate, "candidate_weights")
    return sample, reference, candidate


def _validate_error_matrix(error: torch.Tensor) -> torch.Tensor:
    if error.ndim != 2:
        raise ValueError("Activation error measurements must be rank 2.")
    error = error.float()
    _require_finite(error, "activation error measurement")
    _require_nonnegative(error, "activation error measurement")
    return error


def _validate_scores(scores: torch.Tensor | Sequence[float]) -> torch.Tensor:
    values = (
        scores.float()
        if isinstance(scores, torch.Tensor)
        else torch.tensor(scores, dtype=torch.float32)
    )
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("Evaluation scores must be a non-empty rank-1 vector.")
    _require_finite(values, "evaluation scores")
    _require_nonnegative(values, "evaluation scores")
    return values


def _validate_weights(
    weights: torch.Tensor | Sequence[float],
    scores: torch.Tensor,
) -> torch.Tensor:
    values = (
        weights.float()
        if isinstance(weights, torch.Tensor)
        else torch.tensor(weights, dtype=torch.float32, device=scores.device)
    )
    if values.ndim != 1 or tuple(values.shape) != tuple(scores.shape):
        raise ValueError("Reducer weights must match the score vector shape.")
    _require_finite(values, "reducer weights")
    _require_nonnegative(values, "reducer weights")
    return values


def _safe_ratio(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    *,
    absolute_floor: float = 0.0,
) -> torch.Tensor:
    if not math.isfinite(absolute_floor) or absolute_floor < 0:
        raise ValueError("absolute_floor must be finite and non-negative.")
    _require_finite(numerator, "ratio numerator")
    _require_nonnegative(numerator, "ratio numerator")
    _require_finite(denominator, "ratio denominator")
    _require_nonnegative(denominator, "ratio denominator")
    if absolute_floor == 0.0:
        invalid = (denominator == 0) & (numerator > 0)
        if bool(invalid.any().item()):
            raise ValueError(
                "Relative SSE denominator is zero for non-zero error."
            )
        return torch.where(
            denominator > 0,
            numerator / denominator,
            torch.zeros_like(numerator),
        )
    return numerator / denominator.clamp_min(absolute_floor)


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must be finite.")


def _require_nonnegative(tensor: torch.Tensor, name: str) -> None:
    if bool((tensor < 0).any().item()):
        raise ValueError(f"{name} must be non-negative.")


__all__ = [
    "EvaluationReducer",
    "aggregate_diagonal_relative_sse",
    "aggregate_observed_relative_sse",
    "bias_free_output_energy",
    "diagonal_reference_output_energy",
    "evaluation_relative_sse",
    "global_channel_contribution",
    "row_relative_channel_sse",
    "reduce_evaluation_scores",
    "sampled_diagonal_approximation_ratio",
    "sampled_diagonal_sse",
    "sampled_direction_error",
    "sampled_output_error",
    "sampled_per_row_relative_sse",
    "sampled_relative_sse",
]
