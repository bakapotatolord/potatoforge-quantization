"""Activation-weighted logical reconstruction metrics."""

from typing import NamedTuple

import torch


class ActivationErrorResult(NamedTuple):
    activation_error: float
    weight_error_per_input_sum: torch.Tensor
    activation_energy_sum: float
    input_features: int


def _require_finite(name: str, values: torch.Tensor) -> None:
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError(f"{name} must contain only finite values.")


def activation_weighted_error(
    reference_weight: torch.Tensor,
    candidate_weight: torch.Tensor,
    sum_x2: torch.Tensor,
) -> ActivationErrorResult:
    """Calculate raw diagonal activation-energy weighted reconstruction error."""
    if reference_weight.ndim != 2:
        raise ValueError("reference_weight must be a rank-2 tensor.")
    if candidate_weight.ndim != 2:
        raise ValueError("candidate_weight must be a rank-2 tensor.")
    if sum_x2.ndim != 1:
        raise ValueError("sum_x2 must be a rank-1 tensor.")
    if reference_weight.shape != candidate_weight.shape:
        raise ValueError(
            "reference_weight and candidate_weight must have matching shapes."
        )
    if reference_weight.device != candidate_weight.device:
        raise ValueError(
            "reference_weight and candidate_weight must be on the same device."
        )
    if sum_x2.shape[0] != reference_weight.shape[1]:
        raise ValueError(
            "sum_x2 length must match the input feature dimension."
        )

    reference_float = reference_weight.float()
    candidate_float = candidate_weight.float()
    sum_x2_float = sum_x2.to(
        device=reference_float.device,
        dtype=torch.float32,
    )
    _require_finite("reference_weight", reference_float)
    _require_finite("candidate_weight", candidate_float)
    _require_finite("sum_x2", sum_x2_float)
    if bool((sum_x2_float < 0).any().item()):
        raise ValueError("sum_x2 must contain only non-negative values.")

    weight_error_per_input_sum = (
        (candidate_float - reference_float).square().sum(dim=0)
    )
    activation_error = (weight_error_per_input_sum * sum_x2_float).sum()

    return ActivationErrorResult(
        activation_error=float(activation_error.item()),
        weight_error_per_input_sum=weight_error_per_input_sum,
        activation_energy_sum=float(sum_x2_float.sum().item()),
        input_features=reference_weight.shape[1],
    )
