from typing import NamedTuple

import torch

from ..quantization.convrot_w4a4 import (
    dequantize_convrot_w4a4,
    quantize_convrot_w4a4,
    quantize_convrot_w4a4_mse,
)
from ..quantization.hadamard import CONVROT_GROUP_SIZE
from ..quantization.int6_rowwise import (
    dequantize_int6_convrot,
    dequantize_int6_rowwise,
    quantize_int6_convrot,
    quantize_int6_rowwise,
)
from ..quantization.int8_tensorwise import (
    dequantize_int8_convrot,
    dequantize_int8_tensorwise,
    quantize_int8_convrot,
    quantize_int8_tensorwise,
)
from .activation_error import ActivationErrorResult, activation_weighted_error


def _relative_l2_error_from_float(
    original_float: torch.Tensor,
    reconstructed: torch.Tensor,
    source_l2: torch.Tensor,
) -> float:
    error_norm = torch.linalg.vector_norm(
        original_float - reconstructed.float()
    )

    if source_l2.item() == 0:
        if error_norm == 0:
            return 0.0

        return float("inf")

    return float((error_norm / source_l2).item())


def relative_l2_error(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    if original.shape != reconstructed.shape:
        raise ValueError(
            "Original and reconstructed tensors must have matching shapes."
        )

    original_float = original.float()
    source_l2 = torch.linalg.vector_norm(original_float)
    return _relative_l2_error_from_float(
        original_float,
        reconstructed,
        source_l2,
    )


class AllQuantizationComparison(NamedTuple):
    int8_relative_l2_error: float | None
    int6_relative_l2_error: float | None
    int8_convrot_relative_l2_error: float | None
    int6_convrot_relative_l2_error: float | None
    w4a4_relative_l2_error: float | None
    w4a4_mse_relative_l2_error: float | None
    activation_error: ActivationErrorResult | None = None
    activation_probe_q_per_input: torch.Tensor | None = None
    activation_probe_reference_power_per_input: torch.Tensor | None = None


def _validate_activation_reference(
    activation_reference_label: str | None,
) -> None:
    if activation_reference_label not in (None, "bf16", "int8_convrot"):
        raise ValueError(
            "Unsupported activation reference: "
            f"{activation_reference_label!r}"
        )


def _activation_outputs(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    activation_sum_x2: torch.Tensor | None,
    activation_probe: bool,
) -> tuple[ActivationErrorResult | None, torch.Tensor | None, torch.Tensor | None]:
    activation_error = (
        None
        if activation_sum_x2 is None
        else activation_weighted_error(reference, candidate, activation_sum_x2)
    )
    if not activation_probe:
        return activation_error, None, None
    return (
        activation_error,
        (candidate.float() - reference.float()).square().sum(dim=0),
        reference.float().square().sum(dim=0),
    )


def compare_w4a4_reconstruction(
    weights: torch.Tensor,
    *,
    original_float: torch.Tensor | None = None,
    source_l2: torch.Tensor | None = None,
    activation_sum_x2: torch.Tensor | None = None,
    activation_reference_label: str | None = None,
    activation_probe: bool = False,
) -> AllQuantizationComparison:
    """Compare only ConvRot W4A4, plus an optional activation reference."""
    _validate_activation_reference(activation_reference_label)
    if original_float is None:
        original_float = weights.float()
    if source_l2 is None:
        source_l2 = torch.linalg.vector_norm(original_float)

    if weights.shape[1] % CONVROT_GROUP_SIZE != 0:
        return AllQuantizationComparison(
            int8_relative_l2_error=None,
            int6_relative_l2_error=None,
            int8_convrot_relative_l2_error=None,
            int6_convrot_relative_l2_error=None,
            w4a4_relative_l2_error=None,
            w4a4_mse_relative_l2_error=None,
            activation_error=None,
        )

    int8_convrot_reconstructed = None
    if (
        activation_probe
        or (
            activation_sum_x2 is not None
            and activation_reference_label == "int8_convrot"
        )
    ):
        int8_convrot_reconstructed = dequantize_int8_convrot(
            quantize_int8_convrot(weights),
        )

    w4a4_reconstructed = dequantize_convrot_w4a4(
        quantize_convrot_w4a4(weights),
    )
    w4a4_error = _relative_l2_error_from_float(
        original_float,
        w4a4_reconstructed,
        source_l2,
    )
    (
        activation_error,
        activation_probe_q_per_input,
        activation_probe_reference_power_per_input,
    ) = _activation_outputs(
        (
            int8_convrot_reconstructed
            if activation_reference_label == "int8_convrot"
            else original_float
        ),
        w4a4_reconstructed,
        activation_sum_x2,
        activation_probe,
    )
    return AllQuantizationComparison(
        int8_relative_l2_error=None,
        int6_relative_l2_error=None,
        int8_convrot_relative_l2_error=None,
        int6_convrot_relative_l2_error=None,
        w4a4_relative_l2_error=w4a4_error,
        w4a4_mse_relative_l2_error=None,
        activation_error=activation_error,
        activation_probe_q_per_input=activation_probe_q_per_input,
        activation_probe_reference_power_per_input=(
            activation_probe_reference_power_per_input
        ),
    )


def compare_all_reconstructions(
    weights: torch.Tensor,
    *,
    original_float: torch.Tensor | None = None,
    source_l2: torch.Tensor | None = None,
    activation_sum_x2: torch.Tensor | None = None,
    activation_reference_label: str | None = None,
    activation_probe: bool = False,
) -> AllQuantizationComparison:
    _validate_activation_reference(activation_reference_label)
    if original_float is None:
        original_float = weights.float()
    if source_l2 is None:
        source_l2 = torch.linalg.vector_norm(original_float)

    int8_result = quantize_int8_tensorwise(weights)
    int8_reconstructed = dequantize_int8_tensorwise(int8_result)
    int8_error = _relative_l2_error_from_float(
        original_float,
        int8_reconstructed,
        source_l2,
    )

    del int8_result
    del int8_reconstructed

    if weights.shape[1] % 4 != 0:
        return AllQuantizationComparison(
            int8_relative_l2_error=int8_error,
            int6_relative_l2_error=None,
            int8_convrot_relative_l2_error=None,
            int6_convrot_relative_l2_error=None,
            w4a4_relative_l2_error=None,
            w4a4_mse_relative_l2_error=None,
            activation_error=None,
        )

    int6_result = quantize_int6_rowwise(weights)
    int6_reconstructed = dequantize_int6_rowwise(int6_result)
    int6_error = _relative_l2_error_from_float(
        original_float,
        int6_reconstructed,
        source_l2,
    )

    del int6_result
    del int6_reconstructed

    if weights.shape[1] % CONVROT_GROUP_SIZE != 0:
        return AllQuantizationComparison(
            int8_relative_l2_error=int8_error,
            int6_relative_l2_error=int6_error,
            int8_convrot_relative_l2_error=None,
            int6_convrot_relative_l2_error=None,
            w4a4_relative_l2_error=None,
            w4a4_mse_relative_l2_error=None,
            activation_error=None,
        )

    int8_convrot_result = quantize_int8_convrot(weights)
    int8_convrot_reconstructed = dequantize_int8_convrot(
        int8_convrot_result,
    )
    int8_convrot_error = _relative_l2_error_from_float(
        original_float,
        int8_convrot_reconstructed,
        source_l2,
    )

    keep_int8_convrot_reference = (
        activation_probe
        or (
            activation_sum_x2 is not None
            and activation_reference_label == "int8_convrot"
        )
    )
    del int8_convrot_result
    if not keep_int8_convrot_reference:
        del int8_convrot_reconstructed

    int6_convrot_result = quantize_int6_convrot(weights)
    int6_convrot_reconstructed = dequantize_int6_convrot(int6_convrot_result)
    int6_convrot_error = _relative_l2_error_from_float(
        original_float,
        int6_convrot_reconstructed,
        source_l2,
    )

    del int6_convrot_result
    del int6_convrot_reconstructed

    w4a4_result = quantize_convrot_w4a4(weights)
    w4a4_reconstructed = dequantize_convrot_w4a4(w4a4_result)
    w4a4_error = _relative_l2_error_from_float(
        original_float,
        w4a4_reconstructed,
        source_l2,
    )
    (
        activation_error,
        activation_probe_q_per_input,
        activation_probe_reference_power_per_input,
    ) = _activation_outputs(
        (
            int8_convrot_reconstructed
            if activation_reference_label == "int8_convrot"
            else original_float
        ),
        w4a4_reconstructed,
        activation_sum_x2,
        activation_probe,
    )

    w4a4_mse_result = quantize_convrot_w4a4_mse(weights)
    w4a4_mse_reconstructed = dequantize_convrot_w4a4(w4a4_mse_result)
    w4a4_mse_error = _relative_l2_error_from_float(
        original_float,
        w4a4_mse_reconstructed,
        source_l2,
    )

    return AllQuantizationComparison(
        int8_relative_l2_error=int8_error,
        int6_relative_l2_error=int6_error,
        int8_convrot_relative_l2_error=int8_convrot_error,
        int6_convrot_relative_l2_error=int6_convrot_error,
        w4a4_relative_l2_error=w4a4_error,
        w4a4_mse_relative_l2_error=w4a4_mse_error,
        activation_error=activation_error,
        activation_probe_q_per_input=activation_probe_q_per_input,
        activation_probe_reference_power_per_input=(
            activation_probe_reference_power_per_input
        ),
    )
