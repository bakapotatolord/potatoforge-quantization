"""Candidate-specific local activation measurements."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import torch

from ..calibration import LayerCalibration
from ..planning import (
    TensorDescriptor,
    build_quantized_tensor_plan,
    source_bytes,
)
from ..profiles import QuantizationAction
from ..quantization.convrot_w4a4 import (
    ConvRotW4A4Result,
    dequantize_convrot_w4a4,
    quantize_convrot_w4a4,
    quantize_convrot_w4a4_mse,
)
from ..quantization.int6_rowwise import (
    dequantize_int6_convrot,
    dequantize_int6_convrot_packed,
    dequantize_int6_rowwise,
    quantize_int6_convrot,
    quantize_int6_convrot_packed,
    quantize_int6_rowwise,
)
from ..quantization.int8_tensorwise import (
    Int8TensorwiseResult,
    dequantize_int8_convrot,
    dequantize_int8_tensorwise,
    quantize_int8_convrot,
    quantize_int8_tensorwise,
)
from ..timing import timed_stage
from .activation_metrics import (
    sampled_diagonal_approximation_ratio,
    sampled_diagonal_sse,
    sampled_direction_error,
    sampled_output_error,
)


METHOD_ACTIONS: Final[dict[str, QuantizationAction]] = {
    "bf16": "keep",
    "int8": "int8",
    "int6": "int6_rowwise",
    "int8_convrot": "int8_convrot",
    "int6_convrot": "int6_convrot",
    "convrot_w4a4": "convrot_w4a4",
    "convrot_w4a4_mse": "convrot_w4a4_mse",
}
MEASUREMENT_METHODS: Final[tuple[str, ...]] = (
    "convrot_w4a4",
    "convrot_w4a4_mse",
    "int6_convrot",
    "int8_convrot",
)


@dataclass(frozen=True)
class CandidateMeasurement:
    method: str
    action: QuantizationAction
    available: bool
    storage_bytes: int | None
    error_by_eval_output: torch.Tensor | None
    unavailable_reason: str | None = None
    sample_error_sse: torch.Tensor | None = None
    sample_reference_energy: torch.Tensor | None = None
    sample_direction_error: torch.Tensor | None = None
    sample_exact_sse: float | None = None
    sample_diag_sse: float | None = None
    sample_cross_term_ratio: float | None = None
    sample_unavailable_reason: str | None = None


def measure_activation_candidates(
    tensor_name: str,
    descriptor: TensorDescriptor,
    weights: torch.Tensor,
    calibration: LayerCalibration,
    methods: Sequence[str] | None = None,
    *,
    device: str = "cpu",
    method_timings: dict[str, float] | None = None,
) -> tuple[CandidateMeasurement, ...]:
    """Measure selected quantization candidates for one source weight."""
    requested_methods = MEASUREMENT_METHODS if methods is None else tuple(methods)
    if method_timings is None:
        return tuple(
            measure_activation_candidate(
                tensor_name,
                descriptor,
                weights,
                calibration,
                method,
                device=device,
            )
            for method in requested_methods
        )

    target_device = torch.device(device)
    synchronize = (
        torch.cuda.synchronize if target_device.type == "cuda" else None
    )
    results: list[CandidateMeasurement] = []
    for method in requested_methods:
        with timed_stage(
            method_timings,
            method,
            synchronize=synchronize,
            synchronize_arg=target_device,
        ):
            results.append(
                measure_activation_candidate(
                    tensor_name,
                    descriptor,
                    weights,
                    calibration,
                    method,
                    device=device,
                )
            )
    return tuple(results)


def measure_activation_candidate(
    tensor_name: str,
    descriptor: TensorDescriptor,
    weights: torch.Tensor,
    calibration: LayerCalibration,
    method: str,
    *,
    device: str = "cpu",
) -> CandidateMeasurement:
    action = METHOD_ACTIONS.get(method)
    if action is None:
        raise ValueError(f"Unsupported activation measurement method: {method!r}")
    _validate_inputs(tensor_name, descriptor, weights, calibration)

    try:
        storage_bytes = (
            source_bytes(descriptor)
            if action == "keep"
            else build_quantized_tensor_plan(
                action,
                tensor_name,
                descriptor,
            ).estimated_bytes
        )
    except ValueError as error:
        return CandidateMeasurement(
            method=method,
            action=action,
            available=False,
            storage_bytes=None,
            error_by_eval_output=None,
            unavailable_reason=str(error),
        )

    if action == "keep":
        reconstructed = weights
        error_by_eval_output = torch.zeros(
            (calibration.evaluation_count, calibration.output_features),
            dtype=torch.float32,
        )
    else:
        reconstructed = _reconstruct_candidate(weights, method, device=device)
        error_by_eval_output = _measure_output_error(
            weights,
            reconstructed,
            calibration,
        )

    sampled = _measure_sampled_diagnostics(
        weights,
        reconstructed,
        calibration,
    )

    return CandidateMeasurement(
        method=method,
        action=action,
        available=True,
        storage_bytes=storage_bytes,
        error_by_eval_output=error_by_eval_output,
        **sampled,
    )


def _validate_inputs(
    tensor_name: str,
    descriptor: TensorDescriptor,
    weights: torch.Tensor,
    calibration: LayerCalibration,
) -> None:
    if not tensor_name:
        raise ValueError("Activation measurement tensor_name must not be empty.")
    if weights.ndim != 2:
        raise ValueError("Activation measurement weights must be rank 2.")
    if tuple(descriptor["shape"]) != tuple(weights.shape):
        raise ValueError(
            f"Activation measurement source shape does not match {tensor_name}."
        )
    if tuple(weights.shape) != (
        calibration.output_features,
        calibration.input_features,
    ):
        raise ValueError(
            "Activation measurement weight shape does not match calibration "
            f"for {tensor_name}."
        )
    if weights.dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ):
        raise ValueError(
            "Activation measurement weights must use BF16, F16, or F32."
        )
    if not bool(torch.isfinite(weights).all().item()):
        raise ValueError("Activation measurement weights must be finite.")
    eval_sum_x2 = calibration.eval_sum_x2
    if eval_sum_x2.dtype != torch.float32 or tuple(eval_sum_x2.shape) != (
        calibration.evaluation_count,
        calibration.input_features,
    ):
        raise ValueError(
            "Activation calibration eval_sum_x2 does not match the weight shape."
        )


def _reconstruct_candidate(
    weights: torch.Tensor,
    method: str,
    *,
    device: str = "cpu",
) -> torch.Tensor:
    if method == "int8":
        return dequantize_int8_tensorwise(quantize_int8_tensorwise(weights))
    if method == "int6":
        return dequantize_int6_rowwise(quantize_int6_rowwise(weights))
    if method == "int8_convrot":
        result = quantize_int8_convrot(weights, device=device)
        if torch.device(device).type == "cuda":
            result = Int8TensorwiseResult(
                result.codes.to(device=device),
                result.scales.to(device=device),
            )
        return dequantize_int8_convrot(result)
    if method == "int6_convrot":
        if torch.device(device).type != "cuda":
            return dequantize_int6_convrot(quantize_int6_convrot(weights))
        packed = quantize_int6_convrot_packed(weights, device=device)
        return dequantize_int6_convrot_packed(packed, device=device)
    if method == "convrot_w4a4":
        result = quantize_convrot_w4a4(weights, device=device)
        if torch.device(device).type == "cuda":
            result = ConvRotW4A4Result(
                result.packed_codes.to(device=device),
                result.scales.to(device=device),
            )
        return dequantize_convrot_w4a4(result)
    if method == "convrot_w4a4_mse":
        result = quantize_convrot_w4a4_mse(weights, device=device)
        if torch.device(device).type == "cuda":
            result = ConvRotW4A4Result(
                result.packed_codes.to(device=device),
                result.scales.to(device=device),
            )
        return dequantize_convrot_w4a4(result)
    raise ValueError(f"Unsupported activation measurement method: {method!r}")


def _measure_output_error(
    weights: torch.Tensor,
    reconstructed: torch.Tensor,
    calibration: LayerCalibration,
) -> torch.Tensor:
    reference = weights.to(
        device=reconstructed.device,
        dtype=torch.float32,
    )
    candidate = reconstructed.float()
    if candidate.shape != reference.shape:
        raise ValueError("Quantizer reconstruction shape does not match source weight.")
    delta_sq = (candidate - reference).square()
    activation_energy = calibration.eval_sum_x2.to(
        device=delta_sq.device,
        dtype=torch.float32,
    )
    error_by_eval_output = torch.mm(activation_energy, delta_sq.transpose(0, 1))
    error_by_eval_output = error_by_eval_output.to(
        device="cpu",
        dtype=torch.float32,
    ).contiguous()
    if not bool(torch.isfinite(error_by_eval_output).all().item()):
        raise ValueError("Activation candidate measurement must be finite.")
    if bool((error_by_eval_output < 0).any().item()):
        raise ValueError("Activation candidate measurement must be non-negative.")
    return error_by_eval_output


def _measure_sampled_diagnostics(
    weights: torch.Tensor,
    reconstructed: torch.Tensor,
    calibration: LayerCalibration,
) -> dict[str, object]:
    sample_x = calibration.sample_x
    if sample_x is None:
        return {}
    if sample_x.shape[0] == 0:
        return {
            "sample_unavailable_reason": "Calibration sample_x contains no valid rows."
        }

    reference = weights.to(
        device=reconstructed.device,
        dtype=torch.float32,
    )
    candidate = reconstructed.float()
    sample = sample_x.to(
        device=reconstructed.device,
        dtype=torch.float32,
    )
    error_output = sampled_output_error(sample, reference, candidate)
    sample_error_sse = error_output.square().sum(dim=1).contiguous()
    reference_output = torch.mm(sample, reference.transpose(0, 1))
    sample_reference_energy = reference_output.square().sum(dim=1).contiguous()
    sample_direction_error = sampled_direction_error(sample, reference, candidate)
    sample_exact_sse = float(sample_error_sse.sum().item())
    sample_diag_sse = sampled_diagonal_sse(sample, reference, candidate)
    try:
        sample_cross_term_ratio = sampled_diagonal_approximation_ratio(
            sample,
            reference,
            candidate,
        )
        sample_unavailable_reason = None
    except ValueError as error:
        sample_cross_term_ratio = None
        sample_unavailable_reason = str(error)

    return {
        "sample_error_sse": sample_error_sse.to(
            device="cpu",
            dtype=torch.float32,
        ),
        "sample_reference_energy": sample_reference_energy.to(
            device="cpu",
            dtype=torch.float32,
        ),
        "sample_direction_error": sample_direction_error.to(
            device="cpu",
            dtype=torch.float32,
        ),
        "sample_exact_sse": sample_exact_sse,
        "sample_diag_sse": sample_diag_sse,
        "sample_cross_term_ratio": sample_cross_term_ratio,
        "sample_unavailable_reason": sample_unavailable_reason,
    }


__all__ = [
    "CandidateMeasurement",
    "METHOD_ACTIONS",
    "MEASUREMENT_METHODS",
    "measure_activation_candidate",
    "measure_activation_candidates",
]
