"""Reusable activation-aware candidate measurement caches."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Final

import torch
from safetensors.torch import load_file, save_file

from ..calibration import ActivationCalibration, LayerCalibration
from ..planning import (
    QUANTIZATION_SOURCE_DTYPES,
    is_supported_weight_key,
)
from ..source_payloads import stream_bf16_source_tensors
from .activation_metrics import (
    aggregate_observed_relative_sse,
    bias_free_output_energy,
    evaluation_relative_sse,
    global_channel_contribution,
    reduce_evaluation_scores,
)
from .activation_measurements import (
    CandidateMeasurement,
    METHOD_ACTIONS,
    MEASUREMENT_METHODS,
    measure_activation_candidates,
)


ACTIVATION_AUDIT_FORMAT: Final[str] = "potatoforge_activation_quant_audit"
ACTIVATION_AUDIT_VERSION: Final[int] = 1
ACTIVATION_AUDIT_METRICS: Final[tuple[str, ...]] = (
    "aggregate_observed_relative_sse",
    "eval_mean_observed_relative_sse",
    "eval_p95_observed_relative_sse",
    "eval_cvar20_observed_relative_sse",
    "channel_p95_global_contribution",
    "channel_cvar20_global_contribution",
    "sampled_exact_relative_sse",
    "sampled_cvar20_relative_sse",
    "aggregate_bias_free_observed_relative_sse",
    "eval_mean_bias_free_observed_relative_sse",
    "eval_p95_bias_free_observed_relative_sse",
    "eval_cvar20_bias_free_observed_relative_sse",
    "channel_p95_bias_free_global_contribution",
    "channel_cvar20_bias_free_global_contribution",
)


def activation_audit_pair_paths(
    output_path: str | Path,
) -> tuple[Path, Path]:
    """Return the JSON and Safetensors paths for one activation-audit cache."""
    path = Path(output_path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return path, path.with_suffix(".safetensors")
    if suffix == ".safetensors":
        return path.with_suffix(".json"), path

    if path.name.lower().endswith(".activation-audit"):
        base = path
    else:
        base = path.with_name(f"{path.name}.activation-audit")
    return (
        base.with_name(f"{base.name}.json"),
        base.with_name(f"{base.name}.safetensors"),
    )


@dataclass(frozen=True)
class ActivationAuditCache:
    source_model_path: str
    calibration_metadata_path: str
    calibration_stats_path: str
    calibration_session_id: str
    calibration_version: int
    baseline_label: str | None
    requested_methods: tuple[str, ...]
    measurements: Mapping[str, Mapping[str, CandidateMeasurement]]
    layer_shapes: Mapping[str, tuple[int, int, int]]
    bias_names: Mapping[str, str | None] = field(default_factory=dict)
    bias_free_output_energy: Mapping[str, torch.Tensor] = field(
        default_factory=dict
    )
    bias_unavailable_reasons: Mapping[str, str | None] = field(
        default_factory=dict
    )

    @classmethod
    def load(
        cls,
        metadata_path: str | Path,
        tensors_path: str | Path | None = None,
        calibration: ActivationCalibration | str | Path | None = None,
    ) -> "ActivationAuditCache":
        metadata_file, default_tensors_file = activation_audit_pair_paths(
            metadata_path
        )
        tensors_file = (
            default_tensors_file
            if tensors_path is None
            else Path(tensors_path)
        )
        try:
            document = json.loads(metadata_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid activation audit metadata: {metadata_file}"
            ) from exc

        requested_methods, raw_layers = _validate_metadata(document)
        tensors = load_file(str(tensors_file), device="cpu")
        measurements: dict[str, dict[str, CandidateMeasurement]] = {}
        layer_shapes: dict[str, tuple[int, int, int]] = {}
        bias_names: dict[str, str | None] = {}
        bias_energy: dict[str, torch.Tensor] = {}
        bias_reasons: dict[str, str | None] = {}
        expected_keys: set[str] = set()
        availability: dict[str, dict[str, int]] = {
            method: {"available": 0, "unavailable": 0}
            for method in requested_methods
        }
        next_key = 0

        for layer_index, tensor_name in enumerate(sorted(raw_layers)):
            _require_string(tensor_name, "layer name")
            raw_layer = raw_layers[tensor_name]
            layer = _validate_layer_metadata(
                tensor_name,
                raw_layer,
                requested_methods,
            )
            layer_shapes[tensor_name] = (
                layer["input_features"],
                layer["output_features"],
                layer["evaluation_count"],
            )
            bias_name, bias_value, bias_reason, bias_key = _load_bias_energy(
                tensor_name,
                layer,
                tensors,
                layer_index,
            )
            bias_names[tensor_name] = bias_name
            bias_reasons[tensor_name] = bias_reason
            if bias_value is not None:
                assert bias_key is not None
                bias_energy[tensor_name] = bias_value
                expected_keys.add(bias_key)
            methods: dict[str, CandidateMeasurement] = {}
            for method in requested_methods:
                raw_method = layer["methods"][method]
                candidate, tensor_keys = _load_candidate(
                    tensor_name,
                    method,
                    raw_method,
                    tensors,
                    next_key,
                    evaluation_count=layer["evaluation_count"],
                    output_features=layer["output_features"],
                    sampled_sample_count=layer.get("sampled_sample_count"),
                )
                if tensor_keys is not None:
                    expected_keys.update(tensor_keys)
                    next_key += 1
                    availability[method]["available"] += 1
                else:
                    availability[method]["unavailable"] += 1
                methods[method] = candidate
            measurements[tensor_name] = methods

        if set(tensors) != expected_keys:
            raise ValueError(
                "Activation audit tensors do not match metadata entries."
            )
        if availability != document["method_availability_counts"]:
            raise ValueError(
                "Activation audit method availability counts do not match entries."
            )

        cache = cls(
            source_model_path=document["source_path"],
            calibration_metadata_path=document["calibration_metadata_path"],
            calibration_stats_path=document["calibration_stats_path"],
            calibration_session_id=document["calibration_session_id"],
            calibration_version=document["calibration_version"],
            baseline_label=document.get("baseline_label"),
            requested_methods=requested_methods,
            measurements=measurements,
            layer_shapes=layer_shapes,
            bias_names=bias_names,
            bias_free_output_energy=bias_energy,
            bias_unavailable_reasons=bias_reasons,
        )
        if calibration is not None:
            supplied = (
                calibration
                if isinstance(calibration, ActivationCalibration)
                else ActivationCalibration.load(calibration)
            )
            _validate_calibration_match(cache, supplied)
        return cache

    def get(
        self,
        tensor_name: str,
        method: str,
    ) -> CandidateMeasurement | None:
        layer = self.measurements.get(tensor_name)
        return None if layer is None else layer.get(method)

    def tensor_names(self) -> tuple[str, ...]:
        return tuple(self.measurements)


def run_activation_audit(
    source_path: str | Path,
    calibration: ActivationCalibration,
    output_path: str | Path,
    *,
    calibration_metadata_path: str | Path,
    calibration_stats_path: str | Path | None = None,
    requested_methods: Sequence[str] | None = None,
    tensor_names: Sequence[str] | None = None,
    on_tensor_started: Callable[[int, int, str], None] | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Measure every selected V2 calibration layer and write its cache."""
    if calibration.version != 2:
        raise ValueError("Activation audit requires V2 calibration.")
    source_header = calibration.validate_against_source(source_path)
    descriptors = {
        tensor_name: descriptor
        for tensor_name, descriptor in source_header.tensors.items()
        if descriptor["dtype"] in QUANTIZATION_SOURCE_DTYPES
        and len(descriptor["shape"]) == 2
        and is_supported_weight_key(tensor_name)
    }
    selected_names = (
        tuple(sorted(calibration.tensor_names()))
        if tensor_names is None
        else tuple(tensor_names)
    )
    if len(set(selected_names)) != len(selected_names):
        raise ValueError("Activation audit tensor_names must be unique.")
    missing = [
        tensor_name
        for tensor_name in selected_names
        if tensor_name not in descriptors
    ]
    if missing:
        raise ValueError(
            "Activation audit tensors are not supported source weights: "
            + ", ".join(missing)
        )
    if set(selected_names) != set(calibration.tensor_names()):
        raise ValueError(
            "Activation audit tensor_names must match the calibration layers."
        )

    descriptor_items = tuple(
        (tensor_name, descriptors[tensor_name])
        for tensor_name in selected_names
    )
    bias_descriptors: dict[str, tuple[str, Mapping[str, object]]] = {}
    bias_names: dict[str, str | None] = {}
    bias_reasons: dict[str, str | None] = {}
    for tensor_name in selected_names:
        bias_name = _matching_bias_name(tensor_name)
        bias_descriptor = source_header.tensors.get(bias_name)
        if bias_descriptor is None:
            bias_names[tensor_name] = None
            bias_reasons[tensor_name] = "No matching bias tensor was found."
            continue
        if bias_descriptor["dtype"] not in QUANTIZATION_SOURCE_DTYPES:
            bias_names[tensor_name] = bias_name
            bias_reasons[tensor_name] = (
                f"Matching bias uses unsupported dtype {bias_descriptor['dtype']}."
            )
            continue
        expected_output_features = descriptors[tensor_name]["shape"][0]
        if bias_descriptor["shape"] != [expected_output_features]:
            bias_names[tensor_name] = bias_name
            bias_reasons[tensor_name] = (
                "Matching bias shape does not match the weight output features."
            )
            continue
        bias_descriptors[tensor_name] = (bias_name, bias_descriptor)
        bias_names[tensor_name] = bias_name
        bias_reasons[tensor_name] = None

    bias_values: dict[str, torch.Tensor] = {}
    bias_descriptor_items = tuple(
        (bias_name, descriptor)
        for bias_name, descriptor in (
            bias_descriptors[tensor_name]
            for tensor_name in selected_names
            if tensor_name in bias_descriptors
        )
    )
    for bias_name, bias in stream_bf16_source_tensors(
        source_path,
        bias_descriptor_items,
    ):
        bias_values[bias_name] = bias

    measurements: dict[str, tuple[CandidateMeasurement, ...]] = {}
    bias_energy: dict[str, torch.Tensor] = {}
    for tensor_name, weights in stream_bf16_source_tensors(
        source_path,
        descriptor_items,
        on_tensor_started,
    ):
        layer = calibration.get(tensor_name)
        if not isinstance(layer, LayerCalibration):
            raise ValueError(
                f"Activation audit calibration layer is not V2: {tensor_name}"
            )
        try:
            measurements[tensor_name] = measure_activation_candidates(
                tensor_name,
                descriptors[tensor_name],
                weights,
                layer,
                requested_methods,
            )
            bias_descriptor = bias_descriptors.get(tensor_name)
            if bias_descriptor is not None:
                bias_name, _ = bias_descriptor
                bias_energy[tensor_name] = bias_free_output_energy(
                    layer.eval_sum_y,
                    layer.eval_sum_y2,
                    layer.eval_sample_counts,
                    bias_values[bias_name],
                )
        except ValueError as error:
            raise ValueError(
                f"Activation audit failed: tensor={tensor_name} "
                f"reason={error}"
            ) from error

    return write_activation_audit_cache(
        output_path,
        source_path,
        calibration,
        measurements,
        calibration_metadata_path=calibration_metadata_path,
        calibration_stats_path=calibration_stats_path,
        requested_methods=requested_methods,
        bias_names=bias_names,
        bias_free_output_energy=bias_energy,
        bias_unavailable_reasons=bias_reasons,
        overwrite=overwrite,
    )


def score_activation_audit(
    cache: ActivationAuditCache | str | Path,
    calibration: ActivationCalibration | str | Path,
    *,
    metric: str = "aggregate_observed_relative_sse",
    absolute_floor: float = 0.0,
) -> dict[str, object]:
    """Score cached candidates without reconstructing or requantizing weights."""
    activation_calibration = (
        calibration
        if isinstance(calibration, ActivationCalibration)
        else ActivationCalibration.load(calibration)
    )
    activation_cache = (
        cache
        if isinstance(cache, ActivationAuditCache)
        else ActivationAuditCache.load(cache, calibration=activation_calibration)
    )
    _validate_calibration_match(activation_cache, activation_calibration)
    if metric not in ACTIVATION_AUDIT_METRICS:
        raise ValueError(f"Unsupported activation audit metric: {metric!r}")

    results: list[dict[str, object]] = []
    available_count = 0
    unavailable_count = 0
    for tensor_name in activation_cache.tensor_names():
        layer = activation_calibration.get(tensor_name)
        if not isinstance(layer, LayerCalibration):
            raise ValueError(
                f"Activation audit calibration layer is not V2: {tensor_name}"
            )
        for method in activation_cache.requested_methods:
            candidate = activation_cache.get(tensor_name, method)
            if candidate is None:
                raise ValueError(
                    f"Activation audit candidate is missing: "
                    f"tensor={tensor_name} method={method}"
                )
            cost: float | None = None
            status = "unavailable"
            if candidate.available:
                error = candidate.error_by_eval_output
                if error is None:
                    raise ValueError(
                        f"Activation audit error is missing: "
                        f"tensor={tensor_name} method={method}"
                    )
                try:
                    if metric.startswith("sampled_"):
                        cost = _score_sampled_candidate(
                            metric,
                            candidate,
                            absolute_floor,
                        )
                    else:
                        bias_metric = {
                            "aggregate_bias_free_observed_relative_sse": (
                                "aggregate_observed_relative_sse"
                            ),
                            "eval_mean_bias_free_observed_relative_sse": (
                                "eval_mean_observed_relative_sse"
                            ),
                            "eval_p95_bias_free_observed_relative_sse": (
                                "eval_p95_observed_relative_sse"
                            ),
                            "eval_cvar20_bias_free_observed_relative_sse": (
                                "eval_cvar20_observed_relative_sse"
                            ),
                            "channel_p95_bias_free_global_contribution": (
                                "channel_p95_global_contribution"
                            ),
                            "channel_cvar20_bias_free_global_contribution": (
                                "channel_cvar20_global_contribution"
                            ),
                        }.get(metric)
                        output_energy = None
                        if bias_metric is not None:
                            output_energy = (
                                activation_cache.bias_free_output_energy.get(
                                    tensor_name
                                )
                            )
                            if output_energy is None:
                                raise ValueError(
                                    "Bias-free output energy is unavailable."
                                )
                        cost = _score_candidate(
                            metric if bias_metric is None else bias_metric,
                            error,
                            layer,
                            absolute_floor,
                            output_energy=output_energy,
                        )
                except ValueError as error:
                    raise ValueError(
                        f"Activation scoring failed: tensor={tensor_name} "
                        f"method={method} reason={error}"
                    ) from error
                available_count += 1
                status = "ok"
            else:
                unavailable_count += 1
            results.append(
                {
                    "tensor_name": tensor_name,
                    "method": method,
                    "action": candidate.action,
                    "storage_bytes": candidate.storage_bytes,
                    "available": candidate.available,
                    "unavailable_reason": candidate.unavailable_reason,
                    "objective_cost": cost,
                    "status": status,
                }
            )

    objective: dict[str, object] = {
        "metric": metric,
        "metric_version": 1,
        "absolute_floor": absolute_floor,
    }
    if metric.startswith("sampled_"):
        objective["reference_energy_basis"] = "bias_free_linear_output"
    elif "bias_free" in metric:
        objective["reference_energy_basis"] = "bias_free_observed_output"

    return {
        "format": "potatoforge_activation_score_report",
        "format_version": 1,
        "source_path": activation_cache.source_model_path,
        "calibration_metadata_path": activation_cache.calibration_metadata_path,
        "calibration_session_id": activation_cache.calibration_session_id,
        "activation_audit_format": ACTIVATION_AUDIT_FORMAT,
        "objective": objective,
        "requested_methods": list(activation_cache.requested_methods),
        "summary": {
            "layer_count": len(activation_cache.measurements),
            "candidate_count": len(results),
            "available_candidate_count": available_count,
            "unavailable_candidate_count": unavailable_count,
        },
        "results": results,
    }


def inspect_activation_audit(
    cache: ActivationAuditCache | str | Path,
    calibration: ActivationCalibration | str | Path,
    *,
    tensor_name: str | None = None,
    top_n: int = 5,
) -> dict[str, object]:
    """Return compact per-layer diagnostics from a reusable audit cache."""
    if isinstance(calibration, ActivationCalibration):
        activation_calibration = calibration
    else:
        activation_calibration = ActivationCalibration.load(calibration)
    if isinstance(cache, ActivationAuditCache):
        activation_cache = cache
    else:
        activation_cache = ActivationAuditCache.load(
            cache,
            calibration=activation_calibration,
        )
    _validate_calibration_match(activation_cache, activation_calibration)
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be an integer >= 1.")

    if tensor_name is None:
        selected_tensor_names = activation_cache.tensor_names()
    else:
        if tensor_name not in activation_cache.measurements:
            raise ValueError(f"Activation audit tensor is missing: {tensor_name}")
        selected_tensor_names = (tensor_name,)

    layer_reports: list[dict[str, object]] = []
    available_count = 0
    unavailable_count = 0
    for selected_name in selected_tensor_names:
        layer = activation_calibration.get(selected_name)
        if not isinstance(layer, LayerCalibration):
            raise ValueError(
                f"Activation audit calibration layer is not V2: {selected_name}"
            )
        input_features, output_features, evaluation_count = (
            activation_cache.layer_shapes[selected_name]
        )
        bias_energy = activation_cache.bias_free_output_energy.get(selected_name)
        layer_report: dict[str, object] = {
            "tensor_name": selected_name,
            "shape": [output_features, input_features],
            "evaluation_count": evaluation_count,
            "sample_count": layer.sample_count,
            "invocation_count": layer.invocation_count,
            "input_activation_energy": float(
                layer.aggregate_sum_x2.float().sum().item()
            ),
            "output_activation_energy": float(
                layer.eval_sum_y2.float().sum().item()
            ),
            "bias": {
                "name": activation_cache.bias_names.get(selected_name),
                "available": bias_energy is not None,
                "unavailable_reason": activation_cache.bias_unavailable_reasons.get(
                    selected_name
                ),
            },
            "methods": [],
        }
        method_reports = layer_report["methods"]
        assert isinstance(method_reports, list)
        for method in activation_cache.requested_methods:
            candidate = activation_cache.get(selected_name, method)
            if candidate is None:
                raise ValueError(
                    f"Activation audit candidate is missing: "
                    f"tensor={selected_name} method={method}"
                )
            method_report: dict[str, object] = {
                "method": method,
                "action": candidate.action,
                "available": candidate.available,
                "storage_bytes": candidate.storage_bytes,
                "unavailable_reason": candidate.unavailable_reason,
            }
            if not candidate.available:
                unavailable_count += 1
                method_reports.append(method_report)
                continue

            error = candidate.error_by_eval_output
            if error is None:
                raise ValueError(
                    f"Activation audit error is missing: "
                    f"tensor={selected_name} method={method}"
                )
            try:
                evaluation_scores = evaluation_relative_sse(
                    error,
                    layer.eval_sum_y2,
                )
                channel_contributions = global_channel_contribution(
                    error,
                    layer.eval_sum_y2,
                )
                method_report["metrics"] = {
                    "aggregate_observed_relative_sse": (
                        aggregate_observed_relative_sse(
                            error,
                            layer.eval_sum_y2,
                        )
                    ),
                    "eval_mean_observed_relative_sse": reduce_evaluation_scores(
                        evaluation_scores,
                        "mean",
                    ),
                    "eval_p95_observed_relative_sse": reduce_evaluation_scores(
                        evaluation_scores,
                        "p95",
                    ),
                    "eval_cvar20_observed_relative_sse": reduce_evaluation_scores(
                        evaluation_scores,
                        "cvar20",
                    ),
                    "channel_p95_global_contribution": reduce_evaluation_scores(
                        channel_contributions,
                        "p95",
                    ),
                    "channel_cvar20_global_contribution": (
                        reduce_evaluation_scores(channel_contributions, "cvar20")
                    ),
                }
                if bias_energy is not None:
                    try:
                        method_report["bias_free_metrics"] = {
                            "aggregate_bias_free_observed_relative_sse": (
                                _score_candidate(
                                    "aggregate_observed_relative_sse",
                                    error,
                                    layer,
                                    0.0,
                                    output_energy=bias_energy,
                                )
                            ),
                            "eval_p95_bias_free_observed_relative_sse": (
                                _score_candidate(
                                    "eval_p95_observed_relative_sse",
                                    error,
                                    layer,
                                    0.0,
                                    output_energy=bias_energy,
                                )
                            ),
                            "eval_cvar20_bias_free_observed_relative_sse": (
                                _score_candidate(
                                    "eval_cvar20_observed_relative_sse",
                                    error,
                                    layer,
                                    0.0,
                                    output_energy=bias_energy,
                                )
                            ),
                        }
                    except ValueError as error:
                        raise ValueError(
                            f"Bias-free inspection failed: tensor={selected_name} "
                            f"method={method} reason={error}"
                        ) from error
                if layer.sample_x is None:
                    method_report["sampled"] = {
                        "available": False,
                        "reason": "Calibration layer has no sample_x rows.",
                    }
                elif candidate.sample_exact_sse is None:
                    method_report["sampled"] = {
                        "available": False,
                        "reason": (
                            candidate.sample_unavailable_reason
                            or "Activation audit cache has no sampled diagnostics."
                        ),
                    }
                else:
                    try:
                        method_report["sampled"] = _sampled_candidate_report(
                            candidate
                        )
                    except ValueError as error:
                        raise ValueError(
                            f"Activation sampled inspection failed: "
                            f"tensor={selected_name} method={method} "
                            f"reason={error}"
                        ) from error
            except ValueError as error:
                raise ValueError(
                    f"Activation inspection failed: tensor={selected_name} "
                    f"method={method} reason={error}"
                ) from error

            worst_indices = sorted(
                range(evaluation_scores.shape[0]),
                key=lambda index: (-float(evaluation_scores[index].item()), index),
            )[:top_n]
            method_report["worst_evaluations"] = [
                {
                    "index": index,
                    "score": float(evaluation_scores[index].item()),
                    "metadata": _evaluation_metadata_report(
                        activation_calibration.evaluations[index]
                    )
                    if index < len(activation_calibration.evaluations)
                    else None,
                }
                for index in worst_indices
            ]
            available_count += 1
            method_reports.append(method_report)
        layer_reports.append(layer_report)

    return {
        "format": "potatoforge_activation_inspection",
        "format_version": 1,
        "source_path": activation_cache.source_model_path,
        "calibration_metadata_path": activation_cache.calibration_metadata_path,
        "calibration_stats_path": activation_cache.calibration_stats_path,
        "calibration_session_id": activation_cache.calibration_session_id,
        "baseline_label": activation_cache.baseline_label,
        "requested_methods": list(activation_cache.requested_methods),
        "summary": {
            "layer_count": len(layer_reports),
            "candidate_count": available_count + unavailable_count,
            "available_candidate_count": available_count,
            "unavailable_candidate_count": unavailable_count,
        },
        "layers": layer_reports,
        "rankings": _inspection_rankings(layer_reports, top_n),
    }


def _evaluation_metadata_report(evaluation: Any) -> dict[str, object]:
    return {
        "index": evaluation.index,
        "timestep": evaluation.timestep,
        "sigma": evaluation.sigma,
        "root_input_sum_x2": evaluation.root_input_sum_x2,
        "root_output_sum_y2": evaluation.root_output_sum_y2,
        "time_parameter_name": evaluation.time_parameter_name,
        "time_value": evaluation.time_value,
        "time_value_truncated": evaluation.time_value_truncated,
    }


def _sampled_candidate_report(
    candidate: CandidateMeasurement,
) -> dict[str, object]:
    if (
        candidate.sample_error_sse is None
        or candidate.sample_reference_energy is None
        or candidate.sample_direction_error is None
        or candidate.sample_exact_sse is None
        or candidate.sample_diag_sse is None
    ):
        raise ValueError("sampled diagnostic fields are incomplete.")
    error = candidate.sample_error_sse.float().reshape(-1, 1)
    reference_energy = candidate.sample_reference_energy.float().reshape(-1, 1)
    relative_scores = evaluation_relative_sse(error, reference_energy)
    direction_error = candidate.sample_direction_error.float()
    return {
        "available": True,
        "sample_count": int(error.shape[0]),
        "exact_sse": float(candidate.sample_exact_sse),
        "diagonal_sse": float(candidate.sample_diag_sse),
        "cross_term_ratio": candidate.sample_cross_term_ratio,
        "cross_term_ratio_reason": candidate.sample_unavailable_reason,
        "reference_energy_basis": "bias_free_linear_output",
        "metrics": {
            "sampled_exact_relative_sse": aggregate_observed_relative_sse(
                error,
                reference_energy,
            ),
            "sampled_cvar20_relative_sse": reduce_evaluation_scores(
                relative_scores,
                "cvar20",
            ),
            "sampled_direction_error_mean": reduce_evaluation_scores(
                direction_error,
                "mean",
            ),
            "sampled_direction_error_p95": reduce_evaluation_scores(
                direction_error,
                "p95",
            ),
            "sampled_direction_error_cvar20": reduce_evaluation_scores(
                direction_error,
                "cvar20",
            ),
        },
    }


def _inspection_rankings(
    layer_reports: Sequence[Mapping[str, object]],
    top_n: int,
) -> dict[str, list[dict[str, object]]]:
    temporal: list[dict[str, object]] = []
    channel: list[dict[str, object]] = []
    sampled: list[dict[str, object]] = []
    int8_convrot: dict[str, float] = {}
    convrot_w4a4: dict[str, float] = {}
    for layer in layer_reports:
        tensor_name = layer["tensor_name"]
        methods = layer["methods"]
        if not isinstance(tensor_name, str) or not isinstance(methods, list):
            continue
        for method_report in methods:
            if not isinstance(method_report, Mapping):
                continue
            method = method_report.get("method")
            metrics = method_report.get("metrics")
            if not isinstance(method, str) or not isinstance(metrics, Mapping):
                continue
            for source, destination in (
                ("eval_cvar20_observed_relative_sse", temporal),
                ("channel_cvar20_global_contribution", channel),
            ):
                value = metrics.get(source)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    destination.append(
                        {
                            "tensor_name": tensor_name,
                            "method": method,
                            "value": float(value),
                            "metric": source,
                        }
                    )
            sampled_report = method_report.get("sampled")
            if isinstance(sampled_report, Mapping):
                ratio = sampled_report.get("cross_term_ratio")
                if isinstance(ratio, (int, float)) and float(ratio) > 0:
                    sampled.append(
                        {
                            "tensor_name": tensor_name,
                            "method": method,
                            "value": abs(math.log2(float(ratio))),
                            "cross_term_ratio": float(ratio),
                            "metric": "abs_log2_sample_cross_term_ratio",
                        }
                    )
            value = metrics.get("aggregate_observed_relative_sse")
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                if method == "int8_convrot":
                    int8_convrot[tensor_name] = float(value)
                elif method == "convrot_w4a4":
                    convrot_w4a4[tensor_name] = float(value)

    benefit = [
        {
            "tensor_name": tensor_name,
            "value": convrot_w4a4[tensor_name] - int8_convrot[tensor_name],
            "metric": "convrot_w4a4_minus_int8_convrot",
        }
        for tensor_name in sorted(int8_convrot.keys() & convrot_w4a4.keys())
    ]
    return {
        "most_temporally_fragile": _top_ranked(temporal, top_n),
        "largest_channel_tail": _top_ranked(channel, top_n),
        "largest_diagonal_sampled_disagreement": _top_ranked(sampled, top_n),
        "largest_int8_convrot_benefit_over_w4a4": _top_ranked(benefit, top_n),
    }


def _top_ranked(
    entries: Sequence[dict[str, object]],
    top_n: int,
) -> list[dict[str, object]]:
    return sorted(
        entries,
        key=lambda entry: (-float(entry["value"]), str(entry["tensor_name"])),
    )[:top_n]


def _score_candidate(
    metric: str,
    error: torch.Tensor,
    layer: LayerCalibration,
    absolute_floor: float,
    *,
    output_energy: torch.Tensor | None = None,
) -> float:
    denominator = layer.eval_sum_y2 if output_energy is None else output_energy
    if metric == "aggregate_observed_relative_sse":
        return aggregate_observed_relative_sse(
            error,
            denominator,
            absolute_floor=absolute_floor,
        )
    if metric.startswith("eval_"):
        per_evaluation = evaluation_relative_sse(
            error,
            denominator,
            absolute_floor=absolute_floor,
        )
        reducer = {
            "eval_mean_observed_relative_sse": "mean",
            "eval_p95_observed_relative_sse": "p95",
            "eval_cvar20_observed_relative_sse": "cvar20",
        }[metric]
        return reduce_evaluation_scores(per_evaluation, reducer)

    contributions = global_channel_contribution(
        error,
        denominator,
        absolute_floor=absolute_floor,
    )
    reducer = {
        "channel_p95_global_contribution": "p95",
        "channel_cvar20_global_contribution": "cvar20",
    }[metric]
    return reduce_evaluation_scores(contributions, reducer)


def _score_sampled_candidate(
    metric: str,
    candidate: CandidateMeasurement,
    absolute_floor: float,
) -> float:
    if (
        candidate.sample_error_sse is None
        or candidate.sample_reference_energy is None
    ):
        raise ValueError("sampled diagnostic fields are unavailable.")
    error = candidate.sample_error_sse.float().reshape(-1, 1)
    reference_energy = candidate.sample_reference_energy.float().reshape(-1, 1)
    if metric == "sampled_exact_relative_sse":
        return aggregate_observed_relative_sse(
            error,
            reference_energy,
            absolute_floor=absolute_floor,
        )
    if metric == "sampled_cvar20_relative_sse":
        return reduce_evaluation_scores(
            evaluation_relative_sse(
                error,
                reference_energy,
                absolute_floor=absolute_floor,
            ),
            "cvar20",
        )
    raise ValueError(f"Unsupported activation audit metric: {metric!r}")


def _matching_bias_name(weight_name: str) -> str:
    if weight_name.endswith(".attn.in_proj_weight"):
        return weight_name.removesuffix("_weight") + "_bias"
    if weight_name.endswith(".weight"):
        return weight_name.removesuffix(".weight") + ".bias"
    raise ValueError(f"Unsupported activation audit weight name: {weight_name}")


def write_activation_audit_cache(
    output_path: str | Path,
    source_model_path: str | Path,
    calibration: ActivationCalibration,
    measurements: Mapping[
        str,
        Sequence[CandidateMeasurement]
        | Mapping[str, CandidateMeasurement],
    ],
    *,
    calibration_metadata_path: str | Path,
    calibration_stats_path: str | Path | None = None,
    requested_methods: Sequence[str] | None = None,
    bias_names: Mapping[str, str | None] | None = None,
    bias_free_output_energy: Mapping[str, torch.Tensor] | None = None,
    bias_unavailable_reasons: Mapping[str, str | None] | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write a validated activation-audit cache as an atomic pair."""
    metadata_path, tensors_path = activation_audit_pair_paths(output_path)
    if not overwrite and (metadata_path.exists() or tensors_path.exists()):
        raise FileExistsError(
            f"Activation audit output already exists: {metadata_path}"
        )

    requested = _normalise_requested_methods(requested_methods)
    if calibration.version != 2:
        raise ValueError("Activation audit caches require V2 calibration.")
    session_id = calibration.session_id
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Activation audit calibration session_id is required.")

    layer_names = set(calibration.tensor_names())
    if set(measurements) != layer_names:
        raise ValueError(
            "Activation audit measurement tensor names must match calibration."
        )
    serialized_bias_names = {} if bias_names is None else dict(bias_names)
    serialized_bias_energy = (
        {} if bias_free_output_energy is None else dict(bias_free_output_energy)
    )
    serialized_bias_reasons = (
        {} if bias_unavailable_reasons is None else dict(bias_unavailable_reasons)
    )
    for mapping_name, mapping in (
        ("bias_names", serialized_bias_names),
        ("bias_free_output_energy", serialized_bias_energy),
        ("bias_unavailable_reasons", serialized_bias_reasons),
    ):
        if not set(mapping).issubset(layer_names):
            raise ValueError(
                f"Activation audit {mapping_name} contains unknown layers."
            )

    serialized_tensors: dict[str, torch.Tensor] = {}
    serialized_layers: dict[str, dict[str, object]] = {}
    availability: dict[str, dict[str, int]] = {
        method: {"available": 0, "unavailable": 0}
        for method in requested
    }
    next_key = 0

    for layer_index, tensor_name in enumerate(sorted(layer_names)):
        layer = calibration.get(tensor_name)
        if not isinstance(layer, LayerCalibration):
            raise ValueError(
                f"Activation audit requires V2 layer calibration: {tensor_name}"
            )
        records = _normalise_layer_measurements(
            tensor_name,
            measurements[tensor_name],
            requested,
        )
        serialized_methods: dict[str, dict[str, object]] = {}
        bias_name = serialized_bias_names.get(tensor_name)
        bias_value = serialized_bias_energy.get(tensor_name)
        bias_reason = serialized_bias_reasons.get(tensor_name)
        bias_key: str | None = None
        if bias_name is not None and (
            not isinstance(bias_name, str) or not bias_name
        ):
            raise ValueError(f"Activation audit bias_name is invalid: {tensor_name}")
        if bias_reason is not None and (
            not isinstance(bias_reason, str) or not bias_reason
        ):
            raise ValueError(
                f"Activation audit bias reason is invalid: {tensor_name}"
            )
        if bias_value is not None:
            if bias_name is None or bias_reason is not None:
                raise ValueError(
                    f"Activation audit bias metadata is inconsistent: {tensor_name}"
                )
            bias_value = bias_value.detach().to(device="cpu", dtype=torch.float32)
            expected_bias_shape = (
                layer.evaluation_count,
                layer.output_features,
            )
            if tuple(bias_value.shape) != expected_bias_shape:
                raise ValueError(
                    f"Activation audit bias energy shape mismatch: "
                    f"tensor={tensor_name} expected={expected_bias_shape} "
                    f"observed={tuple(bias_value.shape)}"
                )
            if not bool(torch.isfinite(bias_value).all().item()) or bool(
                (bias_value < 0).any().item()
            ):
                raise ValueError(
                    f"Activation audit bias energy contains invalid values: "
                    f"tensor={tensor_name}"
                )
            bias_key = f"b{layer_index:06d}"
            serialized_tensors[bias_key] = bias_value.contiguous()
        for method in requested:
            candidate = records[method]
            record, tensor_key = _serialize_candidate(
                tensor_name,
                method,
                candidate,
                layer,
                next_key,
            )
            if candidate.available:
                assert tensor_key is not None
                serialized_tensors[tensor_key] = (
                    candidate.error_by_eval_output.detach()
                    .to(device="cpu", dtype=torch.float32)
                    .contiguous()
                )
                for metadata_name, field_name in (
                    ("sample_error_sse_key", "sample_error_sse"),
                    ("sample_reference_energy_key", "sample_reference_energy"),
                    ("sample_direction_error_key", "sample_direction_error"),
                ):
                    sample_key = record.get(metadata_name)
                    if sample_key is not None:
                        sample_tensor = getattr(candidate, field_name)
                        assert isinstance(sample_tensor, torch.Tensor)
                        serialized_tensors[sample_key] = (
                            sample_tensor.detach()
                            .to(device="cpu", dtype=torch.float32)
                            .contiguous()
                        )
                availability[method]["available"] += 1
                next_key += 1
            else:
                availability[method]["unavailable"] += 1
            serialized_methods[method] = record

        serialized_layers[tensor_name] = {
            "input_features": layer.input_features,
            "output_features": layer.output_features,
            "sample_count": layer.sample_count,
            "sampled_sample_count": (
                None
                if layer.sample_x is None
                else int(layer.sample_x.shape[0])
            ),
            "invocation_count": layer.invocation_count,
            "evaluation_count": layer.evaluation_count,
            "bias_name": bias_name,
            "bias_free_output_energy_key": bias_key,
            "bias_unavailable_reason": bias_reason,
            "methods": serialized_methods,
        }

    stats_path = (
        Path(calibration_metadata_path).with_suffix(".safetensors")
        if calibration_stats_path is None
        else Path(calibration_stats_path)
    )
    document = {
        "format": ACTIVATION_AUDIT_FORMAT,
        "format_version": ACTIVATION_AUDIT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_path": str(Path(source_model_path).resolve()),
        "calibration_metadata_path": str(
            Path(calibration_metadata_path).resolve()
        ),
        "calibration_stats_path": str(stats_path.resolve()),
        "calibration_session_id": session_id,
        "calibration_version": calibration.version,
        "baseline_label": calibration.baseline_label,
        "requested_methods": list(requested),
        "layer_count": len(serialized_layers),
        "method_availability_counts": availability,
        "layers": serialized_layers,
    }

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    replaced_paths: list[Path] = []
    previously_existing = {
        metadata_path: metadata_path.exists(),
        tensors_path: tensors_path.exists(),
    }
    try:
        tensors_temp = _temporary_path(tensors_path, ".safetensors")
        metadata_temp = _temporary_path(metadata_path, ".json")
        temporary_paths.extend((tensors_temp, metadata_temp))
        save_file(serialized_tensors, str(tensors_temp))
        metadata_temp.write_text(
            json.dumps(document, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tensors_temp, tensors_path)
        replaced_paths.append(tensors_path)
        os.replace(metadata_temp, metadata_path)
        replaced_paths.append(metadata_path)
    except Exception:
        for replaced_path in replaced_paths:
            if not previously_existing[replaced_path]:
                replaced_path.unlink(missing_ok=True)
        raise
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)
    return metadata_path, tensors_path


def _temporary_path(path: Path, suffix: str) -> Path:
    handle, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=f"{suffix}.tmp",
    )
    os.close(handle)
    return Path(name)


def _normalise_requested_methods(
    methods: Sequence[str] | None,
) -> tuple[str, ...]:
    requested = MEASUREMENT_METHODS if methods is None else tuple(methods)
    if not requested:
        raise ValueError("Activation audit requested_methods must not be empty.")
    if any(not isinstance(method, str) or not method for method in requested):
        raise ValueError("Activation audit methods must be non-empty strings.")
    if len(set(requested)) != len(requested):
        raise ValueError("Activation audit requested_methods must be unique.")
    unsupported = [method for method in requested if method not in METHOD_ACTIONS]
    if unsupported:
        raise ValueError(
            "Unsupported activation audit method(s): "
            + ", ".join(repr(method) for method in unsupported)
        )
    return requested


def _normalise_layer_measurements(
    tensor_name: str,
    records: Sequence[CandidateMeasurement]
    | Mapping[str, CandidateMeasurement],
    requested_methods: tuple[str, ...],
) -> dict[str, CandidateMeasurement]:
    if isinstance(records, Mapping):
        if set(records) != set(requested_methods):
            raise ValueError(
                f"Activation audit methods do not match metadata: {tensor_name}"
            )
        by_method = dict(records)
    else:
        by_method = {}
        for candidate in records:
            if not isinstance(candidate, CandidateMeasurement):
                raise ValueError(
                    f"Activation audit measurement is invalid: {tensor_name}"
                )
            if candidate.method in by_method:
                raise ValueError(
                    f"Duplicate activation audit method {candidate.method!r}: "
                    f"{tensor_name}"
                )
            by_method[candidate.method] = candidate
        if set(by_method) != set(requested_methods):
            raise ValueError(
                f"Activation audit methods do not match metadata: {tensor_name}"
            )
    return {
        method: _validate_candidate_identity(
            tensor_name,
            method,
            by_method[method],
        )
        for method in requested_methods
    }


def _validate_candidate_identity(
    tensor_name: str,
    method: str,
    candidate: CandidateMeasurement,
) -> CandidateMeasurement:
    if not isinstance(candidate, CandidateMeasurement):
        raise ValueError(
            f"Activation audit candidate is invalid: "
            f"tensor={tensor_name} method={method}"
        )
    if candidate.method != method:
        raise ValueError(
            f"Activation audit candidate method mismatch: "
            f"tensor={tensor_name} method={method}"
        )
    if candidate.action != METHOD_ACTIONS[method]:
        raise ValueError(
            f"Activation audit candidate action mismatch: "
            f"tensor={tensor_name} method={method}"
        )
    return candidate


def _serialize_candidate(
    tensor_name: str,
    method: str,
    candidate: CandidateMeasurement,
    layer: LayerCalibration,
    tensor_key_index: int,
) -> tuple[dict[str, object], str | None]:
    if candidate.available:
        if not isinstance(candidate.storage_bytes, int) or isinstance(
            candidate.storage_bytes,
            bool,
        ) or candidate.storage_bytes < 0:
            raise ValueError(
                f"Activation audit storage_bytes is invalid: "
                f"tensor={tensor_name} method={method}"
            )
        error = candidate.error_by_eval_output
        if not isinstance(error, torch.Tensor):
            raise ValueError(
                f"Activation audit error tensor is missing: "
                f"tensor={tensor_name} method={method}"
            )
        if candidate.unavailable_reason is not None:
            raise ValueError(
                f"Available activation audit candidate has a reason: "
                f"tensor={tensor_name} method={method}"
            )
        error = error.detach().to(device="cpu", dtype=torch.float32)
        expected_shape = (layer.evaluation_count, layer.output_features)
        if tuple(error.shape) != expected_shape:
            raise ValueError(
                f"Activation audit error shape mismatch: tensor={tensor_name} "
                f"method={method} expected={expected_shape} "
                f"observed={tuple(error.shape)}"
            )
        if not bool(torch.isfinite(error).all().item()):
            raise ValueError(
                f"Activation audit error is not finite: "
                f"tensor={tensor_name} method={method}"
            )
        if bool((error < 0).any().item()):
            raise ValueError(
                f"Activation audit error is negative: "
                f"tensor={tensor_name} method={method}"
            )
        record: dict[str, object] = {
            "method": method,
            "action": candidate.action,
            "storage_bytes": candidate.storage_bytes,
            "available": True,
            "error_tensor_key": f"m{tensor_key_index:06d}",
            "unavailable_reason": None,
        }
        record.update(
            _serialize_sampled_candidate(
                tensor_name,
                method,
                candidate,
                layer,
                tensor_key_index,
            )
        )
        return record, f"m{tensor_key_index:06d}"

    if candidate.storage_bytes is not None:
        raise ValueError(
            f"Unavailable activation audit candidate has storage_bytes: "
            f"tensor={tensor_name} method={method}"
        )
    if candidate.error_by_eval_output is not None:
        raise ValueError(
            f"Unavailable activation audit candidate has an error tensor: "
            f"tensor={tensor_name} method={method}"
        )
    if any(
        getattr(candidate, field_name) is not None
        for field_name in (
            "sample_error_sse",
            "sample_reference_energy",
            "sample_direction_error",
            "sample_exact_sse",
            "sample_diag_sse",
            "sample_cross_term_ratio",
            "sample_unavailable_reason",
        )
    ):
        raise ValueError(
            f"Unavailable activation audit candidate has sampled data: "
            f"tensor={tensor_name} method={method}"
        )
    if (
        not isinstance(candidate.unavailable_reason, str)
        or not candidate.unavailable_reason
    ):
        raise ValueError(
            f"Activation audit unavailable reason is missing: "
            f"tensor={tensor_name} method={method}"
        )
    return (
        {
            "method": method,
            "action": candidate.action,
            "storage_bytes": None,
            "available": False,
            "error_tensor_key": None,
            "unavailable_reason": candidate.unavailable_reason,
        },
        None,
    )


def _serialize_sampled_candidate(
    tensor_name: str,
    method: str,
    candidate: CandidateMeasurement,
    layer: LayerCalibration,
    tensor_key_index: int,
) -> dict[str, object]:
    sample_x = layer.sample_x
    sampled_fields = (
        "sample_error_sse",
        "sample_reference_energy",
        "sample_direction_error",
        "sample_exact_sse",
        "sample_diag_sse",
        "sample_cross_term_ratio",
        "sample_unavailable_reason",
    )
    if sample_x is None:
        if any(
            getattr(candidate, field_name) is not None
            for field_name in sampled_fields
        ):
            raise ValueError(
                f"Activation audit sampled data has no calibration samples: "
                f"tensor={tensor_name} method={method}"
            )
        return {}

    sample_count = int(sample_x.shape[0])
    if sample_count == 0:
        if any(
            getattr(candidate, field_name) is not None
            for field_name in sampled_fields[:-1]
        ):
            raise ValueError(
                f"Activation audit sampled data is invalid for empty samples: "
                f"tensor={tensor_name} method={method}"
            )
        reason = candidate.sample_unavailable_reason
        if reason is not None and not reason:
            raise ValueError(
                f"Activation audit sampled reason is empty: "
                f"tensor={tensor_name} method={method}"
            )
        return {"sample_unavailable_reason": reason}

    serialized: dict[str, object] = {
        "sample_error_sse_key": f"s{tensor_key_index:06d}",
        "sample_reference_energy_key": f"r{tensor_key_index:06d}",
        "sample_direction_error_key": f"d{tensor_key_index:06d}",
    }
    for field_name in sampled_fields[:3]:
        value = getattr(candidate, field_name)
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"Activation audit sampled tensor is missing: "
                f"tensor={tensor_name} method={method} field={field_name}"
            )
        value = value.detach().to(device="cpu", dtype=torch.float32)
        if tuple(value.shape) != (sample_count,):
            raise ValueError(
                f"Activation audit sampled tensor shape mismatch: "
                f"tensor={tensor_name} method={method} field={field_name}"
            )
        if not bool(torch.isfinite(value).all().item()) or bool(
            (value < 0).any().item()
        ):
            raise ValueError(
                f"Activation audit sampled tensor values are invalid: "
                f"tensor={tensor_name} method={method} field={field_name}"
            )
    exact_sse = _optional_nonnegative_float(
        candidate.sample_exact_sse,
        "sample_exact_sse",
        required=True,
    )
    diag_sse = _optional_nonnegative_float(
        candidate.sample_diag_sse,
        "sample_diag_sse",
        required=True,
    )
    ratio = _optional_nonnegative_float(
        candidate.sample_cross_term_ratio,
        "sample_cross_term_ratio",
    )
    reason = candidate.sample_unavailable_reason
    if reason is not None and not reason:
        raise ValueError(
            f"Activation audit sampled reason is empty: "
            f"tensor={tensor_name} method={method}"
        )
    serialized.update(
        {
            "sample_exact_sse": exact_sse,
            "sample_diag_sse": diag_sse,
            "sample_cross_term_ratio": ratio,
            "sample_unavailable_reason": reason,
        }
    )
    return serialized


def _validate_metadata(
    document: Any,
) -> tuple[tuple[str, ...], Mapping[str, Any]]:
    if not isinstance(document, dict):
        raise ValueError("Activation audit metadata must be an object.")
    if document.get("format") != ACTIVATION_AUDIT_FORMAT:
        raise ValueError(
            f"Unsupported activation audit format: {document.get('format')!r}"
        )
    if (
        type(document.get("format_version")) is not int
        or document["format_version"] != ACTIVATION_AUDIT_VERSION
    ):
        raise ValueError(
            f"Unsupported activation audit version: "
            f"{document.get('format_version')!r}"
        )
    for key in (
        "source_path",
        "calibration_metadata_path",
        "calibration_stats_path",
        "calibration_session_id",
        "created_at",
    ):
        _require_string(document.get(key), key)
    if type(document.get("calibration_version")) is not int or document[
        "calibration_version"
    ] != 2:
        raise ValueError("Activation audit calibration_version must be 2.")

    requested_raw = document.get("requested_methods")
    if not isinstance(requested_raw, list):
        raise ValueError("Activation audit requested_methods must be an array.")
    requested = _normalise_requested_methods(requested_raw)

    raw_layers = document.get("layers")
    if not isinstance(raw_layers, dict):
        raise ValueError("Activation audit layers must be an object.")
    layer_count = _require_integer(document.get("layer_count"), "layer_count")
    if layer_count != len(raw_layers):
        raise ValueError("Activation audit layer_count does not match layers.")

    counts = document.get("method_availability_counts")
    if not isinstance(counts, dict) or set(counts) != set(requested):
        raise ValueError(
            "Activation audit method_availability_counts do not match methods."
        )
    for method in requested:
        count = counts[method]
        if not isinstance(count, dict):
            raise ValueError(
                f"Activation audit availability count is invalid: {method}"
            )
        for key in ("available", "unavailable"):
            _require_integer(count.get(key), f"{method}.{key}")
    baseline_label = document.get("baseline_label")
    if baseline_label is not None:
        _require_string(baseline_label, "baseline_label")
    return requested, raw_layers


def _validate_layer_metadata(
    tensor_name: str,
    raw_layer: Any,
    requested_methods: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(raw_layer, dict):
        raise ValueError(f"Activation audit layer must be an object: {tensor_name}")
    _require_positive_integer(
        raw_layer.get("input_features"),
        f"{tensor_name}.input_features",
    )
    _require_positive_integer(
        raw_layer.get("output_features"),
        f"{tensor_name}.output_features",
    )
    for key in ("sample_count", "invocation_count", "evaluation_count"):
        _require_integer(raw_layer.get(key), f"{tensor_name}.{key}")
    sampled_sample_count = raw_layer.get("sampled_sample_count")
    if sampled_sample_count is not None:
        _require_integer(
            sampled_sample_count,
            f"{tensor_name}.sampled_sample_count",
        )
    bias_name = raw_layer.get("bias_name")
    if bias_name is not None and (
        not isinstance(bias_name, str) or not bias_name
    ):
        raise ValueError(f"Activation audit bias_name is invalid: {tensor_name}")
    bias_key = raw_layer.get("bias_free_output_energy_key")
    if bias_key is not None and (
        not isinstance(bias_key, str) or not bias_key
    ):
        raise ValueError(
            f"Activation audit bias energy key is invalid: {tensor_name}"
        )
    bias_reason = raw_layer.get("bias_unavailable_reason")
    if bias_reason is not None and (
        not isinstance(bias_reason, str) or not bias_reason
    ):
        raise ValueError(
            f"Activation audit bias reason is invalid: {tensor_name}"
        )
    if bias_key is not None and bias_name is None:
        raise ValueError(
            f"Activation audit bias energy requires bias_name: {tensor_name}"
        )
    methods = raw_layer.get("methods")
    if not isinstance(methods, dict) or set(methods) != set(requested_methods):
        raise ValueError(
            f"Activation audit layer methods do not match metadata: {tensor_name}"
        )
    raw_layer["methods"] = methods
    return raw_layer


def _load_candidate(
    tensor_name: str,
    method: str,
    raw_method: Any,
    tensors: Mapping[str, torch.Tensor],
    tensor_key_index: int,
    *,
    evaluation_count: int,
    output_features: int,
    sampled_sample_count: int | None,
) -> tuple[CandidateMeasurement, set[str] | None]:
    if not isinstance(raw_method, dict):
        raise ValueError(
            f"Activation audit method entry must be an object: "
            f"tensor={tensor_name} method={method}"
        )
    if raw_method.get("method") != method:
        raise ValueError(
            f"Activation audit method metadata mismatch: "
            f"tensor={tensor_name} method={method}"
        )
    expected_action = METHOD_ACTIONS[method]
    if raw_method.get("action") != expected_action:
        raise ValueError(
            f"Activation audit action metadata mismatch: "
            f"tensor={tensor_name} method={method}"
        )
    available = raw_method.get("available")
    if not isinstance(available, bool):
        raise ValueError(
            f"Activation audit available must be boolean: "
            f"tensor={tensor_name} method={method}"
        )
    storage_bytes = raw_method.get("storage_bytes")
    tensor_key = raw_method.get("error_tensor_key")
    reason = raw_method.get("unavailable_reason")
    if available:
        if (
            not isinstance(storage_bytes, int)
            or isinstance(storage_bytes, bool)
            or storage_bytes < 0
        ):
            raise ValueError(
                f"Activation audit storage_bytes is invalid: "
                f"tensor={tensor_name} method={method}"
            )
        expected_key = f"m{tensor_key_index:06d}"
        if tensor_key != expected_key:
            raise ValueError(
                f"Activation audit tensor key is not deterministic: "
                f"tensor={tensor_name} method={method}"
            )
        if reason is not None:
            raise ValueError(
                f"Available activation audit candidate has a reason: "
                f"tensor={tensor_name} method={method}"
            )
        tensor = tensors.get(tensor_key)
        if tensor is None:
            raise ValueError(
                f"Activation audit tensor is missing: "
                f"tensor={tensor_name} method={method}"
            )
        if tensor.dtype != torch.float32 or tuple(tensor.shape) != (
            evaluation_count,
            output_features,
        ):
            raise ValueError(
                f"Activation audit error tensor is invalid: "
                f"tensor={tensor_name} method={method}"
            )
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(
                f"Activation audit error is not finite: "
                f"tensor={tensor_name} method={method}"
            )
        if bool((tensor < 0).any().item()):
            raise ValueError(
                f"Activation audit error is negative: "
                f"tensor={tensor_name} method={method}"
            )
        sampled_fields, sampled_keys = _load_sampled_candidate(
            tensor_name,
            method,
            raw_method,
            tensors,
            tensor_key_index,
            sampled_sample_count=sampled_sample_count,
        )
        return (
            CandidateMeasurement(
                method=method,
                action=expected_action,
                available=True,
                storage_bytes=storage_bytes,
                error_by_eval_output=tensor,
                **sampled_fields,
            ),
            {tensor_key, *sampled_keys},
        )

    if storage_bytes is not None or tensor_key is not None or any(
        raw_method.get(field) is not None
        for field in (
            "sample_error_sse_key",
            "sample_reference_energy_key",
            "sample_direction_error_key",
            "sample_exact_sse",
            "sample_diag_sse",
            "sample_cross_term_ratio",
            "sample_unavailable_reason",
        )
    ):
        raise ValueError(
            f"Unavailable activation audit candidate has serialized data: "
            f"tensor={tensor_name} method={method}"
        )
    if not isinstance(reason, str) or not reason:
        raise ValueError(
            f"Activation audit unavailable reason is missing: "
            f"tensor={tensor_name} method={method}"
        )
    return (
        CandidateMeasurement(
            method=method,
            action=expected_action,
            available=False,
            storage_bytes=None,
            error_by_eval_output=None,
            unavailable_reason=reason,
        ),
        None,
    )


def _load_bias_energy(
    tensor_name: str,
    raw_layer: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
    layer_index: int,
) -> tuple[str | None, torch.Tensor | None, str | None, str | None]:
    bias_name = raw_layer.get("bias_name")
    if bias_name is not None and (
        not isinstance(bias_name, str) or not bias_name
    ):
        raise ValueError(f"Activation audit bias_name is invalid: {tensor_name}")
    bias_key = raw_layer.get("bias_free_output_energy_key")
    reason = raw_layer.get("bias_unavailable_reason")
    if reason is not None and (not isinstance(reason, str) or not reason):
        raise ValueError(
            f"Activation audit bias_unavailable_reason is invalid: {tensor_name}"
        )
    if bias_key is None:
        return bias_name, None, reason, None
    if not isinstance(bias_key, str) or bias_key != f"b{layer_index:06d}":
        raise ValueError(
            f"Activation audit bias tensor key is not deterministic: {tensor_name}"
        )
    if bias_name is None or reason is not None:
        raise ValueError(
            f"Activation audit bias metadata is inconsistent: {tensor_name}"
        )
    tensor = tensors.get(bias_key)
    if tensor is None:
        raise ValueError(f"Activation audit bias energy is missing: {tensor_name}")
    expected_shape = (
        int(raw_layer["evaluation_count"]),
        int(raw_layer["output_features"]),
    )
    if tensor.dtype != torch.float32 or tuple(tensor.shape) != expected_shape:
        raise ValueError(f"Activation audit bias energy is invalid: {tensor_name}")
    if not bool(torch.isfinite(tensor).all().item()) or bool((tensor < 0).any().item()):
        raise ValueError(
            f"Activation audit bias energy contains invalid values: {tensor_name}"
        )
    return bias_name, tensor, None, bias_key


def _load_sampled_candidate(
    tensor_name: str,
    method: str,
    raw_method: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
    tensor_key_index: int,
    *,
    sampled_sample_count: int | None,
) -> tuple[dict[str, object], set[str]]:
    key_fields = (
        ("sample_error_sse_key", "sample_error_sse", "s"),
        ("sample_reference_energy_key", "sample_reference_energy", "r"),
        ("sample_direction_error_key", "sample_direction_error", "d"),
    )
    raw_keys = [raw_method.get(metadata_name) for metadata_name, _, _ in key_fields]
    if all(key is None for key in raw_keys):
        if any(
            raw_method.get(name) is not None
            for name in (
                "sample_exact_sse",
                "sample_diag_sse",
                "sample_cross_term_ratio",
            )
        ):
            raise ValueError(
                f"Activation audit sampled scalar has no tensors: "
                f"tensor={tensor_name} method={method}"
            )
        reason = raw_method.get("sample_unavailable_reason")
        if reason is not None and (
            not isinstance(reason, str) or not reason
        ):
            raise ValueError(
                f"Activation audit sampled reason is invalid: "
                f"tensor={tensor_name} method={method}"
            )
        return {"sample_unavailable_reason": reason}, set()
    if any(not isinstance(key, str) or not key for key in raw_keys):
        raise ValueError(
            f"Activation audit sampled tensor keys are incomplete: "
            f"tensor={tensor_name} method={method}"
        )

    loaded: dict[str, torch.Tensor] = {}
    loaded_keys: set[str] = set()
    observed_sample_count = sampled_sample_count
    for metadata_name, field_name, prefix in key_fields:
        key = raw_method[metadata_name]
        expected_key = f"{prefix}{tensor_key_index:06d}"
        if key != expected_key:
            raise ValueError(
                f"Activation audit sampled tensor key is not deterministic: "
                f"tensor={tensor_name} method={method}"
            )
        tensor = tensors.get(key)
        if tensor is None:
            raise ValueError(
                f"Activation audit sampled tensor is missing: "
                f"tensor={tensor_name} method={method}"
            )
        if (
            tensor.dtype != torch.float32
            or tensor.ndim != 1
            or (
                observed_sample_count is not None
                and tensor.shape[0] != observed_sample_count
            )
        ):
            raise ValueError(
                f"Activation audit sampled tensor is invalid: "
                f"tensor={tensor_name} method={method} field={field_name}"
            )
        if observed_sample_count is None:
            observed_sample_count = int(tensor.shape[0])
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(
                f"Activation audit sampled tensor is not finite: "
                f"tensor={tensor_name} method={method} field={field_name}"
            )
        if bool((tensor < 0).any().item()):
            raise ValueError(
                f"Activation audit sampled tensor is negative: "
                f"tensor={tensor_name} method={method} field={field_name}"
            )
        loaded[field_name] = tensor
        loaded_keys.add(key)

    exact_sse = _optional_nonnegative_float(
        raw_method.get("sample_exact_sse"),
        "sample_exact_sse",
        required=True,
    )
    diag_sse = _optional_nonnegative_float(
        raw_method.get("sample_diag_sse"),
        "sample_diag_sse",
        required=True,
    )
    ratio = _optional_nonnegative_float(
        raw_method.get("sample_cross_term_ratio"),
        "sample_cross_term_ratio",
    )
    reason = raw_method.get("sample_unavailable_reason")
    if reason is not None and (not isinstance(reason, str) or not reason):
        raise ValueError(
            f"Activation audit sampled reason is invalid: "
            f"tensor={tensor_name} method={method}"
        )
    loaded.update(
        {
            "sample_exact_sse": exact_sse,
            "sample_diag_sse": diag_sse,
            "sample_cross_term_ratio": ratio,
            "sample_unavailable_reason": reason,
        }
    )
    return loaded, loaded_keys


def _validate_calibration_match(
    cache: ActivationAuditCache,
    calibration: ActivationCalibration,
) -> None:
    if calibration.version != cache.calibration_version:
        raise ValueError("Activation audit and calibration versions do not match.")
    if calibration.session_id != cache.calibration_session_id:
        raise ValueError("Activation audit calibration session_id does not match.")
    if calibration.baseline_label != cache.baseline_label:
        raise ValueError("Activation audit calibration baseline_label does not match.")
    if set(calibration.tensor_names()) != set(cache.measurements):
        raise ValueError("Activation audit and calibration layers do not match.")
    for tensor_name, methods in cache.measurements.items():
        layer = calibration.get(tensor_name)
        if not isinstance(layer, LayerCalibration):
            raise ValueError(
                f"Activation audit calibration layer is not V2: {tensor_name}"
            )
        input_features, output_features, evaluation_count = cache.layer_shapes[
            tensor_name
        ]
        if (
            evaluation_count != layer.evaluation_count
            or input_features != layer.input_features
            or output_features != layer.output_features
        ):
            raise ValueError(
                f"Activation audit and calibration shapes do not match: {tensor_name}"
            )
        bias_energy = cache.bias_free_output_energy.get(tensor_name)
        if bias_energy is not None and tuple(bias_energy.shape) != (
            layer.evaluation_count,
            layer.output_features,
        ):
            raise ValueError(
                f"Activation audit bias-free energy shape does not match: "
                f"{tensor_name}"
            )


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Activation audit {name} must be a non-empty string.")
    return value


def _require_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Activation audit {name} must be an integer >= 0.")
    return value


def _require_positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Activation audit {name} must be an integer >= 1.")
    return value


def _optional_nonnegative_float(
    value: Any,
    name: str,
    *,
    required: bool = False,
) -> float | None:
    if value is None:
        if required:
            raise ValueError(f"Activation audit {name} is required.")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Activation audit {name} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(
            f"Activation audit {name} must be finite and non-negative."
        )
    return result


__all__ = [
    "ACTIVATION_AUDIT_FORMAT",
    "ACTIVATION_AUDIT_VERSION",
    "ActivationAuditCache",
    "activation_audit_pair_paths",
    "inspect_activation_audit",
    "run_activation_audit",
    "score_activation_audit",
    "write_activation_audit_cache",
]
