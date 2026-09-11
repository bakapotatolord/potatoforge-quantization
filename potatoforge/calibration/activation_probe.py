"""Reusable activation probe caches and lightweight calibration scoring."""

from collections import Counter
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any, Literal, NamedTuple

import torch
from safetensors.torch import load_file, save_file

from .activation import (
    ActivationCalibration,
    merge_v2_activation_calibrations,
)


_PROBE_FORMAT = "potatoforge_activation_probe"
_PROBE_VERSION = 2
_SCORE_FORMAT = "potatoforge_activation_score"
_SCORE_VERSION = 2
_CALIBRATION_FORMAT = "potatoforge_activation_calibration"
_CALIBRATION_VERSION = 1
_REFERENCE_FORMAT = "int8_convrot"
_CANDIDATE_FORMAT = "convrot_w4a4"
_METRIC_BASIS = "logical_linear_input"
_METRIC = "diagonal_activation_energy_v1"
_RELATIVE_METRIC = "relative_diagonal_output_error_v1"
_OK_STATUS = "ok"
_UNSUPPORTED_STATUS = "unsupported_candidate"
_ZERO_REFERENCE_OUTPUT_STATUS = "zero_reference_output_energy"


ProbeStatus = Literal["ok", "unsupported_candidate"]


class ActivationProbeRecord(NamedTuple):
    tensor_name: str
    input_features: int
    status: ProbeStatus
    q_per_input: torch.Tensor | None
    reference_power_per_input: torch.Tensor | None


def activation_pair_paths(
    output_path: str | Path,
) -> tuple[Path, Path]:
    path = Path(output_path)
    if path.suffix.lower() == ".json":
        return path, path.with_suffix(".safetensors")
    if path.suffix.lower() == ".safetensors":
        return path.with_suffix(".json"), path
    return path.with_suffix(".json"), path.with_suffix(".safetensors")


class ActivationProbeCache:
    def __init__(
        self,
        *,
        source_model_path: str | None,
        records: Mapping[str, ActivationProbeRecord],
    ) -> None:
        self.source_model_path = source_model_path
        self.reference_format = _REFERENCE_FORMAT
        self.candidate_format = _CANDIDATE_FORMAT
        self.metric_basis = _METRIC_BASIS
        self.metric = _METRIC
        self._records = dict(records)

    @classmethod
    def load(
        cls,
        metadata_path: str | Path,
        tensors_path: str | Path | None = None,
    ) -> "ActivationProbeCache":
        metadata_file = Path(metadata_path)
        metadata_file, default_tensors_file = activation_pair_paths(
            metadata_file
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
                f"Invalid activation probe metadata: {metadata_file}"
            ) from exc

        if not isinstance(document, dict):
            raise ValueError("Activation probe metadata must be an object.")
        _validate_probe_metadata(document)
        tensors = load_file(str(tensors_file), device="cpu")

        raw_records = document["tensors"]
        expected_keys: set[str] = set()
        records: dict[str, ActivationProbeRecord] = {}
        for tensor_name, raw_record in raw_records.items():
            record = _load_probe_record(tensor_name, raw_record, tensors)
            records[tensor_name] = record
            if record.q_per_input is not None:
                expected_keys.add(f"{tensor_name}.q_per_input")
                expected_keys.add(
                    f"{tensor_name}.reference_power_per_input"
                )

        if set(tensors) != expected_keys:
            raise ValueError(
                "Activation probe tensors do not match metadata tensor entries."
            )

        return cls(
            source_model_path=_optional_string(
                document,
                "source_model_path",
            ),
            records=records,
        )

    def get(self, tensor_name: str) -> ActivationProbeRecord | None:
        return self._records.get(tensor_name)

    def tensor_names(self) -> tuple[str, ...]:
        return tuple(self._records)


def write_activation_probe_cache(
    output_path: str | Path,
    source_model_path: str | Path,
    records: Mapping[str, ActivationProbeRecord],
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    metadata_path, tensors_path = activation_pair_paths(output_path)
    if not overwrite and (metadata_path.exists() or tensors_path.exists()):
        raise FileExistsError(
            f"Activation probe output already exists: {metadata_path}"
        )

    serialized_tensors: dict[str, torch.Tensor] = {}
    serialized_records: dict[str, dict[str, object]] = {}
    for tensor_name, record in records.items():
        if record.tensor_name != tensor_name:
            raise ValueError(
                f"Activation probe record name mismatch: {tensor_name}"
            )
        if record.input_features < 1:
            raise ValueError(
                f"Activation probe input_features must be positive: {tensor_name}"
            )
        q_key = None
        reference_power_key = None
        if record.status == _OK_STATUS:
            if record.q_per_input is None:
                raise ValueError(
                    f"Activation probe q_per_input is missing: {tensor_name}"
                )
            if record.reference_power_per_input is None:
                raise ValueError(
                    "Activation probe reference_power_per_input is missing: "
                    f"{tensor_name}"
                )
            q_per_input = record.q_per_input.detach().to(
                device="cpu",
                dtype=torch.float32,
            )
            if q_per_input.ndim != 1:
                raise ValueError(
                    f"Activation probe q_per_input must be rank 1: {tensor_name}"
                )
            if q_per_input.shape[0] != record.input_features:
                raise ValueError(
                    "Activation probe q_per_input length does not match "
                    f"input_features: {tensor_name}"
                )
            if not bool(torch.isfinite(q_per_input).all().item()):
                raise ValueError(
                    f"Activation probe q_per_input must be finite: {tensor_name}"
                )
            if bool((q_per_input < 0).any().item()):
                raise ValueError(
                    "Activation probe q_per_input must be non-negative: "
                    f"{tensor_name}"
                )
            q_key = f"{tensor_name}.q_per_input"
            serialized_tensors[q_key] = q_per_input.contiguous()
            reference_power_per_input = (
                record.reference_power_per_input.detach().to(
                    device="cpu",
                    dtype=torch.float32,
                )
            )
            if reference_power_per_input.ndim != 1:
                raise ValueError(
                    "Activation probe reference_power_per_input must be rank 1: "
                    f"{tensor_name}"
                )
            if reference_power_per_input.shape[0] != record.input_features:
                raise ValueError(
                    "Activation probe reference_power_per_input length does "
                    "not match input_features: "
                    f"{tensor_name}"
                )
            if not bool(torch.isfinite(reference_power_per_input).all().item()):
                raise ValueError(
                    "Activation probe reference_power_per_input must be finite: "
                    f"{tensor_name}"
                )
            if bool((reference_power_per_input < 0).any().item()):
                raise ValueError(
                    "Activation probe reference_power_per_input must be "
                    "non-negative: "
                    f"{tensor_name}"
                )
            reference_power_key = (
                f"{tensor_name}.reference_power_per_input"
            )
            serialized_tensors[reference_power_key] = (
                reference_power_per_input.contiguous()
            )
        elif record.status != _UNSUPPORTED_STATUS:
            raise ValueError(
                f"Unsupported activation probe status: {record.status!r}"
            )
        elif record.q_per_input is not None:
            raise ValueError(
                "Unsupported activation probe records cannot contain q_per_input: "
                f"{tensor_name}"
            )
        elif record.reference_power_per_input is not None:
            raise ValueError(
                "Unsupported activation probe records cannot contain "
                "reference_power_per_input: "
                f"{tensor_name}"
            )

        serialized_records[tensor_name] = {
            "tensor_name": tensor_name,
            "input_features": record.input_features,
            "q_key": q_key,
            "reference_power_key": reference_power_key,
            "status": record.status,
        }

    metadata = {
        "format": _PROBE_FORMAT,
        "version": _PROBE_VERSION,
        "source_model_path": str(Path(source_model_path).resolve()),
        "reference_format": _REFERENCE_FORMAT,
        "candidate_format": _CANDIDATE_FORMAT,
        "metric_basis": _METRIC_BASIS,
        "metric": _METRIC,
        "tensor_count": len(serialized_records),
        "tensors": serialized_records,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(serialized_tensors, str(tensors_path))
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata_path, tensors_path


def score_activation_probe(
    probe: ActivationProbeCache | str | Path,
    calibration: ActivationCalibration | str | Path,
) -> dict[str, object]:
    probe_cache = (
        probe
        if isinstance(probe, ActivationProbeCache)
        else ActivationProbeCache.load(probe)
    )
    activation_calibration = (
        calibration
        if isinstance(calibration, ActivationCalibration)
        else ActivationCalibration.load(calibration)
    )
    if activation_calibration.baseline_label != _REFERENCE_FORMAT:
        raise ValueError(
            "Activation calibration baseline_label must be "
            f"{_REFERENCE_FORMAT!r}."
        )
    if activation_calibration.activation_basis != _METRIC_BASIS:
        raise ValueError(
            "Activation calibration activation_basis must be "
            f"{_METRIC_BASIS!r}."
        )
    if activation_calibration.activation_axis != "last_dimension":
        raise ValueError(
            "Activation calibration activation_axis must be 'last_dimension'."
        )
    results: list[dict[str, object]] = []
    status_counts: Counter[str] = Counter()
    for tensor_name in activation_calibration.tensor_names():
        stats = activation_calibration.get(tensor_name)
        assert stats is not None
        probe_record = probe_cache.get(tensor_name)
        result: dict[str, object] = {
            "tensor_name": tensor_name,
            "activation_error": None,
            "activation_error_mean": None,
            "weight_error_energy": None,
            "input_activation_energy_mean": None,
            "activation_alignment_ratio": None,
            "reference_output_energy": None,
            "relative_output_error_sq": None,
            "relative_output_error": None,
            "relative_output_error_rank": None,
            "activation_rank_raw": None,
            "activation_rank_mean": None,
            "activation_samples": stats.sample_count,
            "activation_invocations": stats.invocation_count,
            "activation_baseline": activation_calibration.baseline_label,
            "activation_reference": probe_cache.reference_format,
            "activation_candidate": probe_cache.candidate_format,
            "activation_status": "missing_probe",
        }
        if probe_record is None:
            status_counts["missing_probe"] += 1
            results.append(result)
            continue
        if probe_record.input_features != stats.input_features:
            raise ValueError(
                "Activation probe and calibration input_features disagree for "
                f"{tensor_name}: {probe_record.input_features} vs "
                f"{stats.input_features}."
            )
        if probe_record.status != _OK_STATUS:
            result["activation_status"] = probe_record.status
            status_counts[probe_record.status] += 1
            results.append(result)
            continue

        q_per_input = probe_record.q_per_input
        if q_per_input is None:
            raise ValueError(
                f"Activation probe q_per_input is missing for {tensor_name}."
            )
        reference_power_per_input = probe_record.reference_power_per_input
        if reference_power_per_input is None:
            raise ValueError(
                "Activation probe reference_power_per_input is missing for "
                f"{tensor_name}."
            )
        if q_per_input.shape[0] != stats.sum_x2.shape[0]:
            raise ValueError(
                "Activation probe and calibration lengths disagree for "
                f"{tensor_name}: {q_per_input.shape[0]} vs "
                f"{stats.sum_x2.shape[0]}."
            )
        if reference_power_per_input.shape[0] != stats.sum_x2.shape[0]:
            raise ValueError(
                "Activation reference power and calibration lengths disagree "
                f"for {tensor_name}: {reference_power_per_input.shape[0]} vs "
                f"{stats.sum_x2.shape[0]}."
            )
        activation_error_tensor = torch.dot(q_per_input, stats.sum_x2)
        activation_error = activation_error_tensor.item()
        if not bool(torch.isfinite(activation_error_tensor).item()):
            raise ValueError(
                f"Activation score is not finite for {tensor_name}."
            )
        result["activation_error"] = float(activation_error)
        if stats.sample_count > 0:
            result["activation_error_mean"] = float(
                (activation_error_tensor / stats.sample_count).item()
            )
        weight_error_energy_tensor = q_per_input.sum()
        input_activation_energy_tensor = stats.sum_x2.sum()
        result["weight_error_energy"] = float(
            weight_error_energy_tensor.item()
        )
        if stats.sample_count > 0:
            result["input_activation_energy_mean"] = float(
                (
                    input_activation_energy_tensor
                    / (stats.sample_count * stats.input_features)
                ).item()
            )
        expected_if_uniform = (
            input_activation_energy_tensor
            * weight_error_energy_tensor
            / stats.input_features
        )
        if expected_if_uniform.item() != 0.0:
            result["activation_alignment_ratio"] = float(
                (activation_error_tensor / expected_if_uniform).item()
            )
        reference_output_energy_tensor = torch.dot(
            reference_power_per_input,
            stats.sum_x2,
        )
        if not bool(torch.isfinite(reference_output_energy_tensor).item()):
            raise ValueError(
                f"Reference output energy is not finite for {tensor_name}."
            )
        reference_output_energy = reference_output_energy_tensor.item()
        result["reference_output_energy"] = float(reference_output_energy)
        if reference_output_energy > 0:
            relative_output_error_sq_tensor = (
                activation_error_tensor / reference_output_energy_tensor
            )
            if not bool(torch.isfinite(relative_output_error_sq_tensor).item()):
                raise ValueError(
                    "Relative activation error is not finite for "
                    f"{tensor_name}."
                )
            result["relative_output_error_sq"] = float(
                relative_output_error_sq_tensor.item()
            )
            result["relative_output_error"] = float(
                torch.sqrt(relative_output_error_sq_tensor).item()
            )
            result["activation_status"] = _OK_STATUS
            status_counts[_OK_STATUS] += 1
        else:
            result["activation_status"] = _ZERO_REFERENCE_OUTPUT_STATUS
            status_counts[_ZERO_REFERENCE_OUTPUT_STATUS] += 1
        results.append(result)

    _assign_activation_rank(
        results,
        value_key="activation_error",
        rank_key="activation_rank_raw",
    )
    _assign_activation_rank(
        results,
        value_key="activation_error_mean",
        rank_key="activation_rank_mean",
    )
    _assign_activation_rank(
        results,
        value_key="relative_output_error",
        rank_key="relative_output_error_rank",
    )

    return {
        "format": _SCORE_FORMAT,
        "version": _SCORE_VERSION,
        "probe_cache": (
            None
            if isinstance(probe, ActivationProbeCache)
            else str(Path(probe).resolve())
        ),
        "activation_calibration": (
            None
            if isinstance(calibration, ActivationCalibration)
            else str(Path(calibration).resolve())
        ),
        "reference_format": probe_cache.reference_format,
        "candidate_format": probe_cache.candidate_format,
        "metric_basis": probe_cache.metric_basis,
        "metrics": [_METRIC, _RELATIVE_METRIC],
        "primary_metric": _RELATIVE_METRIC,
        "summary": {
            "calibration_tensor_count": len(results),
            "status_counts": dict(sorted(status_counts.items())),
            "scored_tensor_count": status_counts[_OK_STATUS],
        },
        "results": results,
    }


def merge_activation_calibrations(
    metadata_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    if not metadata_paths:
        raise ValueError("At least one activation calibration is required.")

    calibrations = [
        ActivationCalibration.load(metadata_path)
        for metadata_path in metadata_paths
    ]
    first = calibrations[0]
    if any(calibration.version != first.version for calibration in calibrations[1:]):
        raise ValueError("Cannot merge activation calibrations across versions.")
    if first.version == 2:
        return merge_v2_activation_calibrations(
            calibrations,
            output_path,
            overwrite=overwrite,
        )
    for index, calibration in enumerate(calibrations[1:], start=2):
        for field in (
            "baseline_label",
            "activation_basis",
            "activation_axis",
            "diffusion_model_class",
        ):
            if getattr(calibration, field) != getattr(first, field):
                raise ValueError(
                    f"Calibration {index} disagrees on {field}."
                )
        if set(calibration.tensor_names()) != set(first.tensor_names()):
            raise ValueError(
                f"Calibration {index} has different canonical tensor names."
            )

    merged_records: dict[str, tuple[int, int, torch.Tensor]] = {}
    for tensor_name in first.tensor_names():
        stats = [calibration.get(tensor_name) for calibration in calibrations]
        if any(item is None for item in stats):
            raise ValueError(f"Missing calibration tensor: {tensor_name}")
        first_stats = stats[0]
        assert first_stats is not None
        for item in stats[1:]:
            assert item is not None
            if item.input_features != first_stats.input_features:
                raise ValueError(
                    "Calibration input_features disagree for "
                    f"{tensor_name}."
                )
        sum_x2 = torch.stack(
            [item.sum_x2 for item in stats if item is not None]
        ).sum(dim=0)
        if not bool(torch.isfinite(sum_x2).all().item()):
            raise ValueError(
                f"Merged activation statistics are not finite: {tensor_name}"
            )
        merged_records[tensor_name] = (
            sum(item.sample_count for item in stats if item is not None),
            sum(item.invocation_count for item in stats if item is not None),
            sum_x2,
        )

    metadata_file, tensors_file = activation_pair_paths(output_path)
    if not overwrite and (metadata_file.exists() or tensors_file.exists()):
        raise FileExistsError(
            f"Merged activation calibration already exists: {metadata_file}"
        )
    tensor_payloads: dict[str, torch.Tensor] = {}
    layers: dict[str, dict[str, object]] = {}
    for tensor_name, (sample_count, invocation_count, sum_x2) in merged_records.items():
        stats_key = f"{tensor_name}.sum_x2"
        tensor_payloads[stats_key] = sum_x2.to(dtype=torch.float32)
        layers[tensor_name] = {
            "input_features": int(sum_x2.shape[0]),
            "sample_count": sample_count,
            "invocation_count": invocation_count,
            "stats_key": stats_key,
        }
    metadata: dict[str, Any] = {
        "format": _CALIBRATION_FORMAT,
        "version": _CALIBRATION_VERSION,
        "session_name": "merged",
        "baseline_label": first.baseline_label,
        "activation_basis": first.activation_basis,
        "activation_axis": first.activation_axis,
        "layer_count": len(layers),
        "layers": layers,
    }
    if first.diffusion_model_class is not None:
        metadata["diffusion_model_class"] = first.diffusion_model_class

    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensor_payloads, str(tensors_file))
    metadata_file.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata_file, tensors_file


def _validate_probe_metadata(document: dict[str, Any]) -> None:
    expected = {
        "format": _PROBE_FORMAT,
        "version": _PROBE_VERSION,
        "reference_format": _REFERENCE_FORMAT,
        "candidate_format": _CANDIDATE_FORMAT,
        "metric_basis": _METRIC_BASIS,
        "metric": _METRIC,
    }
    for field, value in expected.items():
        if document.get(field) != value:
            raise ValueError(
                f"Unsupported activation probe {field}: {document.get(field)!r}"
            )
    tensor_count = document.get("tensor_count")
    if type(tensor_count) is not int or tensor_count < 0:
        raise ValueError("Activation probe tensor_count must be non-negative.")
    tensors = document.get("tensors")
    if not isinstance(tensors, dict):
        raise ValueError("Activation probe tensors must be an object.")
    if tensor_count != len(tensors):
        raise ValueError("Activation probe tensor_count does not match tensors.")
    _optional_string(document, "source_model_path")


def _assign_activation_rank(
    results: list[dict[str, object]],
    *,
    value_key: str,
    rank_key: str,
) -> None:
    ranked = sorted(
        (
            result
            for result in results
            if result[value_key] is not None
        ),
        key=lambda result: (
            -float(result[value_key]),
            str(result["tensor_name"]),
        ),
    )
    for rank, result in enumerate(ranked, start=1):
        result[rank_key] = rank


def _load_probe_record(
    tensor_name: Any,
    raw_record: Any,
    tensors: Mapping[str, torch.Tensor],
) -> ActivationProbeRecord:
    if not isinstance(tensor_name, str) or not tensor_name:
        raise ValueError("Activation probe tensor names must be non-empty strings.")
    if not isinstance(raw_record, dict):
        raise ValueError(f"Activation probe entry must be an object: {tensor_name}")
    if raw_record.get("tensor_name") != tensor_name:
        raise ValueError(f"Activation probe tensor_name mismatch: {tensor_name}")
    input_features = raw_record.get("input_features")
    if type(input_features) is not int or input_features < 1:
        raise ValueError(
            f"Activation probe input_features must be positive: {tensor_name}"
        )
    status = raw_record.get("status")
    q_key = raw_record.get("q_key")
    reference_power_key = raw_record.get("reference_power_key")
    if status == _UNSUPPORTED_STATUS:
        if q_key is not None or reference_power_key is not None:
            raise ValueError(
                "Unsupported activation probe entry has cached vectors: "
                f"{tensor_name}"
            )
        return ActivationProbeRecord(
            tensor_name,
            input_features,
            _UNSUPPORTED_STATUS,
            None,
            None,
        )
    if (
        status != _OK_STATUS
        or not isinstance(q_key, str)
        or not q_key
        or not isinstance(reference_power_key, str)
        or not reference_power_key
    ):
        raise ValueError(f"Invalid activation probe entry: {tensor_name}")
    expected_q_key = f"{tensor_name}.q_per_input"
    if q_key != expected_q_key:
        raise ValueError(
            f"Activation probe q_key must be {expected_q_key!r}: {tensor_name}"
        )
    expected_reference_power_key = (
        f"{tensor_name}.reference_power_per_input"
    )
    if reference_power_key != expected_reference_power_key:
        raise ValueError(
            "Activation probe reference_power_key must be "
            f"{expected_reference_power_key!r}: {tensor_name}"
        )
    if q_key not in tensors:
        raise ValueError(f"Activation probe q tensor is missing: {q_key}")
    if reference_power_key not in tensors:
        raise ValueError(
            "Activation probe reference power tensor is missing: "
            f"{reference_power_key}"
        )
    q_per_input = tensors[q_key]
    if q_per_input.ndim != 1 or q_per_input.shape[0] != input_features:
        raise ValueError(
            "Activation probe q_per_input length does not match input_features: "
            f"{tensor_name}"
        )
    if q_per_input.dtype == torch.bool or q_per_input.is_complex():
        raise ValueError(f"Activation probe q_per_input must be real: {tensor_name}")
    q_per_input = q_per_input.detach().to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(q_per_input).all().item()):
        raise ValueError(f"Activation probe q_per_input must be finite: {tensor_name}")
    if bool((q_per_input < 0).any().item()):
        raise ValueError(
            f"Activation probe q_per_input must be non-negative: {tensor_name}"
        )
    reference_power_per_input = tensors[reference_power_key]
    if (
        reference_power_per_input.ndim != 1
        or reference_power_per_input.shape[0] != input_features
    ):
        raise ValueError(
            "Activation probe reference_power_per_input length does not match "
            f"input_features: {tensor_name}"
        )
    if (
        reference_power_per_input.dtype == torch.bool
        or reference_power_per_input.is_complex()
    ):
        raise ValueError(
            "Activation probe reference_power_per_input must be real: "
            f"{tensor_name}"
        )
    reference_power_per_input = reference_power_per_input.detach().to(
        device="cpu",
        dtype=torch.float32,
    )
    if not bool(torch.isfinite(reference_power_per_input).all().item()):
        raise ValueError(
            "Activation probe reference_power_per_input must be finite: "
            f"{tensor_name}"
        )
    if bool((reference_power_per_input < 0).any().item()):
        raise ValueError(
            "Activation probe reference_power_per_input must be non-negative: "
            f"{tensor_name}"
        )
    return ActivationProbeRecord(
        tensor_name,
        input_features,
        _OK_STATUS,
        q_per_input,
        reference_power_per_input,
    )


def _optional_string(document: Mapping[str, Any], key: str) -> str | None:
    value = document.get(key)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"Activation probe {key} must be a string.")
    return value
