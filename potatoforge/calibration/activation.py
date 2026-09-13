"""Reader for PotatoForge activation calibration artifacts."""

import json
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Real
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..headers.source_header import SourceModelHeader, read_source_model_header


_FORMAT = "potatoforge_activation_calibration"
_ACTIVATION_VERSION = 1
_ACTIVATION_BASIS = "logical_linear_input"
_ACTIVATION_AXIS = "last_dimension"
_ROOT_INPUT_SUM_X2_KEY = "__pf__.root_input_sum_x2"
_ROOT_INPUT_VALID_KEY = "__pf__.root_input_valid"
_ROOT_OUTPUT_SUM_Y2_KEY = "__pf__.root_output_sum_y2"
_ROOT_OUTPUT_VALID_KEY = "__pf__.root_output_valid"


@dataclass(frozen=True)
class EvaluationMetadata:
    index: int
    timestep: float | None = None
    sigma: float | None = None
    root_input_sum_x2: float | None = None
    root_output_sum_y2: float | None = None
    time_parameter_name: str | None = None
    time_value: float | tuple[float, ...] | None = None
    time_value_truncated: bool = False


class _CalibrationTensorStore:
    def __init__(self, tensors_path: Path) -> None:
        self._handle = safe_open(
            str(tensors_path),
            framework="pt",
            device="cpu",
        )
        self.keys = frozenset(self._handle.keys())

    def get(self, key: str) -> torch.Tensor:
        return self._handle.get_tensor(key)


class LayerCalibration:
    """Validated activation statistics for one logical Linear weight."""

    def __init__(
        self,
        *,
        tensor_name: str,
        input_features: int,
        output_features: int,
        sample_count: int,
        invocation_count: int,
        evaluation_count: int,
        tensor_keys: Mapping[str, str],
        tensor_store: _CalibrationTensorStore,
    ) -> None:
        self.tensor_name = tensor_name
        self.input_features = input_features
        self.output_features = output_features
        self.sample_count = sample_count
        self.invocation_count = invocation_count
        self.evaluation_count = evaluation_count
        self._tensor_keys = dict(tensor_keys)
        self._tensor_store = tensor_store
        self._cache: dict[str, torch.Tensor] = {}
        self._sample_x: torch.Tensor | None = None
        self._sample_evaluation_indices: torch.Tensor | None = None

    def _tensor(self, name: str) -> torch.Tensor:
        cached = self._cache.get(name)
        if cached is None:
            cached = self._tensor_store.get(self._tensor_keys[name])
            self._cache[name] = cached
        return cached

    @property
    def aggregate_sum_x2(self) -> torch.Tensor:
        return self._tensor("aggregate_sum_x2")

    @property
    def eval_sum_x(self) -> torch.Tensor:
        return self._tensor("eval_sum_x")

    @property
    def eval_sum_x2(self) -> torch.Tensor:
        return self._tensor("eval_sum_x2")

    @property
    def eval_max_abs_x(self) -> torch.Tensor:
        return self._tensor("eval_max_abs_x")

    @property
    def eval_sum_y(self) -> torch.Tensor:
        return self._tensor("eval_sum_y")

    @property
    def eval_sum_y2(self) -> torch.Tensor:
        return self._tensor("eval_sum_y2")

    @property
    def eval_sample_counts(self) -> torch.Tensor:
        return self._tensor("eval_sample_count").to(dtype=torch.int64)

    @property
    def eval_invocation_counts(self) -> torch.Tensor:
        return self._tensor("eval_invocation_count").to(dtype=torch.int64)

    @property
    def sample_x(self) -> torch.Tensor | None:
        if "sample_x" not in self._tensor_keys:
            return None
        if self._sample_x is None:
            padded = self._tensor("sample_x").to(
                device="cpu",
                dtype=torch.float32,
            )
            valid_counts = self._tensor("sample_x_valid").to(
                device="cpu",
                dtype=torch.int64,
            )
            chunks: list[torch.Tensor] = []
            evaluation_indices: list[int] = []
            for evaluation_index, count in enumerate(valid_counts.tolist()):
                if count:
                    chunks.append(padded[evaluation_index, :count])
                    evaluation_indices.extend([evaluation_index] * count)
            self._sample_x = (
                torch.cat(chunks, dim=0).contiguous()
                if chunks
                else torch.zeros(
                    (0, self.input_features),
                    dtype=torch.float32,
                )
            )
            self._sample_evaluation_indices = torch.tensor(
                evaluation_indices,
                dtype=torch.int64,
            )
        return self._sample_x

    @property
    def sample_evaluation_indices(self) -> torch.Tensor | None:
        if self.sample_x is None:
            return None
        assert self._sample_evaluation_indices is not None
        return self._sample_evaluation_indices

    def clear_cached_tensors(self) -> None:
        """Release materialized calibration tensors for this layer."""
        self._cache.clear()
        self._sample_x = None
        self._sample_evaluation_indices = None


class ActivationCalibration:
    def __init__(
        self,
        *,
        baseline_label: str | None,
        session_id: str | None,
        session_name: str | None,
        activation_basis: str,
        activation_axis: str,
        stats: Mapping[str, LayerCalibration],
        diffusion_model_class: str | None = None,
        version: int = _ACTIVATION_VERSION,
        evaluations: tuple[EvaluationMetadata, ...] = (),
    ) -> None:
        self.baseline_label = baseline_label
        self.session_id = session_id
        self.session_name = session_name
        self.diffusion_model_class = diffusion_model_class
        self.activation_basis = activation_basis
        self.activation_axis = activation_axis
        self.version = version
        self.evaluations = evaluations
        self.evaluation_count = len(evaluations)
        self._stats = dict(stats)

    @classmethod
    def load(
        cls,
        metadata_path: str | Path,
        tensors_path: str | Path | None = None,
    ) -> "ActivationCalibration":
        metadata_file = Path(metadata_path)
        tensors_file = (
            metadata_file.with_suffix(".safetensors")
            if tensors_path is None
            else Path(tensors_path)
        )
        try:
            document = json.loads(metadata_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid activation calibration metadata: {metadata_file}"
            ) from exc

        if not isinstance(document, dict):
            raise ValueError("Activation calibration metadata must be an object.")
        version = document.get("version")
        if type(version) is not int or version != _ACTIVATION_VERSION:
            raise ValueError(
                f"Unsupported activation calibration version: {version!r}"
            )
        return _load_activation_calibration(tensors_file, document)

    def get(self, tensor_name: str) -> LayerCalibration | None:
        return self._stats.get(tensor_name)

    def has(self, tensor_name: str) -> bool:
        return tensor_name in self._stats

    def tensor_names(self) -> tuple[str, ...]:
        return tuple(self._stats)

    @property
    def layers(self) -> Mapping[str, LayerCalibration]:
        return self._stats

    def validate_against_source(
        self,
        source_path: str | Path,
    ) -> SourceModelHeader:
        return validate_activation_calibration_against_source(
            self,
            source_path,
        )


def load_activation_calibration(
    metadata_path: str | Path,
    stats_path: str | Path | None = None,
) -> ActivationCalibration:
    """Load an activation calibration pair with strict schema validation."""
    return ActivationCalibration.load(metadata_path, stats_path)


def merge_activation_calibrations(
    calibrations: Sequence[ActivationCalibration | str | Path],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Merge activation calibration evaluations without reading a source checkpoint."""
    if not calibrations:
        raise ValueError("At least one activation calibration is required.")
    loaded = tuple(
        calibration
        if isinstance(calibration, ActivationCalibration)
        else ActivationCalibration.load(calibration)
        for calibration in calibrations
    )
    if any(calibration.version != _ACTIVATION_VERSION for calibration in loaded):
        raise ValueError("Activation calibration merge requires current pairs.")

    first = loaded[0]
    for index, calibration in enumerate(loaded[1:], start=2):
        for field in (
            "baseline_label",
            "activation_basis",
            "activation_axis",
            "diffusion_model_class",
        ):
            if getattr(calibration, field) != getattr(first, field):
                raise ValueError(f"Calibration {index} disagrees on {field}.")
        if set(calibration.tensor_names()) != set(first.tensor_names()):
            raise ValueError(
                f"Calibration {index} has different canonical tensor names."
            )

    evaluations: list[EvaluationMetadata] = []
    for calibration in loaded:
        offset = len(evaluations)
        evaluations.extend(
            replace(evaluation, index=evaluation.index + offset)
            for evaluation in calibration.evaluations
        )

    tensor_payloads: dict[str, torch.Tensor] = {}
    layers: dict[str, dict[str, object]] = {}
    for tensor_name in sorted(first.tensor_names()):
        source_layers = [calibration.get(tensor_name) for calibration in loaded]
        if any(not isinstance(layer, LayerCalibration) for layer in source_layers):
            raise ValueError(
                "Calibration layer is not a valid activation layer: "
                f"{tensor_name}"
            )
        typed_layers = tuple(
            layer for layer in source_layers if isinstance(layer, LayerCalibration)
        )
        first_layer = typed_layers[0]
        for index, layer in enumerate(typed_layers[1:], start=2):
            if (
                layer.input_features != first_layer.input_features
                or layer.output_features != first_layer.output_features
            ):
                raise ValueError(
                    f"Calibration {index} disagrees on shape for {tensor_name}."
                )

        prefix = tensor_name
        aggregate_sum_x2 = torch.stack(
            [layer.aggregate_sum_x2 for layer in typed_layers]
        ).sum(dim=0)
        eval_sum_x = torch.cat(
            [layer.eval_sum_x for layer in typed_layers],
            dim=0,
        )
        eval_sum_x2 = torch.cat(
            [layer.eval_sum_x2 for layer in typed_layers],
            dim=0,
        )
        eval_max_abs_x = torch.cat(
            [layer.eval_max_abs_x for layer in typed_layers],
            dim=0,
        )
        eval_sum_y = torch.cat(
            [layer.eval_sum_y for layer in typed_layers],
            dim=0,
        )
        eval_sum_y2 = torch.cat(
            [layer.eval_sum_y2 for layer in typed_layers],
            dim=0,
        )
        eval_sample_count = torch.cat(
            [layer.eval_sample_counts for layer in typed_layers],
            dim=0,
        )
        eval_invocation_count = torch.cat(
            [layer.eval_invocation_counts for layer in typed_layers],
            dim=0,
        )
        tensor_payloads[f"{prefix}.sum_x2"] = aggregate_sum_x2.to(
            dtype=torch.float32
        ).contiguous()
        tensor_payloads[f"{prefix}.eval_sum_x"] = eval_sum_x.to(
            dtype=torch.float32
        ).contiguous()
        tensor_payloads[f"{prefix}.eval_sum_x2"] = eval_sum_x2.to(
            dtype=torch.float32
        ).contiguous()
        tensor_payloads[f"{prefix}.eval_max_abs_x"] = eval_max_abs_x.to(
            dtype=torch.float32
        ).contiguous()
        tensor_payloads[f"{prefix}.eval_sum_y"] = eval_sum_y.to(
            dtype=torch.float32
        ).contiguous()
        tensor_payloads[f"{prefix}.eval_sum_y2"] = eval_sum_y2.to(
            dtype=torch.float32
        ).contiguous()
        tensor_payloads[f"{prefix}.eval_sample_count"] = eval_sample_count.to(
            dtype=torch.int64
        ).contiguous()
        tensor_payloads[f"{prefix}.eval_invocation_count"] = (
            eval_invocation_count.to(dtype=torch.int64).contiguous()
        )

        sample_layers = [layer.sample_x for layer in typed_layers]
        if any(sample is not None for sample in sample_layers):
            rows_by_evaluation: list[torch.Tensor] = []
            max_rows = 0
            for layer, sample in zip(typed_layers, sample_layers, strict=True):
                indices = layer.sample_evaluation_indices
                if sample is None or indices is None:
                    rows = [
                        torch.zeros(
                            (0, first_layer.input_features),
                            dtype=torch.float32,
                        )
                        for _ in range(layer.evaluation_count)
                    ]
                else:
                    rows = [
                        sample[indices == evaluation_index]
                        for evaluation_index in range(layer.evaluation_count)
                    ]
                rows_by_evaluation.extend(rows)
                max_rows = max(max_rows, *(row.shape[0] for row in rows))
            if max_rows:
                padded = torch.zeros(
                    (
                        len(rows_by_evaluation),
                        max_rows,
                        first_layer.input_features,
                    ),
                    dtype=torch.float32,
                )
                valid = torch.zeros(
                    len(rows_by_evaluation),
                    dtype=torch.int64,
                )
                for evaluation_index, rows in enumerate(rows_by_evaluation):
                    count = rows.shape[0]
                    if count:
                        padded[evaluation_index, :count] = rows
                        valid[evaluation_index] = count
                tensor_payloads[f"{prefix}.sample_x"] = padded.contiguous()
                tensor_payloads[f"{prefix}.sample_x_valid"] = valid

        layers[tensor_name] = {
            "input_features": first_layer.input_features,
            "output_features": first_layer.output_features,
            "sample_count": sum(layer.sample_count for layer in typed_layers),
            "invocation_count": sum(
                layer.invocation_count for layer in typed_layers
            ),
            "stats_key": f"{prefix}.sum_x2",
            "eval_sum_x_key": f"{prefix}.eval_sum_x",
            "eval_sum_x2_key": f"{prefix}.eval_sum_x2",
            "eval_max_abs_x_key": f"{prefix}.eval_max_abs_x",
            "eval_sum_y_key": f"{prefix}.eval_sum_y",
            "eval_sum_y2_key": f"{prefix}.eval_sum_y2",
            "eval_sample_count_key": f"{prefix}.eval_sample_count",
            "eval_invocation_count_key": f"{prefix}.eval_invocation_count",
            "sample_x_key": (
                f"{prefix}.sample_x"
                if f"{prefix}.sample_x" in tensor_payloads
                else None
            ),
            "sample_x_valid_key": (
                f"{prefix}.sample_x_valid"
                if f"{prefix}.sample_x_valid" in tensor_payloads
                else None
            ),
        }

    root_inputs = [evaluation.root_input_sum_x2 for evaluation in evaluations]
    root_outputs = [evaluation.root_output_sum_y2 for evaluation in evaluations]
    root_input_present = any(value is not None for value in root_inputs)
    root_output_present = any(value is not None for value in root_outputs)
    if root_input_present:
        tensor_payloads[_ROOT_INPUT_SUM_X2_KEY] = torch.tensor(
            [0.0 if value is None else value for value in root_inputs],
            dtype=torch.float32,
        )
        tensor_payloads[_ROOT_INPUT_VALID_KEY] = torch.tensor(
            [value is not None for value in root_inputs],
            dtype=torch.bool,
        )
    if root_output_present:
        tensor_payloads[_ROOT_OUTPUT_SUM_Y2_KEY] = torch.tensor(
            [0.0 if value is None else value for value in root_outputs],
            dtype=torch.float32,
        )
        tensor_payloads[_ROOT_OUTPUT_VALID_KEY] = torch.tensor(
            [value is not None for value in root_outputs],
            dtype=torch.bool,
        )

    session_ids = [calibration.session_id for calibration in loaded]
    session_digest = hashlib.sha256(
        "\n".join(session_ids).encode("utf-8")
    ).hexdigest()[:16]
    metadata: dict[str, object] = {
        "format": _FORMAT,
        "version": _ACTIVATION_VERSION,
        "session_id": f"merged-{session_digest}",
        "session_name": "merged",
        "baseline_label": first.baseline_label,
        "activation_basis": first.activation_basis,
        "activation_axis": first.activation_axis,
        "layer_count": len(layers),
        "evaluation_count": len(evaluations),
        "evaluations": [_evaluation_document(evaluation) for evaluation in evaluations],
        "sample_rows_per_evaluation": max(
            (
                int(tensor.shape[1])
                for key, tensor in tensor_payloads.items()
                if key.endswith(".sample_x")
            ),
            default=0,
        ),
        "layers": layers,
        "merged_session_ids": session_ids,
    }
    if first.diffusion_model_class is not None:
        metadata["diffusion_model_class"] = first.diffusion_model_class

    metadata_file, tensors_file = _activation_pair_paths(output_path)
    if not overwrite and (metadata_file.exists() or tensors_file.exists()):
        raise FileExistsError(
            f"Merged activation calibration already exists: {metadata_file}"
        )
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensor_payloads, str(tensors_file))
    metadata_file.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata_file, tensors_file


def _activation_pair_paths(output_path: str | Path) -> tuple[Path, Path]:
    path = Path(output_path)
    if path.suffix.lower() == ".json":
        return path, path.with_suffix(".safetensors")
    if path.suffix.lower() == ".safetensors":
        return path.with_suffix(".json"), path
    return path.with_suffix(".json"), path.with_suffix(".safetensors")


def _evaluation_document(evaluation: EvaluationMetadata) -> dict[str, object]:
    document: dict[str, object] = {
        "evaluation_index": evaluation.index,
        "time_parameter_name": evaluation.time_parameter_name,
        "time_value": (
            list(evaluation.time_value)
            if isinstance(evaluation.time_value, tuple)
            else evaluation.time_value
        ),
        "time_value_truncated": evaluation.time_value_truncated,
        "timestep": evaluation.timestep,
        "sigma": evaluation.sigma,
        "root_input_sum_x2": evaluation.root_input_sum_x2,
        "root_output_sum_y2": evaluation.root_output_sum_y2,
    }
    return document


def validate_activation_calibration_against_source(
    calibration: ActivationCalibration | str | Path,
    source_path: str | Path,
) -> SourceModelHeader:
    """Validate every activation calibration layer against source Linear shapes."""
    loaded = (
        calibration
        if isinstance(calibration, ActivationCalibration)
        else load_activation_calibration(calibration)
    )
    if loaded.version != _ACTIVATION_VERSION:
        raise ValueError(
            "Source shape validation requires the current activation calibration."
        )

    source_header = read_source_model_header(source_path)
    for tensor_name in loaded.tensor_names():
        layer = loaded.get(tensor_name)
        assert isinstance(layer, LayerCalibration)
        descriptor = source_header.tensors.get(tensor_name)
        if descriptor is None:
            raise ValueError(
                "Activation calibration tensor is missing from the source "
                f"checkpoint: {tensor_name}"
            )
        source_shape = descriptor["shape"]
        if len(source_shape) != 2:
            raise ValueError(
                "Activation calibration source tensor must be rank 2: "
                f"{tensor_name} has shape {source_shape}"
            )
        expected_shape = [layer.output_features, layer.input_features]
        if source_shape != expected_shape:
            raise ValueError(
                "Activation calibration shape does not match source tensor "
                f"{tensor_name}: expected {expected_shape}, observed {source_shape}"
            )
    return source_header


@dataclass(frozen=True)
class _LayerDefinition:
    tensor_name: str
    input_features: int
    output_features: int
    sample_count: int
    invocation_count: int
    tensor_keys: dict[str, str]


def _load_activation_calibration(
    tensors_file: Path,
    document: dict[str, Any],
) -> ActivationCalibration:
    _validate_activation_metadata_header(document)
    evaluation_count = document["evaluation_count"]
    evaluations = _parse_evaluations(document["evaluations"])
    tensor_store = _CalibrationTensorStore(tensors_file)

    layer_definitions: list[_LayerDefinition] = []
    expected_keys: set[str] = set()
    claimed_keys: dict[str, str] = {}
    for tensor_name, layer_document in document["layers"].items():
        definition = _parse_activation_layer_definition(
            tensor_name,
            layer_document,
            evaluation_count,
        )
        layer_definitions.append(definition)
        for field_name, key in definition.tensor_keys.items():
            previous_owner = claimed_keys.get(key)
            owner = f"{tensor_name}.{field_name}"
            if previous_owner is not None:
                raise ValueError(
                    "Activation calibration tensor key is referenced more than "
                    f"once: {key} ({previous_owner}, {owner})"
                )
            claimed_keys[key] = owner
            expected_keys.add(key)

    root_keys = _validate_root_tensor_keys(tensor_store.keys)
    expected_keys.update(root_keys)
    missing_keys = sorted(expected_keys - tensor_store.keys)
    extra_keys = sorted(tensor_store.keys - expected_keys)
    if missing_keys or extra_keys:
        details: list[str] = []
        if missing_keys:
            details.append(f"missing={missing_keys}")
        if extra_keys:
            details.append(f"unexpected={extra_keys}")
        raise ValueError(
            "Activation calibration tensors do not match metadata: "
            + ", ".join(details)
        )

    root_inputs, root_outputs = _load_root_statistics(
        tensor_store,
        evaluation_count,
        root_keys,
    )
    evaluations = tuple(
        replace(
            evaluation,
            root_input_sum_x2=(
                root_inputs[evaluation.index]
                if _ROOT_INPUT_SUM_X2_KEY in root_keys
                else evaluation.root_input_sum_x2
            ),
            root_output_sum_y2=(
                root_outputs[evaluation.index]
                if _ROOT_OUTPUT_SUM_Y2_KEY in root_keys
                else evaluation.root_output_sum_y2
            ),
        )
        for evaluation in evaluations
    )

    layers: dict[str, LayerCalibration] = {}
    for definition in layer_definitions:
        _validate_activation_layer_tensors(
            definition,
            tensor_store,
            evaluation_count,
        )
        layers[definition.tensor_name] = LayerCalibration(
            tensor_name=definition.tensor_name,
            input_features=definition.input_features,
            output_features=definition.output_features,
            sample_count=definition.sample_count,
            invocation_count=definition.invocation_count,
            evaluation_count=evaluation_count,
            tensor_keys=definition.tensor_keys,
            tensor_store=tensor_store,
        )

    return ActivationCalibration(
        baseline_label=_optional_string(document, "baseline_label"),
        session_id=document["session_id"],
        session_name=_optional_string(document, "session_name"),
        diffusion_model_class=_optional_string(
            document,
            "diffusion_model_class",
        ),
        activation_basis=document["activation_basis"],
        activation_axis=document["activation_axis"],
        stats=layers,
        version=_ACTIVATION_VERSION,
        evaluations=evaluations,
    )


def _validate_activation_metadata_header(document: dict[str, Any]) -> None:
    if document.get("format") != _FORMAT:
        raise ValueError(
            "Unsupported activation calibration format: "
            f"{document.get('format')!r}"
        )
    if (
        type(document.get("version")) is not int
        or document["version"] != _ACTIVATION_VERSION
    ):
        raise ValueError(
            f"Unsupported activation calibration version: {document.get('version')!r}"
        )
    if document.get("activation_basis") != _ACTIVATION_BASIS:
        raise ValueError(
            "Unsupported activation calibration basis: "
            f"{document.get('activation_basis')!r}"
        )
    if document.get("activation_axis") != _ACTIVATION_AXIS:
        raise ValueError(
            "Unsupported activation calibration axis: "
            f"{document.get('activation_axis')!r}"
        )
    _require_non_empty_string(document.get("session_id"), "session_id")
    for key in (
        "session_name",
        "diffusion_model_class",
        "include_regex",
        "exclude_regex",
        "evaluation_basis",
    ):
        _optional_string(document, key)

    layers = document.get("layers")
    if not isinstance(layers, dict):
        raise ValueError("Activation calibration layers must be an object.")
    layer_count = _require_integer(
        document.get("layer_count"),
        "layer_count",
        minimum=0,
    )
    if layer_count != len(layers):
        raise ValueError("Activation calibration layer_count does not match layers.")

    evaluation_count = _require_integer(
        document.get("evaluation_count"),
        "evaluation_count",
        minimum=0,
    )
    evaluations = document.get("evaluations")
    if not isinstance(evaluations, list):
        raise ValueError("Activation calibration evaluations must be an array.")
    if len(evaluations) != evaluation_count:
        raise ValueError(
            "Activation calibration evaluation_count does not match evaluations."
        )
    sample_rows_per_evaluation = document.get("sample_rows_per_evaluation")
    if sample_rows_per_evaluation is not None:
        _require_integer(
            sample_rows_per_evaluation,
            "sample_rows_per_evaluation",
            minimum=0,
        )


def _parse_evaluations(
    raw_evaluations: list[Any],
) -> tuple[EvaluationMetadata, ...]:
    parsed: list[EvaluationMetadata] = []
    for expected_index, raw_evaluation in enumerate(raw_evaluations):
        if not isinstance(raw_evaluation, dict):
            raise ValueError(
                f"Activation calibration evaluation {expected_index} must be an object."
            )
        index = _require_integer(
            raw_evaluation.get("evaluation_index"),
            f"evaluations[{expected_index}].evaluation_index",
            minimum=0,
        )
        if index != expected_index:
            raise ValueError(
                "Activation calibration evaluation indices must be contiguous and "
                f"ordered; expected {expected_index}, observed {index}."
            )
        time_parameter_name = raw_evaluation.get("time_parameter_name")
        if time_parameter_name is not None:
            _require_non_empty_string(
                time_parameter_name,
                f"evaluations[{index}].time_parameter_name",
            )
        time_value = _parse_time_value(
            raw_evaluation.get("time_value"),
            f"evaluations[{index}].time_value",
        )
        time_value_truncated = raw_evaluation.get("time_value_truncated", False)
        if not isinstance(time_value_truncated, bool):
            raise ValueError(
                f"evaluations[{index}].time_value_truncated must be a boolean."
            )
        timestep = _optional_float(
            raw_evaluation.get("timestep"),
            f"evaluations[{index}].timestep",
        )
        sigma = _optional_float(
            raw_evaluation.get("sigma"),
            f"evaluations[{index}].sigma",
        )
        if time_parameter_name in {"timestep", "timesteps", "t"}:
            if isinstance(time_value, float):
                timestep = time_value
        elif time_parameter_name in {"sigma", "sigmas"}:
            if isinstance(time_value, float):
                sigma = time_value
        parsed.append(
            EvaluationMetadata(
                index=index,
                timestep=timestep,
                sigma=sigma,
                root_input_sum_x2=_optional_float(
                    raw_evaluation.get("root_input_sum_x2"),
                    f"evaluations[{index}].root_input_sum_x2",
                ),
                root_output_sum_y2=_optional_float(
                    raw_evaluation.get("root_output_sum_y2"),
                    f"evaluations[{index}].root_output_sum_y2",
                ),
                time_parameter_name=time_parameter_name,
                time_value=time_value,
                time_value_truncated=time_value_truncated,
            )
        )
    return tuple(parsed)


def _parse_time_value(value: Any, name: str) -> float | tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, Real) and not isinstance(value, bool):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Activation calibration {name} must be finite.")
        return value
    if isinstance(value, list):
        if len(value) > 16:
            raise ValueError(
                f"Activation calibration {name} contains too many values."
            )
        parsed = tuple(_require_finite_real(item, name) for item in value)
        return parsed
    raise ValueError(
        f"Activation calibration {name} must be null, numeric, or a short numeric array."
    )


def _parse_activation_layer_definition(
    tensor_name: Any,
    layer_document: Any,
    evaluation_count: int,
) -> _LayerDefinition:
    if not isinstance(tensor_name, str) or not tensor_name:
        raise ValueError(
            "Activation calibration tensor names must be non-empty strings."
        )
    if not isinstance(layer_document, dict):
        raise ValueError(f"Calibration entry for {tensor_name!r} must be an object.")
    input_features = _require_integer(
        layer_document.get("input_features"),
        f"{tensor_name}.input_features",
        minimum=1,
    )
    output_features = _require_integer(
        layer_document.get("output_features"),
        f"{tensor_name}.output_features",
        minimum=1,
    )
    sample_count = _require_integer(
        layer_document.get("sample_count"),
        f"{tensor_name}.sample_count",
        minimum=0,
    )
    invocation_count = _require_integer(
        layer_document.get("invocation_count"),
        f"{tensor_name}.invocation_count",
        minimum=0,
    )
    tensor_keys: dict[str, str] = {}
    for metadata_name, internal_name in (
        ("stats_key", "aggregate_sum_x2"),
        ("eval_sum_x_key", "eval_sum_x"),
        ("eval_sum_x2_key", "eval_sum_x2"),
        ("eval_max_abs_x_key", "eval_max_abs_x"),
        ("eval_sum_y_key", "eval_sum_y"),
        ("eval_sum_y2_key", "eval_sum_y2"),
        ("eval_sample_count_key", "eval_sample_count"),
        ("eval_invocation_count_key", "eval_invocation_count"),
    ):
        key = layer_document.get(metadata_name)
        if not isinstance(key, str) or not key:
            raise ValueError(
                f"{tensor_name}.{metadata_name} must be a non-empty string."
            )
        tensor_keys[internal_name] = key

    sample_x_key = layer_document.get("sample_x_key")
    sample_x_valid_key = layer_document.get("sample_x_valid_key")
    if (sample_x_key is None) != (sample_x_valid_key is None):
        raise ValueError(
            f"{tensor_name}.sample_x_key and sample_x_valid_key must be provided together."
        )
    if sample_x_key is not None:
        if not isinstance(sample_x_key, str) or not sample_x_key:
            raise ValueError(f"{tensor_name}.sample_x_key must be a non-empty string.")
        if not isinstance(sample_x_valid_key, str) or not sample_x_valid_key:
            raise ValueError(
                f"{tensor_name}.sample_x_valid_key must be a non-empty string."
            )
        tensor_keys["sample_x"] = sample_x_key
        tensor_keys["sample_x_valid"] = sample_x_valid_key

    return _LayerDefinition(
        tensor_name=tensor_name,
        input_features=input_features,
        output_features=output_features,
        sample_count=sample_count,
        invocation_count=invocation_count,
        tensor_keys=tensor_keys,
    )


def _validate_root_tensor_keys(available_keys: frozenset[str]) -> set[str]:
    expected: set[str] = set()
    for statistic_key, valid_key in (
        (_ROOT_INPUT_SUM_X2_KEY, _ROOT_INPUT_VALID_KEY),
        (_ROOT_OUTPUT_SUM_Y2_KEY, _ROOT_OUTPUT_VALID_KEY),
    ):
        statistic_present = statistic_key in available_keys
        valid_present = valid_key in available_keys
        if statistic_present != valid_present:
            raise ValueError(
                "Activation calibration root statistic and validity tensors must "
                f"be provided together: {statistic_key}, {valid_key}"
            )
        if statistic_present:
            expected.update((statistic_key, valid_key))
    return expected


def _load_root_statistics(
    tensor_store: _CalibrationTensorStore,
    evaluation_count: int,
    root_keys: set[str],
) -> tuple[list[float | None], list[float | None]]:
    inputs = [None] * evaluation_count
    outputs = [None] * evaluation_count
    for statistic_key, valid_key, destination in (
        (
            _ROOT_INPUT_SUM_X2_KEY,
            _ROOT_INPUT_VALID_KEY,
            inputs,
        ),
        (
            _ROOT_OUTPUT_SUM_Y2_KEY,
            _ROOT_OUTPUT_VALID_KEY,
            outputs,
        ),
    ):
        if statistic_key not in root_keys:
            continue
        values = _validate_float_tensor(
            tensor_store.get(statistic_key),
            statistic_key,
            shape=(evaluation_count,),
            nonnegative=True,
        )
        valid = tensor_store.get(valid_key)
        if valid.dtype is not torch.bool or tuple(valid.shape) != (evaluation_count,):
            raise ValueError(
                f"Activation calibration {valid_key} must be bool with shape "
                f"[{evaluation_count}]."
            )
        for index, is_valid in enumerate(valid.tolist()):
            if is_valid:
                destination[index] = float(values[index].item())
    return inputs, outputs


def _validate_activation_layer_tensors(
    definition: _LayerDefinition,
    tensor_store: _CalibrationTensorStore,
    evaluation_count: int,
) -> None:
    name = definition.tensor_name
    aggregate_sum_x2 = _validate_float_tensor(
        tensor_store.get(definition.tensor_keys["aggregate_sum_x2"]),
        f"{name}.sum_x2",
        shape=(definition.input_features,),
        nonnegative=True,
    )
    eval_input_shape = (evaluation_count, definition.input_features)
    eval_output_shape = (evaluation_count, definition.output_features)
    eval_sum_x = _validate_float_tensor(
        tensor_store.get(definition.tensor_keys["eval_sum_x"]),
        f"{name}.eval_sum_x",
        shape=eval_input_shape,
    )
    eval_sum_x2 = _validate_float_tensor(
        tensor_store.get(definition.tensor_keys["eval_sum_x2"]),
        f"{name}.eval_sum_x2",
        shape=eval_input_shape,
        nonnegative=True,
    )
    _validate_float_tensor(
        tensor_store.get(definition.tensor_keys["eval_max_abs_x"]),
        f"{name}.eval_max_abs_x",
        shape=eval_input_shape,
        nonnegative=True,
    )
    _validate_float_tensor(
        tensor_store.get(definition.tensor_keys["eval_sum_y"]),
        f"{name}.eval_sum_y",
        shape=eval_output_shape,
    )
    _validate_float_tensor(
        tensor_store.get(definition.tensor_keys["eval_sum_y2"]),
        f"{name}.eval_sum_y2",
        shape=eval_output_shape,
        nonnegative=True,
    )
    eval_sample_count = _validate_count_tensor(
        tensor_store.get(definition.tensor_keys["eval_sample_count"]),
        f"{name}.eval_sample_count",
        evaluation_count,
    )
    eval_invocation_count = _validate_count_tensor(
        tensor_store.get(definition.tensor_keys["eval_invocation_count"]),
        f"{name}.eval_invocation_count",
        evaluation_count,
    )
    if int(eval_sample_count.sum().item()) != definition.sample_count:
        raise ValueError(
            f"{name}.sample_count does not match eval_sample_count."
        )
    if int(eval_invocation_count.sum().item()) != definition.invocation_count:
        raise ValueError(
            f"{name}.invocation_count does not match eval_invocation_count."
        )
    if not torch.allclose(
        eval_sum_x2.sum(dim=0),
        aggregate_sum_x2,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise ValueError(
            f"{name}.sum_x2 does not match the sum of eval_sum_x2."
        )

    if "sample_x" not in definition.tensor_keys:
        return
    sample_x = tensor_store.get(definition.tensor_keys["sample_x"])
    if sample_x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            f"{name}.sample_x must use BF16, F16, or F32 data."
        )
    if sample_x.ndim != 3 or tuple(sample_x.shape[:1]) != (evaluation_count,):
        raise ValueError(
            f"{name}.sample_x must have shape [evaluation_count, rows, input_features]."
        )
    if sample_x.shape[2] != definition.input_features:
        raise ValueError(f"{name}.sample_x width does not match input_features.")
    if not bool(torch.isfinite(sample_x).all().item()):
        raise ValueError(f"{name}.sample_x must be finite.")
    sample_x_valid = _validate_count_tensor(
        tensor_store.get(definition.tensor_keys["sample_x_valid"]),
        f"{name}.sample_x_valid",
        evaluation_count,
    )
    if bool((sample_x_valid > sample_x.shape[1]).any().item()):
        raise ValueError(
            f"{name}.sample_x_valid cannot exceed the padded sample row count."
        )


def _validate_float_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    shape: tuple[int, ...],
    nonnegative: bool = False,
) -> torch.Tensor:
    if tensor.dtype != torch.float32:
        raise ValueError(f"Activation calibration {name} must be FP32.")
    if tuple(tensor.shape) != shape:
        raise ValueError(
            f"Activation calibration {name} has shape {tuple(tensor.shape)}, "
            f"expected {shape}."
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"Activation calibration {name} must be finite.")
    if nonnegative and bool((tensor < 0).any().item()):
        raise ValueError(f"Activation calibration {name} must be non-negative.")
    return tensor


def _validate_count_tensor(
    tensor: torch.Tensor,
    name: str,
    evaluation_count: int,
) -> torch.Tensor:
    if tensor.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError(f"Activation calibration {name} must be an integer tensor.")
    if tuple(tensor.shape) != (evaluation_count,):
        raise ValueError(
            f"Activation calibration {name} has shape {tuple(tensor.shape)}, "
            f"expected {(evaluation_count,)}."
        )
    values = tensor.to(dtype=torch.int64)
    if bool((values < 0).any().item()):
        raise ValueError(f"Activation calibration {name} must be non-negative.")
    return values


def _require_non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Activation calibration {name} must be a non-empty string.")
    return value


def _require_finite_real(value: Any, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"Activation calibration {name} must contain numeric values.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Activation calibration {name} must be finite.")
    return numeric


def _optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _require_finite_real(value, name)


def _optional_string(document: Mapping[str, Any], key: str) -> str | None:
    value = document.get(key)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"Activation calibration {key} must be a string.")
    return value


def _require_integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"Activation calibration {name} must be an integer >= {minimum}."
        )
    return value
