"""Reader for PotatoForge activation calibration artifacts."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import torch
from safetensors.torch import load_file


_FORMAT = "potatoforge_activation_calibration"
_VERSION = 1
_ACTIVATION_BASIS = "logical_linear_input"
_ACTIVATION_AXIS = "last_dimension"


class ActivationStats(NamedTuple):
    tensor_name: str
    input_features: int
    sample_count: int
    invocation_count: int
    sum_x2: torch.Tensor


class ActivationCalibration:
    def __init__(
        self,
        *,
        baseline_label: str | None,
        session_id: str | None,
        session_name: str | None,
        activation_basis: str,
        activation_axis: str,
        stats: Mapping[str, ActivationStats],
        diffusion_model_class: str | None = None,
    ) -> None:
        self.baseline_label = baseline_label
        self.session_id = session_id
        self.session_name = session_name
        self.diffusion_model_class = diffusion_model_class
        self.activation_basis = activation_basis
        self.activation_axis = activation_axis
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
        _validate_metadata_header(document)

        tensors = load_file(str(tensors_file), device="cpu")
        layers = document["layers"]
        stats = {
            tensor_name: _load_stats(
                tensor_name,
                layer_document,
                tensors,
            )
            for tensor_name, layer_document in layers.items()
        }

        return cls(
            baseline_label=_optional_string(document, "baseline_label"),
            session_id=_optional_string(document, "session_id"),
            session_name=_optional_string(document, "session_name"),
            diffusion_model_class=_optional_string(
                document,
                "diffusion_model_class",
            ),
            activation_basis=document["activation_basis"],
            activation_axis=document["activation_axis"],
            stats=stats,
        )

    def get(self, tensor_name: str) -> ActivationStats | None:
        return self._stats.get(tensor_name)

    def has(self, tensor_name: str) -> bool:
        return tensor_name in self._stats

    def tensor_names(self) -> tuple[str, ...]:
        return tuple(self._stats)


def _validate_metadata_header(document: dict[str, Any]) -> None:
    if document.get("format") != _FORMAT:
        raise ValueError(
            "Unsupported activation calibration format: "
            f"{document.get('format')!r}"
        )
    version = document.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version != _VERSION:
        raise ValueError(
            f"Unsupported activation calibration version: {version!r}"
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

    layers = document.get("layers")
    if not isinstance(layers, dict):
        raise ValueError("Activation calibration layers must be an object.")
    layer_count = document.get("layer_count")
    if layer_count is not None:
        _require_integer(layer_count, "layer_count", minimum=0)
        if layer_count != len(layers):
            raise ValueError(
                "Activation calibration layer_count does not match layers."
            )
    for key in ("baseline_label", "session_id", "session_name"):
        _optional_string(document, key)


def _load_stats(
    tensor_name: Any,
    layer_document: Any,
    tensors: Mapping[str, torch.Tensor],
) -> ActivationStats:
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
    stats_key = layer_document.get("stats_key")
    if not isinstance(stats_key, str) or not stats_key:
        raise ValueError(f"{tensor_name}.stats_key must be a non-empty string.")
    if stats_key not in tensors:
        raise ValueError(
            f"Calibration statistics tensor is missing: {stats_key}"
        )

    sum_x2 = tensors[stats_key]
    if sum_x2.ndim != 1:
        raise ValueError(
            f"Calibration statistics for {tensor_name} must be rank 1."
        )
    if sum_x2.shape[0] != input_features:
        raise ValueError(
            f"Calibration statistics length for {tensor_name} does not match "
            "input_features."
        )
    if sum_x2.dtype == torch.bool or sum_x2.is_complex():
        raise ValueError(
            f"Calibration statistics for {tensor_name} must be real numeric data."
        )

    sum_x2 = sum_x2.detach().to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(sum_x2).all().item()):
        raise ValueError(
            f"Calibration statistics for {tensor_name} must be finite."
        )
    if bool((sum_x2 < 0).any().item()):
        raise ValueError(
            f"Calibration statistics for {tensor_name} must be non-negative."
        )

    return ActivationStats(
        tensor_name=tensor_name,
        input_features=input_features,
        sample_count=sample_count,
        invocation_count=invocation_count,
        sum_x2=sum_x2,
    )


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
