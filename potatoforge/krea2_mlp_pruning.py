"""Cascade-aware calibration, selection, and low-memory Krea2/Kroma pruning."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, prod
from pathlib import Path
from typing import BinaryIO, Literal, cast

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .headers.header_reader import read_raw_data_start
from .headers.source_header import SourceModelHeader, read_source_model_header
from .planning import (
    OutputTensorSpec,
    QUANTIZATION_SOURCE_DTYPES,
    SafetensorsLayout,
    build_layout_from_specs,
    source_bytes,
)
from .quantization.hadamard import CONVROT_GROUP_SIZE
from .safetensors_writer import TensorPayload, write_safetensors_file
from .source_payloads import (
    read_source_tensor_bytes,
    tensor_from_raw_bytes,
)
from .patching import validate_file_ranges


_TARGET_PATTERN = re.compile(
    r"^blocks\.(?P<index>\d+)\.mlp\.(?P<component>gate|up|down)\.weight$"
)
_BIAS_PATTERN = re.compile(
    r"^blocks\.(?P<index>\d+)\.mlp\.(?:gate|up|down)\.bias$"
)
PruneComponent = Literal["gate", "up", "down"]
_COMPONENTS: tuple[PruneComponent, ...] = ("gate", "up", "down")
_DTYPE_BYTES: dict[str, int] = {"BF16": 2, "F16": 2, "F32": 4}
_PRUNING_METADATA_KEY = "potatoforge.pruning"
_PRUNING_CONFIG_METADATA_KEY = "potatoforge.pruning_config"
_PRUNING_LABEL = "krea2_mlp_contiguous_v1"
_PAYLOAD_CHUNK_BYTES = 16 * 1024 * 1024
_SOURCE_FORMAT_ERROR = (
    "Krea2 MLP pruning V1 expects unquantized floating-point main-block "
    "MLP weights. Prune before quantization."
)
KREA2_MLP_WEIGHT_STATS_FORMAT = "potatoforge_krea2_mlp_weight_stats"
KREA2_MLP_WEIGHT_STATS_VERSION = 1
KREA2_MLP_WEIGHT_STATS_TENSOR = "down_column_norm_sq"
KREA2_MLP_ACTIVATION_STATS_FORMAT = "potatoforge_krea2_mlp_activation_stats"
KREA2_MLP_ACTIVATION_STATS_VERSION = 1
KREA2_MLP_PLAN_FORMAT = "potatoforge_krea2_mlp_prune_plan"
KREA2_MLP_PLAN_VERSION = 1
KREA2_MLP_IMPORTANCE_TENSOR = "importance"
KREA2_MLP_SELECTION_METRIC = "mean_sq_activation_x_down_col_norm_sq"
KREA2_MLP_CASCADE_AUDIT_FORMAT = "potatoforge_krea2_mlp_cascade_audit"
KREA2_MLP_CASCADE_AUDIT_VERSION = 1
_DOWN_WEIGHT_PATTERN = re.compile(
    r"^blocks\.(?P<index>\d+)\.mlp\.down\.weight$"
)

PlanReporter = Callable[["Krea2MLPPrunePlan"], None]


@dataclass(frozen=True)
class Krea2MLPPrunePlan:
    block_indices: tuple[int, ...]
    hidden_width: int
    original_mlp_width: int
    start_channel: int
    remove_channels: int
    end_channel: int
    new_mlp_width: int
    estimated_saving_bytes: int
    convrot_group_size: int = CONVROT_GROUP_SIZE
    removed_channels_by_block: Mapping[int, tuple[int, ...]] | None = None
    plan_hash: str | None = None
    selection_metric: str | None = None
    stage: int | None = None


@dataclass(frozen=True)
class Krea2MLPPruneResult:
    output_path: Path
    plan: Krea2MLPPrunePlan
    target_tensors_modified: int
    other_tensors_copied: int
    output_bytes: int


@dataclass(frozen=True)
class Krea2MLPWeightStats:
    down_column_norm_sq: torch.Tensor
    block_indices: tuple[int, ...]
    hidden_width: int
    original_mlp_width: int
    source_model_basename: str
    source_header_fingerprint: str


@dataclass(frozen=True)
class Krea2MLPWeightStatsResult:
    output_path: Path
    stats: Krea2MLPWeightStats
    output_bytes: int


@dataclass(frozen=True)
class Krea2MLPActivationStats:
    sum_squared_activation: torch.Tensor
    observation_count: torch.Tensor
    block_indices: tuple[int, ...]
    hidden_width: int
    original_mlp_width: int
    base_plan_hash: str
    source_model_basename: str
    source_model_fingerprint: str | None = None
    metric: str = "sum_squared_activation"


@dataclass(frozen=True)
class Krea2MLPSelectionResult:
    output_path: Path
    plan: dict[str, object]
    importance_path: Path | None = None
    activation_artifacts: tuple[Path, ...] = ()
    activation_total_observations: float = 0.0


@dataclass(frozen=True)
class Krea2MLPCascadeAudit:
    candidate_plan_hash: str
    forward_count: int
    relative_sse: float
    relative_rmse: float
    mean_per_forward_relative_sse: float
    max_per_forward_relative_sse: float


def _target_key(block_index: int, component: PruneComponent) -> str:
    return f"blocks.{block_index}.mlp.{component}.weight"


def discover_krea2_main_mlp(
    header: SourceModelHeader,
) -> dict[int, dict[PruneComponent, str]]:
    """Find complete main-block gate/up/down weight families."""
    blocks: dict[int, dict[PruneComponent, str]] = {}

    for tensor_name in header.tensors:
        match = _TARGET_PATTERN.fullmatch(tensor_name)
        if match is None:
            continue
        block_index = int(match.group("index"))
        component = cast(PruneComponent, match.group("component"))
        blocks.setdefault(block_index, {})[component] = tensor_name

    if not blocks:
        raise ValueError(
            "No Krea2 main MLP tensors found. Expected blocks.N.mlp.gate/up/down.weight."
        )

    incomplete = {
        block_index: sorted(set(_COMPONENTS) - set(components))
        for block_index, components in blocks.items()
        if set(components) != set(_COMPONENTS)
    }
    if incomplete:
        details = ", ".join(
            f"block {block_index}: missing {', '.join(missing)}"
            for block_index, missing in sorted(incomplete.items())
        )
        raise ValueError(f"Incomplete Krea2 main MLP tensor set: {details}.")

    return blocks


def _discover_krea2_down_weights(
    header: SourceModelHeader,
) -> dict[int, str]:
    main_blocks: set[int] = set()
    down_weights: dict[int, str] = {}
    for tensor_name in header.tensors:
        target_match = _TARGET_PATTERN.fullmatch(tensor_name)
        if target_match is not None:
            main_blocks.add(int(target_match.group("index")))
        down_match = _DOWN_WEIGHT_PATTERN.fullmatch(tensor_name)
        if down_match is not None:
            down_weights[int(down_match.group("index"))] = tensor_name

    if not main_blocks:
        raise ValueError(
            "No Krea2 main MLP tensors found. Expected blocks.N.mlp.down.weight."
        )
    missing = sorted(main_blocks - set(down_weights))
    if missing:
        raise ValueError(
            "Krea2 main MLP is missing mlp.down.weight for blocks: "
            + ", ".join(str(index) for index in missing)
            + "."
        )
    return dict(sorted(down_weights.items()))


def source_header_fingerprint(header: SourceModelHeader) -> str:
    """Return a stable fingerprint for tensor names, shapes, dtypes, and offsets."""
    payload = json.dumps(
        {
            "tensors": {
                name: header.tensors[name]
                for name in sorted(header.tensors)
            }
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_down_weight_source(
    header: SourceModelHeader,
    down_weights: Mapping[int, str],
) -> tuple[int, int]:
    first_shape: tuple[int, int] | None = None
    for block_index, tensor_name in down_weights.items():
        descriptor = header.tensors[tensor_name]
        if descriptor["dtype"] not in QUANTIZATION_SOURCE_DTYPES:
            raise ValueError(_SOURCE_FORMAT_ERROR)
        layer_name = tensor_name.removesuffix(".weight")
        if any(
            f"{layer_name}{suffix}" in header.tensors
            for suffix in (".weight_scale", ".comfy_quant")
        ):
            raise ValueError(_SOURCE_FORMAT_ERROR)

        shape = tuple(descriptor["shape"])
        if len(shape) != 2 or any(dimension <= 0 for dimension in shape):
            raise ValueError(
                f"Krea2 block {block_index} down.weight must be a positive "
                "two-dimensional tensor."
            )
        if first_shape is None:
            first_shape = shape
        elif shape != first_shape:
            raise ValueError(
                "Krea2 main-block down.weight tensors must share one logical "
                f"shape; block {block_index} has {shape}, expected {first_shape}."
            )
        dtype_bytes = _DTYPE_BYTES[descriptor["dtype"]]
        if source_bytes(descriptor) != prod(shape) * dtype_bytes:
            raise ValueError(
                f"Payload size for {tensor_name} does not match its shape and dtype."
            )

    assert first_shape is not None
    return first_shape


def _down_column_norm_sq(
    source_file: BinaryIO,
    raw_data_start: int,
    descriptor: Mapping[str, object],
    tensor_name: str,
) -> torch.Tensor:
    shape = tuple(cast(Sequence[int], descriptor["shape"]))
    dtype = cast(str, descriptor["dtype"])
    hidden_width, original_mlp_width = shape
    row_bytes = original_mlp_width * _DTYPE_BYTES[dtype]
    rows_per_chunk = max(1, _PAYLOAD_CHUNK_BYTES // row_bytes)
    accumulator = torch.zeros(original_mlp_width, dtype=torch.float64)
    for row_start in range(0, hidden_width, rows_per_chunk):
        row_count = min(rows_per_chunk, hidden_width - row_start)
        byte_start = descriptor["data_offsets"][0] + row_start * row_bytes
        byte_end = byte_start + row_count * row_bytes
        raw_bytes = read_source_tensor_bytes(
            source_file,
            raw_data_start,
            (int(byte_start), int(byte_end)),
            row_count * row_bytes,
        )
        rows = tensor_from_raw_bytes(
            raw_bytes,
            (row_count, original_mlp_width),
            dtype,
            tensor_name=tensor_name,
        ).to(dtype=torch.float32)
        accumulator += (rows * rows).sum(dim=0, dtype=torch.float64)
        del rows, raw_bytes
    return accumulator.to(dtype=torch.float32)


def build_krea2_mlp_weight_stats(
    source_path: str | Path,
    output_path: str | Path,
) -> Krea2MLPWeightStatsResult:
    """Compute per-block down.weight column norms without loading the model."""
    source = Path(source_path)
    output = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {source}.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    header = read_source_model_header(source)
    validate_file_ranges(
        source,
        (
            (tensor_name, tuple(descriptor["data_offsets"]))
            for tensor_name, descriptor in header.tensors.items()
        ),
        "Source file",
    )
    down_weights = _discover_krea2_down_weights(header)
    hidden_width, original_mlp_width = _validate_down_weight_source(
        header,
        down_weights,
    )

    norms: list[torch.Tensor] = []
    with source.open("rb") as source_file:
        raw_data_start = read_raw_data_start(
            source_file,
            file_label="Source file",
        )
        for tensor_name in down_weights.values():
            norms.append(
                _down_column_norm_sq(
                    source_file,
                    raw_data_start,
                    header.tensors[tensor_name],
                    tensor_name,
                )
            )
    down_column_norm_sq = torch.stack(norms).contiguous()
    metadata = {
        "format": KREA2_MLP_WEIGHT_STATS_FORMAT,
        "version": str(KREA2_MLP_WEIGHT_STATS_VERSION),
        "architecture": "krea2",
        "block_count": str(len(down_weights)),
        "block_indices": json.dumps(
            list(down_weights),
            separators=(",", ":"),
        ),
        "main_hidden_width": str(hidden_width),
        "original_mlp_width": str(original_mlp_width),
        "metric": "down_column_norm_sq",
        "storage_dtype": "float32",
        "source_model_basename": source.name,
        "source_header_fingerprint": source_header_fingerprint(header),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {KREA2_MLP_WEIGHT_STATS_TENSOR: down_column_norm_sq},
        str(output),
        metadata=metadata,
    )
    stats = Krea2MLPWeightStats(
        down_column_norm_sq=down_column_norm_sq,
        block_indices=tuple(down_weights),
        hidden_width=hidden_width,
        original_mlp_width=original_mlp_width,
        source_model_basename=source.name,
        source_header_fingerprint=metadata["source_header_fingerprint"],
    )
    return Krea2MLPWeightStatsResult(
        output_path=output,
        stats=stats,
        output_bytes=output.stat().st_size,
    )


def _stats_metadata_int(metadata: Mapping[str, str], key: str) -> int:
    raw = metadata.get(key)
    try:
        value = int(raw) if raw is not None else None
    except (TypeError, ValueError) as error:
        raise ValueError(f"Weight stats metadata {key} must be an integer.") from error
    if value is None or value <= 0:
        raise ValueError(f"Weight stats metadata {key} must be positive.")
    return value


def load_krea2_mlp_weight_stats(
    path: str | Path,
) -> Krea2MLPWeightStats:
    """Load and validate a static Krea2/Kroma weight-stat artifact."""
    stats_path = Path(path)
    try:
        with safe_open(str(stats_path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if handle.keys() != [KREA2_MLP_WEIGHT_STATS_TENSOR]:
                raise ValueError(
                    "Weight stats must contain only down_column_norm_sq."
                )
            down_column_norm_sq = handle.get_tensor(KREA2_MLP_WEIGHT_STATS_TENSOR)
    except FileNotFoundError:
        raise FileNotFoundError(f"Weight stats artifact not found: {stats_path}.")

    if metadata.get("format") != KREA2_MLP_WEIGHT_STATS_FORMAT:
        raise ValueError(
            f"Unsupported Krea2 MLP weight stats format: {metadata.get('format')!r}"
        )
    if metadata.get("version") != str(KREA2_MLP_WEIGHT_STATS_VERSION):
        raise ValueError(
            f"Unsupported Krea2 MLP weight stats version: {metadata.get('version')!r}"
        )
    if metadata.get("architecture") != "krea2":
        raise ValueError("Krea2 MLP weight stats architecture must be krea2.")
    if metadata.get("metric") != "down_column_norm_sq":
        raise ValueError("Krea2 MLP weight stats metric is unsupported.")
    if metadata.get("storage_dtype") != "float32":
        raise ValueError("Krea2 MLP weight stats must use float32 storage.")
    if down_column_norm_sq.dtype != torch.float32 or down_column_norm_sq.ndim != 2:
        raise ValueError("down_column_norm_sq must be a rank-2 float32 tensor.")

    block_count = _stats_metadata_int(metadata, "block_count")
    hidden_width = _stats_metadata_int(metadata, "main_hidden_width")
    original_mlp_width = _stats_metadata_int(metadata, "original_mlp_width")
    if tuple(down_column_norm_sq.shape) != (block_count, original_mlp_width):
        raise ValueError(
            "down_column_norm_sq shape does not match weight stats metadata."
        )
    if not torch.isfinite(down_column_norm_sq).all():
        raise ValueError("down_column_norm_sq contains non-finite values.")
    try:
        block_indices = tuple(json.loads(metadata["block_indices"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("Weight stats block_indices metadata is invalid.") from error
    if block_indices != tuple(sorted(block_indices)) or len(block_indices) != block_count:
        raise ValueError("Weight stats block_indices metadata is invalid.")
    if any(type(index) is not int or index < 0 for index in block_indices):
        raise ValueError("Weight stats block_indices metadata is invalid.")
    source_model_basename = metadata.get("source_model_basename")
    source_header_fingerprint_value = metadata.get("source_header_fingerprint")
    if not source_model_basename or not source_header_fingerprint_value:
        raise ValueError("Weight stats source metadata is incomplete.")
    return Krea2MLPWeightStats(
        down_column_norm_sq=down_column_norm_sq,
        block_indices=cast(tuple[int, ...], block_indices),
        hidden_width=hidden_width,
        original_mlp_width=original_mlp_width,
        source_model_basename=source_model_basename,
        source_header_fingerprint=source_header_fingerprint_value,
    )


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Artifact metadata not found: {path}.")
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid artifact metadata: {path}") from error
    if not isinstance(document, dict):
        raise ValueError(f"Artifact metadata must be an object: {path}")
    return cast(dict[str, object], document)


def _artifact_metadata_and_tensors_path(
    path: str | Path,
) -> tuple[dict[str, object], Path]:
    artifact = Path(path)
    if artifact.suffix.lower() == ".json":
        return _load_json_object(artifact), artifact.with_suffix(".safetensors")

    tensors_path = artifact
    try:
        with safe_open(str(tensors_path), framework="pt", device="cpu") as handle:
            raw_metadata = handle.metadata() or {}
    except FileNotFoundError:
        raise FileNotFoundError(f"Activation stats artifact not found: {artifact}.")
    if raw_metadata:
        return cast(dict[str, object], raw_metadata), tensors_path
    sibling = artifact.with_suffix(".json")
    return _load_json_object(sibling), tensors_path


def _metadata_positive_int(
    metadata: Mapping[str, object],
    key: str,
) -> int:
    raw = metadata.get(key)
    if isinstance(raw, str):
        try:
            raw = int(raw)
        except ValueError as error:
            raise ValueError(f"Activation stats metadata {key} must be an integer.") from error
    if type(raw) is not int or raw <= 0:
        raise ValueError(f"Activation stats metadata {key} must be positive.")
    return raw


def _normalise_base_plan_hash(raw: object) -> str:
    if raw is None:
        return "none"
    if not isinstance(raw, str):
        raise ValueError("Activation stats base_plan_hash metadata is invalid.")
    value = raw.strip()
    if value.lower() in {"", "none", "null"}:
        return "none"
    return value


def load_krea2_mlp_activation_stats(
    path: str | Path,
) -> Krea2MLPActivationStats:
    """Load the fixed-size aggregate activation artifact for one stage."""
    metadata, tensors_path = _artifact_metadata_and_tensors_path(path)
    if metadata.get("format") != KREA2_MLP_ACTIVATION_STATS_FORMAT:
        raise ValueError(
            "Unsupported Krea2 MLP activation stats format: "
            f"{metadata.get('format')!r}"
        )
    version = metadata.get("version", metadata.get("format_version"))
    if isinstance(version, str):
        try:
            version = int(version)
        except ValueError as error:
            raise ValueError("Activation stats version must be an integer.") from error
    if version != KREA2_MLP_ACTIVATION_STATS_VERSION:
        raise ValueError(f"Unsupported Krea2 MLP activation stats version: {version!r}")
    if metadata.get("architecture") != "krea2":
        raise ValueError("Krea2 MLP activation stats architecture must be krea2.")

    tensors = load_file(str(tensors_path), device="cpu")
    expected_keys = {"sum_squared_activation", "observation_count"}
    if set(tensors) != expected_keys:
        raise ValueError(
            "Activation stats must contain only sum_squared_activation and "
            "observation_count."
        )
    sum_squared_activation = tensors["sum_squared_activation"]
    observation_count = tensors["observation_count"]
    if not sum_squared_activation.is_floating_point() or sum_squared_activation.ndim != 2:
        raise ValueError(
            "sum_squared_activation must be a rank-2 floating-point tensor."
        )
    integer_dtypes = (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    )
    if observation_count.ndim != 1 or (
        not observation_count.is_floating_point()
        and observation_count.dtype not in integer_dtypes
    ):
        raise ValueError("observation_count must be a rank-1 numeric tensor.")

    block_count = _metadata_positive_int(metadata, "block_count")
    hidden_width = _metadata_positive_int(metadata, "main_hidden_width")
    original_mlp_width = _metadata_positive_int(metadata, "original_mlp_width")
    if tuple(sum_squared_activation.shape) != (block_count, original_mlp_width):
        raise ValueError(
            "sum_squared_activation shape does not match activation stats metadata."
        )
    if tuple(observation_count.shape) != (block_count,):
        raise ValueError(
            "observation_count shape does not match activation stats metadata."
        )
    sum_squared_activation = sum_squared_activation.to(dtype=torch.float32)
    observation_count = observation_count.to(dtype=torch.float64)
    if not torch.isfinite(sum_squared_activation).all():
        raise ValueError("sum_squared_activation contains non-finite values.")
    if (sum_squared_activation < 0).any():
        raise ValueError("sum_squared_activation must contain non-negative values.")
    if not torch.isfinite(observation_count).all() or (observation_count <= 0).any():
        raise ValueError("observation_count must contain positive finite values.")

    raw_indices = metadata.get("block_indices")
    if raw_indices is None:
        block_indices = tuple(range(block_count))
    else:
        if isinstance(raw_indices, str):
            try:
                raw_indices = json.loads(raw_indices)
            except json.JSONDecodeError as error:
                raise ValueError("Activation stats block_indices is invalid.") from error
        if not isinstance(raw_indices, list):
            raise ValueError("Activation stats block_indices is invalid.")
        block_indices = tuple(raw_indices)
        if any(type(index) is not int or index < 0 for index in block_indices):
            raise ValueError("Activation stats block_indices is invalid.")
        if (
            block_indices != tuple(sorted(block_indices))
            or len(set(block_indices)) != len(block_indices)
            or len(block_indices) != block_count
        ):
            raise ValueError("Activation stats block_indices is invalid.")

    if "base_plan_hash" in metadata:
        raw_base_plan_hash = metadata["base_plan_hash"]
    elif "base_plan_sha256" in metadata:
        raw_base_plan_hash = metadata["base_plan_sha256"]
    else:
        raise ValueError("Activation stats base_plan_hash metadata is required.")
    base_plan_hash = _normalise_base_plan_hash(raw_base_plan_hash)
    source_model_basename = metadata.get("source_model_basename")
    if source_model_basename is None:
        source_model_basename = metadata.get("source_model")
    if source_model_basename is None:
        source_model_basename = metadata.get("source_model_path")
    if not isinstance(source_model_basename, str) or not source_model_basename:
        raise ValueError("Activation stats source model metadata is required.")
    metric = metadata.get("metric", "sum_squared_activation")
    if metric != "sum_squared_activation":
        raise ValueError("Activation stats metric is unsupported.")
    source_model_fingerprint: str | None = None
    for key in (
        "source_model_fingerprint",
        "source_header_fingerprint",
        "model_fingerprint",
        "fingerprint",
    ):
        if key in metadata:
            raw_fingerprint = metadata[key]
            if raw_fingerprint is not None:
                if not isinstance(raw_fingerprint, str) or not raw_fingerprint:
                    raise ValueError("Activation stats source fingerprint is invalid.")
                source_model_fingerprint = raw_fingerprint
            break
    return Krea2MLPActivationStats(
        sum_squared_activation=sum_squared_activation,
        observation_count=observation_count,
        block_indices=cast(tuple[int, ...], block_indices),
        hidden_width=hidden_width,
        original_mlp_width=original_mlp_width,
        base_plan_hash=base_plan_hash,
        source_model_basename=source_model_basename,
        source_model_fingerprint=source_model_fingerprint,
        metric=cast(str, metric),
    )


def _embedded_krea2_activation_metadata(path: Path) -> dict[str, object] | None:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
    except Exception:
        return None
    if metadata.get("format") != KREA2_MLP_ACTIVATION_STATS_FORMAT:
        return None
    return cast(dict[str, object], metadata)


def discover_krea2_mlp_activation_stats(
    path: str | Path,
) -> tuple[Path, ...]:
    """Discover direct same-stem activation artifact pairs in a directory."""
    root = Path(path)
    if not root.is_dir():
        return (root,)

    grouped: dict[str, dict[str, list[Path]]] = {}
    for child in root.iterdir():
        if not child.is_file() or child.suffix.lower() not in {".json", ".safetensors"}:
            continue
        grouped.setdefault(child.stem.casefold(), {}).setdefault(
            child.suffix.lower(), []
        ).append(child)

    artifacts: list[Path] = []
    for stem in sorted(grouped):
        files = grouped[stem]
        json_paths = sorted(files.get(".json", []), key=lambda item: item.name.casefold())
        tensor_paths = sorted(
            files.get(".safetensors", []), key=lambda item: item.name.casefold()
        )
        if len(json_paths) > 1 or len(tensor_paths) > 1:
            raise ValueError(f"Ambiguous activation calibration artifact stem: {stem}")
        if json_paths:
            metadata_path = json_paths[0]
            try:
                metadata = _load_json_object(metadata_path)
            except ValueError:
                if tensor_paths:
                    raise
                continue
            if metadata.get("format") == KREA2_MLP_ACTIVATION_STATS_FORMAT:
                if not tensor_paths:
                    raise FileNotFoundError(
                        "Activation calibration metadata is missing its matching "
                        f"Safetensors file: {metadata_path}"
                    )
                artifacts.append(metadata_path)
            elif tensor_paths and _embedded_krea2_activation_metadata(tensor_paths[0]):
                raise ValueError(
                    "Activation calibration metadata format does not match its "
                    f"Safetensors pair: {metadata_path}"
                )
        elif tensor_paths and _embedded_krea2_activation_metadata(tensor_paths[0]):
            artifacts.append(tensor_paths[0])

    if not artifacts:
        raise ValueError(f"No Krea2 activation calibration artifacts found in: {root}")
    return tuple(sorted(artifacts, key=lambda item: (item.name.casefold(), item.name)))


def _activation_artifact_paths(
    path_or_paths: str | Path | Sequence[str | Path],
) -> tuple[Path, ...]:
    if isinstance(path_or_paths, (str, Path)):
        return discover_krea2_mlp_activation_stats(path_or_paths)
    paths = tuple(Path(path) for path in path_or_paths)
    if not paths:
        raise ValueError("At least one activation calibration artifact is required.")
    ordered = tuple(sorted(paths, key=lambda item: (item.name.casefold(), item.name)))
    if len({str(path.resolve()).casefold() for path in ordered}) != len(ordered):
        raise ValueError("Duplicate activation calibration artifact paths are ambiguous.")
    return ordered


def _load_activation_for_aggregation(path: Path) -> Krea2MLPActivationStats:
    try:
        return load_krea2_mlp_activation_stats(path)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Invalid activation calibration artifact {path}: {error}"
        ) from error
    except ValueError as error:
        raise ValueError(f"Invalid activation calibration artifact {path}: {error}") from error


def _raise_activation_metadata_mismatch(
    path: Path,
    field: str,
    expected: object,
    actual: object,
) -> None:
    raise ValueError(
        "Activation calibration metadata mismatch for "
        f"{field}: expected {expected!r}, file {path} has {actual!r}."
    )


def aggregate_krea2_mlp_activation_stats(
    path_or_paths: str | Path | Sequence[str | Path],
) -> Krea2MLPActivationStats:
    """Aggregate raw activation totals from one file or a calibration directory."""
    artifacts = _activation_artifact_paths(path_or_paths)
    first_path = artifacts[0]
    first = _load_activation_for_aggregation(first_path)
    sum_squared_activation = first.sum_squared_activation.to(dtype=torch.float64)
    observation_count = first.observation_count.to(dtype=torch.float64)

    for artifact_path in artifacts[1:]:
        current = _load_activation_for_aggregation(artifact_path)
        for field in (
            "block_indices",
            "hidden_width",
            "original_mlp_width",
            "base_plan_hash",
            "metric",
            "source_model_basename",
            "source_model_fingerprint",
        ):
            expected = getattr(first, field)
            actual = getattr(current, field)
            if actual != expected:
                _raise_activation_metadata_mismatch(
                    artifact_path, field, expected, actual
                )
        sum_squared_activation.add_(current.sum_squared_activation.to(torch.float64))
        observation_count.add_(current.observation_count.to(torch.float64))

    if not torch.isfinite(sum_squared_activation).all():
        raise ValueError("Aggregated sum_squared_activation contains non-finite values.")
    if not torch.isfinite(observation_count).all() or (observation_count <= 0).any():
        raise ValueError(
            "Aggregated observation_count must contain positive finite values."
        )
    return Krea2MLPActivationStats(
        sum_squared_activation=sum_squared_activation.to(torch.float32),
        observation_count=observation_count,
        block_indices=first.block_indices,
        hidden_width=first.hidden_width,
        original_mlp_width=first.original_mlp_width,
        base_plan_hash=first.base_plan_hash,
        source_model_basename=first.source_model_basename,
        source_model_fingerprint=first.source_model_fingerprint,
        metric=first.metric,
    )


def canonical_krea2_mlp_plan_bytes(plan: Mapping[str, object]) -> bytes:
    """Serialize a validated plan without semantically irrelevant whitespace."""
    return json.dumps(
        plan,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def validate_krea2_mlp_plan(
    document: Mapping[str, object],
    *,
    expected_block_indices: Sequence[int] | None = None,
    require_convrot_alignment: bool = False,
) -> dict[str, object]:
    """Validate a cumulative plan shared by selection and physical pruning."""
    if document.get("format") != KREA2_MLP_PLAN_FORMAT:
        raise ValueError(f"Unsupported Krea2 MLP plan format: {document.get('format')!r}")
    if document.get("version") != KREA2_MLP_PLAN_VERSION:
        raise ValueError(f"Unsupported Krea2 MLP plan version: {document.get('version')!r}")
    if document.get("architecture") != "krea2":
        raise ValueError("Krea2 MLP plan architecture must be krea2.")

    def positive_int(key: str) -> int:
        value = document.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"Krea2 MLP plan {key} must be positive.")
        return value

    original_mlp_width = positive_int("original_mlp_width")
    main_hidden_width = positive_int("main_hidden_width")
    block_count = positive_int("block_count")
    removed_per_block_value = document.get("removed_per_block")
    if type(removed_per_block_value) is not int or removed_per_block_value < 0:
        raise ValueError("Krea2 MLP plan removed_per_block must be non-negative.")
    removed_per_block = removed_per_block_value
    new_mlp_width = document.get("new_mlp_width")
    if type(new_mlp_width) is not int or new_mlp_width <= 0:
        raise ValueError("Krea2 MLP plan new_mlp_width must be positive.")
    if new_mlp_width != original_mlp_width - removed_per_block:
        raise ValueError("Krea2 MLP plan new_mlp_width is inconsistent.")
    if require_convrot_alignment and new_mlp_width % CONVROT_GROUP_SIZE != 0:
        raise ValueError(
            "The resulting MLP width must be divisible by "
            f"the ConvRot group size ({CONVROT_GROUP_SIZE})."
        )

    raw_blocks = document.get("blocks")
    if not isinstance(raw_blocks, Mapping):
        raise ValueError("Krea2 MLP plan blocks must be an object.")
    if len(raw_blocks) != block_count:
        raise ValueError("Krea2 MLP plan block count does not match blocks.")
    normalised_indices: list[int] = []
    for raw_index, raw_channels in raw_blocks.items():
        if not isinstance(raw_index, str):
            raise ValueError("Krea2 MLP plan block keys must be strings.")
        try:
            block_index = int(raw_index)
        except ValueError as error:
            raise ValueError("Krea2 MLP plan block key is invalid.") from error
        if str(block_index) != raw_index or block_index < 0:
            raise ValueError("Krea2 MLP plan block key is invalid.")
        if not isinstance(raw_channels, list):
            raise ValueError(f"Krea2 MLP plan block {block_index} must be an array.")
        channels = raw_channels
        if any(type(channel) is not int for channel in channels):
            raise ValueError(f"Krea2 MLP plan block {block_index} has an invalid ID.")
        if channels != sorted(channels) or len(set(channels)) != len(channels):
            raise ValueError(
                f"Krea2 MLP plan block {block_index} IDs must be unique and sorted."
            )
        if any(channel < 0 or channel >= original_mlp_width for channel in channels):
            raise ValueError(f"Krea2 MLP plan block {block_index} has an out-of-range ID.")
        if len(channels) != removed_per_block:
            raise ValueError(
                f"Krea2 MLP plan block {block_index} has the wrong removal count."
            )
        normalised_indices.append(block_index)

    sorted_indices = sorted(normalised_indices)
    expected = document.get("block_indices")
    if expected is not None:
        if not isinstance(expected, list) or expected != sorted_indices:
            raise ValueError("Krea2 MLP plan block_indices does not match blocks.")
    if expected_block_indices is not None and tuple(sorted_indices) != tuple(expected_block_indices):
        raise ValueError("Krea2 MLP plan block set does not match the source model.")
    stage = document.get("stage")
    if stage is not None and (type(stage) is not int or stage <= 0):
        raise ValueError("Krea2 MLP plan stage must be positive.")
    return dict(document)


def load_krea2_mlp_plan(path: str | Path) -> dict[str, object]:
    return validate_krea2_mlp_plan(_load_json_object(Path(path)))


def krea2_mlp_plan_sha256(plan: Mapping[str, object] | str | Path) -> str:
    document = (
        load_krea2_mlp_plan(plan)
        if isinstance(plan, (str, Path))
        else validate_krea2_mlp_plan(plan)
    )
    return hashlib.sha256(canonical_krea2_mlp_plan_bytes(document)).hexdigest()


def remap_krea2_mlp_plan_blocks(
    document: Mapping[str, object],
    old_to_new: Mapping[int, int | None],
) -> dict[str, object]:
    """Move MLP selections to the surviving contiguous block IDs."""
    plan = validate_krea2_mlp_plan(document)
    original_indices = tuple(sorted(int(index) for index in cast(Mapping[str, object], plan["blocks"])))
    if original_indices != tuple(old_to_new):
        raise ValueError("Krea2 MLP plan block set does not match the block-prune plan.")
    blocks = cast(Mapping[str, object], plan["blocks"])
    remapped_blocks = {
        str(new_index): blocks[str(old_index)]
        for old_index in sorted(old_to_new)
        if (new_index := old_to_new[old_index]) is not None
    }
    remapped = {
        **plan,
        "block_count": len(remapped_blocks),
        "block_indices": sorted(int(index) for index in remapped_blocks),
        "blocks": remapped_blocks,
        "source_mlp_plan_hash": krea2_mlp_plan_sha256(plan),
    }
    return validate_krea2_mlp_plan(remapped)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(_PAYLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _krea2_mlp_activation_stats_sha256(stats: Krea2MLPActivationStats) -> str:
    digest = hashlib.sha256()
    metadata = json.dumps(
        {
            "format": KREA2_MLP_ACTIVATION_STATS_FORMAT,
            "version": KREA2_MLP_ACTIVATION_STATS_VERSION,
            "architecture": "krea2",
            "block_indices": stats.block_indices,
            "hidden_width": stats.hidden_width,
            "original_mlp_width": stats.original_mlp_width,
            "base_plan_hash": stats.base_plan_hash,
            "source_model_basename": stats.source_model_basename,
            "source_model_fingerprint": stats.source_model_fingerprint,
            "metric": stats.metric,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(metadata)
    for name, tensor in (
        ("sum_squared_activation", stats.sum_squared_activation.to(torch.float32)),
        ("observation_count", stats.observation_count.to(torch.float64)),
    ):
        tensor = tensor.detach().cpu().contiguous()
        descriptor = json.dumps(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def select_krea2_mlp_plan(
    activation_stats_path: str | Path,
    weight_stats_path: str | Path,
    output_path: str | Path,
    *,
    base_plan_path: str | Path | None = None,
    add_channels_per_block: int = 256,
    importance_output_path: str | Path | None = None,
) -> Krea2MLPSelectionResult:
    """Select a cumulative per-block plan from one calibration stage."""
    if add_channels_per_block <= 0:
        raise ValueError("add_channels_per_block must be greater than zero.")
    activation_input = Path(activation_stats_path)
    activation_artifacts = (
        discover_krea2_mlp_activation_stats(activation_input)
        if activation_input.is_dir()
        else (activation_input,)
    )
    activation = (
        aggregate_krea2_mlp_activation_stats(activation_artifacts)
        if activation_input.is_dir()
        else load_krea2_mlp_activation_stats(activation_input)
    )
    weights = load_krea2_mlp_weight_stats(weight_stats_path)
    if activation.block_indices != weights.block_indices:
        raise ValueError("Activation and weight stats block sets do not match.")
    if activation.original_mlp_width != weights.original_mlp_width:
        raise ValueError("Activation and weight stats MLP widths do not match.")
    if activation.hidden_width != weights.hidden_width:
        raise ValueError("Activation and weight stats hidden widths do not match.")

    base_plan: dict[str, object] | None = None
    parent_hash = "none"
    if base_plan_path is not None:
        base_plan = load_krea2_mlp_plan(base_plan_path)
        validate_krea2_mlp_plan(
            base_plan,
            expected_block_indices=activation.block_indices,
        )
        if (
            base_plan["original_mlp_width"] != activation.original_mlp_width
            or base_plan["main_hidden_width"] != activation.hidden_width
        ):
            raise ValueError("Base plan dimensions do not match calibration stats.")
        parent_hash = krea2_mlp_plan_sha256(base_plan)
    expected_base_hash = activation.base_plan_hash
    if expected_base_hash != parent_hash:
        raise ValueError(
            "Activation stats base_plan_hash does not match the provided base plan."
        )

    mean_sq = activation.sum_squared_activation / activation.observation_count[:, None]
    importance = mean_sq * weights.down_column_norm_sq
    if not torch.isfinite(importance).all():
        raise ValueError("Krea2 MLP importance contains non-finite values.")

    base_blocks = (
        cast(Mapping[str, object], base_plan["blocks"])
        if base_plan is not None
        else {}
    )
    block_documents: dict[str, list[int]] = {}
    for row_index, block_index in enumerate(activation.block_indices):
        previous = base_blocks.get(str(block_index), [])
        if not isinstance(previous, list):
            raise ValueError(f"Base plan block {block_index} is invalid.")
        removed = {channel for channel in previous if type(channel) is int}
        if len(removed) != len(previous):
            raise ValueError(f"Base plan block {block_index} is invalid.")
        available = [
            channel
            for channel in range(activation.original_mlp_width)
            if channel not in removed
        ]
        if add_channels_per_block > len(available):
            raise ValueError(
                f"Cannot add {add_channels_per_block} channels to block {block_index}; "
                f"only {len(available)} remain."
            )
        available.sort(key=lambda channel: (float(importance[row_index, channel]), channel))
        block_documents[str(block_index)] = sorted(
            (*removed, *available[:add_channels_per_block])
        )

    removed_per_block = len(next(iter(block_documents.values())))
    stage = 1 if base_plan is None else int(base_plan.get("stage", 1)) + 1
    plan: dict[str, object] = {
        "format": KREA2_MLP_PLAN_FORMAT,
        "version": KREA2_MLP_PLAN_VERSION,
        "architecture": "krea2",
        "original_mlp_width": activation.original_mlp_width,
        "main_hidden_width": activation.hidden_width,
        "block_count": len(activation.block_indices),
        "block_indices": list(activation.block_indices),
        "removed_per_block": removed_per_block,
        "new_mlp_width": activation.original_mlp_width - removed_per_block,
        "stage": stage,
        "selection": {
            "metric": KREA2_MLP_SELECTION_METRIC,
            "metric_version": 1,
            "added_per_block_this_stage": add_channels_per_block,
            "parent_plan_sha256": parent_hash,
        },
        "calibration": {
            "activation_stats_sha256": _krea2_mlp_activation_stats_sha256(activation),
            "weight_stats_sha256": _sha256_file(weight_stats_path),
        },
        "blocks": block_documents,
    }
    validate_krea2_mlp_plan(plan, expected_block_indices=activation.block_indices)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    importance_path = (
        None if importance_output_path is None else Path(importance_output_path)
    )
    if importance_path is not None and importance_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {importance_path}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if importance_path is not None:
        importance_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {KREA2_MLP_IMPORTANCE_TENSOR: importance.to(dtype=torch.float32).contiguous()},
            str(importance_path),
            metadata={
                "format": "potatoforge_krea2_mlp_importance",
                "version": "1",
                "architecture": "krea2",
                "block_count": str(len(activation.block_indices)),
                "original_mlp_width": str(activation.original_mlp_width),
                "metric": KREA2_MLP_SELECTION_METRIC,
                "plan_stage": str(stage),
            },
        )
    output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return Krea2MLPSelectionResult(
        output_path=output,
        plan=plan,
        importance_path=importance_path,
        activation_artifacts=activation_artifacts,
        activation_total_observations=float(activation.observation_count.sum().item()),
    )


def load_krea2_mlp_cascade_audit(
    path: str | Path,
) -> Krea2MLPCascadeAudit:
    """Read a tiny final-output cascade audit without applying thresholds."""
    document = _load_json_object(Path(path))
    summary = document.get("summary")
    if isinstance(summary, Mapping):
        document = {**document, **summary}
    if document.get("format") not in (None, KREA2_MLP_CASCADE_AUDIT_FORMAT):
        raise ValueError(f"Unsupported Krea2 MLP cascade audit format: {document.get('format')!r}")
    version = document.get("version", document.get("format_version"))
    if version is not None and version != KREA2_MLP_CASCADE_AUDIT_VERSION:
        raise ValueError(f"Unsupported Krea2 MLP cascade audit version: {version!r}")
    candidate_plan_hash = document.get("candidate_plan_hash")
    if not isinstance(candidate_plan_hash, str) or not candidate_plan_hash:
        raise ValueError("Cascade audit candidate_plan_hash is required.")
    forward_count = document.get("forward_count")
    if type(forward_count) is not int or forward_count <= 0:
        raise ValueError("Cascade audit forward_count must be positive.")

    def finite_number(key: str) -> float:
        value = document.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Cascade audit {key} must be numeric.")
        result = float(value)
        if not isfinite(result):
            raise ValueError(f"Cascade audit {key} must be finite.")
        return result

    return Krea2MLPCascadeAudit(
        candidate_plan_hash=candidate_plan_hash,
        forward_count=forward_count,
        relative_sse=finite_number("relative_sse"),
        relative_rmse=finite_number("relative_rmse"),
        mean_per_forward_relative_sse=finite_number(
            "mean_per_forward_relative_sse"
        ),
        max_per_forward_relative_sse=finite_number(
            "max_per_forward_relative_sse"
        ),
    )


def render_krea2_mlp_cascade_audit(audit: Krea2MLPCascadeAudit) -> str:
    lines = [
        "Krea2 MLP cascade audit",
        "-----------------------",
        f"{'Candidate plan hash:':<30}{audit.candidate_plan_hash}",
        f"{'Forward count:':<30}{audit.forward_count}",
        f"{'Relative SSE:':<30}{audit.relative_sse:g}",
        f"{'Relative RMSE:':<30}{audit.relative_rmse:g}",
        f"{'Mean/forward relative SSE:':<30}{audit.mean_per_forward_relative_sse:g}",
        f"{'Max/forward relative SSE:':<30}{audit.max_per_forward_relative_sse:g}",
        "Acceptance: numerical report only; no automatic threshold applied.",
    ]
    return "\n".join(lines)


def validate_prune_request(
    original_mlp_width: int,
    start_channel: int,
    remove_channels: int,
    *,
    group_size: int = CONVROT_GROUP_SIZE,
) -> tuple[int, int]:
    """Validate a contiguous deletion and return ``(end, new_width)``."""
    if group_size <= 0:
        raise ValueError("ConvRot group size must be greater than zero.")
    if start_channel < 0:
        raise ValueError("--start-channel must be non-negative.")
    if remove_channels <= 0:
        raise ValueError("--remove-channels must be greater than zero.")
    if start_channel >= original_mlp_width:
        raise ValueError("--start-channel must be less than the MLP width.")

    end_channel = start_channel + remove_channels
    if end_channel > original_mlp_width:
        raise ValueError("The removed channel range exceeds the MLP width.")

    new_mlp_width = original_mlp_width - remove_channels
    if new_mlp_width <= 0:
        raise ValueError("Pruning must leave at least one MLP channel.")
    if new_mlp_width % group_size != 0:
        raise ValueError(
            "The resulting MLP width must be divisible by "
            f"the ConvRot group size ({group_size})."
        )

    return end_channel, new_mlp_width


def _validate_source_format(
    header: SourceModelHeader,
    blocks: Mapping[int, Mapping[PruneComponent, str]],
) -> None:
    if any(_BIAS_PATTERN.fullmatch(tensor_name) for tensor_name in header.tensors):
        raise ValueError(
            "Krea2 MLP pruning V1 does not support main-block MLP bias tensors."
        )

    for block_index, components in blocks.items():
        for component in _COMPONENTS:
            tensor_name = components[component]
            descriptor = header.tensors[tensor_name]
            if descriptor["dtype"] not in QUANTIZATION_SOURCE_DTYPES:
                raise ValueError(_SOURCE_FORMAT_ERROR)

            layer_name = tensor_name.removesuffix(".weight")
            if any(
                f"{layer_name}{suffix}" in header.tensors
                for suffix in (".weight_scale", ".comfy_quant")
            ):
                raise ValueError(_SOURCE_FORMAT_ERROR)


def _validate_shapes(
    header: SourceModelHeader,
    blocks: Mapping[int, Mapping[PruneComponent, str]],
) -> tuple[int, int]:
    first_shape: tuple[int, int] | None = None

    for block_index in sorted(blocks):
        components = blocks[block_index]
        gate = header.tensors[components["gate"]]
        up = header.tensors[components["up"]]
        down = header.tensors[components["down"]]

        gate_shape = tuple(gate["shape"])
        up_shape = tuple(up["shape"])
        down_shape = tuple(down["shape"])
        if len(gate_shape) != 2 or len(up_shape) != 2 or len(down_shape) != 2:
            raise ValueError(
                f"Krea2 block {block_index} main MLP weights must be two-dimensional."
            )
        if any(dimension <= 0 for shape in (gate_shape, up_shape, down_shape) for dimension in shape):
            raise ValueError(
                f"Krea2 block {block_index} main MLP weights must have positive dimensions."
            )
        if gate_shape != up_shape:
            raise ValueError(
                f"Krea2 block {block_index} gate/up shapes must match: "
                f"gate={gate_shape}, up={up_shape}."
            )
        if down_shape != (gate_shape[1], gate_shape[0]):
            raise ValueError(
                f"Krea2 block {block_index} down shape must be "
                f"({gate_shape[1]}, {gate_shape[0]}), got {down_shape}."
            )

        if first_shape is None:
            first_shape = gate_shape
        elif gate_shape != first_shape:
            raise ValueError(
                "Krea2 main-block MLP weights must share one logical shape; "
                f"block {block_index} has {gate_shape}, expected {first_shape}."
            )

    assert first_shape is not None
    hidden_width, original_mlp_width = first_shape[1], first_shape[0]
    for components in blocks.values():
        for component in _COMPONENTS:
            tensor_name = components[component]
            descriptor = header.tensors[tensor_name]
            dtype_bytes = _DTYPE_BYTES[descriptor["dtype"]]
            if source_bytes(descriptor) != prod(descriptor["shape"]) * dtype_bytes:
                raise ValueError(
                    f"Payload size for {tensor_name} does not match its shape and dtype."
                )

    return hidden_width, original_mlp_width


def build_krea2_mlp_prune_plan(
    header: SourceModelHeader,
    start_channel: int,
    remove_channels: int,
) -> Krea2MLPPrunePlan:
    blocks = discover_krea2_main_mlp(header)
    _validate_source_format(header, blocks)
    hidden_width, original_mlp_width = _validate_shapes(header, blocks)
    end_channel, new_mlp_width = validate_prune_request(
        original_mlp_width,
        start_channel,
        remove_channels,
    )

    estimated_saving_bytes = 0
    for components in blocks.values():
        for component in _COMPONENTS:
            dtype = header.tensors[components[component]]["dtype"]
            estimated_saving_bytes += (
                remove_channels * hidden_width * _DTYPE_BYTES[dtype]
            )

    return Krea2MLPPrunePlan(
        block_indices=tuple(sorted(blocks)),
        hidden_width=hidden_width,
        original_mlp_width=original_mlp_width,
        start_channel=start_channel,
        remove_channels=remove_channels,
        end_channel=end_channel,
        new_mlp_width=new_mlp_width,
        estimated_saving_bytes=estimated_saving_bytes,
    )


def build_krea2_mlp_plan_from_document(
    header: SourceModelHeader,
    document: Mapping[str, object],
    *,
    group_size: int = CONVROT_GROUP_SIZE,
) -> Krea2MLPPrunePlan:
    blocks = discover_krea2_main_mlp(header)
    _validate_source_format(header, blocks)
    hidden_width, original_mlp_width = _validate_shapes(header, blocks)
    validated = validate_krea2_mlp_plan(
        document,
        expected_block_indices=tuple(sorted(blocks)),
        require_convrot_alignment=group_size == CONVROT_GROUP_SIZE,
    )
    if validated["original_mlp_width"] != original_mlp_width:
        raise ValueError("Krea2 MLP plan width does not match the source model.")
    if validated["main_hidden_width"] != hidden_width:
        raise ValueError("Krea2 MLP plan hidden width does not match the source model.")
    if group_size <= 0:
        raise ValueError("ConvRot group size must be greater than zero.")
    new_mlp_width = cast(int, validated["new_mlp_width"])
    if new_mlp_width % group_size != 0:
        raise ValueError(
            "The resulting MLP width must be divisible by "
            f"the ConvRot group size ({group_size})."
        )
    raw_blocks = cast(Mapping[str, object], validated["blocks"])
    removed_by_block = {
        int(block_index): tuple(cast(list[int], channels))
        for block_index, channels in raw_blocks.items()
    }
    remove_channels = cast(int, validated["removed_per_block"])
    estimated_saving_bytes = sum(
        remove_channels
        * hidden_width
        * _DTYPE_BYTES[header.tensors[components[component]]["dtype"]]
        for components in blocks.values()
        for component in _COMPONENTS
    )
    selection = validated.get("selection")
    selection_metric = None
    stage = validated.get("stage")
    if isinstance(selection, Mapping):
        raw_metric = selection.get("metric")
        if isinstance(raw_metric, str):
            selection_metric = raw_metric
    return Krea2MLPPrunePlan(
        block_indices=tuple(sorted(blocks)),
        hidden_width=hidden_width,
        original_mlp_width=original_mlp_width,
        start_channel=0,
        remove_channels=remove_channels,
        end_channel=remove_channels,
        new_mlp_width=new_mlp_width,
        estimated_saving_bytes=estimated_saving_bytes,
        convrot_group_size=group_size,
        removed_channels_by_block=removed_by_block,
        plan_hash=krea2_mlp_plan_sha256(validated),
        selection_metric=selection_metric,
        stage=stage if isinstance(stage, int) else None,
    )


def prune_gate_or_up(
    tensor: torch.Tensor,
    start_channel: int,
    end_channel: int,
) -> torch.Tensor:
    """Remove one contiguous intermediate-channel range from gate/up rows."""
    return torch.cat(
        (tensor[:start_channel, :], tensor[end_channel:, :]),
        dim=0,
    ).contiguous()


def prune_down(
    tensor: torch.Tensor,
    start_channel: int,
    end_channel: int,
) -> torch.Tensor:
    """Remove one contiguous intermediate-channel range from down columns."""
    return torch.cat(
        (tensor[:, :start_channel], tensor[:, end_channel:]),
        dim=1,
    ).contiguous()


def _keep_indices(
    length: int,
    remove_indices: Sequence[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    keep = torch.ones(length, dtype=torch.bool, device=device)
    if remove_indices:
        keep[torch.tensor(remove_indices, dtype=torch.long, device=device)] = False
    return torch.nonzero(keep, as_tuple=False).flatten()


def prune_gate_or_up_indices(
    tensor: torch.Tensor,
    remove_indices: Sequence[int],
) -> torch.Tensor:
    """Remove arbitrary original intermediate-channel rows."""
    return tensor.index_select(
        0,
        _keep_indices(
            tensor.shape[0],
            remove_indices,
            device=tensor.device,
        ),
    ).contiguous()


def prune_down_indices(
    tensor: torch.Tensor,
    remove_indices: Sequence[int],
) -> torch.Tensor:
    """Remove arbitrary original intermediate-channel columns."""
    return tensor.index_select(
        1,
        _keep_indices(
            tensor.shape[1],
            remove_indices,
            device=tensor.device,
        ),
    ).contiguous()


def _component_for_name(tensor_name: str) -> PruneComponent | None:
    match = _TARGET_PATTERN.fullmatch(tensor_name)
    if match is None:
        return None
    return cast(PruneComponent, match.group("component"))


def _tensor_payload_chunks(tensor: torch.Tensor) -> Iterator[bytes]:
    raw_tensor = tensor.view(torch.uint8).flatten()
    for start in range(0, raw_tensor.numel(), _PAYLOAD_CHUNK_BYTES):
        yield raw_tensor[start : start + _PAYLOAD_CHUNK_BYTES].numpy().tobytes()


def _planned_shape(
    tensor_name: str,
    source_shape: Sequence[int],
    plan: Krea2MLPPrunePlan,
) -> tuple[int, ...]:
    component = _component_for_name(tensor_name)
    if component in ("gate", "up"):
        return (plan.new_mlp_width, source_shape[1])
    if component == "down":
        return (source_shape[0], plan.new_mlp_width)
    return tuple(source_shape)


def _build_output_layout(
    header: SourceModelHeader,
    plan: Krea2MLPPrunePlan,
) -> SafetensorsLayout:
    specs = []
    for tensor_name, descriptor in header.tensors.items():
        shape = _planned_shape(tensor_name, descriptor["shape"], plan)
        dtype_bytes = _DTYPE_BYTES.get(descriptor["dtype"])
        component = _component_for_name(tensor_name)
        byte_count = (
            prod(shape) * dtype_bytes
            if dtype_bytes is not None and component is not None
            else source_bytes(descriptor)
        )
        specs.append(
            OutputTensorSpec(
                name=tensor_name,
                dtype=descriptor["dtype"],
                shape=shape,
                byte_count=byte_count,
            )
        )
    return build_layout_from_specs(specs)


def _stream_pruned_payloads(
    source_path: Path,
    header: SourceModelHeader,
    plan: Krea2MLPPrunePlan,
    *,
    on_tensor_started: Callable[[int, int, str], None] | None = None,
) -> Iterator[TensorPayload]:
    with source_path.open("rb") as source_file:
        raw_data_start = read_raw_data_start(
            source_file,
            file_label="Source file",
        )
        tensor_count = len(header.tensors)
        for index, (tensor_name, descriptor) in enumerate(
            header.tensors.items(),
            start=1,
        ):
            if on_tensor_started is not None:
                on_tensor_started(index, tensor_count, tensor_name)

            start, end = descriptor["data_offsets"]
            raw_bytes = read_source_tensor_bytes(
                source_file,
                raw_data_start,
                (start, end),
                end - start,
            )
            component = _component_for_name(tensor_name)
            if component is None:
                yield tensor_name, raw_bytes
                continue

            tensor = tensor_from_raw_bytes(
                raw_bytes,
                descriptor["shape"],
                descriptor["dtype"],
                tensor_name=tensor_name,
            )
            match = _TARGET_PATTERN.fullmatch(tensor_name)
            block_index = None if match is None else int(match.group("index"))
            remove_indices = (
                None
                if plan.removed_channels_by_block is None or block_index is None
                else plan.removed_channels_by_block[block_index]
            )
            if component in ("gate", "up") and remove_indices is not None:
                pruned = prune_gate_or_up_indices(tensor, remove_indices)
            elif component == "down" and remove_indices is not None:
                pruned = prune_down_indices(tensor, remove_indices)
            elif component in ("gate", "up"):
                pruned = prune_gate_or_up(
                    tensor,
                    plan.start_channel,
                    plan.end_channel,
                )
            else:
                pruned = prune_down(
                    tensor,
                    plan.start_channel,
                    plan.end_channel,
                )
            payload_chunks = _tensor_payload_chunks(pruned)
            del tensor, raw_bytes
            yield tensor_name, payload_chunks
            del payload_chunks, pruned


def _validate_output(
    output_path: Path,
    source_header: SourceModelHeader,
    plan: Krea2MLPPrunePlan,
) -> None:
    output_header = read_source_model_header(output_path)
    validate_file_ranges(
        output_path,
        (
            (tensor_name, tuple(descriptor["data_offsets"]))
            for tensor_name, descriptor in output_header.tensors.items()
        ),
        "Output file",
    )
    source_blocks = discover_krea2_main_mlp(source_header)
    output_blocks = discover_krea2_main_mlp(output_header)
    if tuple(sorted(output_blocks)) != plan.block_indices:
        raise ValueError("Output block set does not match the pruning plan.")

    for block_index, components in source_blocks.items():
        for component in _COMPONENTS:
            source_name = components[component]
            output_name = _target_key(block_index, component)
            source_descriptor = source_header.tensors[source_name]
            output_descriptor = output_header.tensors.get(output_name)
            if output_descriptor is None:
                raise ValueError(f"Output is missing {output_name}.")
            expected_shape = _planned_shape(
                output_name,
                source_descriptor["shape"],
                plan,
            )
            if tuple(output_descriptor["shape"]) != expected_shape:
                raise ValueError(
                    f"Output shape for {output_name} is "
                    f"{tuple(output_descriptor['shape'])}, expected {expected_shape}."
                )
            if output_descriptor["dtype"] != source_descriptor["dtype"]:
                raise ValueError(f"Output dtype for {output_name} changed.")


def _pruning_metadata(
    source_metadata: Mapping[str, str],
    plan: Krea2MLPPrunePlan,
) -> dict[str, str]:
    if plan.removed_channels_by_block is None:
        label = _PRUNING_LABEL
        config = {
            "original_intermediate_width": plan.original_mlp_width,
            "new_intermediate_width": plan.new_mlp_width,
            "start_channel": plan.start_channel,
            "remove_channels": plan.remove_channels,
            "end_channel": plan.end_channel,
            "block_count": len(plan.block_indices),
            "convrot_group_size": plan.convrot_group_size,
        }
    else:
        label = "krea2_mlp_plan_v1"
        config = {
            "original_intermediate_width": plan.original_mlp_width,
            "new_intermediate_width": plan.new_mlp_width,
            "remove_channels": plan.remove_channels,
            "block_count": len(plan.block_indices),
            "convrot_group_size": plan.convrot_group_size,
            "plan_sha256": plan.plan_hash,
            "selection_metric": plan.selection_metric,
            "stage": plan.stage,
        }
    return {
        **source_metadata,
        _PRUNING_METADATA_KEY: label,
        _PRUNING_CONFIG_METADATA_KEY: json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def _validate_pruning_metadata_slots(
    source_metadata: Mapping[str, str],
) -> None:
    collisions = sorted(
        key
        for key in (_PRUNING_METADATA_KEY, _PRUNING_CONFIG_METADATA_KEY)
        if key in source_metadata
    )
    if collisions:
        raise ValueError(
            "Source metadata already contains pruning provenance keys: "
            f"{', '.join(collisions)}."
        )


def _write_pruned_checkpoint_core(
    source: Path,
    source_header: SourceModelHeader,
    plan: Krea2MLPPrunePlan,
    staged_output: Path,
    *,
    on_tensor_started: Callable[[int, int, str], None] | None = None,
) -> None:
    """Write and validate an MLP-pruned checkpoint at the supplied staging path."""
    staged_output.parent.mkdir(parents=True, exist_ok=True)
    layout = _build_output_layout(source_header, plan)
    write_safetensors_file(
        staged_output,
        layout,
        _stream_pruned_payloads(
            source,
            source_header,
            plan,
            on_tensor_started=on_tensor_started,
        ),
        metadata=_pruning_metadata(source_header.metadata, plan),
    )
    _validate_output(staged_output, source_header, plan)


def _write_pruned_checkpoint(
    source: Path,
    output: Path,
    source_header: SourceModelHeader,
    plan: Krea2MLPPrunePlan,
    *,
    on_tensor_started: Callable[[int, int, str], None] | None = None,
) -> Krea2MLPPruneResult:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.name}.partial")
    if partial.exists():
        raise FileExistsError(f"Partial output already exists: {partial}")
    target_count = len(plan.block_indices) * len(_COMPONENTS)
    try:
        _write_pruned_checkpoint_core(
            source,
            source_header,
            plan,
            partial,
            on_tensor_started=on_tensor_started,
        )
        partial.rename(output)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise

    return Krea2MLPPruneResult(
        output_path=output,
        plan=plan,
        target_tensors_modified=target_count,
        other_tensors_copied=len(source_header.tensors) - target_count,
        output_bytes=output.stat().st_size,
    )


def _read_prune_source(
    source_path: str | Path,
    output_path: str | Path,
) -> tuple[Path, Path, SourceModelHeader]:
    source = Path(source_path)
    output = Path(output_path)
    if source.resolve() == output.resolve():
        raise ValueError("Source and output paths cannot be the same.")
    if not source.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {source}.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    source_header = read_source_model_header(source)
    validate_file_ranges(
        source,
        (
            (tensor_name, tuple(descriptor["data_offsets"]))
            for tensor_name, descriptor in source_header.tensors.items()
        ),
        "Source file",
    )
    _validate_pruning_metadata_slots(source_header.metadata)
    return source, output, source_header


def prune_krea2_mlp_checkpoint(
    source_path: str | Path,
    output_path: str | Path,
    start_channel: int,
    remove_channels: int,
    *,
    on_plan: PlanReporter | None = None,
    on_tensor_started: Callable[[int, int, str], None] | None = None,
) -> Krea2MLPPruneResult:
    """Physically prune contiguous main-block MLP channels."""
    source, output, source_header = _read_prune_source(source_path, output_path)
    plan = build_krea2_mlp_prune_plan(
        source_header,
        start_channel,
        remove_channels,
    )
    if on_plan is not None:
        on_plan(plan)
    return _write_pruned_checkpoint(
        source,
        output,
        source_header,
        plan,
        on_tensor_started=on_tensor_started,
    )


def apply_krea2_mlp_plan(
    source_path: str | Path,
    plan_path: str | Path,
    output_path: str | Path,
    *,
    group_size: int = CONVROT_GROUP_SIZE,
    on_plan: PlanReporter | None = None,
    on_tensor_started: Callable[[int, int, str], None] | None = None,
) -> Krea2MLPPruneResult:
    """Apply one accepted cumulative plan to the original floating-point source."""
    source, output, source_header = _read_prune_source(source_path, output_path)
    document = load_krea2_mlp_plan(plan_path)
    plan = build_krea2_mlp_plan_from_document(
        source_header,
        document,
        group_size=group_size,
    )
    if on_plan is not None:
        on_plan(plan)
    return _write_pruned_checkpoint(
        source,
        output,
        source_header,
        plan,
        on_tensor_started=on_tensor_started,
    )


prune_krea2_mlp = prune_krea2_mlp_checkpoint
