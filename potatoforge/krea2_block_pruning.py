"""Offline aggregation and cumulative plans for Krea2 block pruning."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file

from .headers.header_reader import read_raw_data_start
from .headers.source_header import SourceModelHeader, read_source_model_header
from .planning import OutputTensorSpec, SafetensorsLayout, build_layout_from_specs, source_bytes
from .profiles import QuantizationProfile, load_profile, resolve_profile
from .safetensors_writer import TensorPayload, write_safetensors_file
from .source_payloads import read_source_tensor_bytes
from .patching import validate_file_ranges


KREA2_BLOCK_PLAN_FORMAT = "potatoforge_krea2_block_prune_plan"
KREA2_BLOCK_PLAN_VERSION = 1
KREA2_BLOCK_CALIBRATION_FORMAT = "potatoforge_krea2_block_calibration"
KREA2_BLOCK_CALIBRATION_VERSION = 1
KREA2_BLOCK_AUDIT_FORMAT = "potatoforge_krea2_block_audit"
KREA2_BLOCK_AUDIT_VERSION = 2
_EPSILON = 1e-12
_MAIN_BLOCK_PATTERN = re.compile(r"^blocks\.(?P<index>\d+)\.")
_BLOCK_PREFIX_PATTERN = re.compile(r"^blocks\.(?P<index>\d+)(?=\.|$)")
_BLOCK_PRUNING_MANIFEST_FORMAT = "potatoforge_krea2_physical_block_prune_manifest"


@dataclass(frozen=True)
class Krea2BlockCalibration:
    path: Path
    block_count: int
    removed_blocks: tuple[int, ...]
    active_blocks: tuple[int, ...]
    base_plan_hash: str
    source_model_fingerprint: str | None
    sum_input_sq: torch.Tensor
    sum_residual_sq: torch.Tensor
    observation_count: torch.Tensor


@dataclass(frozen=True)
class Krea2BlockAudit:
    path: Path
    block_count: int
    base_removed_blocks: tuple[int, ...]
    candidate_block: int
    base_plan_hash: str
    source_model_fingerprint: str | None
    metric: str | None
    metric_version: int | None
    forward_count: int
    mean_relative_sse: float
    total_relative_sse: float
    max_relative_sse: float


@dataclass(frozen=True)
class Krea2BlockApplyResult:
    output_path: Path
    manifest_path: Path
    original_block_count: int
    removed_blocks: tuple[int, ...]
    tensor_count: int
    renamed_tensor_count: int
    removed_tensor_count: int
    removed_payload_bytes: int
    old_to_new: Mapping[int, int | None]
    dry_run: bool


@dataclass(frozen=True)
class Krea2BlockProfileRemapResult:
    output_path: Path
    removed_rules: int
    rewritten_rules: int
    unchanged_rules: int
    verified_assignments: int
    old_to_new: Mapping[int, int | None]
    dry_run: bool


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Artifact not found: {path}.") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Artifact must be a JSON object: {path}")
    return cast(dict[str, object], value)


def _positive_int(document: Mapping[str, object], key: str, *, path: Path) -> int:
    value = document.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{path}: {key} must be a positive integer.")
    return value


def _finite_nonnegative(document: Mapping[str, object], key: str, *, path: Path) -> float:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: {key} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{path}: {key} must be finite and non-negative.")
    return result


def _block_ids(value: object, key: str, block_count: int, *, path: Path) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValueError(f"{path}: {key} must be an array of block IDs.")
    result = tuple(value)
    if result != tuple(sorted(result)) or len(set(result)) != len(result):
        raise ValueError(f"{path}: {key} must be sorted and unique.")
    if any(item < 0 or item >= block_count for item in result):
        raise ValueError(f"{path}: {key} contains an out-of-range block.")
    return result


def _fingerprint(document: Mapping[str, object], *, path: Path) -> str | None:
    value = document.get("source_model_fingerprint")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: source_model_fingerprint is invalid.")
    return value


def _base_plan_hash(document: Mapping[str, object], *, path: Path) -> str:
    value = document.get("base_plan_hash", "none")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: base_plan_hash is invalid.")
    return value


def _require(document: Mapping[str, object], values: Mapping[str, object], *, path: Path) -> None:
    for key, expected in values.items():
        if document.get(key) != expected:
            raise ValueError(f"{path}: unsupported {key}: {document.get(key)!r}.")


def load_krea2_block_calibration(path: str | Path) -> Krea2BlockCalibration:
    metadata_path = Path(path)
    if metadata_path.suffix.lower() == ".safetensors":
        metadata_path = metadata_path.with_suffix(".json")
    document = _load_json(metadata_path)
    _require(document, {
        "format": KREA2_BLOCK_CALIBRATION_FORMAT,
        "version": KREA2_BLOCK_CALIBRATION_VERSION,
        "architecture": "krea2",
        "metric": "relative_residual_energy",
        "metric_version": 1,
    }, path=metadata_path)
    block_count = _positive_int(document, "block_count", path=metadata_path)
    removed = _block_ids(document.get("removed_blocks"), "removed_blocks", block_count, path=metadata_path)
    active = _block_ids(document.get("active_blocks"), "active_blocks", block_count, path=metadata_path)
    if set(active) != set(range(block_count)) - set(removed):
        raise ValueError(f"{metadata_path}: active_blocks does not match removed_blocks.")
    tensors_path = metadata_path.with_suffix(".safetensors")
    tensors = load_file(str(tensors_path), device="cpu")
    if set(tensors) != {"sum_input_sq", "sum_residual_sq", "observation_count"}:
        raise ValueError(f"{tensors_path}: expected sum_input_sq, sum_residual_sq, and observation_count.")
    input_sq, residual_sq, count = (tensors[key] for key in ("sum_input_sq", "sum_residual_sq", "observation_count"))
    if any(value.ndim != 1 or value.numel() != block_count for value in (input_sq, residual_sq, count)):
        raise ValueError(f"{tensors_path}: block-stat tensor shape does not match block_count.")
    input_sq, residual_sq, count = (value.to(torch.float64) for value in (input_sq, residual_sq, count))
    if any(not torch.isfinite(value).all() or (value < 0).any() for value in (input_sq, residual_sq, count)):
        raise ValueError(f"{tensors_path}: block statistics must be finite and non-negative.")
    return Krea2BlockCalibration(metadata_path, block_count, removed, active, _base_plan_hash(document, path=metadata_path), _fingerprint(document, path=metadata_path), input_sq, residual_sq, count)


def discover_krea2_block_calibrations(path: str | Path) -> tuple[Path, ...]:
    root = Path(path)
    if not root.is_dir():
        return (root,)
    found: list[Path] = []
    for child in root.iterdir():
        if not child.is_file() or child.suffix.lower() != ".json":
            continue
        try:
            document = _load_json(child)
        except ValueError:
            continue
        if document.get("format") == KREA2_BLOCK_CALIBRATION_FORMAT:
            found.append(child)
    if not found:
        raise ValueError(f"No Krea2 block calibration artifacts found in: {root}")
    return tuple(sorted(found, key=lambda item: (item.name.casefold(), item.name)))


def _check_match(first: object, current: object, field: str, path: Path) -> None:
    if first != current:
        raise ValueError(f"{path}: {field} does not match the first artifact ({first!r} != {current!r}).")


def rank_krea2_blocks(calibration_path: str | Path, output_path: str | Path, *, top_k: int = 5) -> dict[str, object]:
    if type(top_k) is not int or top_k < 1:
        raise ValueError("--top-k must be a positive integer.")
    paths = discover_krea2_block_calibrations(calibration_path)
    first = load_krea2_block_calibration(paths[0])
    input_sq, residual_sq, count = first.sum_input_sq.clone(), first.sum_residual_sq.clone(), first.observation_count.clone()
    for path in paths[1:]:
        current = load_krea2_block_calibration(path)
        for field in ("block_count", "removed_blocks", "active_blocks", "base_plan_hash", "source_model_fingerprint"):
            _check_match(getattr(first, field), getattr(current, field), field, path)
        input_sq.add_(current.sum_input_sq)
        residual_sq.add_(current.sum_residual_sq)
        count.add_(current.observation_count)
    if not first.active_blocks:
        raise ValueError("No active Krea2 blocks remain to rank.")
    ranking = sorted(({
        "block": block,
        "score": float(residual_sq[block] / (input_sq[block] + _EPSILON)),
        "sum_input_sq": float(input_sq[block]),
        "sum_residual_sq": float(residual_sq[block]),
        "observation_count": int(count[block]),
    } for block in first.active_blocks), key=lambda item: (item["score"], item["block"]))
    for rank, item in enumerate(ranking, start=1):
        item["rank"] = rank
    document: dict[str, object] = {
        "format": "potatoforge_krea2_block_candidates", "version": 1, "architecture": "krea2",
        "source_model_fingerprint": first.source_model_fingerprint, "block_count": first.block_count,
        "base_removed_blocks": list(first.removed_blocks), "base_plan_hash": first.base_plan_hash,
        "metric": "relative_residual_energy", "metric_version": 1, "artifact_count": len(paths),
        "ranking": ranking, "top_k": [item["block"] for item in ranking[:top_k]],
    }
    _write_json(Path(output_path), document)
    return document


def load_krea2_block_audit(path: str | Path) -> Krea2BlockAudit:
    artifact_path = Path(path)
    document = _load_json(artifact_path)
    _require(document, {"format": KREA2_BLOCK_AUDIT_FORMAT, "version": KREA2_BLOCK_AUDIT_VERSION, "architecture": "krea2"}, path=artifact_path)
    block_count = _positive_int(document, "block_count", path=artifact_path)
    base = _block_ids(document.get("base_removed_blocks"), "base_removed_blocks", block_count, path=artifact_path)
    candidate = document.get("candidate_block")
    if type(candidate) is not int or candidate < 0 or candidate >= block_count or candidate in base:
        raise ValueError(f"{artifact_path}: candidate_block must be an active block ID.")
    candidate_removed = document.get("candidate_removed_blocks")
    if not isinstance(candidate_removed, list) or any(type(item) is not int for item in candidate_removed):
        raise ValueError(f"{artifact_path}: candidate_removed_blocks must be an array of block IDs.")
    if len(set(candidate_removed)) != len(candidate_removed) or any(item < 0 or item >= block_count for item in candidate_removed):
        raise ValueError(f"{artifact_path}: candidate_removed_blocks is invalid.")
    if set(candidate_removed) != {*base, candidate}:
        raise ValueError(f"{artifact_path}: candidate_removed_blocks must equal base_removed_blocks plus candidate_block.")
    metric, metric_version = document.get("metric"), document.get("metric_version")
    if (metric, metric_version) not in {(None, None), ("final_output_relative_sse", 1)}:
        raise ValueError(f"{artifact_path}: unsupported audit metric/version.")
    return Krea2BlockAudit(artifact_path, block_count, base, candidate, _base_plan_hash(document, path=artifact_path), _fingerprint(document, path=artifact_path), cast(str | None, metric), cast(int | None, metric_version), _positive_int(document, "forward_count", path=artifact_path), _finite_nonnegative(document, "mean_relative_sse", path=artifact_path), _finite_nonnegative(document, "total_relative_sse", path=artifact_path), _finite_nonnegative(document, "max_relative_sse", path=artifact_path))


def discover_krea2_block_audits(path: str | Path) -> tuple[Path, ...]:
    root = Path(path)
    if not root.is_dir():
        return (root,)
    found = []
    for child in root.iterdir():
        if not child.is_file() or child.suffix.lower() != ".json":
            continue
        try:
            document = _load_json(child)
        except ValueError:
            continue
        if document.get("format") == KREA2_BLOCK_AUDIT_FORMAT:
            found.append(child)
    if not found:
        raise ValueError(f"No Krea2 block audit artifacts found in: {root}")
    return tuple(sorted(found, key=lambda item: (item.name.casefold(), item.name)))


def summarize_krea2_block_audits(audit_path: str | Path, output_path: str | Path) -> dict[str, object]:
    paths = discover_krea2_block_audits(audit_path)
    first = load_krea2_block_audit(paths[0])
    grouped: dict[int, list[Krea2BlockAudit]] = defaultdict(list)
    for path in paths:
        current = load_krea2_block_audit(path)
        for field in ("block_count", "base_removed_blocks", "base_plan_hash", "source_model_fingerprint", "metric", "metric_version"):
            _check_match(getattr(first, field), getattr(current, field), field, path)
        grouped[current.candidate_block].append(current)
    ranking: list[dict[str, object]] = []
    for candidate, audits in grouped.items():
        forwards = sum(item.forward_count for item in audits)
        # v2 does not expose raw numerator/denominator totals. This exact per-forward mean is the strongest merge available.
        mean_relative_sse = sum(item.mean_relative_sse * item.forward_count for item in audits) / forwards
        ranking.append({"candidate_block": candidate, "mean_relative_sse": mean_relative_sse,
                        "mean_total_relative_sse": sum(item.total_relative_sse for item in audits) / len(audits),
                        "max_relative_sse": max(item.max_relative_sse for item in audits),
                        "artifact_count": len(audits), "forward_count": forwards})
    ranking.sort(key=lambda item: (item["mean_relative_sse"], item["candidate_block"]))
    for rank, item in enumerate(ranking, start=1):
        item["rank"] = rank
    document: dict[str, object] = {"format": "potatoforge_krea2_block_audit_summary", "version": 1, "architecture": "krea2",
        "source_model_fingerprint": first.source_model_fingerprint, "block_count": first.block_count, "base_removed_blocks": list(first.base_removed_blocks),
        "base_plan_hash": first.base_plan_hash, "metric": "mean_relative_sse_per_forward", "metric_version": 1,
        "aggregation": "forward_count_weighted_mean; raw error/baseline sums are unavailable in audit v2", "artifact_count": len(paths), "ranking": ranking}
    _write_json(Path(output_path), document)
    return document


def validate_krea2_block_plan(document: Mapping[str, object]) -> dict[str, object]:
    _require(document, {"format": KREA2_BLOCK_PLAN_FORMAT, "version": KREA2_BLOCK_PLAN_VERSION, "architecture": "krea2"}, path=Path("<plan>"))
    original_count = document.get("original_block_count")
    legacy_count = document.get("block_count")
    if original_count is not None and legacy_count is not None and original_count != legacy_count:
        raise ValueError("<plan>: original_block_count does not match block_count.")
    count_key = "original_block_count" if original_count is not None else "block_count"
    block_count = _positive_int(document, count_key, path=Path("<plan>"))
    removed = _block_ids(document.get("removed_blocks"), "removed_blocks", block_count, path=Path("<plan>"))
    remaining = document.get("remaining_block_count")
    if remaining is not None and (type(remaining) is not int or remaining != block_count - len(removed)):
        raise ValueError("<plan>: remaining_block_count does not match removed_blocks.")
    return {"format": KREA2_BLOCK_PLAN_FORMAT, "version": KREA2_BLOCK_PLAN_VERSION, "architecture": "krea2", "block_count": block_count, "removed_blocks": list(removed)}


def canonical_krea2_block_plan_bytes(document: Mapping[str, object]) -> bytes:
    return json.dumps(validate_krea2_block_plan(document), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def krea2_block_plan_sha256(document: Mapping[str, object] | str | Path) -> str:
    plan = _load_json(Path(document)) if isinstance(document, (str, Path)) else document
    return hashlib.sha256(canonical_krea2_block_plan_bytes(plan)).hexdigest()


def krea2_block_old_to_new(
    plan: Mapping[str, object] | str | Path,
) -> dict[int, int | None]:
    """Map original Krea2 block IDs to their final contiguous IDs."""
    document = (
        validate_krea2_block_plan(_load_json(Path(plan)))
        if isinstance(plan, (str, Path))
        else validate_krea2_block_plan(plan)
    )
    block_count = cast(int, document["block_count"])
    removed = set(cast(list[int], document["removed_blocks"]))
    return {
        old_id: None if old_id in removed else old_id - sum(item < old_id for item in removed)
        for old_id in range(block_count)
    }


def _main_block_id(tensor_name: str) -> int | None:
    match = _MAIN_BLOCK_PATTERN.match(tensor_name)
    return None if match is None else int(match.group("index"))


def _validate_source_blocks(header: SourceModelHeader, block_count: int) -> None:
    block_ids = {_main_block_id(name) for name in header.tensors}
    block_ids.discard(None)
    expected = set(range(block_count))
    if block_ids != expected:
        raise ValueError(
            "Input model block count does not match plan block_count. The input may "
            "already have whole-block pruning applied. Apply cumulative original-ID "
            "plans to the original block topology, not to an already-renumbered model."
        )


def _mapped_tensor_name(tensor_name: str, mapping: Mapping[int, int | None]) -> str | None:
    old_id = _main_block_id(tensor_name)
    if old_id is None:
        return tensor_name
    new_id = mapping[old_id]
    if new_id is None:
        return None
    return _MAIN_BLOCK_PATTERN.sub(f"blocks.{new_id}.", tensor_name, count=1)


def _output_layout(
    header: SourceModelHeader,
    mapping: Mapping[int, int | None],
) -> tuple[SafetensorsLayout, tuple[tuple[str, str], ...], int, int]:
    specs: list[OutputTensorSpec] = []
    rewritten: list[tuple[str, str]] = []
    removed_tensor_count = 0
    removed_payload_bytes = 0
    for source_name, descriptor in header.tensors.items():
        output_name = _mapped_tensor_name(source_name, mapping)
        if output_name is None:
            removed_tensor_count += 1
            removed_payload_bytes += source_bytes(descriptor)
            continue
        specs.append(OutputTensorSpec(
            name=output_name,
            dtype=descriptor["dtype"],
            shape=tuple(descriptor["shape"]),
            byte_count=source_bytes(descriptor),
        ))
        rewritten.append((source_name, output_name))
    return build_layout_from_specs(specs), tuple(rewritten), removed_tensor_count, removed_payload_bytes


def _stream_rewritten_payloads(
    source: Path,
    header: SourceModelHeader,
    rewritten: Sequence[tuple[str, str]],
    *,
    on_tensor_started: Callable[[int, int, str], None] | None = None,
) -> Iterator[TensorPayload]:
    with source.open("rb") as source_file:
        raw_data_start = read_raw_data_start(source_file, file_label="Source file")
        for index, (source_name, output_name) in enumerate(rewritten, start=1):
            if on_tensor_started is not None:
                on_tensor_started(index, len(rewritten), source_name)
            descriptor = header.tensors[source_name]
            start, end = descriptor["data_offsets"]
            yield output_name, read_source_tensor_bytes(
                source_file, raw_data_start, (start, end), end - start
            )


def _block_pruning_metadata(
    metadata: Mapping[str, str],
    plan: Mapping[str, object],
    plan_hash: str,
) -> dict[str, str]:
    return {
        **metadata,
        "potatoforge.block_pruning": "krea2",
        "potatoforge.block_pruning_plan_hash": plan_hash,
        "potatoforge.block_pruning_removed_original_blocks": json.dumps(plan["removed_blocks"]),
        "potatoforge.block_pruning_original_block_count": str(plan["block_count"]),
        "potatoforge.block_pruning_remaining_block_count": str(
            int(plan["block_count"]) - len(cast(list[int], plan["removed_blocks"]))
        ),
    }


def _validate_rewritten_output(
    output: Path,
    source_header: SourceModelHeader,
    rewritten: Sequence[tuple[str, str]],
    mapping: Mapping[int, int | None],
) -> None:
    output_header = read_source_model_header(output)
    validate_file_ranges(
        output,
        ((name, tuple(descriptor["data_offsets"])) for name, descriptor in output_header.tensors.items()),
        "Output file",
    )
    expected_ids = [new_id for new_id in mapping.values() if new_id is not None]
    actual_ids = sorted({_main_block_id(name) for name in output_header.tensors} - {None})
    if actual_ids != expected_ids:
        raise ValueError("Output main block IDs are not contiguous.")
    for source_name, output_name in rewritten:
        source = source_header.tensors[source_name]
        target = output_header.tensors.get(output_name)
        if target is None or target["dtype"] != source["dtype"] or target["shape"] != source["shape"]:
            raise ValueError(f"Output tensor changed shape or dtype: {output_name}")


def _write_json_atomic(path: Path, document: Mapping[str, object], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    if partial.exists():
        raise FileExistsError(f"Partial output already exists: {partial}")
    try:
        partial.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        partial.replace(path)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise


def apply_krea2_block_plan(
    source_path: str | Path,
    plan_path: str | Path,
    output_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    old_to_new: Mapping[int, int | None] | None = None,
    on_tensor_started: Callable[[int, int, str], None] | None = None,
) -> Krea2BlockApplyResult:
    """Physically remove original-ID Krea2 main blocks and renumber survivors."""
    source, output = Path(source_path), Path(output_path)
    if source.resolve() == output.resolve():
        raise ValueError("Source and output paths cannot be the same.")
    if not source.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {source}.")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    plan = validate_krea2_block_plan(_load_json(Path(plan_path)))
    mapping = krea2_block_old_to_new(plan) if old_to_new is None else old_to_new
    header = read_source_model_header(source)
    validate_file_ranges(
        source,
        ((name, tuple(descriptor["data_offsets"])) for name, descriptor in header.tensors.items()),
        "Source file",
    )
    _validate_source_blocks(header, cast(int, plan["block_count"]))
    layout, rewritten, removed_tensor_count, removed_payload_bytes = _output_layout(header, mapping)
    plan_hash = krea2_block_plan_sha256(plan)
    manifest = Path(manifest_path) if manifest_path is not None else output.with_name(f"{output.name}.block-prune-manifest.json")
    if manifest.resolve() in {source.resolve(), output.resolve()}:
        raise ValueError("Manifest path must differ from source and output paths.")
    result = Krea2BlockApplyResult(
        output_path=output,
        manifest_path=manifest,
        original_block_count=cast(int, plan["block_count"]),
        removed_blocks=tuple(cast(list[int], plan["removed_blocks"])),
        tensor_count=len(rewritten),
        renamed_tensor_count=sum(source_name != output_name for source_name, output_name in rewritten),
        removed_tensor_count=removed_tensor_count,
        removed_payload_bytes=removed_payload_bytes,
        old_to_new=mapping,
        dry_run=dry_run,
    )
    if dry_run:
        return result
    if manifest.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {manifest}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.name}.partial")
    if partial.exists():
        raise FileExistsError(f"Partial output already exists: {partial}")
    restore = manifest.with_name(f"{manifest.name}.restore")
    if restore.exists():
        raise FileExistsError(f"Restore output already exists: {restore}")
    previous_manifest = manifest.read_bytes() if manifest.exists() else None
    manifest_written = False
    manifest_document: dict[str, object] = {
        "format": _BLOCK_PRUNING_MANIFEST_FORMAT,
        "version": 1,
        "architecture": "krea2",
        "source_model": source.name,
        "original_block_count": plan["block_count"],
        "remaining_block_count": len([item for item in mapping.values() if item is not None]),
        "removed_original_blocks": plan["removed_blocks"],
        "old_to_new": {str(old_id): new_id for old_id, new_id in mapping.items()},
        "plan_hash": plan_hash,
        "source_tensor_count": len(header.tensors),
        "output_tensor_count": len(rewritten),
        "removed_tensor_count": removed_tensor_count,
        "removed_payload_bytes": removed_payload_bytes,
    }
    try:
        write_safetensors_file(
            partial, layout, _stream_rewritten_payloads(source, header, rewritten, on_tensor_started=on_tensor_started),
            metadata=_block_pruning_metadata(header.metadata, plan, plan_hash),
        )
        _validate_rewritten_output(partial, header, rewritten, mapping)
        _write_json_atomic(manifest, manifest_document, overwrite=overwrite)
        manifest_written = True
        partial.replace(output)
    except BaseException:
        if partial.exists():
            partial.unlink()
        if manifest_written:
            if previous_manifest is None:
                manifest.unlink()
            else:
                restore.write_bytes(previous_manifest)
                restore.replace(manifest)
        raise
    return result


def remap_krea2_block_profile(
    profile_path: str | Path,
    plan_path: str | Path,
    output_path: str | Path,
    *,
    source_model_path: str | Path | None = None,
    pruned_model_path: str | Path | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    old_to_new: Mapping[int, int | None] | None = None,
) -> Krea2BlockProfileRemapResult:
    """Remap ordered exact block-prefix rules with the physical block mapping."""
    profile_source, output = Path(profile_path), Path(output_path)
    if profile_source.resolve() == output.resolve():
        raise ValueError("Profile and output paths cannot be the same.")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    original_profile = load_profile(profile_source)
    document = _load_json(profile_source)
    plan = validate_krea2_block_plan(_load_json(Path(plan_path)))
    mapping = krea2_block_old_to_new(plan) if old_to_new is None else old_to_new
    raw_rules = document["rules"]
    assert isinstance(raw_rules, list)
    rules: list[object] = []
    removed = rewritten = unchanged = 0
    for raw_rule in raw_rules:
        assert isinstance(raw_rule, dict)
        prefix = raw_rule["prefix"]
        assert isinstance(prefix, str)
        match = _BLOCK_PREFIX_PATTERN.match(prefix)
        if match is None:
            rules.append(raw_rule)
            unchanged += 1
            continue
        old_id = int(match.group("index"))
        if old_id not in mapping:
            raise ValueError(f"Profile rule targets block {old_id}, outside the plan block count.")
        new_id = mapping[old_id]
        if new_id is None:
            removed += 1
            continue
        rules.append({**raw_rule, "prefix": f"blocks.{new_id}" + prefix[match.end():]})
        rewritten += 1
    remapped = {**document, "rules": rules}
    verified_assignments = 0
    remapped_profile = cast(QuantizationProfile, {
        "default": original_profile["default"],
        "rules": tuple({**cast(dict[str, object], rule), "suffixes": tuple(cast(dict[str, object], rule)["suffixes"])} for rule in rules),
    })
    if pruned_model_path is not None and source_model_path is None:
        raise ValueError("--pruned-model requires --source-model.")
    if source_model_path is not None:
        source_header = read_source_model_header(source_model_path)
        _validate_source_blocks(source_header, cast(int, plan["block_count"]))
        output_header = (
            None if pruned_model_path is None else read_source_model_header(pruned_model_path)
        )
        expected_output_names = set()
        for source_name, source_descriptor in source_header.tensors.items():
            output_name = _mapped_tensor_name(source_name, mapping)
            if output_name is None:
                continue
            expected_output_names.add(output_name)
            if output_header is not None:
                output_descriptor = output_header.tensors.get(output_name)
                if output_descriptor is None:
                    raise ValueError(f"Pruned model is missing surviving tensor: {output_name}")
                if output_descriptor["shape"] != source_descriptor["shape"] or output_descriptor["dtype"] != source_descriptor["dtype"]:
                    raise ValueError(f"Pruned model changed tensor shape or dtype: {output_name}")
            if resolve_profile(original_profile, source_name) != resolve_profile(remapped_profile, output_name):
                raise ValueError(f"Profile assignment changed for surviving tensor: {source_name}")
            verified_assignments += 1
        if output_header is not None and set(output_header.tensors) != expected_output_names:
            raise ValueError("Pruned model tensor keys do not match the block-prune mapping.")
    result = Krea2BlockProfileRemapResult(
        output, removed, rewritten, unchanged, verified_assignments, mapping, dry_run
    )
    if not dry_run:
        _write_json_atomic(output, remapped, overwrite=overwrite)
        load_profile(output)
    return result


def add_krea2_block_to_plan(output_path: str | Path, *, block: int, block_count: int | None = None, base_plan_path: str | Path | None = None) -> dict[str, object]:
    if base_plan_path is None:
        if type(block_count) is not int or block_count < 1:
            raise ValueError("--block-count must be a positive integer when --base-plan is omitted.")
        base = {"format": KREA2_BLOCK_PLAN_FORMAT, "version": 1, "architecture": "krea2", "block_count": block_count, "removed_blocks": []}
    else:
        base = validate_krea2_block_plan(_load_json(Path(base_plan_path)))
        if block_count is not None and block_count != base["block_count"]:
            raise ValueError("--block-count does not match --base-plan.")
    count = cast(int, base["block_count"])
    removed = cast(list[int], base["removed_blocks"])
    if type(block) is not int or block < 0 or block >= count:
        raise ValueError("--block must be within the original block count.")
    if block in removed:
        raise ValueError("--block is already removed by the base plan.")
    plan = {**base, "removed_blocks": sorted([*removed, block])}
    _write_json(Path(output_path), plan)
    return plan


def _write_json(path: Path, document: Mapping[str, object]) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
