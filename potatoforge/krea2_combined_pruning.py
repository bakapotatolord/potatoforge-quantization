"""One-pass orchestration for Krea2 block and MLP structural pruning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, cast

from .headers.source_header import SourceModelHeader, read_source_model_header
from .krea2_block_pruning import (
    _block_pruning_metadata,
    _mapped_tensor_name,
    _validate_source_blocks,
    krea2_block_old_to_new,
    krea2_block_plan_sha256,
    remap_krea2_block_profile,
    validate_krea2_block_plan,
)
from .krea2_mlp_pruning import (
    _validate_pruning_metadata_slots,
    _write_pruned_checkpoint_core,
    build_krea2_mlp_plan_from_document,
    krea2_mlp_plan_sha256,
    load_krea2_mlp_plan,
    remap_krea2_mlp_plan_blocks,
)
from .patching import validate_file_ranges
from .profiles import load_profile, resolve_profile


@dataclass(frozen=True)
class Krea2CombinedPruneResult:
    output_path: Path
    remapped_mlp_plan_path: Path
    output_profile_path: Path
    manifest_path: Path
    original_block_count: int
    removed_blocks: tuple[int, ...]
    original_mlp_width: int
    new_mlp_width: int
    old_to_new: Mapping[int, int | None]
    removed_profile_rules: int
    rewritten_profile_rules: int
    verified_assignments: int
    dry_run: bool


def _load_json(path: Path) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Artifact not found: {path}.") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON artifact: {path}") from error
    if not isinstance(document, dict):
        raise ValueError(f"Artifact must be a JSON object: {path}")
    return cast(dict[str, object], document)


def _write_json(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sidecar_paths(
    output: Path,
    remapped_mlp_plan_path: str | Path | None,
    output_profile_path: str | Path | None,
    manifest_path: str | Path | None,
) -> tuple[Path, Path, Path]:
    return (
        Path(remapped_mlp_plan_path) if remapped_mlp_plan_path is not None else output.with_name(f"{output.stem}.mlp-plan.json"),
        Path(output_profile_path) if output_profile_path is not None else output.with_name(f"{output.stem}.quant-profile.json"),
        Path(manifest_path) if manifest_path is not None else output.with_name(f"{output.stem}.prune-manifest.json"),
    )


def _logical_block_pruned_header(
    source_header: SourceModelHeader,
    block_plan: Mapping[str, object],
    mapping: Mapping[int, int | None],
) -> SourceModelHeader:
    tensors = {}
    for source_name, descriptor in source_header.tensors.items():
        output_name = _mapped_tensor_name(source_name, mapping)
        if output_name is None:
            continue
        if output_name in tensors:
            raise ValueError(f"Duplicate output tensor name: {output_name}")
        tensors[output_name] = descriptor
    return SourceModelHeader(
        tensors=tensors,
        metadata=_block_pruning_metadata(
            source_header.metadata,
            block_plan,
            krea2_block_plan_sha256(block_plan),
        ),
    )


def _require_new_outputs(paths: tuple[Path, ...], *, overwrite: bool) -> None:
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Combined output paths must be distinct.")
    if not overwrite:
        for path in paths:
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def _promote_outputs(staged: tuple[tuple[Path, Path], ...], *, overwrite: bool) -> None:
    restores: dict[Path, Path] = {}
    promoted: list[Path] = []
    try:
        if overwrite:
            for _, output in staged:
                if output.exists():
                    restore = output.with_name(f"{output.name}.restore")
                    if restore.exists():
                        raise FileExistsError(f"Restore output already exists: {restore}")
                    output.replace(restore)
                    restores[output] = restore
        for partial, output in staged:
            partial.replace(output)
            promoted.append(output)
    except BaseException:
        for output in promoted:
            if output.exists():
                output.unlink()
        for output, restore in restores.items():
            if restore.exists():
                restore.replace(output)
        raise
    for restore in restores.values():
        if restore.exists():
            restore.unlink()


def _validate_profile_assignments(
    source_path: Path,
    final_path: Path,
    mapping: Mapping[int, int | None],
    source_profile_path: Path,
    remapped_profile_path: Path,
) -> int:
    source_header = read_source_model_header(source_path)
    final_header = read_source_model_header(final_path)
    expected_names = {
        mapped
        for name in source_header.tensors
        if (mapped := _mapped_tensor_name(name, mapping)) is not None
    }
    if set(final_header.tensors) != expected_names:
        raise ValueError("Final model tensor keys do not match the block-prune mapping.")
    original, remapped = load_profile(source_profile_path), load_profile(remapped_profile_path)
    for source_name in source_header.tensors:
        final_name = _mapped_tensor_name(source_name, mapping)
        if final_name is not None and resolve_profile(original, source_name) != resolve_profile(remapped, final_name):
            raise ValueError(f"Profile assignment changed for surviving tensor: {source_name}")
    return len(expected_names)


def apply_krea2_combined_prune(
    model_path: str | Path,
    block_plan_path: str | Path,
    mlp_plan_path: str | Path,
    quant_profile_path: str | Path,
    output_path: str | Path,
    *,
    remapped_mlp_plan_path: str | Path | None = None,
    output_profile_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> Krea2CombinedPruneResult:
    """Apply original-ID block and MLP plans, then remap the quant profile."""
    source, output = Path(model_path), Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {source}.")
    if source.resolve() == output.resolve():
        raise ValueError("Source and output paths cannot be the same.")
    mlp_output, profile_output, manifest_output = _sidecar_paths(
        output, remapped_mlp_plan_path, output_profile_path, manifest_path
    )
    input_paths = {source.resolve(), Path(block_plan_path).resolve(), Path(mlp_plan_path).resolve(), Path(quant_profile_path).resolve()}
    if any(path.resolve() in input_paths for path in (output, mlp_output, profile_output, manifest_output)):
        raise ValueError("Combined outputs must differ from all input artifacts.")
    _require_new_outputs((output, mlp_output, profile_output, manifest_output), overwrite=overwrite)

    block_plan = validate_krea2_block_plan(_load_json(Path(block_plan_path)))
    mapping = krea2_block_old_to_new(block_plan)
    source_header = read_source_model_header(source)
    validate_file_ranges(
        source,
        ((name, tuple(descriptor["data_offsets"])) for name, descriptor in source_header.tensors.items()),
        "Source file",
    )
    _validate_source_blocks(source_header, cast(int, block_plan["block_count"]))
    _validate_pruning_metadata_slots(source_header.metadata)
    original_mlp_plan = load_krea2_mlp_plan(mlp_plan_path)
    # Validate the original topology and width before making any output.
    build_krea2_mlp_plan_from_document(source_header, original_mlp_plan)
    remapped_mlp_plan = remap_krea2_mlp_plan_blocks(original_mlp_plan, mapping)
    logical_header = _logical_block_pruned_header(source_header, block_plan, mapping)
    remapped_mlp_plan_object = build_krea2_mlp_plan_from_document(
        logical_header, remapped_mlp_plan
    )
    original_width = cast(int, original_mlp_plan["original_mlp_width"])
    new_width = cast(int, original_mlp_plan["new_mlp_width"])
    removed_blocks = tuple(cast(list[int], block_plan["removed_blocks"]))
    result = Krea2CombinedPruneResult(
        output, mlp_output, profile_output, manifest_output,
        cast(int, block_plan["block_count"]), removed_blocks,
        original_width, new_width, mapping, 0, 0, 0, dry_run,
    )
    if dry_run:
        remap_result = remap_krea2_block_profile(
            quant_profile_path, block_plan_path, profile_output,
            source_model_path=source, dry_run=True, old_to_new=mapping,
        )
        return Krea2CombinedPruneResult(
            output, mlp_output, profile_output, manifest_output,
            result.original_block_count, removed_blocks, original_width, new_width,
            mapping, remap_result.removed_rules, remap_result.rewritten_rules,
            remap_result.verified_assignments, True,
        )

    mlp_partial = mlp_output.with_name(f"{mlp_output.name}.partial")
    profile_partial = profile_output.with_name(f"{profile_output.name}.partial")
    manifest_partial = manifest_output.with_name(f"{manifest_output.name}.partial")
    final_partial = output.with_name(f"{output.name}.partial")
    temporary_paths = (final_partial, mlp_partial, profile_partial, manifest_partial)
    if any(path.exists() for path in temporary_paths):
        existing = next(path for path in temporary_paths if path.exists())
        raise FileExistsError(f"Temporary output already exists: {existing}")
    try:
        _write_json(mlp_partial, remapped_mlp_plan)
        _write_pruned_checkpoint_core(
            source, logical_header, remapped_mlp_plan_object, final_partial
        )
        remap_result = remap_krea2_block_profile(
            quant_profile_path, block_plan_path, profile_partial, source_model_path=source, old_to_new=mapping,
        )
        verified_assignments = _validate_profile_assignments(
            source, final_partial, mapping, Path(quant_profile_path), profile_partial,
        )
        manifest = {
            "format": "potatoforge_krea2_combined_prune_manifest",
            "version": 1,
            "architecture": "krea2",
            "source_model": source.name,
            "output_model": output.name,
            "block_pruning": {
                "plan_hash": krea2_block_plan_sha256(block_plan),
                "original_block_count": block_plan["block_count"],
                "removed_original_blocks": list(removed_blocks),
                "remaining_block_count": len([item for item in mapping.values() if item is not None]),
                "old_to_new": {str(old): new for old, new in mapping.items()},
            },
            "mlp_pruning": {
                "source_plan": Path(mlp_plan_path).name,
                "source_plan_hash": krea2_mlp_plan_sha256(original_mlp_plan),
                "remapped_plan": mlp_output.name,
                "original_width": original_width,
                "remaining_width": new_width,
                "removed_channels_per_surviving_block": original_mlp_plan["removed_per_block"],
            },
            "quant_profile": {
                "source_profile": Path(quant_profile_path).name,
                "remapped_profile": profile_output.name,
                "removed_rules": remap_result.removed_rules,
                "rewritten_rules": remap_result.rewritten_rules,
                "assignment_validation": "pass",
                "verified_assignments": verified_assignments,
            },
        }
        _write_json(manifest_partial, manifest)
        _promote_outputs(
            ((final_partial, output), (mlp_partial, mlp_output),
             (profile_partial, profile_output), (manifest_partial, manifest_output)),
            overwrite=overwrite,
        )
        return Krea2CombinedPruneResult(
            output, mlp_output, profile_output, manifest_output,
            result.original_block_count, removed_blocks, original_width, new_width,
            mapping, remap_result.removed_rules, remap_result.rewritten_rules,
            verified_assignments, False,
        )
    except BaseException:
        for path in temporary_paths:
            if path.exists():
                path.unlink()
        raise
