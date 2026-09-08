import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from time import perf_counter
from typing import NamedTuple, TypeAlias

from .headers.source_header import read_source_model_header
from .patch_planning import PatchPlan, build_patch_plan, build_patch_metadata
from .patching import execute_patch_plan
from .planning import TensorDescriptor
from .profiles import QuantizationAction
from .sweep_profiles import SweepProfile, load_sweep_profile


SweepProgressReporter: TypeAlias = Callable[[str], None]


class SweepEntry(NamedTuple):
    layer: str
    family: str
    action: QuantizationAction
    patch_id: str
    filename: str
    source_data_offsets: tuple[int, int]
    plan: PatchPlan


class SweepPlan(NamedTuple):
    profile_id: str
    entries: tuple[SweepEntry, ...]


class SweepResult(NamedTuple):
    plan: SweepPlan
    output_dir: Path
    manifest_path: Path
    generated_patch_count: int
    total_patch_bytes: int
    elapsed_seconds: float


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def plan_patch_sweep(
    source_header: Mapping[str, TensorDescriptor],
    sweep_profile: SweepProfile,
) -> SweepPlan:
    entries: list[SweepEntry] = []
    selected_layers: set[str] = set()
    filenames: set[str] = set()

    for group in sweep_profile["groups"]:
        action = group["action"]
        for layer in group["layers"]:
            if layer in selected_layers:
                raise ValueError(
                    f"Sweep layer {layer} is listed more than once."
                )
            selected_layers.add(layer)
            family = layer.removesuffix(".weight")
            patch_id = (
                f"{sweep_profile['profile_id']}__{family}__{action}"
            )
            filename = f"{_safe_filename(family)}__{action}.safetensors"
            filename_key = filename.casefold()
            if filename_key in filenames:
                raise ValueError(
                    f"Sweep layers produce duplicate output filename: {filename}"
                )
            filenames.add(filename_key)

            plan = build_patch_plan(
                source_header,
                layer,
                action,
                patch_id,
            )
            entries.append(
                SweepEntry(
                    layer=layer,
                    family=family,
                    action=action,
                    patch_id=patch_id,
                    filename=filename,
                    source_data_offsets=plan.entries[0].source_data_offsets,
                    plan=plan,
                )
            )

    if not entries:
        raise ValueError("Sweep profile selected no source layers.")
    return SweepPlan(sweep_profile["profile_id"], tuple(entries))


def _write_manifest(
    path: Path,
    plan: SweepPlan,
    source_path: Path,
) -> None:
    partial = path.with_name(f"{path.name}.partial")
    try:
        with partial.open("x", encoding="utf-8") as file:
            json.dump(
                {
                    "format_version": 1,
                    "profile_id": plan.profile_id,
                    "source": source_path.name,
                    "patches": [
                        {
                            "layer": entry.layer,
                            "family": entry.family,
                            "action": entry.action,
                            "file": entry.filename,
                        }
                        for entry in plan.entries
                    ],
                },
                file,
                indent=2,
            )
            file.write("\n")
        partial.rename(path)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise


def generate_patch_sweep(
    source_path: str | Path,
    output_dir: str | Path,
    sweep_profile: SweepProfile,
    *,
    on_progress: SweepProgressReporter | None = None,
) -> SweepResult:
    source = Path(source_path)
    output = Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {source}")
    if output.exists() and not output.is_dir():
        raise ValueError(f"Sweep output path is not a directory: {output}")

    source_header = read_source_model_header(source)
    plan = plan_patch_sweep(source_header.tensors, sweep_profile)
    manifest = output / "sweep_manifest.json"
    source_resolved = source.resolve()
    if manifest.resolve() == source_resolved:
        raise ValueError("Sweep manifest path cannot be the source checkpoint.")
    if manifest.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {manifest}")
    manifest_partial = manifest.with_name(f"{manifest.name}.partial")
    if manifest_partial.exists():
        raise FileExistsError(f"Partial manifest already exists: {manifest_partial}")
    for entry in plan.entries:
        target = output / entry.filename
        if target.resolve() == source_resolved:
            raise ValueError(
                f"Sweep patch path cannot be the source checkpoint: {target}"
            )
        partial = target.with_name(f"{target.name}.partial")
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing patch: {target}")
        if partial.exists():
            raise FileExistsError(f"Partial patch already exists: {partial}")

    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    generated_bytes = 0
    processing_entries = sorted(
        plan.entries,
        key=lambda entry: entry.source_data_offsets[0],
    )
    for index, entry in enumerate(processing_entries, start=1):
        if on_progress is not None:
            on_progress(
                f"[{index}/{len(processing_entries)}] "
                f"{entry.layer} -> {entry.action}"
            )
        try:
            execute_patch_plan(
                source,
                output / entry.filename,
                entry.plan,
                metadata=build_patch_metadata(
                    entry.plan,
                    source,
                ),
            )
        except Exception as error:
            raise ValueError(
                f"Failed to generate patch for {entry.layer}: {error}"
            ) from error
        generated_bytes += (output / entry.filename).stat().st_size

    _write_manifest(manifest, plan, source)
    return SweepResult(
        plan=plan,
        output_dir=output,
        manifest_path=manifest,
        generated_patch_count=len(plan.entries),
        total_patch_bytes=generated_bytes,
        elapsed_seconds=perf_counter() - started,
    )


def generate_patch_sweep_from_profile(
    source_path: str | Path,
    profile_path: str | Path,
    output_dir: str | Path,
    *,
    on_progress: SweepProgressReporter | None = None,
) -> SweepResult:
    if on_progress is not None:
        on_progress("Resolving sweep profile")
    profile = load_sweep_profile(profile_path)
    if on_progress is not None:
        on_progress(f"Sweep: {profile['profile_id']}")
        on_progress(f"Source: {Path(source_path).name}")
        on_progress(
            "Layers: "
            + str(sum(len(group["layers"]) for group in profile["groups"]))
        )
    return generate_patch_sweep(
        source_path,
        output_dir,
        profile,
        on_progress=on_progress,
    )
