from collections.abc import Callable, Mapping
from pathlib import Path
from time import perf_counter
from typing import NamedTuple, TypeAlias

from .headers.source_header import read_source_model_header
from .patch_planning import PatchPlan, build_patch_plan, build_patch_metadata
from .patching import execute_patch_plan
from .planning import (
    TensorDescriptor,
    build_layout_from_specs,
)
from .profiles import QuantizationAction
from .sweep_profiles import (
    SweepGroup,
    SweepProfile,
    load_sweep_profile,
    validate_sweep_group_id,
)


SweepProgressReporter: TypeAlias = Callable[[str], None]


class SweepEntry(NamedTuple):
    group_index: int
    group_id: str
    action: QuantizationAction
    layers: tuple[str, ...]
    patch_id: str
    filename: str
    plan: PatchPlan


class SweepPlan(NamedTuple):
    profile_id: str
    entries: tuple[SweepEntry, ...]


class SweepResult(NamedTuple):
    plan: SweepPlan
    output_dir: Path
    generated_patch_count: int
    total_patch_bytes: int
    elapsed_seconds: float


def _build_group_plan(
    source_header: Mapping[str, TensorDescriptor],
    group: SweepGroup,
    patch_id: str,
) -> PatchPlan:
    layers = tuple(group["layers"])
    if not layers:
        raise ValueError("Sweep group must select at least one source layer.")
    if len(set(layers)) != len(layers):
        raise ValueError("Sweep group lists a layer more than once.")

    layer_plans = [
        build_patch_plan(
            source_header,
            layer,
            group["action"],
            patch_id,
        )
        for layer in layers
    ]
    entries = tuple(
        entry
        for plan in sorted(
            layer_plans,
            key=lambda plan: plan.entries[0].source_data_offsets[0],
        )
        for entry in plan.entries
    )
    return PatchPlan(
        patch_id=patch_id,
        entries=entries,
        layout=build_layout_from_specs(
            spec
            for entry in entries
            for spec in entry.output_tensors
        ),
        selected_tensor_count=len(entries),
        generated_tensor_count=sum(
            len(entry.output_tensors) for entry in entries
        ),
        source_bytes_to_read=sum(
            entry.source_input_bytes for entry in entries
        ),
        replacement_bytes=sum(entry.estimated_bytes for entry in entries),
    )


def plan_patch_sweep(
    source_header: Mapping[str, TensorDescriptor],
    sweep_profile: SweepProfile,
) -> SweepPlan:
    entries: list[SweepEntry] = []
    filenames: set[str] = set()

    for group_index, group in enumerate(sweep_profile["groups"], start=1):
        action = group["action"]
        raw_group_id = group.get("id")
        group_id = (
            f"group-{group_index}"
            if raw_group_id is None
            else validate_sweep_group_id(
                raw_group_id,
                f"Sweep group {group_index} id",
            )
        )
        patch_id = f"{group_id}-{action}"
        filename = f"{patch_id}.safetensors"
        filename_key = filename.casefold()
        if filename_key in filenames:
            raise ValueError(
                f"Sweep groups produce duplicate output filename: {filename}"
            )
        filenames.add(filename_key)

        plan = _build_group_plan(source_header, group, patch_id)
        entries.append(
            SweepEntry(
                group_index=group_index,
                group_id=group_id,
                action=action,
                layers=tuple(group["layers"]),
                patch_id=patch_id,
                filename=filename,
                plan=plan,
            )
        )

    if not entries:
        raise ValueError("Sweep profile selected no source layers.")
    return SweepPlan(sweep_profile["profile_id"], tuple(entries))


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
    source_resolved = source.resolve()
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
    for index, entry in enumerate(plan.entries, start=1):
        if on_progress is not None:
            on_progress(
                f"[{index}/{len(plan.entries)}] "
                f"{entry.group_id} ({len(entry.layers)} layers) -> "
                f"{entry.action}"
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
                f"Failed to generate patch for group {entry.group_id} "
                f"({', '.join(entry.layers)}): {error}"
            ) from error
        generated_bytes += (output / entry.filename).stat().st_size

    return SweepResult(
        plan=plan,
        output_dir=output,
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
