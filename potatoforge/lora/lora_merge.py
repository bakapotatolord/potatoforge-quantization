import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple, TypeAlias, TypedDict

import torch
from safetensors.torch import load_file

from potatoforge.headers.header_reader import read_raw_data_start
from potatoforge.lora.lora_discovery import (
    DiscoveredAdditiveDelta,
    DiscoveredLinearPair,
    inspect_adapter_header,
)
from potatoforge.lora.lora_math import (
    calculate_additive_tensor_delta,
    calculate_linear_lora_delta,
    merge_tensor_contributions,
)
from potatoforge.planning import (
    OutputTensorSpec,
    PlanEntry,
    SafetensorsLayout,
    build_layout_from_specs,
    source_bytes,
)
from potatoforge.safetensors_writer import (
    TensorPayload,
    write_safetensors_file,
)
from potatoforge.headers.source_header import (
    SourceModelHeader,
    read_source_model_header,
)
from potatoforge.source_payloads import (
    SourcePayloadTransform,
    read_source_tensor_bytes,
    tensor_from_raw_bytes,
    tensor_to_raw_bytes,
)


class ResolvedLinearPair(TypedDict):
    down_key: str
    up_key: str
    source_key: str
    alpha_key: str | None


class ResolvedAdditiveDelta(TypedDict):
    key: str
    source_key: str


class MergePlan(TypedDict):
    linear_pairs: list[ResolvedLinearPair]
    additive_deltas: list[ResolvedAdditiveDelta]


class AdapterMergeInput(NamedTuple):
    path: str | Path
    strength: float


class _PreparedAdapter(NamedTuple):
    strength: float
    plan: MergePlan
    tensors: Mapping[str, torch.Tensor]


class _PatchIndex(NamedTuple):
    linear_pairs: dict[str, tuple[ResolvedLinearPair, ...]]
    additive_deltas: dict[str, tuple[ResolvedAdditiveDelta, ...]]


MergeProgressReporter: TypeAlias = Callable[[int, int, str], None]


_FLOAT_DTYPES = frozenset({
    "F16",
    "BF16",
    "F32",
    "F64",
})


_MERGE_SOURCE_DTYPES = frozenset({
    "BF16",
    "F32",
})


def _has_dot_boundary_suffix(
    first: str,
    second: str,
) -> bool:
    # ponytail: prefix-only matching; add a verified rule for real renames.
    return (
        first == second
        or first.endswith(f".{second}")
        or second.endswith(f".{first}")
    )


def _resolve_source_tensor_key(
    source_header: SourceModelHeader,
    adapter_target: str,
    source_suffix: str,
) -> str:
    if not adapter_target:
        raise ValueError("Adapter target must not be empty.")

    candidate_name = adapter_target + source_suffix

    if candidate_name in source_header.tensors:
        return candidate_name

    matches = sorted(
        source_name
        for source_name in source_header.tensors
        if _has_dot_boundary_suffix(source_name, candidate_name)
    )

    if not matches:
        raise ValueError(
            "No source tensor matched adapter target "
            f"{adapter_target!r}. Tried exact target {candidate_name!r} "
            "and unique dot-boundary suffix matching."
        )

    if len(matches) > 1:
        raise ValueError(
            "Ambiguous source tensor match for adapter target "
            f"{adapter_target!r}. Matches: {matches}"
        )

    return matches[0]


def resolve_linear_pair_source_key(
    source_header: SourceModelHeader,
    pair: DiscoveredLinearPair,
) -> str:
    return _resolve_source_tensor_key(
        source_header,
        adapter_target=pair["target"],
        source_suffix=".weight",
    )


def resolve_additive_delta_source_key(
    source_header: SourceModelHeader,
    delta: DiscoveredAdditiveDelta,
) -> str:
    return _resolve_source_tensor_key(
        source_header,
        adapter_target=delta["target"],
        source_suffix="",
    )


def _validate_adapter_dtype(
    adapter_header: SourceModelHeader,
    key: str,
) -> None:
    dtype = adapter_header.tensors[key]["dtype"]

    if dtype not in _FLOAT_DTYPES:
        raise ValueError(
            f"Adapter tensor {key} must use a floating-point dtype; "
            f"got {dtype}."
        )


def _validate_alpha_tensor(
    adapter_header: SourceModelHeader,
    key: str,
) -> None:
    _validate_adapter_dtype(adapter_header, key)

    shape = adapter_header.tensors[key]["shape"]

    if shape not in ([], [1]):
        raise ValueError(
            f"Alpha tensor {key} must be scalar; got shape {shape}."
        )


def _validate_linear_source_tensor(
    source_header: SourceModelHeader,
    source_key: str,
    pair: DiscoveredLinearPair,
) -> None:
    descriptor = source_header.tensors[source_key]
    expected_shape = [
        pair["output_features"],
        pair["input_features"],
    ]

    if descriptor["dtype"] not in _MERGE_SOURCE_DTYPES:
        raise ValueError(
            f"Source tensor {source_key} must use BF16 or F32; "
            f"got {descriptor['dtype']}."
        )

    if descriptor["shape"] != expected_shape:
        raise ValueError(
            f"Source Linear tensor shape does not match adapter factors "
            f"for {source_key}: expected {expected_shape}, "
            f"got {descriptor['shape']}."
        )


def _validate_additive_source_tensor(
    source_header: SourceModelHeader,
    source_key: str,
    delta: DiscoveredAdditiveDelta,
) -> None:
    descriptor = source_header.tensors[source_key]

    if descriptor["dtype"] not in _MERGE_SOURCE_DTYPES:
        raise ValueError(
            f"Source tensor {source_key} must use BF16 or F32; "
            f"got {descriptor['dtype']}."
        )

    if descriptor["shape"] != delta["shape"]:
        raise ValueError(
            f"Additive delta shape does not match source tensor "
            f"{source_key}: expected {descriptor['shape']}, "
            f"got {delta['shape']}."
        )


def build_merge_plan(
    source_header: SourceModelHeader,
    adapter_header: SourceModelHeader,
) -> MergePlan:
    inspection = inspect_adapter_header(adapter_header)

    invalid_records = [
        record
        for record in inspection["tensors"]
        if record["kind"] in {
            "unpaired_down",
            "unpaired_up",
            "unsupported",
        }
    ]

    if invalid_records:
        invalid_keys = [record["key"] for record in invalid_records]
        unsupported_contracts = sorted({
            record["contract"]
            for record in invalid_records
            if record["contract"] not in {"linear_lora", "unsupported"}
        })
        message = (
            f"Adapter contains unsupported or unpaired tensors: {invalid_keys}."
        )

        if unsupported_contracts:
            message += f" Unsupported contracts: {unsupported_contracts}."

        raise ValueError(message)

    alpha_keys_by_target: dict[str, str] = {}

    for record in inspection["tensors"]:
        if record["kind"] != "alpha":
            continue

        target = record["target"]

        if target is None:
            raise ValueError(
                f"Alpha tensor {record['key']} has no target."
            )

        if target in alpha_keys_by_target:
            raise ValueError(
                f"Multiple alpha tensors target {target!r}."
            )

        _validate_alpha_tensor(adapter_header, record["key"])
        alpha_keys_by_target[target] = record["key"]

    pair_targets = {
        pair["target"]
        for pair in inspection["pairs"]
    }
    orphan_alpha_targets = set(alpha_keys_by_target) - pair_targets

    if orphan_alpha_targets:
        raise ValueError(
            "Alpha tensors have no matching linear pairs: "
            f"{sorted(orphan_alpha_targets)}"
        )

    resolved_pairs: list[ResolvedLinearPair] = []
    seen_pair_targets: set[str] = set()

    for pair in inspection["pairs"]:
        target = pair["target"]

        if target in seen_pair_targets:
            raise ValueError(
                f"Multiple linear pairs target {target!r}."
            )

        seen_pair_targets.add(target)
        _validate_adapter_dtype(adapter_header, pair["down_key"])
        _validate_adapter_dtype(adapter_header, pair["up_key"])

        source_key = resolve_linear_pair_source_key(
            source_header,
            pair,
        )
        _validate_linear_source_tensor(
            source_header,
            source_key,
            pair,
        )

        resolved_pairs.append(
            ResolvedLinearPair(
                down_key=pair["down_key"],
                up_key=pair["up_key"],
                source_key=source_key,
                alpha_key=alpha_keys_by_target.get(target),
            )
        )

    resolved_deltas: list[ResolvedAdditiveDelta] = []

    for delta in inspection["additive_deltas"]:
        _validate_adapter_dtype(adapter_header, delta["key"])

        source_key = resolve_additive_delta_source_key(
            source_header,
            delta,
        )
        _validate_additive_source_tensor(
            source_header,
            source_key,
            delta,
        )

        resolved_deltas.append(
            ResolvedAdditiveDelta(
                key=delta["key"],
                source_key=source_key,
            )
        )

    return MergePlan(
        linear_pairs=resolved_pairs,
        additive_deltas=resolved_deltas,
    )


def _validate_file_layout(
    file_path: Path,
    header: SourceModelHeader,
) -> int:
    with file_path.open("rb") as file:
        raw_data_start = read_raw_data_start(
            file,
            file_label=str(file_path),
        )
    file_size = file_path.stat().st_size

    if raw_data_start > file_size:
        raise ValueError(
            f"{file_path} ends before its safetensors header."
        )

    ranges = sorted(
        (
            descriptor["data_offsets"][0],
            descriptor["data_offsets"][1],
            tensor_name,
        )
        for tensor_name, descriptor in header.tensors.items()
    )

    previous_end = 0

    for start, end, tensor_name in ranges:
        if start < previous_end:
            raise ValueError(
                f"Overlapping payload offsets for {tensor_name}."
            )

        if raw_data_start + end > file_size:
            raise ValueError(
                f"Payload for {tensor_name} extends past the end of "
                f"{file_path}."
            )

        previous_end = max(previous_end, end)

    return raw_data_start


def _build_patch_index(plan: MergePlan) -> _PatchIndex:
    linear_pairs: dict[str, list[ResolvedLinearPair]] = {}
    additive_deltas: dict[str, list[ResolvedAdditiveDelta]] = {}

    for pair in plan["linear_pairs"]:
        linear_pairs.setdefault(pair["source_key"], []).append(pair)

    for delta in plan["additive_deltas"]:
        additive_deltas.setdefault(delta["source_key"], []).append(delta)

    return _PatchIndex(
        linear_pairs={
            source_key: tuple(pairs)
            for source_key, pairs in linear_pairs.items()
        },
        additive_deltas={
            source_key: tuple(deltas)
            for source_key, deltas in additive_deltas.items()
        },
    )


def _merge_source_payload(
    source_key: str,
    source_payload: bytes,
    shape: Sequence[int],
    dtype: str,
    prepared_adapters: Sequence[_PreparedAdapter],
    patch_indexes: Sequence[_PatchIndex],
) -> bytes:
    contributions: list[torch.Tensor] = []

    for prepared_adapter, patch_index in zip(
        prepared_adapters,
        patch_indexes,
    ):
        contributions.extend(
            _iter_adapter_contributions(
                prepared_adapter,
                patch_index,
                source_key,
            )
        )

    if not contributions:
        return source_payload

    source_tensor = tensor_from_raw_bytes(
        source_payload,
        shape,
        dtype,
        tensor_name=source_key,
    )
    merged_tensor = merge_tensor_contributions(
        source_tensor,
        contributions,
    )
    return tensor_to_raw_bytes(merged_tensor)


def build_adapter_merger(
    source_header: SourceModelHeader,
    adapter_inputs: Sequence[AdapterMergeInput],
) -> SourcePayloadTransform | None:
    prepared_adapters = tuple(
        _prepare_adapter(source_header, adapter_input)
        for adapter_input in adapter_inputs
    )
    patch_indexes = tuple(
        _build_patch_index(adapter.plan)
        for adapter in prepared_adapters
    )

    if not prepared_adapters:
        return None

    def merge_source_payload(
        entry: PlanEntry,
        source_payload: bytes,
    ) -> bytes:
        return _merge_source_payload(
            entry["tensor_name"],
            source_payload,
            entry["shape"],
            entry["source_dtype"],
            prepared_adapters,
            patch_indexes,
        )

    return merge_source_payload


def _iter_adapter_contributions(
    prepared_adapter: _PreparedAdapter,
    patch_index: _PatchIndex,
    source_key: str,
) -> Iterator[torch.Tensor]:
    for pair in patch_index.linear_pairs.get(source_key, ()):
        down = prepared_adapter.tensors[pair["down_key"]]
        up = prepared_adapter.tensors[pair["up_key"]]

        alpha = None

        if pair["alpha_key"] is not None:
            alpha = float(
                prepared_adapter.tensors[pair["alpha_key"]].item()
            )
            if not math.isfinite(alpha):
                raise ValueError(
                    f"Alpha tensor {pair['alpha_key']} must contain a finite value."
                )

        yield calculate_linear_lora_delta(
            down=down,
            up=up,
            strength=prepared_adapter.strength,
            alpha=alpha,
        )

    for delta in patch_index.additive_deltas.get(source_key, ()):
        yield calculate_additive_tensor_delta(
            delta=prepared_adapter.tensors[delta["key"]],
            strength=prepared_adapter.strength,
        )


def build_merge_output_layout(
    source_header: SourceModelHeader,
) -> SafetensorsLayout:
    return build_layout_from_specs(
        OutputTensorSpec(
            name=tensor_name,
            dtype=descriptor["dtype"],
            shape=tuple(descriptor["shape"]),
            byte_count=source_bytes(descriptor),
        )
        for tensor_name, descriptor in source_header.tensors.items()
    )


def _stream_merged_payloads(
    source_path: Path,
    source_header: SourceModelHeader,
    source_raw_data_start: int,
    prepared_adapters: Sequence[_PreparedAdapter],
    on_tensor_started: MergeProgressReporter | None,
) -> Iterator[TensorPayload]:
    patch_indexes = tuple(
        _build_patch_index(adapter.plan)
        for adapter in prepared_adapters
    )

    with source_path.open("rb") as source_file:
        tensor_count = len(source_header.tensors)

        for tensor_index, (source_key, descriptor) in enumerate(
            source_header.tensors.items(),
            start=1,
        ):
            if on_tensor_started is not None:
                on_tensor_started(
                    tensor_index,
                    tensor_count,
                    source_key,
                )

            start, end = descriptor["data_offsets"]
            source_payload = read_source_tensor_bytes(
                source_file,
                source_raw_data_start,
                (start, end),
                end - start,
            )
            yield source_key, _merge_source_payload(
                source_key,
                source_payload,
                descriptor["shape"],
                descriptor["dtype"],
                prepared_adapters,
                patch_indexes,
            )


def _prepare_adapter(
    source_header: SourceModelHeader,
    adapter_input: AdapterMergeInput,
) -> _PreparedAdapter:
    adapter_path = Path(adapter_input.path)

    try:
        strength = float(adapter_input.strength)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Adapter strength must be numeric: {adapter_input.strength!r}."
        ) from error

    if not math.isfinite(strength):
        raise ValueError(
            f"Adapter strength must be finite: {strength!r}."
        )

    if not adapter_path.is_file():
        raise FileNotFoundError(
            f"Adapter file does not exist: {adapter_path}"
        )

    adapter_header = read_source_model_header(adapter_path)
    plan = build_merge_plan(
        source_header,
        adapter_header,
    )

    if not plan["linear_pairs"] and not plan["additive_deltas"]:
        raise ValueError(
            f"Adapter {adapter_path} contains no supported patches."
        )

    return _PreparedAdapter(
        strength=strength,
        plan=plan,
        tensors=load_file(str(adapter_path), device="cpu"),
    )


def merge_bf16_adapters(
    source_path: str | Path,
    output_path: str | Path,
    adapters: Sequence[AdapterMergeInput],
    on_tensor_started: MergeProgressReporter | None = None,
) -> None:
    source = Path(source_path)
    output = Path(output_path)
    adapter_inputs = tuple(adapters)

    if not adapter_inputs:
        raise ValueError("At least one adapter is required.")

    if source.resolve() == output.resolve():
        raise ValueError("Source and output paths must be different.")

    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output}"
        )

    partial = output.with_name(f"{output.name}.partial")

    if partial.exists():
        raise FileExistsError(
            f"Partial output already exists: {partial}"
        )

    if not source.is_file():
        raise FileNotFoundError(
            f"Source file does not exist: {source}"
        )

    source_resolved = source.resolve()
    output_resolved = output.resolve()
    partial_resolved = partial.resolve()

    for adapter_input in adapter_inputs:
        adapter_resolved = Path(adapter_input.path).resolve()

        if adapter_resolved == source_resolved:
            raise ValueError(
                "Source and adapter paths must be different."
            )

        if adapter_resolved in {
            output_resolved,
            partial_resolved,
        }:
            raise ValueError(
                "Adapter path must be different from output paths."
            )

    source_header = read_source_model_header(source)

    if not source_header.tensors:
        raise ValueError("Source checkpoint contains no tensors.")

    source_raw_data_start = _validate_file_layout(
        source,
        source_header,
    )
    prepared_adapters = tuple(
        _prepare_adapter(source_header, adapter_input)
        for adapter_input in adapter_inputs
    )
    layout = build_merge_output_layout(source_header)

    write_safetensors_file(
        partial,
        layout,
        _stream_merged_payloads(
            source,
            source_header,
            source_raw_data_start,
            prepared_adapters,
            on_tensor_started,
        ),
        metadata=source_header.metadata or None,
    )
    partial.rename(output)
