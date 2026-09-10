"""Generate one runtime-compatible profile from a measurement-only audit."""

import json
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from math import fsum, isclose, isfinite, sqrt
from multiprocessing import get_context
from pathlib import Path
from typing import Final, NamedTuple, cast

from ..headers.source_header import SourceModelHeader, read_source_model_header
from ..planning import (
    build_output_layout,
    build_plan,
    build_quantization_metadata,
)
from ..profiles import (
    ProfileRule,
    QuantizationAction,
    QuantizationProfile,
)
from ..safetensors_writer import encode_safetensors_header
from .weight_audit import (
    QuantizationMethod,
    WeightAuditDocument,
    WeightAuditRecord,
    select_auditable_bf16_weights,
)


_METHOD_ACTIONS: Final[dict[QuantizationMethod, QuantizationAction]] = {
    "bf16": "keep",
    "int8": "int8",
    "int6": "int6_rowwise",
    "int8_convrot": "int8_convrot",
    "int6_convrot": "int6_convrot",
    "convrot_w4a4": "convrot_w4a4",
    "convrot_w4a4_mse": "convrot_w4a4_mse",
}

MEASURED_METHODS: Final[frozenset[QuantizationMethod]] = frozenset(
    _METHOD_ACTIONS,
)
SUPPORTED_METHODS: Final[frozenset[QuantizationMethod]] = frozenset(
    (
        "bf16",
        "int8",
        "int8_convrot",
        "convrot_w4a4",
        "convrot_w4a4_mse",
    ),
)

_AUDIT_FORMAT_VERSION: Final[int] = 5
_KNAPSACK_BUDGET_UNIT_BYTES: Final[int] = 1 << 20


class MethodChoice(NamedTuple):
    method: QuantizationMethod
    action: QuantizationAction
    storage_bytes: int
    relative_l2_error: float
    reconstruction_sse: float


class OptimizedProfile(NamedTuple):
    profile: QuantizationProfile
    output_bytes: int
    target_bytes: int
    reconstruction_sse: float


class _KnapsackFrontier(NamedTuple):
    scores: dict[int, float]
    storage_bytes: dict[int, int]
    backpointers: tuple[dict[int, tuple[int, int]], ...]


def _audit_version_error() -> ValueError:
    return ValueError(
        "Profile optimization requires a format_version 5 audit. "
        "Regenerate the weight audit to include tensor L2 energy."
    )


def _require_current_audit(audit: object) -> None:
    if (
        not isinstance(audit, dict)
        or audit.get("format_version") != _AUDIT_FORMAT_VERSION
    ):
        raise _audit_version_error()


def load_weight_audit(path: str | Path) -> WeightAuditDocument:
    """Load the current measurement-only audit report."""
    report_path = Path(path)
    try:
        with report_path.open(encoding="utf-8") as report_file:
            document = json.load(report_file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid weight audit: {report_path}") from error

    _require_current_audit(document)
    if not isinstance(document.get("source_path"), str):
        raise ValueError("Weight audit source_path must be a string.")
    if not isinstance(document.get("results"), list):
        raise ValueError("Weight audit results must be a list.")

    audit = cast(WeightAuditDocument, document)
    _validated_results(audit)
    return audit


def _profile_for_assignments(
    assignments: dict[str, MethodChoice],
    profile_id: str,
) -> QuantizationProfile:
    rules: tuple[ProfileRule, ...] = tuple(
        {
            "action": choice.action,
            "prefix": tensor_name,
            "suffixes": ("",),
        }
        for tensor_name, choice in assignments.items()
        if choice.action != "keep"
    )

    return {
        "profile_id": profile_id,
        "description": "Generated from a weight audit by target-size optimization.",
        "default": "keep",
        "rules": rules,
    }


def estimate_profile_bytes(
    source_header: SourceModelHeader,
    profile: QuantizationProfile,
) -> int:
    """Return the exact output-file size the streaming converter will write."""
    plan_entries = build_plan(source_header.tensors, profile)
    layout = build_output_layout(plan_entries)
    header = encode_safetensors_header(
        layout,
        {
            **source_header.metadata,
            **build_quantization_metadata(plan_entries),
        },
    )
    return 8 + len(header) + layout.raw_data_bytes


def _method_choices(
    result: WeightAuditRecord,
    allowed_methods: frozenset[QuantizationMethod],
    max_relative_l2_error: float | None,
) -> tuple[MethodChoice, ...]:
    choices: list[MethodChoice] = []
    for method, action in _METHOD_ACTIONS.items():
        if method not in allowed_methods:
            continue

        measurement = result["methods"].get(method)
        if measurement is None:
            continue
        storage_bytes = measurement["storage_bytes"]
        relative_l2_error = measurement["relative_l2_error"]
        if (
            not isinstance(storage_bytes, int)
            or storage_bytes < 0
            or not isinstance(relative_l2_error, (int, float))
            or not isfinite(relative_l2_error)
            or relative_l2_error < 0
        ):
            continue
        if (
            max_relative_l2_error is not None
            and relative_l2_error > max_relative_l2_error
        ):
            continue
        reconstruction_sse = float(
            relative_l2_error**2 * result["weight_l2_sq"]
        )
        if not isfinite(reconstruction_sse):
            raise ValueError(
                f"Weight audit has an invalid reconstruction SSE for "
                f"{result.get('tensor_name')}."
            )
        choices.append(
            MethodChoice(
                method,
                action,
                storage_bytes,
                float(relative_l2_error),
                reconstruction_sse,
            )
        )

    unique_choices: dict[tuple[int, float], MethodChoice] = {}
    for choice in choices:
        key = (choice.storage_bytes, choice.relative_l2_error)
        existing = unique_choices.get(key)
        if existing is None or choice.method < existing.method:
            unique_choices[key] = choice

    non_dominated = [
        candidate
        for candidate in unique_choices.values()
        if not any(
            other.storage_bytes <= candidate.storage_bytes
            and other.relative_l2_error <= candidate.relative_l2_error
            and (
                other.storage_bytes < candidate.storage_bytes
                or other.relative_l2_error < candidate.relative_l2_error
            )
            for other in unique_choices.values()
        )
    ]
    if not non_dominated:
        raise ValueError(
            f"No allowed measurement choices for {result.get('tensor_name')}."
        )
    bf16_choice = next(
        (choice for choice in choices if choice.method == "bf16"),
        None,
    )
    if (
        bf16_choice is not None
        and all(choice.method != "bf16" for choice in non_dominated)
    ):
        non_dominated.append(bf16_choice)
    return tuple(
        sorted(
            non_dominated,
            key=lambda choice: (
                choice.storage_bytes,
                choice.relative_l2_error,
                choice.method,
            ),
        )
    )


def _choices_by_tensor(
    audit: WeightAuditDocument,
    allowed_methods: frozenset[QuantizationMethod],
    max_relative_l2_error: float | None,
    excluded_prefixes: tuple[str, ...],
    excluded_suffixes: tuple[str, ...],
) -> dict[str, tuple[MethodChoice, ...]]:
    return {
        result["tensor_name"]: _method_choices(
            result,
            (
                frozenset(("bf16",))
                if result["tensor_name"].startswith(excluded_prefixes)
                or result["tensor_name"].endswith(excluded_suffixes)
                else allowed_methods
            ),
            max_relative_l2_error,
        )
        for result in sorted(
            _validated_results(audit),
            key=lambda item: item["tensor_name"],
        )
    }


def _validate_optimizer_inputs(
    profile_id: str,
    allowed_methods: frozenset[QuantizationMethod],
    max_relative_l2_error: float | None,
    excluded_prefixes: tuple[str, ...],
    excluded_suffixes: tuple[str, ...],
) -> frozenset[QuantizationMethod]:
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError("profile_id must be non-empty.")
    if not allowed_methods:
        raise ValueError("allowed_methods must not be empty.")
    unknown_methods = allowed_methods - MEASURED_METHODS
    if unknown_methods:
        raise ValueError("allowed_methods contains unsupported methods.")
    if (
        max_relative_l2_error is not None
        and (
            not isinstance(max_relative_l2_error, (int, float))
            or isinstance(max_relative_l2_error, bool)
            or not isfinite(max_relative_l2_error)
            or max_relative_l2_error < 0
        )
    ):
        raise ValueError("max_relative_l2_error must be finite and non-negative.")
    if any(
        not isinstance(prefix, str) or not prefix
        for prefix in excluded_prefixes
    ):
        raise ValueError("excluded_prefixes must contain non-empty strings.")
    if any(
        not isinstance(suffix, str) or not suffix
        for suffix in excluded_suffixes
    ):
        raise ValueError("excluded_suffixes must contain non-empty strings.")
    return allowed_methods | frozenset(("bf16",))


def next_method_choice(
    result: WeightAuditRecord,
    current_method: QuantizationMethod,
    allowed_methods: frozenset[QuantizationMethod] = SUPPORTED_METHODS,
    max_relative_l2_error: float | None = None,
) -> MethodChoice | None:
    """Return the next non-dominated upgrade for one selected tensor method."""
    choices = _method_choices(
        result,
        allowed_methods | frozenset(("bf16",)),
        max_relative_l2_error,
    )
    for index, choice in enumerate(choices):
        if choice.method == current_method:
            return choices[index + 1] if index + 1 < len(choices) else None
    return None


def _validated_results(
    audit: WeightAuditDocument,
) -> tuple[WeightAuditRecord, ...]:
    results: list[WeightAuditRecord] = []
    tensor_names: set[str] = set()
    for result in audit["results"]:
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("tensor_name"), str)
            or not isinstance(result.get("shape"), list)
            or any(type(dimension) is not int for dimension in result["shape"])
            or not isinstance(result.get("methods"), dict)
        ):
            raise ValueError("Weight audit contains an invalid measurement result.")
        weight_l2_sq = result.get("weight_l2_sq")
        if (
            not isinstance(weight_l2_sq, (int, float))
            or isinstance(weight_l2_sq, bool)
            or not isfinite(weight_l2_sq)
            or weight_l2_sq < 0
        ):
            raise ValueError("Weight audit contains an invalid weight_l2_sq.")
        if result["tensor_name"] in tensor_names:
            raise ValueError("Weight audit contains duplicate tensor measurements.")
        if frozenset(result["methods"]) != MEASURED_METHODS:
            raise ValueError(
                "Weight audit contains unsupported or incomplete method records."
            )
        for method in MEASURED_METHODS:
            measurement = result["methods"].get(method)
            if not isinstance(measurement, dict):
                raise ValueError("Weight audit contains an invalid method record.")
            storage_bytes = measurement.get("storage_bytes")
            relative_l2_error = measurement.get("relative_l2_error")
            if storage_bytes is None and relative_l2_error is None:
                continue
            if (
                type(storage_bytes) is not int
                or storage_bytes < 0
                or not isinstance(relative_l2_error, (int, float))
                or isinstance(relative_l2_error, bool)
                or not isfinite(relative_l2_error)
                or relative_l2_error < 0
            ):
                raise ValueError("Weight audit contains an invalid method record.")
        tensor_names.add(result["tensor_name"])
        results.append(cast(WeightAuditRecord, result))
    return tuple(results)


def method_for_action(action: QuantizationAction) -> QuantizationMethod:
    for method, mapped_action in _METHOD_ACTIONS.items():
        if mapped_action == action:
            return method
    raise ValueError(f"Unsupported quantization action: {action}.")


def validate_weight_audit_against_source(
    audit: WeightAuditDocument,
) -> SourceModelHeader:
    _require_current_audit(audit)
    if not isinstance(audit.get("source_path"), str):
        raise ValueError("Weight audit source_path must be a string.")

    source_header = read_source_model_header(audit["source_path"])
    results = _validated_results(audit)
    if any(
        result["tensor_name"] not in source_header.tensors
        for result in results
    ):
        raise ValueError("Weight audit does not match the source checkpoint header.")

    auditable_names = {
        tensor_name
        for tensor_name, _ in select_auditable_bf16_weights(
            source_header.tensors,
        )
    }
    if {result["tensor_name"] for result in results} != auditable_names:
        raise ValueError("Weight audit does not cover the source checkpoint.")
    if any(
        result["shape"] != source_header.tensors[result["tensor_name"]]["shape"]
        for result in results
    ):
        raise ValueError("Weight audit does not match the source checkpoint shapes.")

    return source_header


def _tensor_names(
    choices_by_tensor: dict[str, tuple[MethodChoice, ...]],
) -> tuple[str, ...]:
    return tuple(sorted(choices_by_tensor))


def _profile_for_indices(
    choices_by_tensor: dict[str, tuple[MethodChoice, ...]],
    indices: dict[str, int],
    profile_id: str,
) -> QuantizationProfile:
    return _profile_for_assignments(
        {
            tensor_name: choices_by_tensor[tensor_name][indices[tensor_name]]
            for tensor_name in _tensor_names(choices_by_tensor)
        },
        profile_id,
    )


def _assignment_sse(
    choices_by_tensor: dict[str, tuple[MethodChoice, ...]],
    indices: dict[str, int],
) -> float:
    return fsum(
        choices_by_tensor[tensor_name][indices[tensor_name]].reconstruction_sse
        for tensor_name in _tensor_names(choices_by_tensor)
    )


def _bf16_indices(
    choices_by_tensor: dict[str, tuple[MethodChoice, ...]],
) -> dict[str, int]:
    indices: dict[str, int] = {}
    for tensor_name in _tensor_names(choices_by_tensor):
        for index, choice in enumerate(choices_by_tensor[tensor_name]):
            if choice.method == "bf16":
                indices[tensor_name] = index
                break
        else:
            raise ValueError(f"No BF16 fallback choice for {tensor_name}.")
    return indices


def _maximum_indices(
    choices_by_tensor: dict[str, tuple[MethodChoice, ...]],
) -> dict[str, int]:
    """Return the highest-storage choice, including binary scored choices."""
    indices: dict[str, int] = {}
    for tensor_name in _tensor_names(choices_by_tensor):
        choices = choices_by_tensor[tensor_name]
        for index, choice in enumerate(choices):
            if choice.method == "bf16":
                indices[tensor_name] = index
                break
        else:
            indices[tensor_name] = len(choices) - 1
    return indices


def _better_state(
    candidate_score: float,
    candidate_storage: int,
    current_score: float,
    current_storage: int,
) -> bool:
    if not isclose(
        candidate_score,
        current_score,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        return candidate_score > current_score
    return candidate_storage < current_storage


def _build_knapsack_frontier(
    choices_by_tensor: dict[str, tuple[MethodChoice, ...]],
    budget_units: int,
) -> _KnapsackFrontier:
    if type(budget_units) is not int or budget_units < 0:
        raise ValueError("Knapsack budget must be a non-negative integer.")

    current_scores = {0: 0.0}
    current_storage = {0: 0}
    backpointers: list[dict[int, tuple[int, int]]] = []

    for tensor_name in _tensor_names(choices_by_tensor):
        choices = choices_by_tensor[tensor_name]
        if not choices:
            raise ValueError(f"No allowed measurement choices for {tensor_name}.")
        base_choice = choices[0]
        options = []
        for choice_index, choice in enumerate(choices):
            extra_bytes = choice.storage_bytes - base_choice.storage_bytes
            if extra_bytes < 0:
                raise ValueError(
                    f"Choices for {tensor_name} are not ordered by storage."
                )
            options.append(
                (
                    choice_index,
                    (extra_bytes + _KNAPSACK_BUDGET_UNIT_BYTES - 1)
                    // _KNAPSACK_BUDGET_UNIT_BYTES,
                    extra_bytes,
                    base_choice.reconstruction_sse
                    - choice.reconstruction_sse,
                )
            )

        next_scores: dict[int, float] = {}
        next_storage: dict[int, int] = {}
        layer_backpointers: dict[int, tuple[int, int]] = {}
        for used_units in sorted(current_scores):
            previous_score = current_scores[used_units]
            previous_storage = current_storage[used_units]
            for choice_index, cost_units, extra_bytes, sse_reduction in options:
                next_units = used_units + cost_units
                if next_units > budget_units:
                    continue
                candidate_score = previous_score + sse_reduction
                candidate_storage = previous_storage + extra_bytes
                existing_score = next_scores.get(next_units)
                if existing_score is not None and not _better_state(
                    candidate_score,
                    candidate_storage,
                    existing_score,
                    next_storage[next_units],
                ):
                    continue
                next_scores[next_units] = candidate_score
                next_storage[next_units] = candidate_storage
                layer_backpointers[next_units] = (used_units, choice_index)

        current_scores = next_scores
        current_storage = next_storage
        backpointers.append(layer_backpointers)

    return _KnapsackFrontier(
        current_scores,
        current_storage,
        tuple(backpointers),
    )


def _best_frontier_state(
    frontier: _KnapsackFrontier,
    budget_units: int,
) -> int:
    reachable = [
        used_units
        for used_units in frontier.scores
        if used_units <= budget_units
    ]
    if not reachable:
        raise ValueError("No feasible multiple-choice knapsack assignment.")

    best_units = min(reachable)
    for used_units in sorted(reachable):
        if _better_state(
            frontier.scores[used_units],
            frontier.storage_bytes[used_units],
            frontier.scores[best_units],
            frontier.storage_bytes[best_units],
        ):
            best_units = used_units
    return best_units


def _indices_from_frontier(
    choices_by_tensor: dict[str, tuple[MethodChoice, ...]],
    frontier: _KnapsackFrontier,
    budget_units: int,
) -> dict[str, int]:
    tensor_names = _tensor_names(choices_by_tensor)
    state_units = _best_frontier_state(frontier, budget_units)
    indices: dict[str, int] = {}
    for layer_index in range(len(tensor_names) - 1, -1, -1):
        previous_units, choice_index = frontier.backpointers[layer_index][
            state_units
        ]
        indices[tensor_names[layer_index]] = choice_index
        state_units = previous_units
    return indices


def _refine_exact_size(
    source_header: SourceModelHeader,
    profile_id: str,
    target_bytes: int,
    choices_by_tensor: dict[str, tuple[MethodChoice, ...]],
    indices: dict[str, int],
) -> tuple[dict[str, int], int]:
    profile = _profile_for_indices(choices_by_tensor, indices, profile_id)
    output_bytes = estimate_profile_bytes(source_header, profile)

    while output_bytes > target_bytes:
        candidates: list[tuple[float, float, int, str, int]] = []
        for tensor_name in _tensor_names(choices_by_tensor):
            current_index = indices[tensor_name]
            current_choice = choices_by_tensor[tensor_name][current_index]
            for new_index in range(current_index):
                candidate_indices = dict(indices)
                candidate_indices[tensor_name] = new_index
                candidate_output_bytes = estimate_profile_bytes(
                    source_header,
                    _profile_for_indices(
                        choices_by_tensor,
                        candidate_indices,
                        profile_id,
                    ),
                )
                recovered_bytes = output_bytes - candidate_output_bytes
                if recovered_bytes <= 0:
                    continue
                candidate_choice = choices_by_tensor[tensor_name][new_index]
                penalty = (
                    candidate_choice.reconstruction_sse
                    - current_choice.reconstruction_sse
                )
                candidates.append(
                    (
                        penalty / recovered_bytes,
                        penalty,
                        -recovered_bytes,
                        tensor_name,
                        new_index,
                    )
                )

        if not candidates:
            raise ValueError(
                "Unable to repair an over-target profile within the exact size."
            )

        _, _, _, tensor_name, new_index = min(candidates)
        indices[tensor_name] = new_index
        profile = _profile_for_indices(choices_by_tensor, indices, profile_id)
        output_bytes = estimate_profile_bytes(source_header, profile)

    while True:
        candidates: list[tuple[float, float, int, str, int, int]] = []
        for tensor_name in _tensor_names(choices_by_tensor):
            current_index = indices[tensor_name]
            if current_index + 1 >= len(choices_by_tensor[tensor_name]):
                continue
            candidate_indices = dict(indices)
            candidate_indices[tensor_name] = current_index + 1
            candidate_output_bytes = estimate_profile_bytes(
                source_header,
                _profile_for_indices(
                    choices_by_tensor,
                    candidate_indices,
                    profile_id,
                ),
            )
            if candidate_output_bytes > target_bytes:
                continue
            current_choice = choices_by_tensor[tensor_name][current_index]
            candidate_choice = choices_by_tensor[tensor_name][current_index + 1]
            sse_reduction = (
                current_choice.reconstruction_sse
                - candidate_choice.reconstruction_sse
            )
            if sse_reduction <= 0:
                continue
            extra_bytes = candidate_output_bytes - output_bytes
            candidates.append(
                (
                    -(sse_reduction / max(extra_bytes, 1)),
                    -sse_reduction,
                    candidate_output_bytes,
                    tensor_name,
                    current_index + 1,
                    extra_bytes,
                )
            )

        if not candidates:
            break

        _, _, candidate_output_bytes, tensor_name, new_index, _ = min(candidates)
        indices[tensor_name] = new_index
        output_bytes = candidate_output_bytes

    return indices, output_bytes


def _profile_bounds(
    source_header: SourceModelHeader,
    profile_id: str,
    choices_by_tensor: dict[str, tuple[MethodChoice, ...]],
) -> tuple[int, int]:
    base_indices = {
        tensor_name: 0
        for tensor_name in _tensor_names(choices_by_tensor)
    }
    minimum_output_bytes = estimate_profile_bytes(
        source_header,
        _profile_for_indices(choices_by_tensor, base_indices, profile_id),
    )
    maximum_output_bytes = estimate_profile_bytes(
        source_header,
        _profile_for_indices(
            choices_by_tensor,
            _maximum_indices(choices_by_tensor),
            profile_id,
        ),
    )
    return minimum_output_bytes, max(maximum_output_bytes, minimum_output_bytes)


def _optimized_from_frontier(
    source_header: SourceModelHeader,
    profile_id: str,
    target_bytes: int,
    choices_by_tensor: dict[str, tuple[MethodChoice, ...]],
    minimum_output_bytes: int,
    maximum_output_bytes: int,
    frontier: _KnapsackFrontier,
) -> OptimizedProfile:
    if target_bytes >= maximum_output_bytes:
        indices = _maximum_indices(choices_by_tensor)
    else:
        remaining_budget_bytes = target_bytes - minimum_output_bytes
        indices = _indices_from_frontier(
            choices_by_tensor,
            frontier,
            remaining_budget_bytes // _KNAPSACK_BUDGET_UNIT_BYTES,
        )

    return _optimized_from_indices(
        source_header,
        profile_id,
        target_bytes,
        choices_by_tensor,
        indices,
    )


def _optimized_from_indices(
    source_header: SourceModelHeader,
    profile_id: str,
    target_bytes: int,
    choices_by_tensor: dict[str, tuple[MethodChoice, ...]],
    indices: dict[str, int],
) -> OptimizedProfile:
    indices, output_bytes = _refine_exact_size(
        source_header,
        profile_id,
        target_bytes,
        choices_by_tensor,
        indices,
    )
    if output_bytes > target_bytes:
        raise ValueError(
            "Generated profile exceeds the requested exact output size."
        )
    return OptimizedProfile(
        profile=_profile_for_indices(choices_by_tensor, indices, profile_id),
        output_bytes=output_bytes,
        target_bytes=target_bytes,
        reconstruction_sse=_assignment_sse(choices_by_tensor, indices),
    )


def _optimize_profile_target_worker(
    task: tuple[
        SourceModelHeader,
        str,
        int,
        dict[str, tuple[MethodChoice, ...]],
        dict[str, int],
    ],
) -> OptimizedProfile:
    return _optimized_from_indices(
        task[0],
        task[1],
        task[2],
        task[3],
        task[4],
    )


def _optimize_target_size_from_choices(
    source_header: SourceModelHeader,
    profile_id: str,
    target_bytes: int,
    choices_by_tensor: dict[str, tuple[MethodChoice, ...]],
) -> OptimizedProfile:
    minimum_output_bytes, maximum_output_bytes = _profile_bounds(
        source_header,
        profile_id,
        choices_by_tensor,
    )
    if minimum_output_bytes > target_bytes:
        raise ValueError(
            "Target is below the smallest allowed profile: "
            f"needs at least {minimum_output_bytes / 1_000_000_000:.2f} GB."
        )
    frontier = _build_knapsack_frontier(
        choices_by_tensor,
        (target_bytes - minimum_output_bytes)
        // _KNAPSACK_BUDGET_UNIT_BYTES,
    )
    return _optimized_from_frontier(
        source_header,
        profile_id,
        target_bytes,
        choices_by_tensor,
        minimum_output_bytes,
        maximum_output_bytes,
        frontier,
    )


def optimize_scored_choices(
    source_header: SourceModelHeader,
    profile_id: str,
    target_bytes: int,
    choices_by_tensor: dict[str, tuple[MethodChoice, ...]],
) -> OptimizedProfile:
    """Optimize an already validated set of storage/loss choices."""
    if type(target_bytes) is not int or target_bytes <= 0:
        raise ValueError("target_bytes must be positive.")
    if not choices_by_tensor:
        raise ValueError("At least one scored tensor is required.")
    return _optimize_target_size_from_choices(
        source_header,
        profile_id,
        target_bytes,
        choices_by_tensor,
    )


def optimize_target_size(
    audit: WeightAuditDocument,
    profile_id: str,
    target_bytes: int,
    allowed_methods: frozenset[QuantizationMethod] = SUPPORTED_METHODS,
    max_relative_l2_error: float | None = None,
    excluded_prefixes: tuple[str, ...] = (),
    excluded_suffixes: tuple[str, ...] = (),
) -> OptimizedProfile:
    """Minimize audited reconstruction SSE while keeping the file under budget."""
    if type(target_bytes) is not int or target_bytes <= 0:
        raise ValueError("target_bytes must be positive.")
    _require_current_audit(audit)
    allowed_methods = _validate_optimizer_inputs(
        profile_id,
        allowed_methods,
        max_relative_l2_error,
        excluded_prefixes,
        excluded_suffixes,
    )
    choices_by_tensor = _choices_by_tensor(
        audit,
        allowed_methods,
        max_relative_l2_error,
        excluded_prefixes,
        excluded_suffixes,
    )
    source_header = validate_weight_audit_against_source(audit)
    return _optimize_target_size_from_choices(
        source_header,
        profile_id,
        target_bytes,
        choices_by_tensor,
    )


def audited_global_relative_l2(
    audit: WeightAuditDocument,
    reconstruction_sse: float,
) -> float | None:
    _require_current_audit(audit)
    total_weight_l2_sq = fsum(
        result["weight_l2_sq"] for result in _validated_results(audit)
    )
    if total_weight_l2_sq <= 0:
        return None
    return sqrt(reconstruction_sse / total_weight_l2_sq)


def generate_profile_sweep(
    audit: WeightAuditDocument,
    profile_id: str,
    step_bytes: int,
    allowed_methods: frozenset[QuantizationMethod] = SUPPORTED_METHODS,
    max_relative_l2_error: float | None = None,
    excluded_prefixes: tuple[str, ...] = (),
    excluded_suffixes: tuple[str, ...] = (),
    selected_target_bytes: int | None = None,
    on_profile_started: Callable[[int, int, int], None] | None = None,
) -> tuple[OptimizedProfile, ...]:
    """Generate feasible target profiles from the smallest to all-BF16 output."""
    if type(step_bytes) is not int or step_bytes <= 0:
        raise ValueError("step_bytes must be positive.")
    _require_current_audit(audit)
    allowed_methods = _validate_optimizer_inputs(
        profile_id,
        allowed_methods,
        max_relative_l2_error,
        excluded_prefixes,
        excluded_suffixes,
    )
    if selected_target_bytes is not None and (
        type(selected_target_bytes) is not int or selected_target_bytes <= 0
    ):
        raise ValueError("selected_target_bytes must be positive.")

    source_header = validate_weight_audit_against_source(audit)
    choices_by_tensor = _choices_by_tensor(
        audit,
        allowed_methods,
        max_relative_l2_error,
        excluded_prefixes,
        excluded_suffixes,
    )
    minimum_profile = _profile_for_assignments(
        {
            tensor_name: choices[0]
            for tensor_name, choices in choices_by_tensor.items()
        },
        profile_id,
    )
    minimum_output_bytes = estimate_profile_bytes(source_header, minimum_profile)
    maximum_output_bytes = estimate_profile_bytes(
        source_header,
        {
            "profile_id": profile_id,
            "description": "All tensors kept in BF16.",
            "default": "keep",
            "rules": (),
        },
    )
    maximum_output_bytes = max(maximum_output_bytes, minimum_output_bytes)
    if (
        selected_target_bytes is not None
        and selected_target_bytes < minimum_output_bytes
    ):
        raise ValueError(
            "Target is below the smallest allowed profile: "
            f"needs at least {minimum_output_bytes / 1_000_000_000:.2f} GB."
        )

    targets = list(
        range(minimum_output_bytes, maximum_output_bytes + 1, step_bytes)
    )
    if not targets or targets[-1] != maximum_output_bytes:
        targets.append(maximum_output_bytes)
    if selected_target_bytes is not None:
        selected_target_bytes = min(selected_target_bytes, maximum_output_bytes)
        if selected_target_bytes not in targets:
            targets.append(selected_target_bytes)
            targets.sort()

    maximum_budget_units = (
        maximum_output_bytes - minimum_output_bytes
    ) // _KNAPSACK_BUDGET_UNIT_BYTES
    frontier = _build_knapsack_frontier(
        choices_by_tensor,
        maximum_budget_units,
    )

    tasks = [
        (
            source_header,
            profile_id,
            target_bytes,
            choices_by_tensor,
            (
                _bf16_indices(choices_by_tensor)
                if target_bytes >= maximum_output_bytes
                else _indices_from_frontier(
                    choices_by_tensor,
                    frontier,
                    (target_bytes - minimum_output_bytes)
                    // _KNAPSACK_BUDGET_UNIT_BYTES,
                )
            ),
        )
        for target_bytes in targets
    ]
    for index, target_bytes in enumerate(targets, start=1):
        if on_profile_started is not None:
            on_profile_started(index, len(targets), target_bytes)

    if len(tasks) < 3:
        return tuple(_optimize_profile_target_worker(task) for task in tasks)

    with ProcessPoolExecutor(
        max_workers=min(4, len(tasks)),
        mp_context=get_context("spawn"),
    ) as executor:
        return tuple(executor.map(_optimize_profile_target_worker, tasks))


def write_profile(
    output_path: str | Path,
    profile: QuantizationProfile,
    overwrite: bool = False,
) -> None:
    """Write a generated profile, refusing replacement by default."""
    rules: list[dict[str, object]] = []
    for rule in profile["rules"]:
        serialized_rule: dict[str, object] = {
            "action": rule["action"],
            "prefix": rule["prefix"],
            "suffixes": list(rule["suffixes"]),
        }
        fallback = rule.get("fallback")
        if fallback is not None:
            serialized_rule["fallback"] = fallback
        rules.append(serialized_rule)

    document = {
        "format_version": 1,
        "profile_id": profile["profile_id"],
        "description": profile.get("description", ""),
        "default": profile["default"],
        "rules": rules,
    }
    mode = "w" if overwrite else "x"
    with Path(output_path).open(mode, encoding="utf-8") as output_file:
        json.dump(document, output_file, indent=2)
        output_file.write("\n")
