"""Generate one runtime-compatible profile from a measurement-only audit."""

import json
from heapq import heappop, heappush
from math import isfinite
from pathlib import Path
from typing import Final, NamedTuple, cast

from ..headers.source_header import SourceModelHeader, read_source_model_header
from ..planning import build_output_layout, build_plan
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
}

MEASURED_METHODS: Final[frozenset[QuantizationMethod]] = frozenset(
    _METHOD_ACTIONS,
)
SUPPORTED_METHODS: Final[frozenset[QuantizationMethod]] = frozenset(
    ("bf16", "int8", "int8_convrot", "convrot_w4a4"),
)
_POTATOFORGE_INT6_METHODS: Final[frozenset[QuantizationMethod]] = frozenset(
    ("int6", "int6_convrot"),
)


class MethodChoice(NamedTuple):
    method: QuantizationMethod
    action: QuantizationAction
    storage_bytes: int
    relative_l2_error: float


class OptimizedProfile(NamedTuple):
    profile: QuantizationProfile
    output_bytes: int
    target_bytes: int
    proxy_distortion: float


def load_weight_audit(path: str | Path) -> WeightAuditDocument:
    """Load the current measurement-only audit report."""
    report_path = Path(path)
    try:
        with report_path.open(encoding="utf-8") as report_file:
            document = json.load(report_file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid weight audit: {report_path}") from error

    if not isinstance(document, dict) or document.get("format_version") != 3:
        raise ValueError("Profile optimization requires a format_version 3 audit.")
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
        "description": "Generated from a weight audit by target-size search.",
        "default": "keep",
        "rules": rules,
    }


def estimate_profile_bytes(
    source_header: SourceModelHeader,
    profile: QuantizationProfile,
) -> int:
    """Return the exact output-file size the streaming converter will write."""
    layout = build_output_layout(build_plan(source_header.tensors, profile))
    header = encode_safetensors_header(layout, source_header.metadata or None)
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
        choices.append(
            MethodChoice(
                method,
                action,
                storage_bytes,
                float(relative_l2_error),
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
        if result["tensor_name"] in tensor_names:
            raise ValueError("Weight audit contains duplicate tensor measurements.")
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


def optimize_target_size(
    audit: WeightAuditDocument,
    profile_id: str,
    target_bytes: int,
    allowed_methods: frozenset[QuantizationMethod] = SUPPORTED_METHODS,
    max_relative_l2_error: float | None = None,
    allow_potatoforge_int6_runtime: bool = False,
    excluded_prefixes: tuple[str, ...] = (),
    excluded_suffixes: tuple[str, ...] = (),
) -> OptimizedProfile:
    """Greedily reduce measured error while keeping the real file under budget."""
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError("profile_id must be non-empty.")
    if type(target_bytes) is not int or target_bytes <= 0:
        raise ValueError("target_bytes must be positive.")
    if not allowed_methods:
        raise ValueError("allowed_methods must not be empty.")
    unknown_methods = allowed_methods - MEASURED_METHODS
    if unknown_methods:
        raise ValueError("allowed_methods contains unsupported methods.")
    if (
        allowed_methods & _POTATOFORGE_INT6_METHODS
        and not allow_potatoforge_int6_runtime
    ):
        raise ValueError(
            "INT6 methods require allow_potatoforge_int6_runtime=True."
        )
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

    allowed_methods = allowed_methods | frozenset(("bf16",))
    if not isinstance(audit.get("source_path"), str):
        raise ValueError("Weight audit source_path must be a string.")
    source_header = read_source_model_header(audit["source_path"])
    results = _validated_results(audit)
    choices_by_tensor = {
        result["tensor_name"]: _method_choices(
            result,
            (
                frozenset(("bf16",))
                if any(
                    result["tensor_name"].startswith(prefix)
                    for prefix in excluded_prefixes
                )
                or any(
                    result["tensor_name"].endswith(suffix)
                    for suffix in excluded_suffixes
                )
                else allowed_methods
            ),
            max_relative_l2_error,
        )
        for result in results
    }
    if any(
        tensor_name not in source_header.tensors
        for tensor_name in choices_by_tensor
    ):
        raise ValueError("Weight audit does not match the source checkpoint header.")
    auditable_names = {
        tensor_name
        for tensor_name, _ in select_auditable_bf16_weights(
            source_header.tensors,
        )
    }
    if set(choices_by_tensor) != auditable_names:
        raise ValueError("Weight audit does not cover the source checkpoint.")
    if any(
        result["shape"] != source_header.tensors[result["tensor_name"]]["shape"]
        for result in results
    ):
        raise ValueError("Weight audit does not match the source checkpoint shapes.")

    assignments = {
        tensor_name: choices[0]
        for tensor_name, choices in choices_by_tensor.items()
    }
    profile = _profile_for_assignments(assignments, profile_id)
    output_bytes = estimate_profile_bytes(source_header, profile)
    if output_bytes > target_bytes:
        raise ValueError(
            "Target is below the smallest allowed profile: "
            f"needs at least {output_bytes} bytes."
        )

    upgrades: list[tuple[float, str, int]] = []

    def push_upgrade(tensor_name: str, next_index: int) -> None:
        choices = choices_by_tensor[tensor_name]
        if next_index >= len(choices):
            return
        current = choices[next_index - 1]
        upgraded = choices[next_index]
        score = (
            current.relative_l2_error - upgraded.relative_l2_error
        ) / (upgraded.storage_bytes - current.storage_bytes)
        heappush(upgrades, (-score, tensor_name, next_index))

    for tensor_name in choices_by_tensor:
        push_upgrade(tensor_name, 1)

    # ponytail: greedy proxy search; use a multi-choice knapsack only if it
    # proves unable to find acceptable profiles at the requested size targets.
    while upgrades:
        _, tensor_name, next_index = heappop(upgrades)
        current = assignments[tensor_name]
        upgraded = choices_by_tensor[tensor_name][next_index]
        assignments[tensor_name] = upgraded
        candidate_bytes = estimate_profile_bytes(
            source_header,
            _profile_for_assignments(assignments, profile_id),
        )
        if candidate_bytes > target_bytes:
            assignments[tensor_name] = current
            continue

        output_bytes = candidate_bytes
        push_upgrade(tensor_name, next_index + 1)

    profile = _profile_for_assignments(assignments, profile_id)
    return OptimizedProfile(
        profile=profile,
        output_bytes=output_bytes,
        target_bytes=target_bytes,
        proxy_distortion=sum(
            choice.relative_l2_error for choice in assignments.values()
        ),
    )


def write_profile(
    output_path: str | Path,
    profile: QuantizationProfile,
    overwrite: bool = False,
) -> None:
    """Write a generated profile, refusing replacement by default."""
    document = {
        "format_version": 1,
        "profile_id": profile["profile_id"],
        "description": profile.get("description", ""),
        "default": profile["default"],
        "rules": [
            {
                "action": rule["action"],
                "prefix": rule["prefix"],
                "suffixes": list(rule["suffixes"]),
            }
            for rule in profile["rules"]
        ],
    }
    mode = "w" if overwrite else "x"
    with Path(output_path).open(mode, encoding="utf-8") as output_file:
        json.dump(document, output_file, indent=2)
        output_file.write("\n")
