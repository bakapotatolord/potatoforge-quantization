"""Mixed-precision profiles driven by activation-audit costs."""

from collections import Counter
from collections.abc import Mapping
from math import isfinite
from pathlib import Path
from typing import NamedTuple

from ..calibration import ActivationCalibration
from ..headers.source_header import SourceModelHeader, read_source_model_header
from ..planning import source_bytes
from ..profiles import (
    QuantizationAction,
    QuantizationProfile,
    find_profile_rule,
    resolve_profile,
)
from .activation_audit import ActivationAuditCache, score_activation_audit
from .profile_compaction import compact_runtime_rules
from .profile_optimizer import (
    MethodChoice,
    OptimizedProfile,
    _profile_for_assignments,
    estimate_profile_bytes,
    optimize_scored_choices,
)


class ActivationProfileResult(NamedTuple):
    optimized: OptimizedProfile
    summary: dict[str, object]


def _validate_profile_semantics(
    profile: QuantizationProfile,
    desired_actions: Mapping[str, QuantizationAction],
    *,
    label: str = "Generated compact profile",
) -> None:
    for tensor_name, action in sorted(desired_actions.items()):
        matching_rule = find_profile_rule(profile, tensor_name)
        actual_action = resolve_profile(profile, tensor_name)
        if actual_action != action:
            raise ValueError(
                f"{label} is not semantically equivalent for {tensor_name}: "
                f"expected {action}, actual {actual_action}; matching compact "
                f"rule: {matching_rule!r}."
            )


def _validate_profile_output_bytes(
    source_header: SourceModelHeader,
    profile: QuantizationProfile,
    expected_output_bytes: int,
    *,
    error_message: str,
) -> int:
    output_bytes = estimate_profile_bytes(source_header, profile)
    if output_bytes != expected_output_bytes:
        raise ValueError(error_message)
    return output_bytes


def generate_activation_cache_profile(
    activation_cache: ActivationAuditCache | str | Path,
    activation_calibration: ActivationCalibration | str | Path,
    *,
    source_path: str | Path | None = None,
    source_header: SourceModelHeader | None = None,
    target_bytes: int | None = None,
    promotion_budget_bytes: int | None = None,
    profile_id: str = "activation-audit",
    metric: str = "aggregate_observed_relative_sse",
    score_report: Mapping[str, object] | None = None,
    allowed_methods: frozenset[str] | None = None,
    baseline_method: str = "bf16",
    excluded_prefixes: tuple[str, ...] = (),
    excluded_suffixes: tuple[str, ...] = (),
) -> ActivationProfileResult:
    """Generate a runtime profile from cache-only activation costs."""
    if target_bytes is not None and promotion_budget_bytes is not None:
        raise ValueError("Choose target_bytes or promotion_budget_bytes, not both.")
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError("profile_id must be non-empty.")
    calibration = (
        activation_calibration
        if isinstance(activation_calibration, ActivationCalibration)
        else ActivationCalibration.load(activation_calibration)
    )
    cache = (
        activation_cache
        if isinstance(activation_cache, ActivationAuditCache)
        else ActivationAuditCache.load(
            activation_cache,
            calibration=calibration,
        )
    )
    source = cache.source_model_path if source_path is None else source_path
    if source_header is None:
        source_header = read_source_model_header(source)
    if baseline_method not in {"bf16", *cache.requested_methods}:
        raise ValueError(
            "Activation optimizer baseline method is not present in the cache: "
            + baseline_method
        )
    methods = (
        set(cache.requested_methods)
        if allowed_methods is None
        else set(allowed_methods)
    )
    methods.add(baseline_method)
    if baseline_method == "bf16":
        methods.add("bf16")
    unknown_methods = methods - set(cache.requested_methods) - {"bf16"}
    if unknown_methods:
        raise ValueError(
            "Activation optimizer methods are not present in the cache: "
            + ", ".join(sorted(unknown_methods))
        )
    if score_report is None:
        score_report = score_activation_audit(
            cache,
            calibration,
            metric=metric,
        )
    else:
        score_report = dict(score_report)
    score_by_method = {
        (result["tensor_name"], result["method"]): result
        for result in score_report["results"]
    }
    choices_by_tensor: dict[str, tuple[MethodChoice, ...]] = {}
    for tensor_name in cache.tensor_names():
        descriptor = source_header.tensors.get(tensor_name)
        if descriptor is None:
            raise ValueError(
                "Activation cache tensor is missing from source: "
                f"{tensor_name}"
            )
        selected_methods = (
            {"bf16"}
            if tensor_name.startswith(excluded_prefixes)
            or tensor_name.endswith(excluded_suffixes)
            else methods
        )
        if baseline_method == "bf16" or selected_methods == {"bf16"}:
            choices: list[MethodChoice] = [
                MethodChoice(
                    "bf16",
                    "keep",
                    source_bytes(descriptor),
                    0.0,
                    0.0,
                )
            ]
        else:
            baseline_candidate = cache.get(tensor_name, baseline_method)
            baseline_score = score_by_method.get((tensor_name, baseline_method))
            if (
                baseline_candidate is None
                or not baseline_candidate.available
                or baseline_score is None
            ):
                raise ValueError(
                    "Activation profile baseline is unavailable: "
                    f"tensor={tensor_name} method={baseline_method}"
                )
            baseline_cost = baseline_score.get("objective_cost")
            if (
                not isinstance(baseline_candidate.storage_bytes, int)
                or baseline_candidate.storage_bytes < 0
                or not isinstance(baseline_cost, (int, float))
                or isinstance(baseline_cost, bool)
                or not isfinite(baseline_cost)
                or baseline_cost < 0
            ):
                raise ValueError(
                    "Activation profile baseline has invalid measurements: "
                    f"tensor={tensor_name} method={baseline_method}"
                )
            choices = [
                MethodChoice(
                    baseline_method,
                    baseline_candidate.action,
                    baseline_candidate.storage_bytes,
                    float(baseline_cost),
                    float(baseline_cost),
                )
            ]
        for method in sorted(selected_methods - {baseline_method, "bf16"}):
            candidate = cache.get(tensor_name, method)
            score = score_by_method.get((tensor_name, method))
            if candidate is None or not candidate.available or score is None:
                continue
            cost = score.get("objective_cost")
            if (
                not isinstance(candidate.storage_bytes, int)
                or candidate.storage_bytes < 0
                or not isinstance(cost, (int, float))
                or isinstance(cost, bool)
                or not isfinite(cost)
                or cost < 0
            ):
                continue
            choices.append(
                MethodChoice(
                    method,
                    candidate.action,
                    candidate.storage_bytes,
                    float(cost),
                    float(cost),
                )
            )
        if "bf16" in selected_methods and baseline_method != "bf16":
            choices.append(
                MethodChoice(
                    "bf16",
                    "keep",
                    source_bytes(descriptor),
                    0.0,
                    0.0,
                )
            )
        unique_choices = {
            (choice.storage_bytes, choice.reconstruction_sse): choice
            for choice in choices
        }
        choices_by_tensor[tensor_name] = tuple(
            sorted(
                unique_choices.values(),
                key=lambda choice: (
                    choice.storage_bytes,
                    choice.reconstruction_sse,
                    choice.method,
                ),
            )
        )

    minimum_profile = _profile_for_assignments(
        {name: choices[0] for name, choices in choices_by_tensor.items()},
        f"{profile_id}-minimum",
    )
    minimum_output_bytes = estimate_profile_bytes(source_header, minimum_profile)
    if promotion_budget_bytes is not None:
        if type(promotion_budget_bytes) is not int or promotion_budget_bytes < 0:
            raise ValueError("promotion_budget_bytes must be non-negative.")
        target_bytes = minimum_output_bytes + promotion_budget_bytes
    if target_bytes is None:
        raise ValueError("Provide target_bytes or promotion_budget_bytes.")
    optimized = optimize_scored_choices(
        source_header,
        profile_id,
        target_bytes,
        choices_by_tensor,
    )
    runtime_tensor_names = set(source_header.tensors)
    desired_actions = {
        tensor_name: resolve_profile(optimized.profile, tensor_name)
        for tensor_name in runtime_tensor_names
    }
    compact_profile = {
        **optimized.profile,
        "default": "keep",
        "rules": compact_runtime_rules(desired_actions, "keep"),
    }
    _validate_profile_semantics(
        compact_profile,
        desired_actions,
        label="Generated compact activation profile",
    )
    compact_output_bytes = _validate_profile_output_bytes(
        source_header,
        compact_profile,
        optimized.output_bytes,
        error_message=(
            "Generated compact activation profile storage differs from "
            "optimizer estimate."
        ),
    )
    optimized = optimized._replace(
        profile=compact_profile,
        output_bytes=compact_output_bytes,
    )
    selected_methods = Counter(
        resolve_profile(optimized.profile, tensor_name)
        for tensor_name in choices_by_tensor
    )
    summary = {
        "format": "potatoforge_activation_profile_summary",
        "version": 1,
        "metric": metric,
        "metric_version": score_report["objective"]["metric_version"],
        "baseline_method": baseline_method,
        "allowed_methods": sorted(methods),
        "source_path": str(Path(source).resolve()),
        "calibration_session_id": cache.calibration_session_id,
        "target_size_bytes": optimized.target_bytes,
        "estimated_final_size_bytes": optimized.output_bytes,
        "minimum_output_bytes": minimum_output_bytes,
        "profile_compacted": True,
        "profile_rule_count": len(compact_profile["rules"]),
        "selected_action_counts": dict(sorted(selected_methods.items())),
        "layer_count": len(choices_by_tensor),
        "objective": (
            "sum of per-layer activation costs; tail reductions are local proxies"
        ),
    }
    return ActivationProfileResult(optimized, summary)
