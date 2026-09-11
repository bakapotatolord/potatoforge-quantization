"""Offline mixed-precision profiles driven by activation-score reports."""

from collections import Counter
from collections.abc import Mapping
import json
from math import fsum, isfinite
from pathlib import Path
import re
from typing import Final, NamedTuple, cast

from ..calibration import ActivationCalibration
from ..headers.source_header import SourceModelHeader, read_source_model_header
from ..planning import plan_convrot_w4a4, plan_int8_convrot, source_bytes
from ..profiles import (
    QuantizationAction,
    QuantizationProfile,
    find_profile_rule,
    resolve_profile,
)
from .profile_optimizer import (
    MethodChoice,
    OptimizedProfile,
    _profile_for_assignments,
    estimate_profile_bytes,
    load_weight_audit,
    optimize_scored_choices,
    validate_weight_audit_against_source,
)
from .profile_compaction import compact_runtime_rules
from .weight_audit import WeightAuditDocument, WeightAuditRecord
from .activation_audit import ActivationAuditCache, score_activation_audit


_SCORE_FORMAT: Final[str] = "potatoforge_activation_score"
_SCORE_VERSION: Final[int] = 2
_METRIC_BASIS: Final[str] = "logical_linear_input"
_RELATIVE_METRIC: Final[str] = "relative_output_error"
_RELATIVE_SSE_METRIC: Final[str] = "relative_output_sse"
_LEGACY_METRIC: Final[str] = "relative_l2_error"
_REFERENCE_FORMAT: Final[str] = "int8_convrot"
_CANDIDATE_FORMAT: Final[QuantizationAction] = "convrot_w4a4"
_DEFAULT_INCLUDE_REGEX: Final[str] = r"^blocks\."
_MIB: Final[int] = 1024**2


class ActivationProfileResult(NamedTuple):
    optimized: OptimizedProfile
    summary: dict[str, object]


def load_activation_score(path: str | Path) -> dict[str, object]:
    score_path = Path(path)
    try:
        document = json.loads(score_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid activation-score report: {score_path}") from error
    if not isinstance(document, dict):
        raise ValueError("Activation-score report must be an object.")
    return load_activation_score_document(document)


def _score_results(document: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    results = document["results"]
    assert isinstance(results, list)
    by_name: dict[str, Mapping[str, object]] = {}
    for result in results:
        if not isinstance(result, dict) or not isinstance(
            result.get("tensor_name"), str
        ):
            raise ValueError("Activation score contains an invalid tensor result.")
        tensor_name = result["tensor_name"]
        if tensor_name in by_name:
            raise ValueError(f"Activation score contains duplicate tensor: {tensor_name}")
        by_name[tensor_name] = result
    return by_name


def _audit_results(
    audit: WeightAuditDocument | str | Path,
) -> tuple[SourceModelHeader, dict[str, WeightAuditRecord]]:
    document = (
        audit
        if isinstance(audit, dict)
        else load_weight_audit(audit)
    )
    typed_document = cast(WeightAuditDocument, document)
    source_header = validate_weight_audit_against_source(typed_document)
    results = {
        result["tensor_name"]: result
        for result in typed_document["results"]
    }
    return source_header, results


def _measurement(
    result: WeightAuditRecord,
    method: str,
) -> Mapping[str, object]:
    measurement = result["methods"].get(method)  # type: ignore[arg-type]
    if not isinstance(measurement, dict):
        raise ValueError(f"Missing {method} measurement for {result['tensor_name']}.")
    return measurement


def _nonnegative_float(value: object, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
        or value < 0
    ):
        raise ValueError(f"Invalid {label}: {value!r}")
    return float(value)


def _activation_score_key(metric: str) -> str:
    return (
        "relative_output_error_sq"
        if metric == _RELATIVE_SSE_METRIC
        else "relative_output_error"
    )


def _storage_bytes(
    result: WeightAuditRecord,
    method: str,
    source_header: SourceModelHeader,
) -> int:
    value = _measurement(result, method).get("storage_bytes")
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError(
            f"Invalid storage_bytes for {method}: {result['tensor_name']}"
        )
    if type(value) is int:
        return value

    descriptor = source_header.tensors.get(result["tensor_name"])
    if descriptor is None:
        raise ValueError(
            f"Missing source tensor for {method}: {result['tensor_name']}"
        )
    if method == "convrot_w4a4":
        return plan_convrot_w4a4(
            result["tensor_name"], descriptor
        ).estimated_bytes
    if method == "int8_convrot":
        return plan_int8_convrot(
            result["tensor_name"], descriptor
        ).estimated_bytes
    raise ValueError(f"Cannot derive storage_bytes for {method}.")


def _loss_value(
    result: WeightAuditRecord,
    score: Mapping[str, object] | None,
    metric: str,
) -> tuple[float, float]:
    if metric in (_RELATIVE_METRIC, _RELATIVE_SSE_METRIC):
        if score is None or score.get("activation_status") != "ok":
            raise ValueError("Activation-relative profile requires a valid score.")
        score_key = _activation_score_key(metric)
        return _nonnegative_float(
            score.get(score_key),
            f"{score_key} for {result['tensor_name']}",
        ), 0.0
    baseline = _nonnegative_float(
        _measurement(result, "convrot_w4a4").get("relative_l2_error"),
        f"relative_l2_error for {result['tensor_name']}",
    )
    promoted = _nonnegative_float(
        _measurement(result, "int8_convrot").get("relative_l2_error"),
        f"int8_convrot relative_l2_error for {result['tensor_name']}",
    )
    return baseline, promoted


def _build_choices(
    audit_results: Mapping[str, WeightAuditRecord],
    source_header: SourceModelHeader,
    score_results: Mapping[str, Mapping[str, object]],
    metric: str,
    include_regex: str,
) -> dict[str, tuple[MethodChoice, ...]]:
    try:
        pattern = re.compile(include_regex)
    except re.error as error:
        raise ValueError(f"Invalid --include-regex: {include_regex!r}") from error

    choices: dict[str, tuple[MethodChoice, ...]] = {}
    for tensor_name in sorted(audit_results):
        if not tensor_name.endswith(".weight") or pattern.search(tensor_name) is None:
            continue
        result = audit_results[tensor_name]
        score = score_results.get(tensor_name)
        baseline_bytes = _storage_bytes(
            result,
            "convrot_w4a4",
            source_header,
        )
        if metric in (_RELATIVE_METRIC, _RELATIVE_SSE_METRIC) and (
            score is None
            or score.get("activation_status") != "ok"
            or score.get(_activation_score_key(metric)) is None
        ):
            choices[tensor_name] = (
                MethodChoice(
                    "convrot_w4a4",
                    "convrot_w4a4",
                    baseline_bytes,
                    0.0,
                    0.0,
                ),
            )
            continue
        promoted_bytes = _storage_bytes(
            result,
            "int8_convrot",
            source_header,
        )
        if promoted_bytes < baseline_bytes:
            raise ValueError(
                "int8_convrot storage is smaller than convrot_w4a4 for "
                f"{tensor_name}; binary promotion choices are invalid."
            )
        baseline_loss, promoted_loss = _loss_value(result, score, metric)
        choices[tensor_name] = (
            MethodChoice(
                "convrot_w4a4",
                "convrot_w4a4",
                baseline_bytes,
                baseline_loss,
                baseline_loss,
            ),
            MethodChoice(
                "int8_convrot",
                "int8_convrot",
                promoted_bytes,
                promoted_loss,
                promoted_loss,
            ),
        )
    if not choices:
        raise ValueError("No eligible tensors have usable profile measurements.")
    return choices


def _runtime_profile(
    profile_id: str,
    description: str,
    eligible_names: set[str],
    audit_names: set[str],
    promoted_names: set[str],
    runtime_tensor_names: set[str],
    *,
    default_action: QuantizationAction,
) -> QuantizationProfile:
    """Build a profile over the complete runtime tensor namespace."""
    missing_runtime_names = (
        eligible_names | audit_names | promoted_names
    ) - runtime_tensor_names
    if missing_runtime_names:
        raise ValueError(
            "Runtime tensor namespace is missing names required by the "
            "generated profile: "
            + ", ".join(sorted(missing_runtime_names))
        )
    desired_actions: dict[str, QuantizationAction] = {
        tensor_name: default_action for tensor_name in runtime_tensor_names
    }
    desired_actions.update(
        {
            tensor_name: "keep"
            for tensor_name in audit_names - eligible_names
        }
    )
    desired_actions.update(
        {tensor_name: "int8_convrot" for tensor_name in promoted_names}
    )
    profile: QuantizationProfile = {
        "profile_id": profile_id,
        "description": description,
        "default": default_action,
        "rules": compact_runtime_rules(desired_actions, default_action),
    }
    for tensor_name, action in sorted(desired_actions.items()):
        matching_rule = find_profile_rule(profile, tensor_name)
        actual_action = resolve_profile(profile, tensor_name)
        if actual_action != action:
            raise ValueError(
                "Generated compact profile is not semantically equivalent "
                f"for {tensor_name}: expected {action}, "
                f"actual {actual_action}; matching compact rule: "
                f"{matching_rule!r}."
            )
    return profile


def generate_activation_profile(
    activation_score: Mapping[str, object] | str | Path,
    weight_audit: WeightAuditDocument | str | Path,
    *,
    target_bytes: int | None = None,
    promotion_budget_bytes: int | None = None,
    profile_id: str = "activation-relative",
    metric: str = _RELATIVE_SSE_METRIC,
    include_regex: str = _DEFAULT_INCLUDE_REGEX,
) -> ActivationProfileResult:
    if metric not in (_RELATIVE_SSE_METRIC, _RELATIVE_METRIC, _LEGACY_METRIC):
        raise ValueError(
            "metric must be relative_output_sse, relative_output_error, "
            "or relative_l2_error."
        )
    if target_bytes is not None and promotion_budget_bytes is not None:
        raise ValueError("Choose target_bytes or promotion_budget_bytes, not both.")
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError("profile_id must be non-empty.")

    score_document = (
        dict(activation_score)
        if isinstance(activation_score, Mapping)
        else load_activation_score(activation_score)
    )
    if isinstance(activation_score, Mapping):
        score_document = load_activation_score_document(score_document)
    score_results = _score_results(score_document)
    source_header, audit_results = _audit_results(weight_audit)
    choices = _build_choices(
        audit_results,
        source_header,
        score_results,
        metric,
        include_regex,
    )
    baseline_profile = _profile_for_assignments(
        {name: tensor_choices[0] for name, tensor_choices in choices.items()},
        f"{profile_id}-baseline",
    )
    minimum_output_bytes = estimate_profile_bytes(source_header, baseline_profile)
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
        choices,
    )
    promoted: list[dict[str, object]] = []
    remaining_losses: list[float] = []
    before_losses: list[float] = []
    promotion_bytes = 0
    for tensor_name, tensor_choices in choices.items():
        baseline_choice = tensor_choices[0]
        promoted_choice = (
            tensor_choices[1] if len(tensor_choices) > 1 else None
        )
        before_losses.append(baseline_choice.reconstruction_sse)
        action = resolve_profile(optimized.profile, tensor_name)
        choice = (
            promoted_choice
            if action == "int8_convrot" and promoted_choice is not None
            else baseline_choice
        )
        remaining_losses.append(choice.reconstruction_sse)
        if action == "int8_convrot" and promoted_choice is not None:
            cost = promoted_choice.storage_bytes - baseline_choice.storage_bytes
            promotion_bytes += cost
            promoted.append(
                {
                    "tensor_name": tensor_name,
                    "source_metric_value": baseline_choice.reconstruction_sse,
                    "w4a4_size": baseline_choice.storage_bytes,
                    "int8_convrot_size": promoted_choice.storage_bytes,
                    "promotion_cost": cost,
                }
            )

    total_loss = fsum(before_losses)
    remaining_loss = fsum(remaining_losses)
    recovered_loss = total_loss - remaining_loss
    percent_recovered = (
        None if total_loss <= 0 else recovered_loss / total_loss * 100.0
    )
    description = (
        f"{metric} profile: convrot_w4a4 baseline with int8_convrot "
        f"promotions for {include_regex!r}."
    )
    runtime_tensor_names = set(source_header.tensors)
    runtime_profile = _runtime_profile(
        profile_id,
        description,
        set(choices),
        set(audit_results),
        {item["tensor_name"] for item in promoted},
        runtime_tensor_names,
        default_action=_CANDIDATE_FORMAT,
    )
    runtime_output_bytes = estimate_profile_bytes(
        source_header,
        runtime_profile,
    )
    if runtime_output_bytes != optimized.output_bytes:
        raise ValueError(
            "Generated profile storage differs from optimizer estimate."
        )
    optimized = optimized._replace(
        profile=runtime_profile,
        output_bytes=runtime_output_bytes,
    )
    summary: dict[str, object] = {
        "format": "potatoforge_activation_profile_summary",
        "version": 1,
        "metric": metric,
        "metric_basis": _METRIC_BASIS,
        "reference_format": _REFERENCE_FORMAT,
        "candidate_format": _CANDIDATE_FORMAT,
        "include_regex": include_regex,
        "target_size_bytes": optimized.target_bytes,
        "target_size_mib": optimized.target_bytes / _MIB,
        "estimated_final_size_bytes": optimized.output_bytes,
        "estimated_final_size_mib": optimized.output_bytes / _MIB,
        "baseline_w4a4_tensor_count": len(choices),
        "promoted_int8cr_tensor_count": len(promoted),
        "promotion_bytes": promotion_bytes,
        "promotion_mib": promotion_bytes / _MIB,
        "total_loss_before_promotion": total_loss,
        "remaining_loss_after_promotion": remaining_loss,
        "recovered_loss": recovered_loss,
        "percent_loss_recovered": percent_recovered,
        "promoted_tensors": promoted,
    }
    if metric == _RELATIVE_METRIC:
        summary.update(
            {
                "total_relative_output_error_before_promotion": total_loss,
                "remaining_relative_output_error": remaining_loss,
                "recovered_relative_output_error": recovered_loss,
                "percent_relative_output_error_recovered": percent_recovered,
            }
        )
    elif metric == _RELATIVE_SSE_METRIC:
        summary.update(
            {
                "total_relative_output_sse_before_promotion": total_loss,
                "remaining_relative_output_sse": remaining_loss,
                "recovered_relative_output_sse": recovered_loss,
                "percent_relative_output_sse_recovered": percent_recovered,
            }
        )
    else:
        summary.update(
            {
                "total_legacy_loss_before_promotion": total_loss,
                "remaining_legacy_loss": remaining_loss,
                "recovered_legacy_loss": recovered_loss,
                "percent_legacy_loss_recovered": percent_recovered,
            }
        )
    summary["loss_metric"] = metric
    return ActivationProfileResult(optimized, summary)


def load_activation_score_document(document: Mapping[str, object]) -> dict[str, object]:
    """Validate an already loaded score document without filesystem access."""
    if document.get("format") != _SCORE_FORMAT:
        raise ValueError("Unsupported activation-score format.")
    if document.get("version") != _SCORE_VERSION:
        raise ValueError(
            "relative_output_error is unavailable in this activation-score "
            "version; regenerate the score report."
        )
    if document.get("reference_format") != _REFERENCE_FORMAT:
        raise ValueError("Activation score reference_format must be int8_convrot.")
    if document.get("candidate_format") != _CANDIDATE_FORMAT:
        raise ValueError("Activation score candidate_format must be convrot_w4a4.")
    if document.get("metric_basis") != _METRIC_BASIS:
        raise ValueError("Activation score metric_basis must be logical_linear_input.")
    metrics = document.get("metrics")
    if metrics is not None and (
        not isinstance(metrics, list)
        or any(not isinstance(metric, str) for metric in metrics)
    ):
        raise ValueError("Activation score metrics must be a string list.")
    primary_metric = document.get("primary_metric")
    if primary_metric is not None and not isinstance(primary_metric, str):
        raise ValueError("Activation score primary_metric must be a string.")
    if not isinstance(document.get("results"), list):
        raise ValueError("Activation score results must be a list.")
    return dict(document)


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
        baseline_bytes = choices[0].storage_bytes
        if any(choice.storage_bytes < baseline_bytes for choice in choices[1:]):
            raise ValueError(
                "Activation profile baseline is not the smallest available "
                f"choice: tensor={tensor_name} method={baseline_method}"
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
    for tensor_name, expected_action in desired_actions.items():
        actual_action = resolve_profile(compact_profile, tensor_name)
        if actual_action != expected_action:
            raise ValueError(
                "Generated compact activation profile is not semantically "
                f"equivalent for {tensor_name}: expected {expected_action}, "
                f"actual {actual_action}."
            )
    compact_output_bytes = estimate_profile_bytes(
        source_header,
        compact_profile,
    )
    if compact_output_bytes != optimized.output_bytes:
        raise ValueError(
            "Generated compact activation profile storage differs from "
            "optimizer estimate."
        )
    optimized = optimized._replace(profile=compact_profile)
    selected_methods = Counter(
        resolve_profile(optimized.profile, tensor_name)
        for tensor_name in choices_by_tensor
    )
    summary = {
        "format": "potatoforge_activation_profile_summary",
        "version": 2,
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
