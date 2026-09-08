from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

from .profile_documents import (
    load_json_object,
    require_format_version,
    require_nonempty_string,
    validate_document_fields,
)


QuantizationAction = Literal[
    "keep",
    "int8",
    "int6_rowwise",
    "int6_convrot",
    "int8_convrot",
    "convrot_w4a4",
    "convrot_w4a4_mse",
]
KeepDType = Literal["BF16"]

_SUPPORTED_ACTIONS: tuple[QuantizationAction, ...] = (
    "keep",
    "int8",
    "int6_rowwise",
    "int6_convrot",
    "int8_convrot",
    "convrot_w4a4",
    "convrot_w4a4_mse",
)


class ProfileRule(TypedDict):
    action: QuantizationAction
    fallback: NotRequired[QuantizationAction]
    prefix: str
    suffixes: tuple[str, ...]


class QuantizationProfile(TypedDict):
    default: QuantizationAction
    rules: tuple[ProfileRule, ...]
    keep_dtype: NotRequired[KeepDType]
    profile_id: NotRequired[str]
    description: NotRequired[str]


def validate_quantization_action(
    value: object,
    field_name: str,
    *,
    allow_keep: bool = True,
) -> QuantizationAction:
    supported_actions = (
        _SUPPORTED_ACTIONS
        if allow_keep
        else tuple(action for action in _SUPPORTED_ACTIONS if action != "keep")
    )
    if not isinstance(value, str) or value not in supported_actions:
        raise ValueError(
            f"{field_name} must be one of: "
            + ", ".join(supported_actions)
            + f"; got {value!r}"
        )

    return cast(QuantizationAction, value)


def validate_profile_rule(
    raw_rule: object,
    rule_label: str,
) -> ProfileRule:
    if not isinstance(raw_rule, dict):
        raise ValueError(f"{rule_label} must be an object")

    required_rule_fields = {"action", "prefix", "suffixes"}
    missing_rule_fields = required_rule_fields - set(raw_rule)
    if missing_rule_fields:
        raise ValueError(
            f"{rule_label} is missing fields: "
            + ", ".join(sorted(missing_rule_fields))
        )

    unknown_rule_fields = set(raw_rule) - required_rule_fields - {"fallback"}
    if unknown_rule_fields:
        raise ValueError(
            f"{rule_label} contains unknown fields: "
            + ", ".join(sorted(unknown_rule_fields))
        )

    action = validate_quantization_action(
        raw_rule["action"],
        f"{rule_label} action",
    )

    fallback: QuantizationAction | None = None
    if "fallback" in raw_rule:
        fallback = validate_quantization_action(
            raw_rule["fallback"],
            f"{rule_label} fallback",
            allow_keep=False,
        )
        if fallback == action:
            raise ValueError(
                f"{rule_label} fallback must differ from action"
            )
        if action == "keep":
            raise ValueError(
                f"{rule_label} fallback requires a quantizing action"
            )
    prefix = raw_rule["prefix"]
    if not isinstance(prefix, str):
        raise ValueError(f"{rule_label} prefix must be a string")

    raw_suffixes = raw_rule["suffixes"]
    if not isinstance(raw_suffixes, list) or not raw_suffixes:
        raise ValueError(
            f"{rule_label} suffixes must be a non-empty list"
        )

    if any(not isinstance(suffix, str) for suffix in raw_suffixes):
        raise ValueError(
            f"{rule_label} suffixes must contain only strings"
        )

    validated_rule: ProfileRule = {
        "action": action,
        "prefix": prefix,
        "suffixes": tuple(cast(str, suffix) for suffix in raw_suffixes),
    }
    if fallback is not None:
        validated_rule["fallback"] = fallback

    return validated_rule


def profile_rule_matches(rule: ProfileRule, tensor_name: str) -> bool:
    return (
        tensor_name.startswith(rule["prefix"])
        and tensor_name.endswith(rule["suffixes"])
    )


def find_profile_rule(
    profile: QuantizationProfile,
    tensor_name: str,
) -> ProfileRule | None:
    for rule in profile["rules"]:
        if profile_rule_matches(rule, tensor_name):
            return rule

    return None


def resolve_profile(
    profile: QuantizationProfile,
    tensor_name: str,
) -> QuantizationAction:
    rule = find_profile_rule(profile, tensor_name)
    if rule is not None:
        return rule["action"]

    return profile["default"]


def load_profile(profile_path: str | Path) -> QuantizationProfile:
    required_fields = {
        "format_version",
        "profile_id",
        "default",
        "rules",
    }
    allowed_fields = required_fields | {"description", "keep_dtype"}
    document = load_json_object(profile_path, "profile")
    validate_document_fields(
        document,
        label="Profile",
        required=required_fields,
        allowed=allowed_fields,
    )
    require_format_version(document["format_version"], "Profile")
    profile_id = require_nonempty_string(
        document["profile_id"],
        "Profile profile_id",
    )

    if "description" in document and not isinstance(
        document["description"],
        str,
    ):
        raise ValueError("Profile description must be a string")

    if "keep_dtype" in document and document["keep_dtype"] != "BF16":
        raise ValueError("Profile keep_dtype must be BF16")

    default_action = validate_quantization_action(
        document["default"],
        "Profile default",
    )

    raw_rules = document["rules"]
    if not isinstance(raw_rules, list):
        raise ValueError("Profile rules must be a list")

    validated_rules: list[ProfileRule] = []

    for rule_index, raw_rule in enumerate(raw_rules):
        validated_rules.append(
            validate_profile_rule(
                raw_rule,
                f"Profile rule {rule_index}",
            )
        )

    loaded_profile: QuantizationProfile = {
        "default": default_action,
        "rules": tuple(validated_rules),
        "profile_id": profile_id,
    }

    if "description" in document:
        loaded_profile["description"] = cast(
            str,
            document["description"],
        )

    if "keep_dtype" in document:
        loaded_profile["keep_dtype"] = "BF16"

    return loaded_profile
