import json
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast


QuantizationAction = Literal[
    "keep",
    "int8",
    "int6_rowwise",
    "int6_convrot",
    "int8_convrot",
    "convrot_w4a4",
]
KeepDType = Literal["BF16"]

_SUPPORTED_ACTIONS: tuple[QuantizationAction, ...] = (
    "keep",
    "int8",
    "int6_rowwise",
    "int6_convrot",
    "int8_convrot",
    "convrot_w4a4",
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
    path = Path(profile_path)

    try:
        with path.open("r", encoding="utf-8") as file:
            document = json.load(file)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON profile: {path}") from error

    if not isinstance(document, dict):
        raise ValueError("Profile must contain a top-level JSON object")

    required_fields = {
        "format_version",
        "profile_id",
        "default",
        "rules",
    }
    allowed_fields = required_fields | {"description", "keep_dtype"}

    missing_fields = required_fields - set(document)
    if missing_fields:
        raise ValueError(
            "Profile is missing required fields: "
            + ", ".join(sorted(missing_fields))
        )

    unknown_fields = set(document) - allowed_fields
    if unknown_fields:
        raise ValueError(
            "Profile contains unknown fields: "
            + ", ".join(sorted(unknown_fields))
        )

    format_version = document["format_version"]
    if type(format_version) is not int:
        raise ValueError("Profile format_version must be an integer")
    if format_version != 1:
        raise ValueError("Profile format_version must be 1")

    profile_id = document["profile_id"]
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError("Profile profile_id must be a non-empty string")

    if "description" in document and not isinstance(
        document["description"],
        str,
    ):
        raise ValueError("Profile description must be a string")

    if "keep_dtype" in document and document["keep_dtype"] != "BF16":
        raise ValueError("Profile keep_dtype must be BF16")

    def validate_action(value: object, field_name: str) -> QuantizationAction:
        if not isinstance(value, str) or value not in _SUPPORTED_ACTIONS:
            raise ValueError(
                f"{field_name} must be one of: "
                + ", ".join(_SUPPORTED_ACTIONS)
            )

        return cast(QuantizationAction, value)

    default_action = validate_action(document["default"], "Profile default")

    raw_rules = document["rules"]
    if not isinstance(raw_rules, list):
        raise ValueError("Profile rules must be a list")

    validated_rules: list[ProfileRule] = []

    for rule_index, raw_rule in enumerate(raw_rules):
        if not isinstance(raw_rule, dict):
            raise ValueError(f"Profile rule {rule_index} must be an object")

        required_rule_fields = {"action", "prefix", "suffixes"}
        missing_rule_fields = required_rule_fields - set(raw_rule)
        if missing_rule_fields:
            raise ValueError(
                f"Profile rule {rule_index} is missing fields: "
                + ", ".join(sorted(missing_rule_fields))
            )

        unknown_rule_fields = set(raw_rule) - required_rule_fields - {"fallback"}
        if unknown_rule_fields:
            raise ValueError(
                f"Profile rule {rule_index} contains unknown fields: "
                + ", ".join(sorted(unknown_rule_fields))
            )

        action = validate_action(
            raw_rule["action"],
            f"Profile rule {rule_index} action",
        )
        fallback: QuantizationAction | None = None
        if "fallback" in raw_rule:
            fallback = validate_quantization_action(
                raw_rule["fallback"],
                f"Profile rule {rule_index} fallback",
                allow_keep=False,
            )
            if fallback == action:
                raise ValueError(
                    f"Profile rule {rule_index} fallback must differ from action"
                )
            if action == "keep":
                raise ValueError(
                    f"Profile rule {rule_index} fallback requires a quantizing action"
                )

        prefix = raw_rule["prefix"]
        if not isinstance(prefix, str):
            raise ValueError(
                f"Profile rule {rule_index} prefix must be a string"
            )

        raw_suffixes = raw_rule["suffixes"]
        if not isinstance(raw_suffixes, list) or not raw_suffixes:
            raise ValueError(
                f"Profile rule {rule_index} suffixes must be a non-empty list"
            )

        if any(not isinstance(suffix, str) for suffix in raw_suffixes):
            raise ValueError(
                f"Profile rule {rule_index} suffixes must contain only strings"
            )

        validated_rule: ProfileRule = {
            "action": action,
            "prefix": prefix,
            "suffixes": tuple(cast(str, suffix) for suffix in raw_suffixes),
        }
        if fallback is not None:
            validated_rule["fallback"] = fallback
        validated_rules.append(validated_rule)

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
