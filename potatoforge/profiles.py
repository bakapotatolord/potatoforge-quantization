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
    prefix: str
    suffixes: tuple[str, ...]

class QuantizationProfile(TypedDict):
    default: QuantizationAction
    rules: tuple[ProfileRule, ...]
    keep_dtype: NotRequired[KeepDType]
    profile_id: NotRequired[str]
    description: NotRequired[str]


def resolve_profile(profile: QuantizationProfile, tensor_name: str) -> QuantizationAction:
    for rule in profile["rules"]:
        if (
            tensor_name.startswith(rule["prefix"])
            and tensor_name.endswith(rule["suffixes"])
        ):
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

        unknown_rule_fields = set(raw_rule) - required_rule_fields
        if unknown_rule_fields:
            raise ValueError(
                f"Profile rule {rule_index} contains unknown fields: "
                + ", ".join(sorted(unknown_rule_fields))
            )

        action = validate_action(
            raw_rule["action"],
            f"Profile rule {rule_index} action",
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

        validated_rules.append(
            {
                "action": action,
                "prefix": prefix,
                "suffixes": tuple(cast(str, suffix) for suffix in raw_suffixes),
            }
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
