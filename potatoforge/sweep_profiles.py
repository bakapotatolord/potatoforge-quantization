from pathlib import Path
from typing import TypedDict

from .profile_documents import (
    load_json_object,
    require_format_version,
    require_nonempty_string,
    validate_document_fields,
)
from .profiles import QuantizationAction, validate_quantization_action


class SweepGroup(TypedDict):
    action: QuantizationAction
    layers: tuple[str, ...]


class SweepProfile(TypedDict):
    profile_id: str
    groups: tuple[SweepGroup, ...]


def load_sweep_profile(profile_path: str | Path) -> SweepProfile:
    required_fields = {"format_version", "profile_id", "groups"}
    document = load_json_object(profile_path, "sweep profile")
    validate_document_fields(
        document,
        label="Sweep profile",
        required=required_fields,
        allowed=required_fields,
    )
    require_format_version(document["format_version"], "Sweep profile")
    profile_id = require_nonempty_string(
        document["profile_id"],
        "Sweep profile profile_id",
    )

    raw_groups = document["groups"]
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("Sweep profile groups must be a non-empty list")

    groups: list[SweepGroup] = []
    for group_index, raw_group in enumerate(raw_groups):
        label = f"Sweep profile group {group_index}"
        if not isinstance(raw_group, dict):
            raise ValueError(f"{label} must be an object")
        required_group_fields = {"action", "layers"}
        missing_group_fields = required_group_fields - set(raw_group)
        if missing_group_fields:
            raise ValueError(
                f"{label} is missing fields: "
                + ", ".join(sorted(missing_group_fields))
            )
        unknown_group_fields = set(raw_group) - required_group_fields
        if unknown_group_fields:
            raise ValueError(
                f"{label} contains unknown fields: "
                + ", ".join(sorted(unknown_group_fields))
            )

        action = validate_quantization_action(
            raw_group["action"],
            f"{label} action",
            allow_keep=False,
        )
        raw_layers = raw_group["layers"]
        if not isinstance(raw_layers, list) or not raw_layers:
            raise ValueError(f"{label} layers must be a non-empty list")
        if any(
            not isinstance(layer, str) or not layer.strip()
            for layer in raw_layers
        ):
            raise ValueError(
                f"{label} layers must contain non-empty strings"
            )

        groups.append(
            {
                "action": action,
                "layers": tuple(raw_layers),
            }
        )

    return {"profile_id": profile_id, "groups": tuple(groups)}
