import re
from pathlib import Path
from typing import NotRequired, TypedDict

from .profile_documents import (
    load_json_object,
    require_format_version,
    require_nonempty_string,
    validate_document_fields,
)
from .profiles import QuantizationAction, validate_quantization_action


class SweepGroup(TypedDict):
    id: NotRequired[str]
    action: QuantizationAction
    layers: tuple[str, ...]


class SweepProfile(TypedDict):
    profile_id: str
    groups: tuple[SweepGroup, ...]


def validate_sweep_group_id(value: object, label: str) -> str:
    group_id = require_nonempty_string(value, label)
    if (
        group_id in (".", "..")
        or group_id.casefold().endswith(".safetensors")
        or re.fullmatch(r"[A-Za-z0-9._-]+", group_id) is None
    ):
        raise ValueError(
            f"{label} must be a safe filename stem without an extension"
        )
    return group_id


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
        allowed_group_fields = {"action", "layers", "id"}
        missing_group_fields = required_group_fields - set(raw_group)
        if missing_group_fields:
            raise ValueError(
                f"{label} is missing fields: "
                + ", ".join(sorted(missing_group_fields))
            )
        unknown_group_fields = set(raw_group) - allowed_group_fields
        if unknown_group_fields:
            raise ValueError(
                f"{label} contains unknown fields: "
                + ", ".join(sorted(unknown_group_fields))
            )

        group_id = None
        if "id" in raw_group:
            group_id = validate_sweep_group_id(
                raw_group["id"],
                f"{label} id",
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

        group: SweepGroup = {
            "action": action,
            "layers": tuple(raw_layers),
        }
        if group_id is not None:
            group["id"] = group_id
        groups.append(group)

    return {"profile_id": profile_id, "groups": tuple(groups)}
