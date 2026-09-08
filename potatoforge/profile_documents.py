"""Shared validation for versioned JSON profile documents."""

import json
from pathlib import Path


def load_json_object(path: str | Path, label: str) -> dict[str, object]:
    profile_path = Path(path)
    try:
        with profile_path.open("r", encoding="utf-8") as file:
            document = json.load(file)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON {label}: {profile_path}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{label.capitalize()} must contain a top-level JSON object")
    return document


def validate_document_fields(
    document: dict[str, object],
    *,
    label: str,
    required: set[str],
    allowed: set[str],
) -> None:
    missing_fields = required - set(document)
    if missing_fields:
        raise ValueError(
            f"{label} is missing required fields: "
            + ", ".join(sorted(missing_fields))
        )
    unknown_fields = set(document) - allowed
    if unknown_fields:
        raise ValueError(
            f"{label} contains unknown fields: "
            + ", ".join(sorted(unknown_fields))
        )


def require_format_version(value: object, label: str) -> None:
    if type(value) is not int:
        raise ValueError(f"{label} format_version must be an integer")
    if value != 1:
        raise ValueError(f"{label} format_version must be 1")


def require_nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value
