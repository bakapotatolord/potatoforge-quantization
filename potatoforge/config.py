"""Strict TOML configuration for repeatable direct commands."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, cast

from .audits.profile_optimizer import MEASURED_METHODS
from .audits.weight_audit import QuantizationMethod
from .lora.lora_merge import AdapterMergeInput
from .planning import IOMode


_TOP_LEVEL_FIELDS = frozenset(
    ("format_version", "paths", "optimize", "quantize")
)
_PATH_FIELDS = frozenset(
    ("source", "audit_report", "profile", "quantized_output")
)
@dataclass(frozen=True)
class ConfigPaths:
    source: Path | None = None
    audit_report: Path | None = None
    profile: Path | None = None
    quantized_output: Path | None = None


@dataclass(frozen=True)
class OptimizeConfig:
    paths: ConfigPaths
    profile_id: str
    target_size_gib: float
    methods: tuple[QuantizationMethod, ...]
    enable_potatoforge_int6_runtime: bool
    max_relative_l2_error: float | None
    exclude_prefixes: tuple[str, ...]
    exclude_suffixes: tuple[str, ...]
    overwrite: bool


@dataclass(frozen=True)
class QuantizeConfig:
    paths: ConfigPaths
    io_mode: IOMode
    input_buffer_gib: float | None
    adapters: tuple[AdapterMergeInput, ...] = ()


def _table(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a TOML table.")
    return cast(dict[str, object], value)


def _validate_keys(table: Mapping[str, object], allowed: frozenset[str], label: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ValueError(f"Unknown {label} field(s): {', '.join(unknown)}.")


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string.")
    return value


def _find_project_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ValueError("Could not find the project root containing pyproject.toml.")


def _resolve_path(value: object, label: str, project_root: Path) -> Path:
    raw = _nonempty_string(value, label)
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _load_paths(raw: object, project_root: Path) -> ConfigPaths:
    table = _table(raw, "[paths]")
    _validate_keys(table, _PATH_FIELDS, "[paths]")
    values = {
        field: _resolve_path(value, f"paths.{field}", project_root)
        for field, value in table.items()
    }
    source = values.get("source")
    outputs = {
        field: values[field]
        for field in ("audit_report", "profile", "quantized_output")
        if field in values
    }
    if source is not None and any(source == output for output in outputs.values()):
        raise ValueError("Source and output paths must be different.")
    output_items = list(outputs.items())
    for index, (field, path) in enumerate(output_items):
        for other_field, other_path in output_items[index + 1 :]:
            if path == other_path:
                raise ValueError(
                    f"Output paths {field!r} and {other_field!r} must be different."
                )
    return ConfigPaths(
        source=values.get("source"),
        audit_report=values.get("audit_report"),
        profile=values.get("profile"),
        quantized_output=values.get("quantized_output"),
    )


def _load_quantize_adapters(
    value: object | None,
    project_root: Path,
) -> tuple[AdapterMergeInput, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("quantize.adapters must be an array of tables.")

    adapters: list[AdapterMergeInput] = []
    for index, raw_adapter in enumerate(value, start=1):
        label = f"quantize.adapters[{index}]"
        table = _table(raw_adapter, label)
        _validate_keys(table, frozenset(("path", "strength")), label)

        strength = table.get("strength")
        if (
            not isinstance(strength, (int, float))
            or isinstance(strength, bool)
            or not math.isfinite(float(strength))
        ):
            raise ValueError(
                f"{label}.strength must be a finite number."
            )

        adapters.append(
            AdapterMergeInput(
                path=_resolve_path(
                    table.get("path"),
                    f"{label}.path",
                    project_root,
                ),
                strength=float(strength),
            )
        )

    return tuple(adapters)


def _load_quantize_settings(
    value: object | None,
) -> tuple[IOMode, float | None, tuple[AdapterMergeInput, ...]]:
    if value is None:
        return "batched", None, ()

    table = _table(value, "[quantize]")
    _validate_keys(
        table,
        frozenset(("io_mode", "input_buffer_gib", "adapters")),
        "[quantize]",
    )

    io_mode = table.get("io_mode", "batched")
    if io_mode not in ("serial", "batched"):
        raise ValueError(
            "quantize.io_mode must be serial or batched."
        )

    input_buffer_gib = table.get("input_buffer_gib")
    if input_buffer_gib is not None and (
        not isinstance(input_buffer_gib, (int, float))
        or isinstance(input_buffer_gib, bool)
        or not math.isfinite(float(input_buffer_gib))
        or input_buffer_gib <= 0
    ):
        raise ValueError(
            "quantize.input_buffer_gib must be finite and positive."
        )

    return (
        cast(IOMode, io_mode),
        None if input_buffer_gib is None else float(input_buffer_gib),
        _load_quantize_adapters(
            table.get("adapters"),
            _find_project_root(),
        ),
    )


def _string_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{label} must contain non-empty strings.")
    return tuple(value)


def _load_optimize_settings(value: object) -> dict[str, object]:
    table = _table(value, "[optimize]")
    allowed = frozenset(
        (
            "profile_id",
            "target_size_gib",
            "methods",
            "enable_potatoforge_int6_runtime",
            "max_relative_l2_error",
            "exclude_prefixes",
            "exclude_suffixes",
            "overwrite",
        )
    )
    _validate_keys(table, allowed, "[optimize]")
    profile_id = _nonempty_string(table.get("profile_id"), "optimize.profile_id")
    target_size_gib = table.get("target_size_gib")
    if (
        not isinstance(target_size_gib, (int, float))
        or isinstance(target_size_gib, bool)
        or not math.isfinite(float(target_size_gib))
        or target_size_gib <= 0
    ):
        raise ValueError("optimize.target_size_gib must be finite and positive.")
    methods_value = table.get("methods")
    if not isinstance(methods_value, list) or any(not isinstance(method, str) for method in methods_value):
        raise ValueError("optimize.methods must be a non-empty string array.")
    if not methods_value:
        raise ValueError("optimize.methods must not be empty.")
    if len(set(methods_value)) != len(methods_value):
        raise ValueError("optimize.methods must not contain duplicates.")
    unknown = sorted(set(methods_value) - MEASURED_METHODS)
    if unknown:
        raise ValueError(f"Unknown optimize method(s): {', '.join(unknown)}.")
    enable_int6 = table.get("enable_potatoforge_int6_runtime", False)
    if type(enable_int6) is not bool:
        raise ValueError("optimize.enable_potatoforge_int6_runtime must be a boolean.")
    if {"int6", "int6_convrot"} & set(methods_value) and not enable_int6:
        raise ValueError("INT6 methods require optimize.enable_potatoforge_int6_runtime = true.")
    max_error = table.get("max_relative_l2_error")
    if max_error is not None and (
        not isinstance(max_error, (int, float))
        or isinstance(max_error, bool)
        or not math.isfinite(float(max_error))
        or max_error < 0
    ):
        raise ValueError("optimize.max_relative_l2_error must be finite and non-negative.")
    overwrite = table.get("overwrite", False)
    if type(overwrite) is not bool:
        raise ValueError("optimize.overwrite must be a boolean.")
    return {
        "profile_id": profile_id,
        "target_size_gib": float(target_size_gib),
        "methods": tuple(cast(QuantizationMethod, method) for method in methods_value),
        "enable_potatoforge_int6_runtime": enable_int6,
        "max_relative_l2_error": None if max_error is None else float(max_error),
        "exclude_prefixes": _string_list(table.get("exclude_prefixes", []), "optimize.exclude_prefixes"),
        "exclude_suffixes": _string_list(table.get("exclude_suffixes", []), "optimize.exclude_suffixes"),
        "overwrite": overwrite,
    }


def _load_config_document(config_path: str | Path) -> dict[str, object]:
    path = Path(config_path)
    try:
        with path.open("rb") as config_file:
            document = tomllib.load(config_file)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"Invalid TOML config: {error}") from error

    _validate_keys(document, _TOP_LEVEL_FIELDS, "config")
    if document.get("format_version") != 1 or type(document.get("format_version")) is not int:
        raise ValueError("format_version must be exactly integer 1.")
    return document


def _load_config_base(
    document: Mapping[str, object],
    required_paths: tuple[str, ...],
    config_label: str,
) -> ConfigPaths:
    paths = _load_paths(document.get("paths"), _find_project_root())
    if any(getattr(paths, field) is None for field in required_paths):
        required = " and ".join(f"paths.{field}" for field in required_paths)
        raise ValueError(f"{config_label} configs require {required}.")
    return paths


def load_optimize_config(config_path: str | Path) -> OptimizeConfig:
    """Load and validate the optimize portion of a version-one config."""
    document = _load_config_document(config_path)
    paths = _load_config_base(
        document,
        ("audit_report", "profile"),
        "Optimize",
    )
    if "optimize" not in document:
        raise ValueError("Optimize config requires an [optimize] section.")
    settings = _load_optimize_settings(document["optimize"])
    return OptimizeConfig(
        paths=paths,
        **settings,
    )


def load_quantize_config(config_path: str | Path) -> QuantizeConfig:
    """Load and validate a direct quantize config."""
    document = _load_config_document(config_path)
    paths = _load_config_base(
        document,
        ("source", "profile", "quantized_output"),
        "Quantize",
    )
    io_mode, input_buffer_gib, adapters = _load_quantize_settings(
        document.get("quantize")
    )
    return QuantizeConfig(
        paths=paths,
        io_mode=io_mode,
        input_buffer_gib=input_buffer_gib,
        adapters=adapters,
    )
