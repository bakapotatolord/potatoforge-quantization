"""Persistent local model, profile, and output-folder configuration."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from ..audits.activation_audit import (
    ACTIVATION_AUDIT_FORMAT,
    ACTIVATION_AUDIT_VERSION,
    activation_audit_pair_paths,
)
from ..calibration.activation import (
    ACTIVATION_CALIBRATION_VERSIONS,
    _ACTIVATION_V2_VERSION,
    _FORMAT as ACTIVATION_CALIBRATION_FORMAT,
)
from ..headers.source_header import read_source_model_header
from ..planning import QUANTIZATION_SOURCE_DTYPES


DEFAULT_CONFIG_PATH = Path(__file__).with_name("model_library.toml")

# Folder kinds in config-file order, with the label used in error messages.
FOLDER_KINDS: dict[str, str] = {
    "model": "Model",
    "lora": "LoRA",
    "profile": "Profile",
    "output": "Output",
    "activation": "Activation",
}


def _write_config(path: Path, folders: dict[str, list[Path]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["format_version = 1"]
    for kind in FOLDER_KINDS:
        lines.append(f"{kind}_folders = [")
        lines.extend(f"  {json.dumps(str(folder))}," for folder in folders[kind])
        lines.append("]")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_config(config_path: Path) -> dict[str, list[Path]]:
    if not config_path.exists():
        empty: dict[str, list[Path]] = {kind: [] for kind in FOLDER_KINDS}
        _write_config(config_path, empty)
        return empty

    with config_path.open("rb") as config_file:
        document = tomllib.load(config_file)
    if document.get("format_version") != 1:
        raise ValueError("Model library config format_version must be 1.")
    folders = {}
    for kind in FOLDER_KINDS:
        key = f"{kind}_folders"
        raw_folders = document.get(key, [])
        if not isinstance(raw_folders, list) or any(
            not isinstance(folder, str) or not folder for folder in raw_folders
        ):
            raise ValueError(f"{key} must be a list of non-empty paths.")
        folders[kind] = [Path(folder).expanduser().resolve() for folder in raw_folders]
    return folders


def load_library_folders(kind: str, config_path: Path = DEFAULT_CONFIG_PATH) -> list[Path]:
    return _load_config(config_path)[kind]


def add_library_folder(
    kind: str,
    folder: str | Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> list[Path]:
    path = Path(folder).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"{FOLDER_KINDS[kind]} folder does not exist: {path}")
    folders = _load_config(config_path)
    if path not in folders[kind]:
        folders[kind].append(path)
        _write_config(config_path, folders)
    return folders[kind]


def remove_library_folder(
    kind: str,
    folder: str | Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> list[Path]:
    """Forget a folder; it may already be gone from disk."""
    path = Path(folder).expanduser().resolve()
    folders = _load_config(config_path)
    folders[kind] = [configured for configured in folders[kind] if configured != path]
    _write_config(config_path, folders)
    return folders[kind]


def is_in_library(kind: str, path: Path, config_path: Path = DEFAULT_CONFIG_PATH) -> bool:
    resolved = path.expanduser().resolve()
    return any(
        resolved.is_relative_to(folder)
        for folder in load_library_folders(kind, config_path)
        if folder.is_dir()
    )


def is_quantizable_model(model_path: Path) -> bool:
    try:
        tensors = read_source_model_header(model_path).tensors
    except (OSError, ValueError):
        return False
    return bool(tensors) and all(
        descriptor["dtype"] in QUANTIZATION_SOURCE_DTYPES
        for descriptor in tensors.values()
    )


def discover_models(config_path: Path = DEFAULT_CONFIG_PATH) -> list[dict[str, str | bool]]:
    models = {}
    for folder in load_library_folders("model", config_path):
        if not folder.is_dir():
            continue
        for model in folder.rglob("*.safetensors"):
            if model.is_file():
                resolved = model.resolve()
                models[resolved] = {
                    "name": model.name,
                    "path": str(resolved),
                    "quantizable": is_quantizable_model(resolved),
                }
    return sorted(models.values(), key=lambda model: (model["name"].casefold(), model["path"].casefold()))


def discover_loras(config_path: Path = DEFAULT_CONFIG_PATH) -> list[dict[str, str]]:
    loras = {
        lora.resolve(): {"name": lora.name, "path": str(lora.resolve())}
        for folder in load_library_folders("lora", config_path)
        if folder.is_dir()
        for lora in folder.rglob("*.safetensors")
        if lora.is_file()
    }
    return sorted(loras.values(), key=lambda lora: (lora["name"].casefold(), lora["path"].casefold()))


def _artifact_folder(root: Path, metadata_path: Path) -> str:
    relative_parent = metadata_path.parent.relative_to(root)
    return root.name if relative_parent == Path(".") else relative_parent.as_posix()


def _discover_json_files(kind: str, config_path: Path) -> list[dict[str, str]]:
    found = {}
    for root in load_library_folders(kind, config_path):
        if not root.is_dir():
            continue
        for document in root.rglob("*.json"):
            if document.is_file():
                resolved = document.resolve()
                found[resolved] = {
                    "folder": _artifact_folder(root, document),
                    "name": document.name,
                    "path": str(resolved),
                }
    return sorted(
        found.values(),
        key=lambda item: (item["folder"].casefold(), item["name"].casefold(), item["path"].casefold()),
    )


def discover_profiles(config_path: Path = DEFAULT_CONFIG_PATH) -> list[dict[str, str]]:
    return _discover_json_files("profile", config_path)


def discover_audit_reports(config_path: Path = DEFAULT_CONFIG_PATH) -> list[dict[str, str]]:
    return _discover_json_files("output", config_path)


def discover_activation_calibrations(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> list[dict[str, object]]:
    calibrations: dict[Path, dict[str, object]] = {}
    for activation_folder in load_library_folders("activation", config_path):
        if not activation_folder.is_dir():
            continue
        for metadata in activation_folder.rglob("*.json"):
            if not metadata.is_file():
                continue
            try:
                document = json.loads(metadata.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if not isinstance(document, dict) or document.get("format") != ACTIVATION_CALIBRATION_FORMAT:
                continue
            version = document.get("version")
            if type(version) is not int or version not in ACTIVATION_CALIBRATION_VERSIONS:
                continue
            collection_mode = document.get("collection_mode")
            if version == _ACTIVATION_V2_VERSION and (
                type(collection_mode) is not str
                or collection_mode not in ("cvar", "full")
            ):
                continue
            if version == 1 and collection_mode is not None:
                continue
            resolved = metadata.resolve()
            stats_path = resolved.with_suffix(".safetensors")
            if not stats_path.is_file():
                continue
            calibrations[resolved] = {
                "folder": _artifact_folder(activation_folder, resolved),
                "name": resolved.name,
                "path": str(resolved),
                "version": version,
                "collection_mode": collection_mode,
            }
    return sorted(
        calibrations.values(),
        key=lambda calibration: (
            calibration["folder"].casefold(),
            calibration["name"].casefold(),
            calibration["path"].casefold(),
        ),
    )


def discover_activation_audit_caches(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> list[dict[str, str]]:
    caches: dict[Path, dict[str, str]] = {}
    for activation_folder in load_library_folders("activation", config_path):
        if not activation_folder.is_dir():
            continue
        for metadata in activation_folder.rglob("*.json"):
            if not metadata.is_file():
                continue
            try:
                document = json.loads(metadata.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if not isinstance(document, dict) or document.get("format") != ACTIVATION_AUDIT_FORMAT:
                continue
            if (
                type(document.get("format_version")) is not int
                or document["format_version"] != ACTIVATION_AUDIT_VERSION
            ):
                continue
            resolved = metadata.resolve()
            _, tensors_path = activation_audit_pair_paths(resolved)
            if not tensors_path.is_file():
                continue
            caches[resolved] = {
                "folder": _artifact_folder(activation_folder, resolved),
                "name": resolved.name,
                "path": str(resolved),
            }
    return sorted(
        caches.values(),
        key=lambda cache: (
            cache["folder"].casefold(),
            cache["name"].casefold(),
            cache["path"].casefold(),
        ),
    )
