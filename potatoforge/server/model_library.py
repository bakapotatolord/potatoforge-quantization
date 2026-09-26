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


def _write_config(
    path: Path,
    model_folders: list[Path],
    lora_folders: list[Path],
    profile_folders: list[Path],
    output_folders: list[Path],
    activation_folders: list[Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["format_version = 1", "model_folders = ["]
    lines.extend(f"  {json.dumps(str(folder))}," for folder in model_folders)
    lines.extend(["]", "lora_folders = ["])
    lines.extend(f"  {json.dumps(str(folder))}," for folder in lora_folders)
    lines.extend(["]", "profile_folders = ["])
    lines.extend(f"  {json.dumps(str(folder))}," for folder in profile_folders)
    lines.extend(["]", "output_folders = ["])
    lines.extend(f"  {json.dumps(str(folder))}," for folder in output_folders)
    lines.extend(["]", "activation_folders = ["])
    lines.extend(f"  {json.dumps(str(folder))}," for folder in activation_folders)
    lines.append("]")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_config(
    config_path: Path,
) -> tuple[list[Path], list[Path], list[Path], list[Path], list[Path]]:
    if not config_path.exists():
        _write_config(config_path, [], [], [], [], [])
        return [], [], [], [], []

    with config_path.open("rb") as config_file:
        document = tomllib.load(config_file)
    if document.get("format_version") != 1:
        raise ValueError("Model library config format_version must be 1.")
    folders = []
    for key in (
        "model_folders",
        "lora_folders",
        "profile_folders",
        "output_folders",
        "activation_folders",
    ):
        raw_folders = document.get(key, [])
        if not isinstance(raw_folders, list) or any(
            not isinstance(folder, str) or not folder for folder in raw_folders
        ):
            raise ValueError(f"{key} must be a list of non-empty paths.")
        folders.append([Path(folder).expanduser().resolve() for folder in raw_folders])
    return folders[0], folders[1], folders[2], folders[3], folders[4]


def load_folders(config_path: Path = DEFAULT_CONFIG_PATH) -> list[Path]:
    return _load_config(config_path)[0]


def load_profile_folders(config_path: Path = DEFAULT_CONFIG_PATH) -> list[Path]:
    return _load_config(config_path)[2]


def load_lora_folders(config_path: Path = DEFAULT_CONFIG_PATH) -> list[Path]:
    return _load_config(config_path)[1]


def load_output_folders(config_path: Path = DEFAULT_CONFIG_PATH) -> list[Path]:
    return _load_config(config_path)[3]


def load_activation_folders(config_path: Path = DEFAULT_CONFIG_PATH) -> list[Path]:
    return _load_config(config_path)[4]


def add_folder(
    folder: str | Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> list[Path]:
    path = Path(folder).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Model folder does not exist: {path}")
    (
        folders,
        lora_folders,
        profile_folders,
        output_folders,
        activation_folders,
    ) = _load_config(config_path)
    if path not in folders:
        _write_config(
            config_path,
            [*folders, path],
            lora_folders,
            profile_folders,
            output_folders,
            activation_folders,
        )
        folders.append(path)
    return folders


def remove_folder(
    folder: str | Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> list[Path]:
    path = Path(folder).expanduser().resolve()
    (
        model_folders,
        lora_folders,
        profile_folders,
        output_folders,
        activation_folders,
    ) = _load_config(config_path)
    folders = [configured for configured in model_folders if configured != path]
    _write_config(
        config_path,
        folders,
        lora_folders,
        profile_folders,
        output_folders,
        activation_folders,
    )
    return folders


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
    for folder in load_folders(config_path):
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


def add_lora_folder(
    folder: str | Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> list[Path]:
    path = Path(folder).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"LoRA folder does not exist: {path}")
    (
        model_folders,
        lora_folders,
        profile_folders,
        output_folders,
        activation_folders,
    ) = _load_config(config_path)
    if path not in lora_folders:
        _write_config(
            config_path,
            model_folders,
            [*lora_folders, path],
            profile_folders,
            output_folders,
            activation_folders,
        )
        lora_folders.append(path)
    return lora_folders


def remove_lora_folder(
    folder: str | Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> list[Path]:
    path = Path(folder).expanduser().resolve()
    (
        model_folders,
        lora_folders,
        profile_folders,
        output_folders,
        activation_folders,
    ) = _load_config(config_path)
    folders = [configured for configured in lora_folders if configured != path]
    _write_config(
        config_path,
        model_folders,
        folders,
        profile_folders,
        output_folders,
        activation_folders,
    )
    return folders


def discover_loras(config_path: Path = DEFAULT_CONFIG_PATH) -> list[dict[str, str]]:
    loras = {
        lora.resolve(): {"name": lora.name, "path": str(lora.resolve())}
        for folder in load_lora_folders(config_path)
        if folder.is_dir()
        for lora in folder.rglob("*.safetensors")
        if lora.is_file()
    }
    return sorted(loras.values(), key=lambda lora: (lora["name"].casefold(), lora["path"].casefold()))


def add_profile_folder(
    folder: str | Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> list[Path]:
    path = Path(folder).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Profile folder does not exist: {path}")
    (
        model_folders,
        lora_folders,
        profile_folders,
        output_folders,
        activation_folders,
    ) = _load_config(config_path)
    if path not in profile_folders:
        _write_config(
            config_path,
            model_folders,
            lora_folders,
            [*profile_folders, path],
            output_folders,
            activation_folders,
        )
        profile_folders.append(path)
    return profile_folders


def remove_profile_folder(
    folder: str | Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> list[Path]:
    path = Path(folder).expanduser().resolve()
    (
        model_folders,
        lora_folders,
        profile_folders,
        output_folders,
        activation_folders,
    ) = _load_config(config_path)
    folders = [configured for configured in profile_folders if configured != path]
    _write_config(
        config_path,
        model_folders,
        lora_folders,
        folders,
        output_folders,
        activation_folders,
    )
    return folders


def discover_profiles(config_path: Path = DEFAULT_CONFIG_PATH) -> list[dict[str, str]]:
    profiles = {}
    for profile_folder in load_profile_folders(config_path):
        if not profile_folder.is_dir():
            continue
        for profile in profile_folder.rglob("*.json"):
            if not profile.is_file():
                continue
            resolved = profile.resolve()
            relative_parent = profile.parent.relative_to(profile_folder)
            folder = (
                profile_folder.name
                if relative_parent == Path(".")
                else relative_parent.as_posix()
            )
            profiles[resolved] = {
                "folder": folder,
                "name": profile.name,
                "path": str(resolved),
            }
    return sorted(
        profiles.values(),
        key=lambda profile: (
            profile["folder"].casefold(),
            profile["name"].casefold(),
            profile["path"].casefold(),
        ),
    )


def discover_audit_reports(config_path: Path = DEFAULT_CONFIG_PATH) -> list[dict[str, str]]:
    reports = {}
    for output_folder in load_output_folders(config_path):
        if not output_folder.is_dir():
            continue
        for report in output_folder.rglob("*.json"):
            if not report.is_file():
                continue
            resolved = report.resolve()
            relative_parent = report.parent.relative_to(output_folder)
            folder = (
                output_folder.name
                if relative_parent == Path(".")
                else relative_parent.as_posix()
            )
            reports[resolved] = {
                "folder": folder,
                "name": report.name,
                "path": str(resolved),
            }
    return sorted(
        reports.values(),
        key=lambda report: (
            report["folder"].casefold(),
            report["name"].casefold(),
            report["path"].casefold(),
        ),
    )


def _artifact_folder(root: Path, metadata_path: Path) -> str:
    relative_parent = metadata_path.parent.relative_to(root)
    return root.name if relative_parent == Path(".") else relative_parent.as_posix()


def discover_activation_calibrations(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> list[dict[str, object]]:
    calibrations: dict[Path, dict[str, object]] = {}
    for activation_folder in load_activation_folders(config_path):
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
    for activation_folder in load_activation_folders(config_path):
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


def add_activation_folder(
    folder: str | Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> list[Path]:
    path = Path(folder).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Activation folder does not exist: {path}")
    (
        model_folders,
        lora_folders,
        profile_folders,
        output_folders,
        activation_folders,
    ) = _load_config(config_path)
    if path not in activation_folders:
        activation_folders = [*activation_folders, path]
        _write_config(
            config_path,
            model_folders,
            lora_folders,
            profile_folders,
            output_folders,
            activation_folders,
        )
    return activation_folders


def remove_activation_folder(
    folder: str | Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> list[Path]:
    path = Path(folder).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Activation folder does not exist: {path}")
    (
        model_folders,
        lora_folders,
        profile_folders,
        output_folders,
        activation_folders,
    ) = _load_config(config_path)
    remaining = [configured for configured in activation_folders if configured != path]
    _write_config(
        config_path,
        model_folders,
        lora_folders,
        profile_folders,
        output_folders,
        remaining,
    )
    return remaining


def add_output_folder(
    folder: str | Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> list[Path]:
    path = Path(folder).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Output folder does not exist: {path}")
    (
        model_folders,
        lora_folders,
        profile_folders,
        output_folders,
        activation_folders,
    ) = _load_config(config_path)
    if path not in output_folders:
        _write_config(
            config_path,
            model_folders,
            lora_folders,
            profile_folders,
            [*output_folders, path],
            activation_folders,
        )
        output_folders.append(path)
    return output_folders


def remove_output_folder(
    folder: str | Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> list[Path]:
    path = Path(folder).expanduser().resolve()
    (
        model_folders,
        lora_folders,
        profile_folders,
        output_folders,
        activation_folders,
    ) = _load_config(config_path)
    folders = [configured for configured in output_folders if configured != path]
    _write_config(
        config_path,
        model_folders,
        lora_folders,
        profile_folders,
        folders,
        activation_folders,
    )
    return folders


def is_configured_model(
    model_path: Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> bool:
    resolved = model_path.expanduser().resolve()
    return any(
        resolved.is_relative_to(folder)
        for folder in load_folders(config_path)
        if folder.is_dir()
    )


def is_configured_profile(
    profile_path: Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> bool:
    resolved = profile_path.expanduser().resolve()
    return any(
        resolved.is_relative_to(folder)
        for folder in load_profile_folders(config_path)
        if folder.is_dir()
    )


def is_configured_lora(
    lora_path: Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> bool:
    resolved = lora_path.expanduser().resolve()
    return any(
        resolved.is_relative_to(folder)
        for folder in load_lora_folders(config_path)
        if folder.is_dir()
    )


def is_configured_output(
    output_path: Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> bool:
    resolved = output_path.expanduser().resolve()
    return any(
        resolved.is_relative_to(folder)
        for folder in load_output_folders(config_path)
        if folder.is_dir()
    )


def is_configured_activation(
    activation_path: Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> bool:
    resolved = activation_path.expanduser().resolve()
    return any(
        resolved.is_relative_to(folder)
        for folder in load_activation_folders(config_path)
        if folder.is_dir()
    )
