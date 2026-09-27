"""Validate API workflows, datasets, and literal input bindings."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any


def _reject_constant(value: str) -> None:
    raise ValueError(f"JSON cannot contain {value}.")


def load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.suffix.lower() != ".json":
        raise ValueError(f"Expected a JSON file: {path}")
    raw = path.read_bytes()
    if len(raw) > 10_000_000:
        raise ValueError(f"JSON file is too large: {path}")
    value = json.loads(raw, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value, raw


def inspect_workflow(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    if not workflow or not all(isinstance(node_id, str) and isinstance(node, dict)
                               and isinstance(node.get("class_type"), str)
                               and isinstance(node.get("inputs"), dict)
                               for node_id, node in workflow.items()):
        raise ValueError("Expected an exported ComfyUI API-format workflow (node IDs with class_type and inputs).")
    nodes = []
    for node_id, node in workflow.items():
        inputs = []
        for name, value in node["inputs"].items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                inputs.append({"name": name, "value": value, "type": type_name(value)})
        metadata = node.get("_meta")
        nodes.append({"id": node_id, "class_type": node["class_type"],
                      "title": metadata.get("title", "") if isinstance(metadata, dict) else "", "inputs": inputs})
    return nodes


def type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return "null"


def inspect_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    if dataset.get("format") != "potatoforge_comfy_dataset" or dataset.get("version") != 1:
        raise ValueError("Dataset format must be potatoforge_comfy_dataset version 1.")
    for name in ("name", "description"):
        if name in dataset and not isinstance(dataset[name], str):
            raise ValueError(f"Dataset {name} must be a string.")
    runs = dataset.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Dataset runs must be a non-empty array.")
    # ponytail: 1000 rows bounds the replayable SSE snapshots; stream deltas if larger jobs matter.
    if len(runs) > 1000:
        raise ValueError("Dataset exceeds 1000 runs.")
    fields: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(runs, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Run {index} must be an object.")
        for name, value in row.items():
            if not isinstance(name, str) or not name or not (value is None or isinstance(value, (str, int, float, bool))):
                raise ValueError(f"Run {index} field {name!r} must be a JSON primitive.")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Run {index} field {name} must be finite.")
            entry = fields.setdefault(name, {"name": name, "count": 0, "types": []})
            entry["count"] += 1
            kind = type_name(value)
            if kind not in entry["types"]:
                entry["types"].append(kind)
        if "seed" in row and (type(row["seed"]) is not int or not 0 <= row["seed"] <= 18446744073709551615):
            raise ValueError(f"Run {index} seed must be a non-negative unsigned 64-bit integer.")
        if "resolution" in row and (type(row["resolution"]) not in (int, float) or not 0 <= row["resolution"] <= 1):
            raise ValueError(f"Run {index} resolution must be between 0 and 1.")
    return {"name": dataset.get("name"), "run_count": len(runs), "fields": list(fields.values())}


def validate_bindings(workflow: dict[str, Any], dataset: dict[str, Any], bindings: list[dict[str, str]],
                      definitions: dict[str, Any] | None = None) -> None:
    inspect_workflow(workflow)
    inspect_dataset(dataset)
    if not bindings:
        raise ValueError("Select at least one workflow input binding.")
    targets: set[tuple[str, str]] = set()
    for binding in bindings:
        field, node_id, class_type, input_name = (binding.get(key) for key in
            ("dataset_field", "node_id", "class_type", "input_name"))
        if not all(isinstance(value, str) and value for value in (field, node_id, class_type, input_name)):
            raise ValueError("Each binding needs dataset_field, node_id, class_type, and input_name.")
        target = (node_id, input_name)
        if target in targets:
            raise ValueError(f"Input {node_id}.{input_name} is bound more than once.")
        targets.add(target)
        node = workflow.get(node_id)
        if not isinstance(node, dict) or node.get("class_type") != class_type:
            raise ValueError(f"Stale binding: node {node_id} is missing or changed class.")
        original = node["inputs"].get(input_name)
        if input_name not in node["inputs"] or isinstance(original, (list, dict)):
            raise ValueError(f"Input {node_id}.{input_name} is missing or connected.")
        for index, row in enumerate(dataset["runs"], 1):
            if field not in row:
                raise ValueError(f"Run {index} is missing bound field {field}.")
            value = row[field]
            if original is not None and (type(original) in (int, float) and type(value) in (int, float) and type(value) is not bool):
                if type(original) is int and type(value) is not int:
                    raise ValueError(f"Run {index} field {field} must be an integer for {node_id}.{input_name}.")
            elif original is not None and type(value) is not type(original):
                raise ValueError(f"Run {index} field {field} has the wrong type for {node_id}.{input_name}.")
            if definitions:
                definition = definitions.get(class_type, {}).get("input", {})
                spec = definition.get("required", {}).get(input_name) or definition.get("optional", {}).get(input_name)
                if isinstance(spec, (list, tuple)) and spec:
                    allowed = spec[0]
                    options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
                    if isinstance(allowed, list) and value not in allowed:
                        raise ValueError(f"Run {index} field {field} is not an allowed value for {node_id}.{input_name}.")
                    if type(value) in (int, float) and (("min" in options and value < options["min"]) or
                                                      ("max" in options and value > options["max"])):
                        raise ValueError(f"Run {index} field {field} is outside the range for {node_id}.{input_name}.")


def apply_bindings(workflow: dict[str, Any], row: dict[str, Any], bindings: list[dict[str, str]]) -> dict[str, Any]:
    result = copy.deepcopy(workflow)
    for binding in bindings:
        result[binding["node_id"]]["inputs"][binding["input_name"]] = row[binding["dataset_field"]]
    return result
