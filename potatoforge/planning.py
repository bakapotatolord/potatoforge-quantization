import json
from math import prod
from typing import Final, NamedTuple, NotRequired, TypeAlias, TypedDict
from collections.abc import Callable, Iterable, Mapping

from .conversion_method_types import QuantizationAction
from .profiles import (
    QuantizationProfile,
    find_profile_rule,
)
from .quantization.hadamard import CONVROT_GROUP_SIZE


class TensorDescriptor(TypedDict):
    dtype: str
    shape: list[int]
    data_offsets: list[int]


class OutputTensorSpec(NamedTuple):
    name: str
    dtype: str
    shape: tuple[int, ...]
    byte_count: int


class QuantizedPlan(NamedTuple):
    output_tensors: tuple[OutputTensorSpec, ...]
    estimated_bytes: int


class PlanEntry(TypedDict):
    tensor_name: str
    source_dtype: str
    shape: tuple[int, ...]
    input_bytes: int
    action: QuantizationAction
    estimated_bytes: int
    output_tensors: tuple[OutputTensorSpec, ...]
    reason: NotRequired[str]
    source_data_offsets: tuple[int, int]


class ScheduledOutputTensor(NamedTuple):
    spec: OutputTensorSpec
    data_offsets: tuple[int, int]


class SafetensorsLayout(NamedTuple):
    tensors: tuple[ScheduledOutputTensor, ...]
    raw_data_bytes: int

TensorHeader: TypeAlias = Mapping[str, TensorDescriptor]

FLOAT32_BYTES_PER_ELEMENT: Final = 4
BFLOAT16_BYTES_PER_ELEMENT: Final = 2
QUANTIZATION_SOURCE_DTYPES: Final[frozenset[str]] = frozenset(
    ("BF16", "F16", "F32")
)
_SUPPORTED_WEIGHT_SUFFIXES: Final[tuple[str, ...]] = (
    ".weight",
    ".attn.in_proj_weight",
)
METADATA_KEY = "__metadata__"
QUANTIZATION_METADATA_KEY = "potatoforge.quantization"
QUANTIZATION_LAYERS_METADATA_KEY = "potatoforge.quantization_layers"

def _marker_payload(marker: Mapping[str, object]) -> bytes:
    return json.dumps(marker).encode("utf-8")


def _convrot_marker(format_name: str) -> dict[str, object]:
    return {"format": format_name, "convrot": True, "convrot_groupsize": CONVROT_GROUP_SIZE}


INT8_MARKER = {"format": "int8_tensorwise"}
INT6_ROWWISE_MARKER = {"format": "int6_rowwise"}
INT7_ROWWISE_MARKER = {"format": "int7_rowwise"}
INT5_CONVROT_MARKER = _convrot_marker("int5_rowwise")
INT6_CONVROT_MARKER = _convrot_marker("int6_rowwise")
INT7_CONVROT_MARKER = _convrot_marker("int7_rowwise")
INT8_CONVROT_MARKER = _convrot_marker("int8_tensorwise")
CONVROT_W4A4_MARKER = {"format": "convrot_w4a4", "convrot_groupsize": CONVROT_GROUP_SIZE}

INT8_MARKER_PAYLOAD: Final[bytes] = _marker_payload(INT8_MARKER)
INT6_ROWWISE_MARKER_PAYLOAD: Final[bytes] = _marker_payload(INT6_ROWWISE_MARKER)
INT7_ROWWISE_MARKER_PAYLOAD: Final[bytes] = _marker_payload(INT7_ROWWISE_MARKER)
INT5_CONVROT_MARKER_PAYLOAD: Final[bytes] = _marker_payload(INT5_CONVROT_MARKER)
INT6_CONVROT_MARKER_PAYLOAD: Final[bytes] = _marker_payload(INT6_CONVROT_MARKER)
INT7_CONVROT_MARKER_PAYLOAD: Final[bytes] = _marker_payload(INT7_CONVROT_MARKER)
INT8_CONVROT_MARKER_PAYLOAD: Final[bytes] = _marker_payload(INT8_CONVROT_MARKER)
CONVROT_W4A4_MARKER_PAYLOAD: Final[bytes] = _marker_payload(CONVROT_W4A4_MARKER)


def source_bytes(descriptor: TensorDescriptor) -> int:
    start, end = descriptor["data_offsets"]
    return end - start


def is_supported_weight_key(name: str) -> bool:
    return name.endswith(_SUPPORTED_WEIGHT_SUFFIXES)


def logical_layer_name_for_weight(weight_name: str) -> str:
    if weight_name.endswith(".attn.in_proj_weight"):
        return weight_name.removesuffix("_weight")
    if weight_name.endswith(".weight"):
        return weight_name.removesuffix(".weight")
    raise ValueError(
        "Quantization layer families must start from a supported weight tensor."
    )


def weight_key_for_layer(layer_name: str) -> str:
    if layer_name.endswith(".attn.in_proj"):
        return f"{layer_name}_weight"
    return f"{layer_name}.weight"


def _validate_quantization_source_dtype(
    descriptor: TensorDescriptor,
    format_name: str,
) -> None:
    if descriptor["dtype"] not in QUANTIZATION_SOURCE_DTYPES:
        raise ValueError(
            f"{format_name} currently expects BF16, F16, or F32 "
            "source weights."
        )


def _validate_quantization_weight(
    tensor_name: str,
    descriptor: TensorDescriptor,
    format_name: str,
) -> tuple[int, int]:
    _validate_quantization_source_dtype(descriptor, format_name)
    shape = descriptor["shape"]
    if len(shape) != 2:
        raise ValueError(f"{format_name} weights must be two-dimensional.")
    if not is_supported_weight_key(tensor_name):
        raise ValueError(
            f"{format_name} selection must target a weight tensor."
        )
    return shape[0], shape[1]


def quantization_family_names(weight_name: str) -> tuple[str, ...]:
    layer_name = logical_layer_name_for_weight(weight_name)
    physical_weight_name = (
        weight_key_for_layer(layer_name)
        if weight_name.endswith(".attn.in_proj_weight")
        else weight_name
    )
    return (
        physical_weight_name,
        f"{layer_name}.weight_scale",
        f"{layer_name}.comfy_quant",
    )


def _build_quantized_plan(
    tensor_name: str,
    code_shape: tuple[int, ...],
    scale_shape: tuple[int, ...],
    marker_payload: bytes,
    storage_dtype: str = "I8",
) -> QuantizedPlan:
    weight_name, scale_name, marker_name = quantization_family_names(
        tensor_name
    )

    output_tensors = (
        OutputTensorSpec(
            name=weight_name,
            dtype=storage_dtype,
            shape=code_shape,
            byte_count=prod(code_shape),
        ),
        OutputTensorSpec(
            name=scale_name,
            dtype="F32",
            shape=scale_shape,
            byte_count=prod(scale_shape) * FLOAT32_BYTES_PER_ELEMENT,
        ),
        OutputTensorSpec(
            name=marker_name,
            dtype="U8",
            shape=(len(marker_payload),),
            byte_count=len(marker_payload),
        ),
    )

    return QuantizedPlan(
        output_tensors=output_tensors,
        estimated_bytes=sum(tensor.byte_count for tensor in output_tensors),
    )


def _rowwise_planner(
    format_label: str,
    marker_payload: bytes,
    *,
    width_multiple: int | None = None,
    packing: tuple[int, int] = (1, 1),
    storage_dtype: str = "I8",
    vector_scale: bool = False,
) -> Callable[[str, TensorDescriptor], QuantizedPlan]:
    """Plan codes, one scale per row, and the marker.

    ``packing`` is (logical values, stored bytes) per packed group.
    """
    values_per_group, bytes_per_group = packing

    def plan(tensor_name: str, descriptor: TensorDescriptor) -> QuantizedPlan:
        rows, columns = _validate_quantization_weight(tensor_name, descriptor, format_label)
        if width_multiple is not None and columns % width_multiple != 0:
            raise ValueError(
                f"{format_label} input features must be divisible by {width_multiple}."
            )
        return _build_quantized_plan(
            tensor_name,
            code_shape=(rows, columns // values_per_group * bytes_per_group),
            scale_shape=(rows,) if vector_scale else (rows, 1),
            marker_payload=marker_payload,
            storage_dtype=storage_dtype,
        )

    return plan


plan_int8_tensorwise = _rowwise_planner("INT8 tensorwise", INT8_MARKER_PAYLOAD)
plan_int6_rowwise = _rowwise_planner(
    "INT6 rowwise", INT6_ROWWISE_MARKER_PAYLOAD,
    width_multiple=4, packing=(4, 3), storage_dtype="U8",
)
plan_int7_rowwise = _rowwise_planner(
    "INT7 rowwise", INT7_ROWWISE_MARKER_PAYLOAD,
    width_multiple=8, packing=(8, 7), storage_dtype="U8",
)
plan_int5_convrot = _rowwise_planner(
    "ConvRot INT5", INT5_CONVROT_MARKER_PAYLOAD,
    width_multiple=CONVROT_GROUP_SIZE, packing=(8, 5), storage_dtype="U8",
)
plan_int6_convrot = _rowwise_planner(
    "ConvRot INT6", INT6_CONVROT_MARKER_PAYLOAD,
    width_multiple=CONVROT_GROUP_SIZE, packing=(4, 3), storage_dtype="U8",
)
plan_int7_convrot = _rowwise_planner(
    "ConvRot INT7", INT7_CONVROT_MARKER_PAYLOAD,
    width_multiple=CONVROT_GROUP_SIZE, packing=(8, 7), storage_dtype="U8",
)
plan_int8_convrot = _rowwise_planner(
    "ConvRot INT8", INT8_CONVROT_MARKER_PAYLOAD, width_multiple=CONVROT_GROUP_SIZE,
)
plan_convrot_w4a4 = _rowwise_planner(
    "ConvRot W4A4", CONVROT_W4A4_MARKER_PAYLOAD,
    width_multiple=CONVROT_GROUP_SIZE, packing=(2, 1), vector_scale=True,
)


def _grouped_int8_marker(format_name: str) -> bytes:
    return json.dumps({
        "format": format_name,
        "group_size": 16,
        "convrot_groupsize": CONVROT_GROUP_SIZE,
    }).encode("utf-8")


W4A8_MARKER_PAYLOAD: Final[bytes] = _grouped_int8_marker("asym_w4a8_int8")
W6A8_MARKER_PAYLOAD: Final[bytes] = _grouped_int8_marker("w6a8_int8")
W4A8_MARKER = json.loads(W4A8_MARKER_PAYLOAD)
W6A8_MARKER = json.loads(W6A8_MARKER_PAYLOAD)


def _plan_grouped_int8(
    tensor_name: str,
    descriptor: TensorDescriptor,
    format_label: str,
    bits: int,
    marker_payload: bytes,
    codebook_levels: int | None,
) -> QuantizedPlan:
    """ComfyUI grouped ConvRot layers: packed codes (int8 [N, K*bits/8]), fp8
    group scales [N, K/16], the marker, fp32 channel scales, then the float32
    codebook when the format has one."""
    rows, columns = _validate_quantization_weight(tensor_name, descriptor, format_label)
    if columns % CONVROT_GROUP_SIZE != 0:
        raise ValueError(
            f"{format_label} input features must be divisible by {CONVROT_GROUP_SIZE}."
        )
    weight_name, _scale_name, marker_name = quantization_family_names(tensor_name)
    layer_name = logical_layer_name_for_weight(tensor_name)
    code_width, groups = columns * bits // 8, columns // 16
    outputs = [
        OutputTensorSpec(weight_name, "I8", (rows, code_width), rows * code_width),
        OutputTensorSpec(f"{layer_name}.weight_s_rel", "F8_E4M3", (rows, groups), rows * groups),
        OutputTensorSpec(marker_name, "U8", (len(marker_payload),), len(marker_payload)),
        OutputTensorSpec(f"{layer_name}.weight_s_channel", "F32", (rows,), rows * FLOAT32_BYTES_PER_ELEMENT),
    ]
    if codebook_levels is not None:
        outputs.append(OutputTensorSpec(
            f"{layer_name}.weight_codebook", "F32", (codebook_levels,), codebook_levels * FLOAT32_BYTES_PER_ELEMENT,
        ))
    return QuantizedPlan(tuple(outputs), sum(tensor.byte_count for tensor in outputs))


def plan_w4a8(tensor_name: str, descriptor: TensorDescriptor) -> QuantizedPlan:
    return _plan_grouped_int8(tensor_name, descriptor, "W4A8", 4, W4A8_MARKER_PAYLOAD, 16)


def plan_w6a8(tensor_name: str, descriptor: TensorDescriptor) -> QuantizedPlan:
    return _plan_grouped_int8(tensor_name, descriptor, "W6A8", 6, W6A8_MARKER_PAYLOAD, None)


def build_quantized_tensor_plan(
    action: QuantizationAction,
    tensor_name: str,
    descriptor: TensorDescriptor,
) -> QuantizedPlan:
    if action == "keep":
        raise ValueError(
            "The keep action does not produce a quantized tensor plan."
        )

    from .conversion_methods import get_conversion_method

    return get_conversion_method(action).plan(tensor_name, descriptor)


def build_plan(
    header: TensorHeader,
    profile: QuantizationProfile,
) -> list[PlanEntry]:
    entries: list[PlanEntry] = []

    for tensor_name, descriptor in header.items():
        input_bytes = source_bytes(descriptor)
        source_start, source_end = descriptor["data_offsets"]

        entry: PlanEntry = {
            "tensor_name": tensor_name,
            "source_dtype": descriptor["dtype"],
            "shape": tuple(descriptor["shape"]),
            "source_data_offsets": (source_start, source_end),
            "input_bytes": input_bytes,
            "action": "keep",
            "estimated_bytes": input_bytes,
            "output_tensors": (
                OutputTensorSpec(
                    name=tensor_name,
                    dtype=descriptor["dtype"],
                    shape=tuple(descriptor["shape"]),
                    byte_count=input_bytes,
                ),
            ),
        }

        matched_rule = find_profile_rule(profile, tensor_name)
        selected_action = (
            profile["default"]
            if matched_rule is None
            else matched_rule["action"]
        )
        fallback_action = (
            None if matched_rule is None else matched_rule.get("fallback")
        )

        if (
            selected_action == "keep"
            and profile.get("keep_dtype") == "BF16"
            and descriptor["dtype"] in ("BF16", "F16", "F32")
        ):
            entry["output_tensors"] = (
                OutputTensorSpec(
                    name=tensor_name,
                    dtype="BF16",
                    shape=tuple(descriptor["shape"]),
                    byte_count=(
                        prod(descriptor["shape"])
                        * BFLOAT16_BYTES_PER_ELEMENT
                    ),
                ),
            )
            entry["estimated_bytes"] = entry["output_tensors"][0].byte_count

        if selected_action != "keep":
            actions = (selected_action,)
            if fallback_action is not None:
                actions += (fallback_action,)

            last_error: ValueError | None = None
            for action in actions:
                try:
                    quantized_plan = build_quantized_tensor_plan(
                        action,
                        tensor_name,
                        descriptor,
                    )
                except ValueError as error:
                    last_error = error
                    continue

                entry["action"] = action
                entry["estimated_bytes"] = quantized_plan.estimated_bytes
                entry["output_tensors"] = quantized_plan.output_tensors
                break

            if last_error is not None and entry["action"] == "keep":
                entry["reason"] = str(last_error)

        entries.append(entry)

    return entries


def build_quantization_metadata(
    entries: Iterable[PlanEntry],
) -> dict[str, str]:
    return _quantization_metadata_for_layers(
        {
            entry["tensor_name"]: entry["action"]
            for entry in entries
            if entry["action"] != "keep"
        }
    )


def _quantization_metadata_for_layers(
    layer_actions: Mapping[str, str],
) -> dict[str, str]:
    runtime_actions = {
        "int8_convrot_mse": "int8_convrot",
        "convrot_w4a4_mse": "convrot_w4a4",
        "convrot_w4a4_mse_gptq": "convrot_w4a4",
        "int5_convrot_mse": "int5_convrot",
        "int6_convrot_mse": "int6_convrot",
        "int7_convrot_mse": "int7_convrot",
    }
    layer_actions = {
        name: runtime_actions.get(action, action)
        for name, action in layer_actions.items()
    }
    actions = sorted(set(layer_actions.values()))
    summary = (
        "none"
        if not actions
        else actions[0]
        if len(actions) == 1
        else "mixed"
    )
    return {
        QUANTIZATION_METADATA_KEY: summary,
        QUANTIZATION_LAYERS_METADATA_KEY: json.dumps(
            dict(layer_actions),
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def parse_quantization_layers(raw_layers: str) -> dict[str, str]:
    try:
        decoded_layers = json.loads(raw_layers)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Invalid PotatoForge quantization layer metadata."
        ) from error

    if not isinstance(decoded_layers, dict) or any(
        not isinstance(name, str) or not isinstance(action, str)
        for name, action in decoded_layers.items()
    ):
        raise ValueError("Invalid PotatoForge quantization layer metadata.")

    return decoded_layers


def update_quantization_metadata(
    metadata: Mapping[str, str],
    layer_actions: Mapping[str, str],
) -> dict[str, str]:
    existing_layers: dict[str, str] = {}
    raw_layers = metadata.get(QUANTIZATION_LAYERS_METADATA_KEY)
    if raw_layers is not None:
        existing_layers.update(parse_quantization_layers(raw_layers))

    existing_layers.update(layer_actions)
    return {
        **metadata,
        **_quantization_metadata_for_layers(existing_layers),
    }


def build_output_layout(
    entries: Iterable[PlanEntry],
) -> SafetensorsLayout:
    return build_layout_from_specs(
        spec
        for entry in entries
        for spec in entry["output_tensors"]
    )


def build_layout_from_specs(
    specs: Iterable[OutputTensorSpec],
) -> SafetensorsLayout:
    tensors: list[ScheduledOutputTensor] = []
    seen_names: set[str] = set()
    next_offset = 0

    for spec in specs:
        if spec.name in seen_names:
            raise ValueError(
                f"Duplicate output tensor name: {spec.name}"
            )

        end_offset = next_offset + spec.byte_count

        tensors.append(
            ScheduledOutputTensor(
                spec=spec,
                data_offsets=(
                    next_offset,
                    end_offset,
                ),
            )
        )

        seen_names.add(spec.name)
        next_offset = end_offset

    return SafetensorsLayout(
        tensors=tuple(tensors),
        raw_data_bytes=next_offset,
    )


def layout_to_header(
    layout: SafetensorsLayout,
) -> dict[str, TensorDescriptor]:
    return {
        tensor.spec.name: {
            "dtype": tensor.spec.dtype,
            "shape": list(tensor.spec.shape),
            "data_offsets": list(tensor.data_offsets),
        }
        for tensor in layout.tensors
    }
