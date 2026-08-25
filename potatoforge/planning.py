import json
from math import prod
from typing import (
    Callable,
    Final,
    Literal,
    NamedTuple,
    NotRequired,
    TypeAlias,
    TypedDict,
)
from collections.abc import Iterable, Mapping

from .profiles import (
    QuantizationAction,
    QuantizationProfile,
    resolve_profile,
)
from .quantization.hadamard import CONVROT_GROUP_SIZE


IOMode = Literal["serial", "batched"]


class ResolvedIOMode(NamedTuple):
    mode: IOMode
    reason: str


def resolve_io_mode(
    requested_mode: IOMode,
    input_buffer_bytes: int | None,
) -> ResolvedIOMode:
    if requested_mode not in ("serial", "batched"):
        raise ValueError(f"Unknown I/O mode: {requested_mode}")
    if requested_mode == "batched" and input_buffer_bytes is None:
        return ResolvedIOMode(
            "serial",
            "batched mode has no input buffer; falling back to serial",
        )
    return ResolvedIOMode(requested_mode, "requested mode")


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


PlanBuilder: TypeAlias = Callable[[str, TensorDescriptor], QuantizedPlan]

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


class TensorReadPlan(NamedTuple):
    name: str
    data_start: int
    data_end: int
    nbytes: int


class InputBatchPlan(NamedTuple):
    tensors: tuple[TensorReadPlan, ...]
    total_input_bytes: int
    oversized: bool


class ScheduledOutputTensor(NamedTuple):
    spec: OutputTensorSpec
    data_offsets: tuple[int, int]

class SafetensorsLayout(NamedTuple):
    tensors: tuple[ScheduledOutputTensor, ...]
    raw_data_bytes: int

TensorHeader: TypeAlias = Mapping[str, TensorDescriptor]

FLOAT32_BYTES_PER_ELEMENT: Final = 4
BFLOAT16_BYTES_PER_ELEMENT: Final = 2
METADATA_KEY = "__metadata__"

INT8_MARKER = {"format": "int8_tensorwise"}
INT8_MARKER_PAYLOAD: Final[bytes] = json.dumps(
    INT8_MARKER,
).encode("utf-8")

INT6_ROWWISE_MARKER = {"format": "int6_rowwise"}
INT6_ROWWISE_MARKER_PAYLOAD: Final[bytes] = json.dumps(
    INT6_ROWWISE_MARKER,
).encode("utf-8")
INT6_ROWWISE_MARKER_BYTE_COUNT: Final[int] = len(
    INT6_ROWWISE_MARKER_PAYLOAD,
)

INT6_CONVROT_MARKER = {
    "format": "int6_rowwise",
    "convrot": True,
    "convrot_groupsize": CONVROT_GROUP_SIZE,
}
INT6_CONVROT_MARKER_PAYLOAD: Final[bytes] = json.dumps(
    INT6_CONVROT_MARKER,
).encode("utf-8")
INT6_CONVROT_MARKER_BYTE_COUNT: Final[int] = len(
    INT6_CONVROT_MARKER_PAYLOAD,
)

INT8_CONVROT_MARKER = {
    "format": "int8_tensorwise",
    "convrot": True,
    "convrot_groupsize": CONVROT_GROUP_SIZE,
}
INT8_CONVROT_MARKER_PAYLOAD: Final[bytes] = json.dumps(
    INT8_CONVROT_MARKER,
).encode("utf-8")
INT8_CONVROT_MARKER_BYTE_COUNT: Final[int] = len(
    INT8_CONVROT_MARKER_PAYLOAD,
)

CONVROT_W4A4_MARKER = {
    "format": "convrot_w4a4",
    "convrot_groupsize": CONVROT_GROUP_SIZE,
}
CONVROT_W4A4_MARKER_PAYLOAD: Final[bytes] = json.dumps(
    CONVROT_W4A4_MARKER,
).encode("utf-8")
CONVROT_W4A4_MARKER_BYTE_COUNT: Final[int] = len(
    CONVROT_W4A4_MARKER_PAYLOAD,
)


def source_bytes(descriptor: TensorDescriptor) -> int:
    start, end = descriptor["data_offsets"]
    return end - start


def _build_quantized_plan(
    tensor_name: str,
    code_shape: tuple[int, ...],
    scale_shape: tuple[int, ...],
    marker_payload: bytes,
    storage_dtype: str = "I8",
) -> QuantizedPlan:
    layer_name = tensor_name.removesuffix(".weight")

    output_tensors = (
        OutputTensorSpec(
            name=tensor_name,
            dtype=storage_dtype,
            shape=code_shape,
            byte_count=prod(code_shape),
        ),
        OutputTensorSpec(
            name=f"{layer_name}.weight_scale",
            dtype="F32",
            shape=scale_shape,
            byte_count=prod(scale_shape) * FLOAT32_BYTES_PER_ELEMENT,
        ),
        OutputTensorSpec(
            name=f"{layer_name}.comfy_quant",
            dtype="U8",
            shape=(len(marker_payload),),
            byte_count=len(marker_payload),
        ),
    )

    return QuantizedPlan(
        output_tensors=output_tensors,
        estimated_bytes=sum(tensor.byte_count for tensor in output_tensors),
    )


def plan_int8_tensorwise(
    tensor_name: str,
    descriptor: TensorDescriptor,
) -> QuantizedPlan:
    if descriptor["dtype"] not in ("BF16", "F16"):
        raise ValueError(
            "INT8 baseline currently expects BF16 or F16 source weights."
        )

    shape = descriptor["shape"]

    if len(shape) != 2:
        raise ValueError("INT8 tensorwise weights must be two-dimensional.")

    if not tensor_name.endswith(".weight"):
        raise ValueError("INT8 tensorwise selection must target a weight tensor.")

    out_features, in_features = shape
    return _build_quantized_plan(
        tensor_name,
        code_shape=(out_features, in_features),
        scale_shape=(out_features, 1),
        marker_payload=INT8_MARKER_PAYLOAD,
    )


def plan_int6_rowwise(
    tensor_name: str,
    descriptor: TensorDescriptor,
) -> QuantizedPlan:
    """Plan physically packed rowwise INT6 codes."""
    if descriptor["dtype"] not in ("BF16", "F16", "F32"):
        raise ValueError(
            "INT6 rowwise currently expects BF16, F16, or F32 source weights."
        )

    shape = descriptor["shape"]

    if len(shape) != 2:
        raise ValueError("INT6 rowwise weights must be two-dimensional.")

    if not tensor_name.endswith(".weight"):
        raise ValueError("INT6 rowwise selection must target a weight tensor.")

    out_features, in_features = shape
    if in_features % 4 != 0:
        raise ValueError(
            "INT6 rowwise input features must be divisible by 4."
        )

    return _build_quantized_plan(
        tensor_name,
        code_shape=(out_features, (in_features // 4) * 3),
        scale_shape=(out_features, 1),
        marker_payload=INT6_ROWWISE_MARKER_PAYLOAD,
        storage_dtype="U8",
    )


def plan_int6_convrot(
    tensor_name: str,
    descriptor: TensorDescriptor,
) -> QuantizedPlan:
    """Plan packed rowwise W6 weights with the fixed ConvRot contract."""

    plan_int6_rowwise(tensor_name, descriptor)
    out_features, in_features = descriptor["shape"]
    if in_features % CONVROT_GROUP_SIZE != 0:
        raise ValueError(
            "ConvRot INT6 input features must be divisible by "
            f"{CONVROT_GROUP_SIZE}."
        )
    return _build_quantized_plan(
        tensor_name,
        code_shape=(out_features, (in_features // 4) * 3),
        scale_shape=(out_features, 1),
        marker_payload=INT6_CONVROT_MARKER_PAYLOAD,
        storage_dtype="U8",
    )


def plan_int8_convrot(
    tensor_name: str,
    descriptor: TensorDescriptor,
) -> QuantizedPlan:
    if descriptor["dtype"] not in ("BF16", "F16"):
        raise ValueError(
            "ConvRot INT8 currently expects BF16 or F16 source weights."
        )

    shape = descriptor["shape"]

    if len(shape) != 2:
        raise ValueError(
            "ConvRot INT8 weights must be two-dimensional."
        )

    if not tensor_name.endswith(".weight"):
        raise ValueError(
            "ConvRot INT8 selection must target a weight tensor."
        )

    out_features, in_features = shape

    if in_features % CONVROT_GROUP_SIZE != 0:
        raise ValueError(
            "ConvRot INT8 input features must be divisible by "
            f"{CONVROT_GROUP_SIZE}."
        )

    return _build_quantized_plan(
        tensor_name,
        code_shape=(out_features, in_features),
        scale_shape=(out_features, 1),
        marker_payload=INT8_CONVROT_MARKER_PAYLOAD,
    )


def plan_convrot_w4a4(
    tensor_name: str,
    descriptor: TensorDescriptor,
) -> QuantizedPlan:
    if descriptor["dtype"] not in ("BF16", "F16"):
        raise ValueError(
            "ConvRot W4A4 currently expects BF16 or F16 source weights."
        )

    shape = descriptor["shape"]

    if len(shape) != 2:
        raise ValueError("ConvRot W4A4 weights must be two-dimensional.")

    if not tensor_name.endswith(".weight"):
        raise ValueError("ConvRot W4A4 selection must target a weight tensor.")

    out_features, in_features = shape

    if in_features % CONVROT_GROUP_SIZE != 0:
        raise ValueError(
            "ConvRot W4A4 input features must be divisible by "
            f"{CONVROT_GROUP_SIZE}."
        )

    return _build_quantized_plan(
        tensor_name,
        code_shape=(out_features, in_features // 2),
        scale_shape=(out_features,),
        marker_payload=CONVROT_W4A4_MARKER_PAYLOAD,
    )


_PLAN_BUILDERS: Mapping[QuantizationAction, PlanBuilder] = {
    "int8": plan_int8_tensorwise,
    "int6_rowwise": plan_int6_rowwise,
    "int6_convrot": plan_int6_convrot,
    "int8_convrot": plan_int8_convrot,
    "convrot_w4a4": plan_convrot_w4a4,
}

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

        selected_action = resolve_profile(
            profile,
            tensor_name,
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

        plan_builder = _PLAN_BUILDERS.get(selected_action)

        if plan_builder is not None:
            try:
                quantized_plan = plan_builder(
                    tensor_name,
                    descriptor,
                )
            except ValueError as error:
                entry["reason"] = str(error)
            else:
                entry["action"] = selected_action
                entry["estimated_bytes"] = (
                    quantized_plan.estimated_bytes
                )
                entry["output_tensors"] = (
                    quantized_plan.output_tensors
                )


        entries.append(entry)

    return entries


def plan_input_batches(
    entries: Iterable[PlanEntry],
    input_buffer_bytes: int,
) -> tuple[InputBatchPlan, ...]:
    """Plan output-ordered batches with physical-order reads within each batch."""
    if type(input_buffer_bytes) is not int or input_buffer_bytes <= 0:
        raise ValueError("Input staging budget must be a positive integer.")

    batches: list[InputBatchPlan] = []
    current: list[TensorReadPlan] = []
    current_bytes = 0

    def emit(*, oversized: bool) -> None:
        nonlocal current, current_bytes
        batches.append(
            InputBatchPlan(
                tensors=tuple(
                    sorted(
                        current,
                        key=lambda tensor: (
                            tensor.data_start,
                            tensor.data_end,
                            tensor.name,
                        ),
                    )
                ),
                total_input_bytes=current_bytes,
                oversized=oversized,
            )
        )
        current = []
        current_bytes = 0

    for entry in entries:
        data_start, data_end = entry["source_data_offsets"]
        if data_start < 0 or data_end < data_start:
            raise ValueError(
                f"Invalid source offsets for {entry['tensor_name']}."
            )
        if entry["input_bytes"] != data_end - data_start:
            raise ValueError(
                f"Source offsets do not match input bytes for "
                f"{entry['tensor_name']}."
            )

        tensor = TensorReadPlan(
            name=entry["tensor_name"],
            data_start=data_start,
            data_end=data_end,
            nbytes=entry["input_bytes"],
        )
        if current and current_bytes + tensor.nbytes > input_buffer_bytes:
            emit(oversized=False)

        current.append(tensor)
        current_bytes += tensor.nbytes

        if len(current) == 1 and tensor.nbytes > input_buffer_bytes:
            emit(oversized=True)

    if current:
        emit(oversized=False)

    return tuple(batches)


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
