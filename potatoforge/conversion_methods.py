"""Built-in conversion catalog.

Adding a method requires its format implementation, one definition here,
focused format tests, and an optional profile. Audit and optimizer support are
separately scoped.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from .conversion_method_types import (
    ConversionMethodDefinition,
    EncodedQuantizedPayload,
    QuantizationAction,
)
from .planning import (
    CONVROT_W4A4_MARKER_PAYLOAD,
    INT5_CONVROT_MARKER_PAYLOAD,
    INT6_CONVROT_MARKER_PAYLOAD,
    INT6_ROWWISE_MARKER_PAYLOAD,
    INT7_CONVROT_MARKER_PAYLOAD,
    INT7_ROWWISE_MARKER_PAYLOAD,
    INT8_CONVROT_MARKER_PAYLOAD,
    INT8_MARKER_PAYLOAD,
    plan_convrot_w4a4,
    plan_int5_convrot,
    plan_int6_convrot,
    plan_int6_rowwise,
    plan_int7_convrot,
    plan_int7_rowwise,
    plan_int8_convrot,
    plan_int8_tensorwise,
)
from .quantization import (
    quantize_int8_convrot,
    quantize_int8_convrot_mse,
    quantize_int8_tensorwise,
)
from .quantization.convrot_w4a4 import (
    quantize_convrot_w4a4,
    quantize_convrot_w4a4_mse,
)
from .quantization.int5_rowwise import (
    quantize_int5_convrot_mse_packed,
    quantize_int5_convrot_packed,
)
from .quantization.int6_packing import pack_int6_row_major
from .quantization.int6_rowwise import (
    quantize_int6_convrot,
    quantize_int6_convrot_mse,
    quantize_int6_convrot_mse_packed,
    quantize_int6_convrot_packed,
    quantize_int6_rowwise,
)
from .quantization.int7_packing import pack_int7_row_major
from .quantization.int7_rowwise import (
    quantize_int7_convrot,
    quantize_int7_convrot_mse,
    quantize_int7_convrot_mse_packed,
    quantize_int7_convrot_packed,
    quantize_int7_rowwise,
)
from .timing import (
    CONVROT_INTERNAL_STAGES,
    INT5_MSE_INTERNAL_STAGES,
    INT6_MSE_INTERNAL_STAGES,
    INT7_MSE_INTERNAL_STAGES,
    INT8_MSE_INTERNAL_STAGES,
    W4A4_MSE_INTERNAL_STAGES,
    TensorTiming,
    timed_stage,
)


GPTQ_INTERNAL_STAGES = (
    "gram_load", "gram_transform", "hessian_factor", "hessian_transfer", "prepare", "rotation",
    "mse_scale", "gptq_solve", "pack", "finalize",
)


def _run_quantize(
    record: TensorTiming | None,
    operation: Callable[[], Any],
) -> Any:
    with timed_stage(record, "quantize"):
        return operation()


def _encode_int8(
    weights: torch.Tensor,
    device: str | torch.device,
    record: TensorTiming | None,
) -> EncodedQuantizedPayload:
    result = _run_quantize(record, lambda: quantize_int8_tensorwise(weights))
    return EncodedQuantizedPayload(result.codes, result.scales)


def _encode_int6_rowwise(
    weights: torch.Tensor,
    device: str | torch.device,
    record: TensorTiming | None,
) -> EncodedQuantizedPayload:
    result = _run_quantize(record, lambda: quantize_int6_rowwise(weights))
    with timed_stage(record, "pack"):
        packed = pack_int6_row_major(result.codes)
    return EncodedQuantizedPayload(packed.packed_codes, result.scales)


def _encode_int7_rowwise(
    weights: torch.Tensor,
    device: str | torch.device,
    record: TensorTiming | None,
) -> EncodedQuantizedPayload:
    result = _run_quantize(record, lambda: quantize_int7_rowwise(weights))
    with timed_stage(record, "pack"):
        packed = pack_int7_row_major(result.codes)
    return EncodedQuantizedPayload(packed.packed_codes, result.scales)


def _encode_int5_convrot(
    weights: torch.Tensor,
    device: str | torch.device,
    record: TensorTiming | None,
) -> EncodedQuantizedPayload:
    if device == "cpu":
        result = _run_quantize(
            record,
            lambda: quantize_int5_convrot_packed(weights),
        )
    elif record is None:
        result = _run_quantize(
            None,
            lambda: quantize_int5_convrot_packed(weights, device=device),
        )
    else:
        result = _run_quantize(
            record,
            lambda: quantize_int5_convrot_packed(
                weights,
                device=device,
                internal_timings=record.internal_stages,
            ),
        )
    return EncodedQuantizedPayload(result.packed_codes, result.scales)


def _encode_int5_convrot_mse(
    weights: torch.Tensor,
    device: str | torch.device,
    record: TensorTiming | None,
) -> EncodedQuantizedPayload:
    if device == "cpu":
        result = _run_quantize(
            record,
            lambda: quantize_int5_convrot_mse_packed(weights),
        )
    elif record is None:
        result = _run_quantize(
            None,
            lambda: quantize_int5_convrot_mse_packed(
                weights,
                device=device,
            ),
        )
    else:
        result = _run_quantize(
            record,
            lambda: quantize_int5_convrot_mse_packed(
                weights,
                device=device,
                internal_timings=record.internal_stages,
            ),
        )
    return EncodedQuantizedPayload(result.packed_codes, result.scales)


def _encode_int6_convrot(
    weights: torch.Tensor,
    device: str | torch.device,
    record: TensorTiming | None,
) -> EncodedQuantizedPayload:
    if device == "cpu":
        result = _run_quantize(record, lambda: quantize_int6_convrot(weights))
        with timed_stage(record, "pack"):
            packed = pack_int6_row_major(result.codes)
        return EncodedQuantizedPayload(packed.packed_codes, result.scales)

    if record is None:
        result = _run_quantize(
            None,
            lambda: quantize_int6_convrot_packed(weights, device=device),
        )
    else:
        result = _run_quantize(
            record,
            lambda: quantize_int6_convrot_packed(
                weights,
                device=device,
                internal_timings=record.internal_stages,
            ),
        )
    return EncodedQuantizedPayload(result.packed_codes, result.scales)


def _encode_int6_convrot_mse(
    weights: torch.Tensor,
    device: str | torch.device,
    record: TensorTiming | None,
) -> EncodedQuantizedPayload:
    if device == "cpu":
        result = _run_quantize(
            record,
            lambda: quantize_int6_convrot_mse(weights),
        )
        with timed_stage(record, "pack"):
            packed = pack_int6_row_major(result.codes)
        return EncodedQuantizedPayload(packed.packed_codes, result.scales)

    if record is None:
        result = _run_quantize(
            None,
            lambda: quantize_int6_convrot_mse_packed(
                weights,
                device=device,
            ),
        )
    else:
        result = _run_quantize(
            record,
            lambda: quantize_int6_convrot_mse_packed(
                weights,
                device=device,
                internal_timings=record.internal_stages,
            ),
        )
    return EncodedQuantizedPayload(result.packed_codes, result.scales)


def _encode_int7_convrot(
    weights: torch.Tensor,
    device: str | torch.device,
    record: TensorTiming | None,
) -> EncodedQuantizedPayload:
    if device == "cpu":
        result = _run_quantize(record, lambda: quantize_int7_convrot(weights))
        with timed_stage(record, "pack"):
            packed = pack_int7_row_major(result.codes)
        return EncodedQuantizedPayload(packed.packed_codes, result.scales)

    if record is None:
        result = _run_quantize(
            None,
            lambda: quantize_int7_convrot_packed(weights, device=device),
        )
    else:
        result = _run_quantize(
            record,
            lambda: quantize_int7_convrot_packed(
                weights,
                device=device,
                internal_timings=record.internal_stages,
            ),
        )
    return EncodedQuantizedPayload(result.packed_codes, result.scales)


def _encode_int7_convrot_mse(
    weights: torch.Tensor,
    device: str | torch.device,
    record: TensorTiming | None,
) -> EncodedQuantizedPayload:
    if device == "cpu":
        result = _run_quantize(
            record,
            lambda: quantize_int7_convrot_mse(weights),
        )
        with timed_stage(record, "pack"):
            packed = pack_int7_row_major(result.codes)
        return EncodedQuantizedPayload(packed.packed_codes, result.scales)

    if record is None:
        result = _run_quantize(
            None,
            lambda: quantize_int7_convrot_mse_packed(
                weights,
                device=device,
            ),
        )
    else:
        result = _run_quantize(
            record,
            lambda: quantize_int7_convrot_mse_packed(
                weights,
                device=device,
                internal_timings=record.internal_stages,
            ),
        )
    return EncodedQuantizedPayload(result.packed_codes, result.scales)


def _encode_int8_convrot(
    weights: torch.Tensor,
    device: str | torch.device,
    record: TensorTiming | None,
) -> EncodedQuantizedPayload:
    if device == "cpu":
        if record is None:
            result = _run_quantize(None, lambda: quantize_int8_convrot(weights))
        else:
            result = _run_quantize(
                record,
                lambda: quantize_int8_convrot(
                    weights,
                    internal_timings=record.internal_stages,
                ),
            )
    elif record is None:
        result = _run_quantize(
            None,
            lambda: quantize_int8_convrot(weights, device=device),
        )
    else:
        result = _run_quantize(
            record,
            lambda: quantize_int8_convrot(
                weights,
                device=device,
                internal_timings=record.internal_stages,
            ),
        )
    return EncodedQuantizedPayload(result.codes, result.scales)


def _encode_int8_convrot_mse(
    weights: torch.Tensor,
    device: str | torch.device,
    record: TensorTiming | None,
) -> EncodedQuantizedPayload:
    result = _run_quantize(
        record,
        lambda: quantize_int8_convrot_mse(
            weights,
            device=device,
            internal_timings=None if record is None else record.internal_stages,
        ),
    )
    return EncodedQuantizedPayload(result.codes, result.scales)


def _encode_convrot_w4a4(
    weights: torch.Tensor,
    device: str | torch.device,
    record: TensorTiming | None,
) -> EncodedQuantizedPayload:
    if device == "cpu":
        result = _run_quantize(record, lambda: quantize_convrot_w4a4(weights))
    elif record is None:
        result = _run_quantize(
            None,
            lambda: quantize_convrot_w4a4(weights, device=device),
        )
    else:
        result = _run_quantize(
            record,
            lambda: quantize_convrot_w4a4(
                weights,
                device=device,
                internal_timings=record.internal_stages,
            ),
        )
    return EncodedQuantizedPayload(result.packed_codes, result.scales)


def _encode_convrot_w4a4_mse(
    weights: torch.Tensor,
    device: str | torch.device,
    record: TensorTiming | None,
) -> EncodedQuantizedPayload:
    if device == "cpu":
        result = _run_quantize(
            record,
            lambda: quantize_convrot_w4a4_mse(weights),
        )
    elif record is None:
        result = _run_quantize(
            None,
            lambda: quantize_convrot_w4a4_mse(weights, device=device),
        )
    else:
        result = _run_quantize(
            record,
            lambda: quantize_convrot_w4a4_mse(
                weights,
                device=device,
                internal_timings=record.internal_stages,
            ),
        )
    return EncodedQuantizedPayload(result.packed_codes, result.scales)


def _encode_convrot_w4a4_mse_gptq(
    weights: torch.Tensor,
    device: str | torch.device,
    record: TensorTiming | None,
) -> EncodedQuantizedPayload:
    raise ValueError("GPTQ encoding requires a calibration artifact and weight name.")


_METHODS: tuple[ConversionMethodDefinition, ...] = (
    ConversionMethodDefinition(
        action="int8",
        label="INT8",
        is_convrot=False,
        internal_stages=(),
        plan=plan_int8_tensorwise,
        marker_payload=INT8_MARKER_PAYLOAD,
        encode=_encode_int8,
    ),
    ConversionMethodDefinition(
        action="int6_rowwise",
        label="INT6 Rowwise",
        is_convrot=False,
        internal_stages=(),
        plan=plan_int6_rowwise,
        marker_payload=INT6_ROWWISE_MARKER_PAYLOAD,
        encode=_encode_int6_rowwise,
    ),
    ConversionMethodDefinition(
        action="int7_rowwise",
        label="INT7 Rowwise",
        is_convrot=False,
        internal_stages=(),
        plan=plan_int7_rowwise,
        marker_payload=INT7_ROWWISE_MARKER_PAYLOAD,
        encode=_encode_int7_rowwise,
    ),
    ConversionMethodDefinition(
        action="int8_convrot",
        label="INT8 ConvRot",
        is_convrot=True,
        internal_stages=CONVROT_INTERNAL_STAGES,
        plan=plan_int8_convrot,
        marker_payload=INT8_CONVROT_MARKER_PAYLOAD,
        encode=_encode_int8_convrot,
    ),
    ConversionMethodDefinition(
        action="int8_convrot_mse",
        label="INT8 ConvRot MSE",
        is_convrot=True,
        internal_stages=INT8_MSE_INTERNAL_STAGES,
        plan=plan_int8_convrot,
        marker_payload=INT8_CONVROT_MARKER_PAYLOAD,
        encode=_encode_int8_convrot_mse,
    ),
    ConversionMethodDefinition(
        action="convrot_w4a4",
        label="W4A4 ConvRot",
        is_convrot=True,
        internal_stages=CONVROT_INTERNAL_STAGES,
        plan=plan_convrot_w4a4,
        marker_payload=CONVROT_W4A4_MARKER_PAYLOAD,
        encode=_encode_convrot_w4a4,
    ),
    ConversionMethodDefinition(
        action="convrot_w4a4_mse",
        label="W4A4 ConvRot MSE",
        is_convrot=True,
        internal_stages=W4A4_MSE_INTERNAL_STAGES,
        plan=plan_convrot_w4a4,
        marker_payload=CONVROT_W4A4_MARKER_PAYLOAD,
        encode=_encode_convrot_w4a4_mse,
    ),
    ConversionMethodDefinition(
        action="convrot_w4a4_mse_gptq",
        label="W4A4 ConvRot MSE GPTQ",
        is_convrot=True,
        internal_stages=GPTQ_INTERNAL_STAGES,
        plan=plan_convrot_w4a4,
        marker_payload=CONVROT_W4A4_MARKER_PAYLOAD,
        encode=_encode_convrot_w4a4_mse_gptq,
    ),
    ConversionMethodDefinition(
        action="int5_convrot",
        label="INT5 ConvRot",
        is_convrot=True,
        internal_stages=CONVROT_INTERNAL_STAGES,
        plan=plan_int5_convrot,
        marker_payload=INT5_CONVROT_MARKER_PAYLOAD,
        encode=_encode_int5_convrot,
    ),
    ConversionMethodDefinition(
        action="int5_convrot_mse",
        label="INT5 ConvRot MSE",
        is_convrot=True,
        internal_stages=INT5_MSE_INTERNAL_STAGES,
        plan=plan_int5_convrot,
        marker_payload=INT5_CONVROT_MARKER_PAYLOAD,
        encode=_encode_int5_convrot_mse,
    ),
    ConversionMethodDefinition(
        action="int6_convrot",
        label="INT6 ConvRot",
        is_convrot=True,
        internal_stages=CONVROT_INTERNAL_STAGES,
        plan=plan_int6_convrot,
        marker_payload=INT6_CONVROT_MARKER_PAYLOAD,
        encode=_encode_int6_convrot,
    ),
    ConversionMethodDefinition(
        action="int6_convrot_mse",
        label="INT6 ConvRot MSE",
        is_convrot=True,
        internal_stages=INT6_MSE_INTERNAL_STAGES,
        plan=plan_int6_convrot,
        marker_payload=INT6_CONVROT_MARKER_PAYLOAD,
        encode=_encode_int6_convrot_mse,
    ),
    ConversionMethodDefinition(
        action="int7_convrot",
        label="INT7 ConvRot",
        is_convrot=True,
        internal_stages=CONVROT_INTERNAL_STAGES,
        plan=plan_int7_convrot,
        marker_payload=INT7_CONVROT_MARKER_PAYLOAD,
        encode=_encode_int7_convrot,
    ),
    ConversionMethodDefinition(
        action="int7_convrot_mse",
        label="INT7 ConvRot MSE",
        is_convrot=True,
        internal_stages=INT7_MSE_INTERNAL_STAGES,
        plan=plan_int7_convrot,
        marker_payload=INT7_CONVROT_MARKER_PAYLOAD,
        encode=_encode_int7_convrot_mse,
    ),
)


def _validate_catalog() -> None:
    actions = tuple(method.action for method in _METHODS)
    if len(actions) != len(set(actions)):
        raise ValueError(
            "Conversion method catalog contains duplicate actions."
        )
    if "keep" in actions:
        raise ValueError("Conversion method catalog must not define keep.")

    for method in _METHODS:
        if not method.label.strip():
            raise ValueError(
                f"Conversion method {method.action} must have a label."
            )
        if not method.marker_payload:
            raise ValueError(
                f"Conversion method {method.action} must have marker bytes."
            )
        if not callable(method.plan) or not callable(method.encode):
            raise ValueError(
                f"Conversion method {method.action} must define "
                "plan and encode."
            )


_validate_catalog()


def get_conversion_method(action: str) -> ConversionMethodDefinition:
    for method in _METHODS:
        if method.action == action:
            return method
    raise ValueError(f"Unsupported quantization action: {action}.")


def supported_quantizing_actions() -> tuple[QuantizationAction, ...]:
    return tuple(method.action for method in _METHODS)


def convrot_conversion_methods() -> tuple[ConversionMethodDefinition, ...]:
    return tuple(method for method in _METHODS if method.is_convrot)
