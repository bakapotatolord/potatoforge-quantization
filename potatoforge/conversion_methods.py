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
    plan_w3a8,
    plan_w4a8,
    plan_w5a8,
    plan_w6a8,
    W3A8_MARKER_PAYLOAD,
    W4A8_MARKER_PAYLOAD,
    W5A8_MARKER_PAYLOAD,
    W6A8_MARKER_PAYLOAD,
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
from .quantization.w3a8 import quantize_w3a8
from .quantization.w4a8 import quantize_w4a8
from .quantization.w5a8 import quantize_w5a8
from .quantization.w6a8 import quantize_w6a8
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

Encoder = Callable[[torch.Tensor, "str | torch.device", "TensorTiming | None"], EncodedQuantizedPayload]


def _run_quantize(
    record: TensorTiming | None,
    operation: Callable[[], Any],
) -> Any:
    with timed_stage(record, "quantize"):
        return operation()


def _quantizer(name: str) -> Callable[..., Any]:
    # Resolved at call time so tests can patch the module attribute.
    return globals()[name]


def _device_encoder(name: str, *, codes: str = "packed_codes", cpu_timings: bool = False) -> Encoder:
    """Encode with a quantizer taking ``device`` and ``internal_timings``.

    ``device`` is passed only off-CPU; internal timings are passed when a
    record exists and the device is not CPU, unless ``cpu_timings`` is set.
    """

    def encode(
        weights: torch.Tensor,
        device: str | torch.device,
        record: TensorTiming | None,
    ) -> EncodedQuantizedPayload:
        on_cpu = device == "cpu"
        options: dict[str, Any] = {} if on_cpu else {"device": device}
        if record is not None and (cpu_timings or not on_cpu):
            options["internal_timings"] = record.internal_stages
        result = _run_quantize(record, lambda: _quantizer(name)(weights, **options))
        return EncodedQuantizedPayload(getattr(result, codes), result.scales)

    return encode


def _pack_encoder(name: str, pack: Callable[[torch.Tensor], Any]) -> Encoder:
    """Quantize on the current device, then pack codes as a separate stage."""

    def encode(
        weights: torch.Tensor,
        device: str | torch.device,
        record: TensorTiming | None,
    ) -> EncodedQuantizedPayload:
        result = _run_quantize(record, lambda: _quantizer(name)(weights))
        with timed_stage(record, "pack"):
            packed = pack(result.codes)
        return EncodedQuantizedPayload(packed.packed_codes, result.scales)

    return encode


def _cpu_or_device(cpu: Encoder, accelerated: Encoder) -> Encoder:
    return lambda weights, device, record: (cpu if device == "cpu" else accelerated)(
        weights, device, record
    )


def _encode_int8(
    weights: torch.Tensor,
    device: str | torch.device,
    record: TensorTiming | None,
) -> EncodedQuantizedPayload:
    result = _run_quantize(record, lambda: _quantizer("quantize_int8_tensorwise")(weights))
    return EncodedQuantizedPayload(result.codes, result.scales)


def _encode_w3a8(
    weights: torch.Tensor,
    device: str | torch.device,
    record: TensorTiming | None,
) -> EncodedQuantizedPayload:
    result = _run_quantize(record, lambda: quantize_w3a8(weights, device=device))
    return EncodedQuantizedPayload(
        result.packed_codes,
        result.group_scales,
        extras=(result.channel_scales, result.codebook),
    )


def _encode_w5a8(
    weights: torch.Tensor,
    device: str | torch.device,
    record: TensorTiming | None,
) -> EncodedQuantizedPayload:
    result = _run_quantize(record, lambda: quantize_w5a8(weights, device=device))
    return EncodedQuantizedPayload(
        result.packed_codes,
        result.group_scales,
        extras=(result.channel_scales,),
    )


def _encode_w4a8(
    weights: torch.Tensor,
    device: str | torch.device,
    record: TensorTiming | None,
) -> EncodedQuantizedPayload:
    result = _run_quantize(record, lambda: quantize_w4a8(weights, device=device))
    return EncodedQuantizedPayload(
        result.packed_codes,
        result.group_scales,
        extras=(result.channel_scales, result.codebook),
    )


def _encode_w6a8(
    weights: torch.Tensor,
    device: str | torch.device,
    record: TensorTiming | None,
) -> EncodedQuantizedPayload:
    result = _run_quantize(record, lambda: quantize_w6a8(weights, device=device))
    return EncodedQuantizedPayload(
        result.packed_codes,
        result.group_scales,
        extras=(result.channel_scales,),
    )


def _encode_convrot_w4a4_mse_gptq(
    weights: torch.Tensor,
    device: str | torch.device,
    record: TensorTiming | None,
) -> EncodedQuantizedPayload:
    raise ValueError("GPTQ encoding requires a calibration artifact and weight name.")


# action, label, is_convrot, internal_stages, plan, marker_payload, encode
_METHOD_TABLE: tuple[tuple[Any, ...], ...] = (
    ("int8", "INT8", False, (), plan_int8_tensorwise, INT8_MARKER_PAYLOAD, _encode_int8),
    ("int6_rowwise", "INT6 Rowwise", False, (), plan_int6_rowwise, INT6_ROWWISE_MARKER_PAYLOAD,
     _pack_encoder("quantize_int6_rowwise", pack_int6_row_major)),
    ("int7_rowwise", "INT7 Rowwise", False, (), plan_int7_rowwise, INT7_ROWWISE_MARKER_PAYLOAD,
     _pack_encoder("quantize_int7_rowwise", pack_int7_row_major)),
    ("int8_convrot", "INT8 ConvRot", True, CONVROT_INTERNAL_STAGES, plan_int8_convrot,
     INT8_CONVROT_MARKER_PAYLOAD,
     _device_encoder("quantize_int8_convrot", codes="codes", cpu_timings=True)),
    ("int8_convrot_mse", "INT8 ConvRot MSE", True, INT8_MSE_INTERNAL_STAGES, plan_int8_convrot,
     INT8_CONVROT_MARKER_PAYLOAD,
     _device_encoder("quantize_int8_convrot_mse", codes="codes", cpu_timings=True)),
    ("convrot_w4a4", "W4A4 ConvRot", True, CONVROT_INTERNAL_STAGES, plan_convrot_w4a4,
     CONVROT_W4A4_MARKER_PAYLOAD, _device_encoder("quantize_convrot_w4a4")),
    ("convrot_w4a4_mse", "W4A4 ConvRot MSE", True, W4A4_MSE_INTERNAL_STAGES, plan_convrot_w4a4,
     CONVROT_W4A4_MARKER_PAYLOAD, _device_encoder("quantize_convrot_w4a4_mse")),
    ("convrot_w4a4_mse_gptq", "W4A4 ConvRot MSE GPTQ", True, GPTQ_INTERNAL_STAGES,
     plan_convrot_w4a4, CONVROT_W4A4_MARKER_PAYLOAD, _encode_convrot_w4a4_mse_gptq),
    ("int5_convrot", "INT5 ConvRot", True, CONVROT_INTERNAL_STAGES, plan_int5_convrot,
     INT5_CONVROT_MARKER_PAYLOAD, _device_encoder("quantize_int5_convrot_packed")),
    ("int5_convrot_mse", "INT5 ConvRot MSE", True, INT5_MSE_INTERNAL_STAGES, plan_int5_convrot,
     INT5_CONVROT_MARKER_PAYLOAD, _device_encoder("quantize_int5_convrot_mse_packed")),
    ("int6_convrot", "INT6 ConvRot", True, CONVROT_INTERNAL_STAGES, plan_int6_convrot,
     INT6_CONVROT_MARKER_PAYLOAD,
     _cpu_or_device(_pack_encoder("quantize_int6_convrot", pack_int6_row_major),
                    _device_encoder("quantize_int6_convrot_packed"))),
    ("int6_convrot_mse", "INT6 ConvRot MSE", True, INT6_MSE_INTERNAL_STAGES, plan_int6_convrot,
     INT6_CONVROT_MARKER_PAYLOAD,
     _cpu_or_device(_pack_encoder("quantize_int6_convrot_mse", pack_int6_row_major),
                    _device_encoder("quantize_int6_convrot_mse_packed"))),
    ("int7_convrot", "INT7 ConvRot", True, CONVROT_INTERNAL_STAGES, plan_int7_convrot,
     INT7_CONVROT_MARKER_PAYLOAD,
     _cpu_or_device(_pack_encoder("quantize_int7_convrot", pack_int7_row_major),
                    _device_encoder("quantize_int7_convrot_packed"))),
    ("int7_convrot_mse", "INT7 ConvRot MSE", True, INT7_MSE_INTERNAL_STAGES, plan_int7_convrot,
     INT7_CONVROT_MARKER_PAYLOAD,
     _cpu_or_device(_pack_encoder("quantize_int7_convrot_mse", pack_int7_row_major),
                    _device_encoder("quantize_int7_convrot_mse_packed"))),
    ("w4a8", "W4A8 ConvRot", True, (), plan_w4a8, W4A8_MARKER_PAYLOAD, _encode_w4a8),
    ("w6a8", "W6A8 ConvRot", True, (), plan_w6a8, W6A8_MARKER_PAYLOAD, _encode_w6a8),
    ("w3a8", "W3A8 ConvRot", True, (), plan_w3a8, W3A8_MARKER_PAYLOAD, _encode_w3a8),
    ("w5a8", "W5A8 ConvRot", True, (), plan_w5a8, W5A8_MARKER_PAYLOAD, _encode_w5a8),
)

_METHODS: tuple[ConversionMethodDefinition, ...] = tuple(
    ConversionMethodDefinition(*row) for row in _METHOD_TABLE
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
