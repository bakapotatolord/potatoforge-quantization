from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from .timing import TensorTiming

QuantizationAction: TypeAlias = str


@dataclass(frozen=True, slots=True)
class EncodedQuantizedPayload:
    codes: Any
    scales: Any
    # Tensors written after the marker, in plan order (output_tensors[3:]).
    extras: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class ConversionMethodDefinition:
    action: QuantizationAction
    label: str
    is_convrot: bool
    internal_stages: tuple[str, ...]
    plan: Callable[[str, Mapping[str, Any]], Any]
    marker_payload: bytes
    encode: Callable[
        [Any, Any, TensorTiming | None],
        EncodedQuantizedPayload,
    ]
