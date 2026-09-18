"""CPU/PyTorch reference packing for logical rowwise INT5 values."""

from __future__ import annotations

from typing import Final, NamedTuple

import torch


INT5_VALUES_PER_GROUP: Final[int] = 8
INT5_PACKED_BYTES_PER_GROUP: Final[int] = 5
INT5_STORAGE_QMIN: Final[int] = -16
INT5_STORAGE_QMAX: Final[int] = 15


class Int5PackedResult(NamedTuple):
    """Packed W5 bytes plus the original logical matrix shape."""

    packed_codes: torch.Tensor
    original_shape: tuple[int, int]


def _validate_codes(codes: torch.Tensor) -> None:
    if codes.dtype != torch.int8:
        raise ValueError("INT5 codes must use torch.int8 storage.")
    if codes.ndim != 2:
        raise ValueError("INT5 codes must be a rank-2 matrix.")
    if bool((codes < INT5_STORAGE_QMIN).any()) or bool(
        (codes > INT5_STORAGE_QMAX).any()
    ):
        raise ValueError("INT5 codes must be in the signed range [-16, 15].")


def pack_int5_row_major(codes: torch.Tensor) -> Int5PackedResult:
    """Pack eight signed W5 codes into five uint8 bytes per row group."""

    _validate_codes(codes)
    out_features, in_features = codes.shape
    if in_features % INT5_VALUES_PER_GROUP != 0:
        raise ValueError("INT5 packed input features must be divisible by 8.")

    group_count = in_features // INT5_VALUES_PER_GROUP
    logical_codes = codes.to(torch.int64) & 0x1F
    grouped_codes = logical_codes.reshape(
        out_features,
        group_count,
        INT5_VALUES_PER_GROUP,
    )
    packed_word = sum(
        grouped_codes[:, :, index] << (5 * index)
        for index in range(INT5_VALUES_PER_GROUP)
    )
    packed_codes = torch.stack(
        tuple(
            (packed_word >> (8 * index)) & 0xFF
            for index in range(INT5_PACKED_BYTES_PER_GROUP)
        ),
        dim=-1,
    ).reshape(
        out_features,
        group_count * INT5_PACKED_BYTES_PER_GROUP,
    ).to(torch.uint8)

    return Int5PackedResult(
        packed_codes=packed_codes,
        original_shape=(out_features, in_features),
    )


def _validate_packed_result(result: Int5PackedResult) -> None:
    if result.packed_codes.dtype != torch.uint8:
        raise ValueError("Packed INT5 data must use torch.uint8 storage.")
    if result.packed_codes.ndim != 2:
        raise ValueError("Packed INT5 data must be a rank-2 matrix.")
    if (
        len(result.original_shape) != 2
        or any(type(dimension) is not int for dimension in result.original_shape)
        or any(dimension < 0 for dimension in result.original_shape)
    ):
        raise ValueError(
            "INT5 original shape must contain two non-negative integers."
        )

    out_features, in_features = result.original_shape
    if in_features % INT5_VALUES_PER_GROUP != 0:
        raise ValueError("INT5 packed input features must be divisible by 8.")

    group_count = in_features // INT5_VALUES_PER_GROUP
    expected_shape = (
        out_features,
        group_count * INT5_PACKED_BYTES_PER_GROUP,
    )
    if tuple(result.packed_codes.shape) != expected_shape:
        raise ValueError(
            "Packed INT5 shape does not match the original shape."
        )


def unpack_int5_row_major(result: Int5PackedResult) -> torch.Tensor:
    """Unpack uint8 W5 bytes into the original logical matrix shape."""

    _validate_packed_result(result)
    out_features, in_features = result.original_shape
    group_count = in_features // INT5_VALUES_PER_GROUP
    packed_bytes = result.packed_codes.to(torch.int64).reshape(
        out_features,
        group_count,
        INT5_PACKED_BYTES_PER_GROUP,
    )
    packed_word = sum(
        packed_bytes[:, :, index] << (8 * index)
        for index in range(INT5_PACKED_BYTES_PER_GROUP)
    )
    unsigned_codes = torch.stack(
        tuple(
            (packed_word >> (5 * index)) & 0x1F
            for index in range(INT5_VALUES_PER_GROUP)
        ),
        dim=-1,
    ).reshape(out_features, group_count * INT5_VALUES_PER_GROUP)
    return torch.where(
        unsigned_codes < 16,
        unsigned_codes,
        unsigned_codes - 32,
    ).to(torch.int8)
