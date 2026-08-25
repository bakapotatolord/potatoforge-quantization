"""CPU/PyTorch reference packing for logical rowwise INT6 values."""

from __future__ import annotations

from typing import Final, NamedTuple

import torch


INT6_QMAX: Final[int] = 31
INT6_QMIN: Final[int] = -31
INT6_VALUES_PER_GROUP: Final[int] = 4
INT6_PACKED_BYTES_PER_GROUP: Final[int] = 3


class Int6PackedResult(NamedTuple):
    """Packed W6 bytes plus the original logical matrix shape."""

    packed_codes: torch.Tensor
    original_shape: tuple[int, int]


def _validate_codes(codes: torch.Tensor) -> None:
    if codes.dtype != torch.int8:
        raise ValueError("INT6 codes must use torch.int8 storage.")
    if codes.ndim != 2:
        raise ValueError("INT6 codes must be a rank-2 matrix.")
    if bool((codes < INT6_QMIN).any()) or bool((codes > INT6_QMAX).any()):
        raise ValueError("INT6 codes must be in the signed range [-31, 31].")


def pack_int6_row_major(codes: torch.Tensor) -> Int6PackedResult:
    """Pack four signed W6 codes into three uint8 bytes per row group."""

    _validate_codes(codes)
    out_features, in_features = codes.shape
    if in_features % INT6_VALUES_PER_GROUP != 0:
        raise ValueError("INT6 packed input features must be divisible by 4.")

    group_count = in_features // INT6_VALUES_PER_GROUP
    logical_codes = codes.to(torch.int32) & 0x3F
    grouped_codes = logical_codes.reshape(
        out_features,
        group_count,
        INT6_VALUES_PER_GROUP,
    )
    packed_word = (
        grouped_codes[:, :, 0]
        | (grouped_codes[:, :, 1] << 6)
        | (grouped_codes[:, :, 2] << 12)
        | (grouped_codes[:, :, 3] << 18)
    )
    packed_codes = torch.stack(
        (
            packed_word & 0xFF,
            (packed_word >> 8) & 0xFF,
            (packed_word >> 16) & 0xFF,
        ),
        dim=-1,
    ).reshape(
        out_features,
        group_count * INT6_PACKED_BYTES_PER_GROUP,
    ).to(torch.uint8)

    return Int6PackedResult(
        packed_codes=packed_codes,
        original_shape=(out_features, in_features),
    )


def _validate_packed_result(result: Int6PackedResult) -> None:
    if result.packed_codes.dtype != torch.uint8:
        raise ValueError("Packed INT6 data must use torch.uint8 storage.")
    if result.packed_codes.ndim != 2:
        raise ValueError("Packed INT6 data must be a rank-2 matrix.")
    if (
        len(result.original_shape) != 2
        or any(type(dimension) is not int for dimension in result.original_shape)
        or any(dimension < 0 for dimension in result.original_shape)
    ):
        raise ValueError("INT6 original shape must contain two non-negative integers.")
    out_features, in_features = result.original_shape
    if in_features % INT6_VALUES_PER_GROUP != 0:
        raise ValueError("INT6 packed input features must be divisible by 4.")

    group_count = in_features // INT6_VALUES_PER_GROUP
    expected_shape = (
        out_features,
        group_count * INT6_PACKED_BYTES_PER_GROUP,
    )
    if tuple(result.packed_codes.shape) != expected_shape:
        raise ValueError(
            "Packed INT6 shape does not match the original shape."
        )


def unpack_int6_row_major(result: Int6PackedResult) -> torch.Tensor:
    """Unpack uint8 W6 bytes into the original logical matrix shape."""

    _validate_packed_result(result)
    out_features, in_features = result.original_shape
    group_count = in_features // INT6_VALUES_PER_GROUP
    packed_bytes = result.packed_codes.to(torch.int32).reshape(
        out_features,
        group_count,
        INT6_PACKED_BYTES_PER_GROUP,
    )
    packed_word = (
        packed_bytes[:, :, 0]
        | (packed_bytes[:, :, 1] << 8)
        | (packed_bytes[:, :, 2] << 16)
    )
    unsigned_codes = torch.stack(
        (
            packed_word & 0x3F,
            (packed_word >> 6) & 0x3F,
            (packed_word >> 12) & 0x3F,
            (packed_word >> 18) & 0x3F,
        ),
        dim=-1,
    ).reshape(out_features, group_count * INT6_VALUES_PER_GROUP)
    signed_codes = torch.where(
        unsigned_codes < 32,
        unsigned_codes,
        unsigned_codes - 64,
    ).to(torch.int8)

    return signed_codes
