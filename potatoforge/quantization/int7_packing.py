"""CPU/PyTorch reference packing for logical rowwise INT7 values."""

from __future__ import annotations

from typing import Final, NamedTuple

import torch


INT7_QMAX: Final[int] = 63
INT7_QMIN: Final[int] = -63
INT7_VALUES_PER_GROUP: Final[int] = 8
INT7_PACKED_BYTES_PER_GROUP: Final[int] = 7


class Int7PackedResult(NamedTuple):
    """Packed W7 bytes plus the original logical matrix shape."""

    packed_codes: torch.Tensor
    original_shape: tuple[int, int]


def _validate_codes(codes: torch.Tensor) -> None:
    if codes.dtype != torch.int8:
        raise ValueError("INT7 codes must use torch.int8 storage.")
    if codes.ndim != 2:
        raise ValueError("INT7 codes must be a rank-2 matrix.")
    if bool((codes < INT7_QMIN).any()) or bool((codes > INT7_QMAX).any()):
        raise ValueError("INT7 codes must be in the signed range [-63, 63].")


def pack_int7_row_major(codes: torch.Tensor) -> Int7PackedResult:
    """Pack eight signed W7 codes into seven uint8 bytes per row group."""

    _validate_codes(codes)
    out_features, in_features = codes.shape
    if in_features % INT7_VALUES_PER_GROUP != 0:
        raise ValueError("INT7 packed input features must be divisible by 8.")

    group_count = in_features // INT7_VALUES_PER_GROUP
    logical_codes = codes.to(torch.int64) & 0x7F
    grouped_codes = logical_codes.reshape(
        out_features,
        group_count,
        INT7_VALUES_PER_GROUP,
    )
    packed_word = sum(
        grouped_codes[:, :, index] << (7 * index)
        for index in range(INT7_VALUES_PER_GROUP)
    )
    packed_codes = torch.stack(
        tuple(
            (packed_word >> (8 * index)) & 0xFF
            for index in range(INT7_PACKED_BYTES_PER_GROUP)
        ),
        dim=-1,
    ).reshape(
        out_features,
        group_count * INT7_PACKED_BYTES_PER_GROUP,
    ).to(torch.uint8)

    return Int7PackedResult(
        packed_codes=packed_codes,
        original_shape=(out_features, in_features),
    )


def _validate_packed_result(result: Int7PackedResult) -> None:
    if not isinstance(result.packed_codes, torch.Tensor):
        raise ValueError("Packed INT7 data must use torch.uint8 storage.")
    if result.packed_codes.dtype != torch.uint8:
        raise ValueError("Packed INT7 data must use torch.uint8 storage.")
    if result.packed_codes.ndim != 2:
        raise ValueError("Packed INT7 data must be a rank-2 matrix.")
    if (
        not isinstance(result.original_shape, tuple)
        or len(result.original_shape) != 2
        or any(type(dimension) is not int for dimension in result.original_shape)
        or any(dimension < 0 for dimension in result.original_shape)
    ):
        raise ValueError("INT7 original shape must contain two non-negative integers.")

    out_features, in_features = result.original_shape
    if in_features % INT7_VALUES_PER_GROUP != 0:
        raise ValueError("INT7 packed input features must be divisible by 8.")

    group_count = in_features // INT7_VALUES_PER_GROUP
    expected_shape = (
        out_features,
        group_count * INT7_PACKED_BYTES_PER_GROUP,
    )
    if tuple(result.packed_codes.shape) != expected_shape:
        raise ValueError(
            "Packed INT7 shape does not match the original shape."
        )


def unpack_int7_row_major(result: Int7PackedResult) -> torch.Tensor:
    """Unpack uint8 W7 bytes into the original logical matrix shape."""

    _validate_packed_result(result)
    out_features, in_features = result.original_shape
    group_count = in_features // INT7_VALUES_PER_GROUP
    packed_bytes = result.packed_codes.to(torch.int64).reshape(
        out_features,
        group_count,
        INT7_PACKED_BYTES_PER_GROUP,
    )
    packed_word = sum(
        packed_bytes[:, :, index] << (8 * index)
        for index in range(INT7_PACKED_BYTES_PER_GROUP)
    )
    unsigned_codes = torch.stack(
        tuple(
            (packed_word >> (7 * index)) & 0x7F
            for index in range(INT7_VALUES_PER_GROUP)
        ),
        dim=-1,
    ).reshape(out_features, group_count * INT7_VALUES_PER_GROUP)
    return torch.where(
        unsigned_codes < 64,
        unsigned_codes,
        unsigned_codes - 128,
    ).to(torch.int8)
