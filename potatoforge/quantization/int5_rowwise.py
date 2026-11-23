"""Rowwise INT5 and ConvRot W5 quantization (see rowwise_convrot)."""

from __future__ import annotations

from typing import Final

from .int5_packing import (
    INT5_STORAGE_QMIN,
    Int5PackedResult,
    pack_int5_row_major,
    unpack_int5_row_major,
)
from .rowwise_convrot import ConvRotPackedResult, RowwiseFormat, RowwiseResult, native_mse_candidate
from .w4a4_mse_native import load_int5_convrot_mse_candidate


INT5_QMAX: Final[int] = 15
INT5_QMIN: Final[int] = -15
INT5_FORMAT_NAME: Final[str] = "INT5"

INT5 = RowwiseFormat(
    name=INT5_FORMAT_NAME,
    qmin=INT5_QMIN,
    qmax=INT5_QMAX,
    storage_qmin=INT5_STORAGE_QMIN,
    pack=pack_int5_row_major,
    unpack=lambda codes, shape: unpack_int5_row_major(Int5PackedResult(codes, shape)),
    mse_candidate=native_mse_candidate(load_int5_convrot_mse_candidate),
    fused_mse_search=True,
)

Int5RowwiseResult = RowwiseResult
Int5ConvRotPackedResult = ConvRotPackedResult

quantize_int5_rowwise = INT5.quantize_rowwise
select_int5_mse_scale = INT5.select_mse_scale
dequantize_int5_rowwise = INT5.dequantize_rowwise
quantize_int5_convrot = INT5.quantize_convrot
quantize_int5_convrot_mse = INT5.quantize_convrot_mse
quantize_int5_convrot_packed = INT5.quantize_convrot_packed
quantize_int5_convrot_mse_packed = INT5.quantize_convrot_mse_packed
dequantize_int5_convrot = INT5.dequantize_convrot
dequantize_int5_convrot_packed = INT5.dequantize_convrot_packed
