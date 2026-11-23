"""Rowwise INT6 and ConvRot W6 quantization (see rowwise_convrot)."""

from __future__ import annotations

from typing import Final

from .int6_packing import (
    Int6PackedResult,
    pack_int6_row_major,
    unpack_int6_row_major,
)
from .rowwise_convrot import ConvRotPackedResult, RowwiseFormat, RowwiseResult, native_mse_candidate
from .w4a4_mse_native import load_int6_convrot_mse_candidate


INT6_QMAX: Final[int] = 31
INT6_QMIN: Final[int] = -31
INT6_FORMAT_NAME: Final[str] = "INT6"

INT6 = RowwiseFormat(
    name=INT6_FORMAT_NAME,
    qmin=INT6_QMIN,
    qmax=INT6_QMAX,
    storage_qmin=INT6_QMIN,
    pack=pack_int6_row_major,
    unpack=lambda codes, shape: unpack_int6_row_major(Int6PackedResult(codes, shape)),
    mse_candidate=native_mse_candidate(load_int6_convrot_mse_candidate),
    fused_mse_search=True,
)

Int6RowwiseResult = RowwiseResult
Int6ConvRotPackedResult = ConvRotPackedResult

quantize_int6_rowwise = INT6.quantize_rowwise
select_int6_mse_scale = INT6.select_mse_scale
dequantize_int6_rowwise = INT6.dequantize_rowwise
quantize_int6_convrot = INT6.quantize_convrot
quantize_int6_convrot_mse = INT6.quantize_convrot_mse
quantize_int6_convrot_packed = INT6.quantize_convrot_packed
quantize_int6_convrot_mse_packed = INT6.quantize_convrot_mse_packed
dequantize_int6_convrot = INT6.dequantize_convrot
dequantize_int6_convrot_packed = INT6.dequantize_convrot_packed
