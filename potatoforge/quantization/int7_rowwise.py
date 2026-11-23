"""Rowwise INT7 and ConvRot W7 quantization (see rowwise_convrot)."""

from __future__ import annotations

from typing import Final

from .int7_packing import (
    Int7PackedResult,
    pack_int7_row_major,
    unpack_int7_row_major,
)
from .rowwise_convrot import ConvRotPackedResult, RowwiseFormat, RowwiseResult
from .w4a4_mse_native import create_exact_mse_candidate


INT7_QMAX: Final[int] = 63
INT7_QMIN: Final[int] = -63
INT7_FORMAT_NAME: Final[str] = "INT7"

INT7 = RowwiseFormat(
    name=INT7_FORMAT_NAME,
    qmin=INT7_QMIN,
    qmax=INT7_QMAX,
    storage_qmin=INT7_QMIN,
    pack=pack_int7_row_major,
    unpack=lambda codes, shape: unpack_int7_row_major(Int7PackedResult(codes, shape)),
    mse_candidate=lambda rotated, device: create_exact_mse_candidate(rotated, bits=7),
    fused_mse_search=False,
)

Int7RowwiseResult = RowwiseResult
Int7ConvRotPackedResult = ConvRotPackedResult

quantize_int7_rowwise = INT7.quantize_rowwise
select_int7_mse_scale = INT7.select_mse_scale
dequantize_int7_rowwise = INT7.dequantize_rowwise
quantize_int7_convrot = INT7.quantize_convrot
quantize_int7_convrot_mse = INT7.quantize_convrot_mse
quantize_int7_convrot_packed = INT7.quantize_convrot_packed
quantize_int7_convrot_mse_packed = INT7.quantize_convrot_mse_packed
dequantize_int7_convrot = INT7.dequantize_convrot
dequantize_int7_convrot_packed = INT7.dequantize_convrot_packed
