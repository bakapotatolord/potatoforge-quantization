import torch
from typing import NamedTuple

from .hadamard import CONVROT_GROUP_SIZE, apply_hadamard_rotation

class W4RowwiseResult(NamedTuple):
    packed_codes: torch.Tensor
    scales: torch.Tensor

class ConvRotW4A4Result(NamedTuple):
    packed_codes: torch.Tensor
    scales: torch.Tensor


def pack_signed_int4_row_major(codes: torch.Tensor) -> torch.Tensor:
    if codes.dtype != torch.int8:
        raise ValueError("INT4 codes must use torch.int8 storage")

    if codes.ndim != 2:
        raise ValueError("INT4 codes must be a 2D matrix")

    if codes.shape[1] % 2 != 0:
        raise ValueError("INT4 code rows must have an even width")

    if torch.any(codes < -8) or torch.any(codes > 7):
        raise ValueError("INT4 codes must be in the range [-8, 7]")

    low_nibbles = codes[:, 0::2].to(torch.int32) & 0x0F
    high_nibbles = codes[:,1::2].to(torch.int32) & 0x0F

    return (low_nibbles | (high_nibbles << 4)).to(torch.int8)

def unpack_signed_int4_row_major(packed: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.int8:
        raise ValueError("packed INT4 data must use torch.int8 storage")

    if packed.ndim != 2:
        raise ValueError("packed INT4 data must be a 2D matrix")

    packed_i32 = packed.to(torch.int32)

    low_nibbles = packed_i32 & 0x0F
    high_nibbles = (packed_i32 >> 4) & 0x0F

    low_codes = torch.where(
        low_nibbles >= 8,
        low_nibbles - 16,
        low_nibbles,
    )
    high_codes = torch.where(
        high_nibbles >= 8,
        high_nibbles - 16,
        high_nibbles,
    )

    return torch.stack(
        (low_codes, high_codes),
        dim=2,
    ).flatten(start_dim=1).to(torch.int8)

def quantize_w4_rowwise(weights: torch.Tensor) -> W4RowwiseResult:
    if weights.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(
            "W4 weights must use torch.bfloat16, torch.float16, "
            "or torch.float32"
        )

    if weights.ndim != 2:
        raise ValueError("W4 weights must be a 2D matrix")

    math_weights = (
        weights.float()
        if weights.dtype == torch.float16
        else weights
    )

    scales = (
        math_weights.abs()
        .amax(dim=1, keepdim=True)
        .float()
        .div(7)
        .clamp_min(1e-10)
    )

    scale_math = scales.to(math_weights.dtype)

    codes = (
        torch.round(math_weights / scale_math)
        .clamp(-7, 7)
        .to(torch.int8)
    )

    packed_codes = pack_signed_int4_row_major(codes)

    return W4RowwiseResult(
        packed_codes=packed_codes,
        scales=scales,
    )

def dequantize_w4_rowwise(result: W4RowwiseResult) -> torch.Tensor:
    codes = unpack_signed_int4_row_major(result.packed_codes)

    return codes.float() * result.scales

def quantize_convrot_w4a4(weights: torch.Tensor) -> ConvRotW4A4Result:
    rotated_weights = apply_hadamard_rotation(
        weights,
        group_size=CONVROT_GROUP_SIZE,
    )
    rowwise_result = quantize_w4_rowwise(rotated_weights)

    return ConvRotW4A4Result(
        packed_codes=rowwise_result.packed_codes,
        scales=rowwise_result.scales.squeeze(dim=1),
    )

def dequantize_convrot_w4a4(result: ConvRotW4A4Result) -> torch.Tensor:
    rotated_codes = unpack_signed_int4_row_major(
        result.packed_codes,
    )

    rotated_weights = (
        rotated_codes.float()
        * result.scales.unsqueeze(dim=1)
    )

    return apply_hadamard_rotation(
        rotated_weights,
        group_size=CONVROT_GROUP_SIZE,
    )
