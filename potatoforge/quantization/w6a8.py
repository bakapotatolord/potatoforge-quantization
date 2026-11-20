"""W6A8 ConvRot weights in ComfyUI's ``w6a8_int8`` storage contract.

A port of comfy-kitchen's eager 6-bit recipe (``comfy_kitchen/backends/eager/
w4a8_int8.py`` with ``bits=6`` at upstream e769274), loaded by ComfyUI through
the same ``AsymW4A8Int8Layout`` and kernels as W4A8:

- ConvRot: 256-wide normalized Hadamard rotation of the input dimension.
- Groups of 16 inputs share one scale. Levels are the uniform integers
  -31..31 (no codebook); two ALS passes refine the group scale.
- Each fp8 group scale then picks the best of itself and its two e4m3
  neighbours by squared error on the decoded INT8 grid the kernels use.
- ``weight`` int8 [N, 3K/4]: the K/2-byte nibble plane of each code's low
  4 bits, then the K/4-byte plane of its top 2 bits (column c in byte c//4
  at bit 2*(c%4)). Codes are stored as level + 32, in 1..63.
- ``weight_s_rel`` float8_e4m3fn [N, K/16]; ``weight_s_channel`` float32 [N].
"""

from __future__ import annotations

from typing import Final, NamedTuple

import torch

from .hadamard import CONVROT_GROUP_SIZE, apply_hadamard_rotation
from .rowwise import resolve_convrot_device
from .w4a8 import (
    QUANT_ROW_ELEM_BUDGET,
    W4A8_GROUP_SIZE,
    _assign_grid,
    pack_w4a8_codes,
    validate_w4a8_weights,
)

W6A8_FORMAT_NAME: Final[str] = "W6A8"
W6A8_LEVELS: Final[int] = 31
W6A8_ZERO_POINT: Final[int] = 32
_ALS_ITERS: Final[int] = 2
_E4M3_MAX_FINITE_BITS: Final[int] = 0x7E


class W6A8Result(NamedTuple):
    packed_codes: torch.Tensor
    group_scales: torch.Tensor
    channel_scales: torch.Tensor


def pack_w6a8_codes(codes: torch.Tensor) -> torch.Tensor:
    """Unsigned 6-bit codes [N, K] -> int8 [N, 3K/4]: nibble plane, then 2-bit plane."""
    if codes.ndim != 2 or codes.shape[1] % 4:
        raise ValueError("W6A8 codes must be a 2D matrix with a column count divisible by 4.")
    high = codes >> 4
    plane = high[:, 0::4] & 3
    for index in range(1, 4):
        plane |= (high[:, index::4] & 3) << (2 * index)
    return torch.cat((pack_w4a8_codes(codes & 0xF), plane.to(torch.int8)), dim=1).contiguous()


def unpack_w6a8_codes(packed: torch.Tensor) -> torch.Tensor:
    rows, width = packed.shape
    columns = width * 4 // 3
    unsigned = packed.view(torch.uint8).to(torch.int32)
    low = unsigned[:, : columns // 2]
    low = torch.stack((low & 0xF, low >> 4), dim=-1).reshape(rows, columns)
    high = unsigned[:, columns // 2 :]
    shifts = torch.arange(0, 8, 2, device=packed.device, dtype=torch.int32)
    high = ((high.unsqueeze(-1) >> shifts) & 3).reshape(rows, columns)
    return low | (high << 4)


def _grid(levels: torch.Tensor, group_scales: torch.Tensor) -> torch.Tensor:
    return (levels.view(1, 1, -1) * group_scales.float().unsqueeze(-1)).round_().clamp_(-127, 127)


def _search_group_scales(
    grouped: torch.Tensor,
    group_scales: torch.Tensor,
    channel_scales: torch.Tensor,
    levels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per group, the e4m3 value (itself, one step down, one step up) whose decoded
    INT8 levels fit the group best; ties keep the earlier candidate."""
    target = grouped / channel_scales.view(-1, 1, 1)
    raw = group_scales.view(torch.uint8)
    candidates = (
        group_scales.float(),
        (raw - 1).clamp_(min=1).view(torch.float8_e4m3fn).float(),
        (raw + 1).clamp_(max=_E4M3_MAX_FINITE_BITS).view(torch.float8_e4m3fn).float(),
    )

    def score(candidate):
        grid = _grid(levels, candidate)
        index = _assign_grid(grouped, grid, channel_scales)
        error = torch.gather(grid, 2, index.long()).sub_(target).pow_(2).sum(-1)
        return error, index

    best_scale = candidates[0]
    best_error, best_index = score(best_scale)
    for candidate in candidates[1:]:
        error, index = score(candidate)
        better = error < best_error
        best_error = torch.where(better, error, best_error)
        best_scale = torch.where(better, candidate, best_scale)
        best_index = torch.where(better.unsqueeze(-1), index, best_index)
    return best_scale.to(torch.float8_e4m3fn).contiguous(), best_index


def _quantize_rotated(rotated: torch.Tensor) -> W6A8Result:
    rows, columns = rotated.shape
    grouped = rotated.float().view(rows, columns // W4A8_GROUP_SIZE, W4A8_GROUP_SIZE)
    levels = torch.arange(-W6A8_LEVELS, W6A8_LEVELS + 1, device=rotated.device, dtype=torch.float32)

    def assign(normalized):
        return (normalized.round_().clamp_(-W6A8_LEVELS, W6A8_LEVELS) + W6A8_LEVELS).to(torch.int32)

    group_scale = (grouped.abs().amax(dim=-1, keepdim=True) / W6A8_LEVELS).clamp(min=1e-8)
    codes = assign(grouped / group_scale)
    for _ in range(_ALS_ITERS):
        quantized = levels[codes]
        group_scale = (
            (grouped * quantized).sum(-1, keepdim=True)
            / (quantized * quantized).sum(-1, keepdim=True).clamp(min=1e-8)
        ).clamp(min=1e-8)
        codes = assign(grouped / group_scale)
    shifted = levels[codes] * group_scale
    channel_scales = (shifted.abs().amax(dim=(1, 2)) / 127.0).clamp(min=1e-8)
    group_scales = (group_scale.squeeze(-1) / channel_scales.unsqueeze(1)).float().to(torch.float8_e4m3fn)
    group_scales, index = _search_group_scales(grouped, group_scales, channel_scales, levels)
    stored = (index + 1).view(rows, columns)  # level index 0..62 -> stored code 1..63
    return W6A8Result(pack_w6a8_codes(stored), group_scales, channel_scales.float().contiguous())


def quantize_w6a8(weights: torch.Tensor, device: str | torch.device = "cpu") -> W6A8Result:
    """Rotate and quantize a floating weight; tensors are returned on the CPU."""
    validate_w4a8_weights(weights)
    target = resolve_convrot_device(device, format_name=W6A8_FORMAT_NAME)
    weights = weights.to(device=target, dtype=torch.float32)
    rows, columns = weights.shape
    block = max(1, QUANT_ROW_ELEM_BUDGET // columns)
    parts = []
    for start in range(0, rows, block):
        rotated = apply_hadamard_rotation(weights[start:start + block].contiguous(), CONVROT_GROUP_SIZE)
        parts.append(_quantize_rotated(rotated))
        del rotated
    return W6A8Result(
        torch.cat([part.packed_codes for part in parts]).cpu(),
        torch.cat([part.group_scales for part in parts]).cpu(),
        torch.cat([part.channel_scales for part in parts]).cpu(),
    )


def dequantize_w6a8_int8(result: W6A8Result) -> torch.Tensor:
    """The rotated INT8 grid the kernels consume: round((code - 32) * s_rel)."""
    codes = unpack_w6a8_codes(result.packed_codes)
    rows, columns = codes.shape
    levels = (codes.float() - W6A8_ZERO_POINT).view(rows, columns // W4A8_GROUP_SIZE, W4A8_GROUP_SIZE)
    scaled = levels * result.group_scales.float().unsqueeze(-1)
    return scaled.round_().clamp_(-127, 127).view(rows, columns)


def dequantize_w6a8(result: W6A8Result) -> torch.Tensor:
    """Float32 weights in the original (unrotated) basis."""
    rotated = dequantize_w6a8_int8(result) * result.channel_scales.float().unsqueeze(1)
    return apply_hadamard_rotation(rotated, CONVROT_GROUP_SIZE)
