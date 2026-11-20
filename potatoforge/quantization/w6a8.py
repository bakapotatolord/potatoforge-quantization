"""W6A8 ConvRot weights in ComfyUI's ``w6a8_int8`` storage contract.

kitchen's 6-bit recipe on the shared grouped core (``grouped_int8``): levels
are the uniform integers -31..31 (no codebook), and each fp8 group scale then
takes the best of itself and its two e4m3 neighbours by squared error on the
decoded INT8 grid.

- ``weight`` int8 [N, 3K/4]: the K/2-byte nibble plane of each code's low
  4 bits, then the K/4-byte plane of its top 2 bits (column c in byte c//4
  at bit 2*(c%4)). Codes are stored as level + 32, in 1..63.
- ``weight_s_rel`` float8_e4m3fn [N, K/16]; ``weight_s_channel`` float32 [N].
"""

from __future__ import annotations

from typing import Final, NamedTuple

import torch

from . import grouped_int8 as core
from .w4a8 import pack_w4a8_codes

W6A8_FORMAT_NAME: Final[str] = "W6A8"
W6A8_LEVELS: Final[int] = 31
W6A8_ZERO_POINT: Final[int] = 32
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


def _search_group_scales(
    groups: torch.Tensor,
    group_scales: torch.Tensor,
    channel_scales: torch.Tensor,
    level_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per group, the e4m3 value (itself, one step down, one step up) whose decoded
    INT8 levels fit the group best; ties keep the earlier candidate.

    A scale that underflowed to 0 wraps to bit pattern 0xFF, a NaN in e4m3; its
    error is NaN, never "better", so it is never chosen (as in kitchen).
    """
    target = core.grid_target(groups, channel_scales)
    raw = group_scales.view(torch.uint8)
    candidates = (
        group_scales.float(),
        (raw - 1).clamp_(min=1).view(torch.float8_e4m3fn).float(),
        (raw + 1).clamp_(max=_E4M3_MAX_FINITE_BITS).view(torch.float8_e4m3fn).float(),
    )

    def score(candidate):
        grid = core.decoded_grid(level_table, candidate)
        index = core.assign_grid(groups, grid, target)
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


def _assign_uniform(normalized: torch.Tensor) -> torch.Tensor:
    return (normalized.round_().clamp_(-W6A8_LEVELS, W6A8_LEVELS) + W6A8_LEVELS).to(torch.int32)


def _quantize_rotated(rotated: torch.Tensor) -> tuple[torch.Tensor, ...]:
    groups = core.grouped(rotated)
    level_table = torch.arange(-W6A8_LEVELS, W6A8_LEVELS + 1, device=rotated.device, dtype=torch.float32)
    start = (groups.abs().amax(dim=-1, keepdim=True) / W6A8_LEVELS).clamp(min=1e-8)
    group_scale, codes = core.refine_group_scales(groups, level_table, start, _assign_uniform)
    channel_scales, group_scales = core.stored_scales(level_table, codes, group_scale)
    group_scales, index = _search_group_scales(groups, group_scales, channel_scales, level_table)
    stored = (index + 1).view(rotated.shape)  # level index 0..62 -> stored code 1..63
    return pack_w6a8_codes(stored), group_scales, channel_scales


def quantize_w6a8(weights: torch.Tensor, device: str | torch.device = "cpu") -> W6A8Result:
    """Rotate and quantize a floating weight; tensors are returned on the CPU."""
    prepared = core.prepare_weights(weights, device, W6A8_FORMAT_NAME)
    return W6A8Result(*core.quantize_in_chunks(prepared, _quantize_rotated))


def dequantize_w6a8_int8(result: W6A8Result) -> torch.Tensor:
    """The rotated INT8 grid the kernels consume: round((code - 32) * s_rel)."""
    codes = unpack_w6a8_codes(result.packed_codes)
    return core.int8_grid(codes.float() - W6A8_ZERO_POINT, result.group_scales)


def dequantize_w6a8(result: W6A8Result) -> torch.Tensor:
    """Float32 weights in the original (unrotated) basis."""
    return core.unrotate(dequantize_w6a8_int8(result), result.channel_scales)
