"""W5A8 ConvRot weights in ComfyUI's ``w5a8_int8`` storage contract.

kitchen's 5-bit recipe on the shared grouped core (``grouped_int8``): the
W6A8 recipe one bit down. Levels are the uniform integers -15..15 (no
codebook) and each fp8 group scale takes the best of itself and its two e4m3
neighbours by squared error on the decoded INT8 grid.

- ``weight`` int8 [N, 5K/8]: the K/2-byte nibble plane of each code's low 4
  bits, then the K/8-byte plane of its top bit (column c in byte c//8 at bit
  c%8). Codes are stored as level + 16, in 1..31. ConvRot already requires K
  divisible by 256, which keeps the 1-bit plane 8-byte aligned.
- ``weight_s_rel`` float8_e4m3fn [N, K/16]; ``weight_s_channel`` float32 [N].
"""

from __future__ import annotations

from typing import Final, NamedTuple

import torch

from . import grouped_int8 as core
from .w4a8 import pack_w4a8_codes, unpack_w4a8_codes
from .w6a8 import _search_group_scales

W5A8_FORMAT_NAME: Final[str] = "W5A8"
W5A8_LEVELS: Final[int] = 15
W5A8_ZERO_POINT: Final[int] = 16


class W5A8Result(NamedTuple):
    packed_codes: torch.Tensor
    group_scales: torch.Tensor
    channel_scales: torch.Tensor


def pack_w5a8_codes(codes: torch.Tensor) -> torch.Tensor:
    """Unsigned 5-bit codes [N, K] -> int8 [N, 5K/8]: nibble plane, then 1-bit plane."""
    if codes.ndim != 2 or codes.shape[1] % 8:
        raise ValueError("W5A8 codes must be a 2D matrix with a column count divisible by 8.")
    return torch.cat((pack_w4a8_codes(codes & 0xF), core.pack_plane(codes, 4, 1)), dim=1).contiguous()


def unpack_w5a8_codes(packed: torch.Tensor) -> torch.Tensor:
    columns = packed.shape[1] * 8 // 5
    unsigned = packed.view(torch.uint8).to(torch.int32)
    low = unpack_w4a8_codes(unsigned[:, : columns // 2].to(torch.uint8))
    return low | (core.unpack_plane(unsigned[:, columns // 2 :], columns, 1) << 4)


def _assign_uniform(normalized: torch.Tensor) -> torch.Tensor:
    return (normalized.round_().clamp_(-W5A8_LEVELS, W5A8_LEVELS) + W5A8_LEVELS).to(torch.int32)


def _quantize_rotated(rotated: torch.Tensor) -> tuple[torch.Tensor, ...]:
    groups = core.grouped(rotated)
    level_table = torch.arange(-W5A8_LEVELS, W5A8_LEVELS + 1, device=rotated.device, dtype=torch.float32)
    start = (groups.abs().amax(dim=-1, keepdim=True) / W5A8_LEVELS).clamp(min=1e-8)
    group_scale, codes = core.refine_group_scales(groups, level_table, start, _assign_uniform)
    channel_scales, group_scales = core.stored_scales(level_table, codes, group_scale)
    group_scales, index = _search_group_scales(groups, group_scales, channel_scales, level_table)
    stored = (index + 1).view(rotated.shape)  # level index 0..30 -> stored code 1..31
    return pack_w5a8_codes(stored), group_scales, channel_scales


def quantize_w5a8(weights: torch.Tensor, device: str | torch.device = "cpu") -> W5A8Result:
    """Rotate and quantize a floating weight; tensors are returned on the CPU."""
    prepared = core.prepare_weights(weights, device, W5A8_FORMAT_NAME)
    return W5A8Result(*core.quantize_in_chunks(prepared, _quantize_rotated))


def dequantize_w5a8_int8(result: W5A8Result) -> torch.Tensor:
    """The rotated INT8 grid the kernels consume: round((code - 16) * s_rel)."""
    codes = unpack_w5a8_codes(result.packed_codes)
    return core.int8_grid(codes.float() - W5A8_ZERO_POINT, result.group_scales)


def dequantize_w5a8(result: W5A8Result) -> torch.Tensor:
    """Float32 weights in the original (unrotated) basis."""
    return core.unrotate(dequantize_w5a8_int8(result), result.channel_scales)
