"""W3A8 ConvRot weights in ComfyUI's ``w3a8_int8`` storage contract.

kitchen's 3-bit recipe on the shared grouped core (``grouped_int8``): the
W4A8 codebook recipe with an 8-entry Lloyd-Max table (kitchen's frozen table
unless the rotated weight is heavy-tailed, then a per-tensor fit). A uniform
8-level grid measured about 7% worse, and there is no fp8 scale search.

- ``weight`` int8 [N, 3K/8]: the K/4-byte plane of each code's low 2 bits
  (column c in byte c//4 at bit 2*(c%4)), then the K/8-byte plane of its top
  bit (byte c//8, bit c%8). ConvRot already requires K divisible by 256, which
  keeps the 1-bit plane 8-byte aligned.
- ``weight_s_rel`` float8_e4m3fn [N, K/16]; ``weight_s_channel`` float32 [N].
- ``weight_codebook`` float32 [8].
"""

from __future__ import annotations

from typing import Final, NamedTuple

import torch

from . import grouped_int8 as core
from .w4a8 import _assign_codes, _decide_codebook

W3A8_FORMAT_NAME: Final[str] = "W3A8"
W3A8_CODEBOOK_LEVELS: Final[int] = 8

# Lloyd-Max levels for a group-normalized Gaussian (kitchen's _FIXED_LUT8).
FIXED_W3A8_CODEBOOK: Final[tuple[float, ...]] = (
    -0.923975, -0.594572, -0.336347, -0.109145, 0.109145, 0.336347, 0.594572, 0.923975,
)


class W3A8Result(NamedTuple):
    packed_codes: torch.Tensor
    group_scales: torch.Tensor
    channel_scales: torch.Tensor
    codebook: torch.Tensor


def default_w3a8_codebook(device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.tensor(FIXED_W3A8_CODEBOOK, dtype=torch.float32, device=device)


def pack_w3a8_codes(codes: torch.Tensor) -> torch.Tensor:
    """Unsigned 3-bit codes [N, K] -> int8 [N, 3K/8]: 2-bit plane, then 1-bit plane."""
    if codes.ndim != 2 or codes.shape[1] % 8:
        raise ValueError("W3A8 codes must be a 2D matrix with a column count divisible by 8.")
    return torch.cat((core.pack_plane(codes, 0, 2), core.pack_plane(codes, 2, 1)), dim=1).contiguous()


def unpack_w3a8_codes(packed: torch.Tensor) -> torch.Tensor:
    columns = packed.shape[1] * 8 // 3
    unsigned = packed.view(torch.uint8).to(torch.int32)
    low = core.unpack_plane(unsigned[:, : columns // 4], columns, 2)
    return low | (core.unpack_plane(unsigned[:, columns // 4 :], columns, 1) << 2)


def _quantize_rotated(rotated: torch.Tensor, codebook: torch.Tensor) -> tuple[torch.Tensor, ...]:
    groups = core.grouped(rotated)
    start = groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    group_scale, codes = core.refine_group_scales(groups, codebook, start, lambda x: _assign_codes(x, codebook))
    channel_scales, group_scales = core.stored_scales(codebook, codes, group_scale)
    grid = core.decoded_grid(codebook, group_scales)
    codes = core.assign_grid(groups, grid, core.grid_target(groups, channel_scales))
    return pack_w3a8_codes(codes.view(rotated.shape)), group_scales, channel_scales


def quantize_w3a8(weights: torch.Tensor, device: str | torch.device = "cpu") -> W3A8Result:
    """Rotate and quantize a floating weight; tensors are returned on the CPU."""
    prepared = core.prepare_weights(weights, device, W3A8_FORMAT_NAME)
    codebook = _decide_codebook(prepared, default_w3a8_codebook())
    packed, group_scales, channel_scales = core.quantize_in_chunks(
        prepared, lambda rotated: _quantize_rotated(rotated, codebook)
    )
    return W3A8Result(packed, group_scales, channel_scales, codebook.cpu())


def dequantize_w3a8_int8(result: W3A8Result) -> torch.Tensor:
    """The rotated INT8 grid the kernels consume: round(codebook[code] * s_rel)."""
    return core.int8_grid(result.codebook.float()[unpack_w3a8_codes(result.packed_codes)], result.group_scales)


def dequantize_w3a8(result: W3A8Result) -> torch.Tensor:
    """Float32 weights in the original (unrotated) basis."""
    return core.unrotate(dequantize_w3a8_int8(result), result.channel_scales)
