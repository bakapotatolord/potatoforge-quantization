"""Shared core of ComfyUI's grouped ConvRot formats (W4A8 codebook, W6A8 uniform).

Both follow comfy-kitchen's eager recipe (``comfy_kitchen/backends/eager/
w4a8_int8.py`` at upstream e769274): rotate 256-wide input groups with the
normalized Hadamard, give every 16 inputs one scale refined by ALS, store that
scale as float8_e4m3fn relative to a per-row float32 scale onto the INT8 grid,
and assign codes against the decoded grid ``round(level * s_rel)`` the kernels
multiply. The formats differ only in their level table, how codes are first
assigned, and W6A8's search over neighbouring fp8 scales.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final

import torch

from .hadamard import CONVROT_GROUP_SIZE, apply_hadamard_rotation
from .rowwise import divide_exactly, resolve_convrot_device

GROUP_SIZE: Final[int] = 16
_ALS_ITERS: Final[int] = 2
# Rows per chunk (fp32 working set). Results never depend on it: rows are
# independent and any per-tensor choice is made before chunking.
QUANT_ROW_ELEM_BUDGET: Final[int] = 1 << 22


def prepare_weights(weights: torch.Tensor, device: str | torch.device, format_name: str) -> torch.Tensor:
    """Validate, move to the quantization device, and return float32 weights.

    The source dtype is transferred as-is and widened on the device, which also
    runs the finiteness check, so a CUDA run does no full-tensor CPU pass.
    """
    if weights.ndim != 2:
        raise ValueError(f"{format_name} weights must be a 2D matrix.")
    if weights.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"{format_name} weights must be BF16, F16, or F32.")
    rows, columns = weights.shape
    if rows < 1 or columns < 1 or columns % CONVROT_GROUP_SIZE:
        raise ValueError(
            f"{format_name} needs at least one row and input features divisible by {CONVROT_GROUP_SIZE}."
        )
    target = resolve_convrot_device(device, format_name=format_name)
    prepared = weights.to(device=target).float()
    if not bool(torch.isfinite(prepared).all()):
        raise ValueError(f"{format_name} weights must be finite.")
    return prepared


def grouped(rotated: torch.Tensor) -> torch.Tensor:
    rows, columns = rotated.shape
    return rotated.view(rows, columns // GROUP_SIZE, GROUP_SIZE)


def refine_group_scales(
    groups: torch.Tensor,
    level_table: torch.Tensor,
    group_scale: torch.Tensor,
    assign: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """ALS: least-squares group scale for the current codes, then reassign."""
    codes = assign(groups / group_scale)
    for _ in range(_ALS_ITERS):
        levels = level_table[codes]
        group_scale = (
            (groups * levels).sum(-1, keepdim=True)
            / (levels * levels).sum(-1, keepdim=True).clamp(min=1e-8)
        ).clamp(min=1e-8)
        codes = assign(groups / group_scale)
    return group_scale, codes


def stored_scales(
    level_table: torch.Tensor,
    codes: torch.Tensor,
    group_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row float32 scale onto the INT8 grid and fp8 group scales relative to it."""
    shifted = level_table[codes] * group_scale
    channel_scales = divide_exactly(shifted.abs().amax(dim=(1, 2)), 127.0).clamp(min=1e-8)
    group_scales = (group_scale.squeeze(-1) / channel_scales.unsqueeze(1)).to(torch.float8_e4m3fn)
    return channel_scales, group_scales


def decoded_grid(level_table: torch.Tensor, group_scales: torch.Tensor) -> torch.Tensor:
    """The INT8 value every level decodes to in every group: round(level * s_rel)."""
    return (level_table.view(1, 1, -1) * group_scales.float().unsqueeze(-1)).round_().clamp_(-127, 127)


def assign_grid(groups: torch.Tensor, grid: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Nearest decoded level per weight (``target`` = weights / channel scale);
    ties resolve to the lower index."""
    rows, group_count, group_size = groups.shape
    last = grid.shape[-1] - 1
    level_rows = grid.reshape(rows * group_count, last + 1).contiguous()
    flat_target = target.reshape(rows * group_count, group_size).contiguous()
    position = torch.searchsorted(level_rows, flat_target)
    low = (position - 1).clamp(0, last)
    high = position.clamp(0, last)
    low_distance = flat_target.sub(torch.gather(level_rows, 1, low)).abs_()
    high_distance = flat_target.sub(torch.gather(level_rows, 1, high)).abs_()
    return torch.where(high_distance < low_distance, high, low).to(torch.int32).reshape(rows, group_count, group_size)


def grid_target(groups: torch.Tensor, channel_scales: torch.Tensor) -> torch.Tensor:
    return groups / channel_scales.view(-1, 1, 1)


def quantize_in_chunks(
    weights: torch.Tensor,
    quantize_rotated: Callable[[torch.Tensor], Sequence[torch.Tensor]],
) -> tuple[torch.Tensor, ...]:
    """Rotate and quantize row chunks; concatenate each field on the CPU."""
    rows, columns = weights.shape
    block = max(1, QUANT_ROW_ELEM_BUDGET // columns)
    parts = []
    for start in range(0, rows, block):
        rotated = apply_hadamard_rotation(weights[start:start + block].contiguous(), CONVROT_GROUP_SIZE)
        parts.append(quantize_rotated(rotated))
        del rotated
    return tuple(torch.cat(fields).cpu() for fields in zip(*parts))


def pack_plane(codes: torch.Tensor, shift: int, width: int) -> torch.Tensor:
    """``width``-bit field ``(codes >> shift)`` of every code, 8/width codes per
    byte (column c in byte c // (8/width) at bit width * (c % (8/width)))."""
    per_byte = 8 // width
    mask = (1 << width) - 1
    plane = (codes[:, 0::per_byte] >> shift) & mask
    for index in range(1, per_byte):
        plane |= ((codes[:, index::per_byte] >> shift) & mask) << (width * index)
    return plane.to(torch.int8)


def unpack_plane(plane: torch.Tensor, columns: int, width: int) -> torch.Tensor:
    """Inverse of ``pack_plane``: int32 bytes [N, columns*width/8] -> int32 [N, columns]."""
    shifts = torch.arange(0, 8, width, device=plane.device, dtype=torch.int32)
    return ((plane.unsqueeze(-1) >> shifts) & ((1 << width) - 1)).reshape(plane.shape[0], columns)


def unrotate(int8_grid: torch.Tensor, channel_scales: torch.Tensor) -> torch.Tensor:
    """Float32 weights in the original basis. The normalized Hadamard is symmetric
    and orthogonal, so applying it again inverts the rotation."""
    return apply_hadamard_rotation(int8_grid * channel_scales.float().unsqueeze(1), CONVROT_GROUP_SIZE)


def int8_grid(level_values: torch.Tensor, group_scales: torch.Tensor) -> torch.Tensor:
    """Decode per-weight level values [N, K] to the rotated INT8 grid."""
    rows, columns = level_values.shape
    scaled = grouped(level_values) * group_scales.float().unsqueeze(-1)
    return scaled.round_().clamp_(-127, 127).view(rows, columns)
