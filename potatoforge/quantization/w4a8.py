"""W4A8 ConvRot weights in ComfyUI's ``asym_w4a8_int8`` storage contract.

A port of comfy-kitchen's eager symmetric 4-bit codebook recipe
(``comfy_kitchen/backends/eager/w4a8_int8.py`` at upstream e769274), so ComfyUI
loads the result through ``AsymW4A8Int8Layout`` and the kitchen W4A8 kernels:

- ConvRot: 256-wide normalized Hadamard rotation of the input dimension.
- Groups of 16 inputs share one scale. Codes index a 16-entry Lloyd-Max codebook
  (kitchen's frozen table unless the rotated weight is heavy-tailed, then a
  per-tensor fit), refined by two ALS passes of the group scale.
- ``weight`` int8 [N, K/2]: nibble plane, even column in the low nibble.
- ``weight_s_rel`` float8_e4m3fn [N, K/16]: group scale / channel scale.
- ``weight_s_channel`` float32 [N]: per-output-row scale onto the INT8 grid.
- ``weight_codebook`` float32 [16].

Codes are assigned against the decoded INT8 grid ``round(level * s_rel)``, the
values the kernels actually multiply. Weights are rotated in float32.
"""

from __future__ import annotations

from typing import Final, NamedTuple

import torch

from .hadamard import CONVROT_GROUP_SIZE, apply_hadamard_rotation
from .rowwise import resolve_convrot_device

W4A8_FORMAT_NAME: Final[str] = "W4A8"
W4A8_GROUP_SIZE: Final[int] = 16
W4A8_CODEBOOK_LEVELS: Final[int] = 16

# Lloyd-Max levels for a group-normalized Gaussian (kitchen's _FIXED_LUT, the
# streamed decode kernel's built-in table).
FIXED_W4A8_CODEBOOK: Final[tuple[float, ...]] = (
    -0.980602, -0.794529, -0.638165, -0.500986, -0.377321, -0.263187, -0.155210, -0.050720,
    0.052541, 0.156985, 0.265284, 0.379533, 0.502636, 0.638953, 0.794876, 0.980671,
)

_GATE_KURTOSIS: Final[float] = -0.1
_KURTOSIS_SAMPLE: Final[int] = 1 << 19
_ALS_ITERS: Final[int] = 2
_FIT_ITERATIONS: Final[int] = 25
_FIT_SAMPLE: Final[int] = 300000
# Rows per chunk (fp32 working set) and the codebook-decision sample; both match
# kitchen so the chosen table and the packed bytes never depend on chunking.
QUANT_ROW_ELEM_BUDGET: Final[int] = 1 << 22
_CODEBOOK_SAMPLE_ELEMS: Final[int] = 1 << 22


class W4A8Result(NamedTuple):
    packed_codes: torch.Tensor
    group_scales: torch.Tensor
    channel_scales: torch.Tensor
    codebook: torch.Tensor


def default_w4a8_codebook(device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.tensor(FIXED_W4A8_CODEBOOK, dtype=torch.float32, device=device)


def validate_w4a8_weights(weights: torch.Tensor) -> tuple[int, int]:
    if weights.ndim != 2:
        raise ValueError("W4A8 weights must be a 2D matrix.")
    if weights.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("W4A8 weights must be BF16, F16, or F32.")
    rows, columns = weights.shape
    if rows < 1 or columns < 1 or columns % CONVROT_GROUP_SIZE:
        raise ValueError(
            f"W4A8 needs at least one row and input features divisible by {CONVROT_GROUP_SIZE}."
        )
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("W4A8 weights must be finite.")
    return rows, columns


def _fit_codebook(normalized: torch.Tensor) -> torch.Tensor:
    samples = normalized.flatten()
    if samples.numel() > _FIT_SAMPLE:
        generator = torch.Generator(device=samples.device).manual_seed(0)
        indices = torch.randint(0, samples.numel(), (_FIT_SAMPLE,), device=samples.device, generator=generator)
        samples = samples[indices]
    samples = samples.float()
    codebook = torch.quantile(samples, torch.linspace(0, 1, W4A8_CODEBOOK_LEVELS, device=samples.device))
    for _ in range(_FIT_ITERATIONS):
        assignments = (samples.unsqueeze(-1) - codebook).abs().argmin(-1)
        updated = codebook.clone()
        for index in range(W4A8_CODEBOOK_LEVELS):
            selected = assignments == index
            if selected.any():
                updated[index] = samples[selected].mean()
        codebook = updated
    return codebook.contiguous()


def _codebook_for(normalized: torch.Tensor) -> torch.Tensor:
    x = normalized.detach().flatten()
    if x.numel() > _KURTOSIS_SAMPLE:
        generator = torch.Generator(device=x.device).manual_seed(0)
        x = x[torch.randint(0, x.numel(), (_KURTOSIS_SAMPLE,), device=x.device, generator=generator)]
    x = x.float()
    excess_kurtosis = ((x - x.mean()) / (x.std() + 1e-9)).pow(4).mean() - 3.0
    if excess_kurtosis.item() <= _GATE_KURTOSIS:
        return default_w4a8_codebook(normalized.device)
    return _fit_codebook(normalized)


def _decide_codebook(weights: torch.Tensor) -> torch.Tensor:
    """One table for the whole tensor from evenly strided rows spanning every chunk."""
    rows, columns = weights.shape
    max_rows = max(1, _CODEBOOK_SAMPLE_ELEMS // columns)
    if rows > max_rows:
        index = torch.linspace(0, rows - 1, max_rows, device=weights.device).long()
        sample = weights.index_select(0, index)
    else:
        sample = weights
    rotated = apply_hadamard_rotation(sample.contiguous(), CONVROT_GROUP_SIZE)
    grouped = rotated.view(sample.shape[0], columns // W4A8_GROUP_SIZE, W4A8_GROUP_SIZE)
    return _codebook_for(grouped / grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8))


def _assign_codes(normalized: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    last = codebook.numel() - 1
    position = torch.searchsorted(codebook, normalized.contiguous())
    low = (position - 1).clamp(0, last)
    high = position.clamp(0, last)
    low_distance = (normalized - codebook[low]).abs_()
    high_distance = (normalized - codebook[high]).abs_()
    return torch.where(high_distance < low_distance, high, low).to(torch.int32)


def _assign_grid(grouped: torch.Tensor, levels: torch.Tensor, channel_scales: torch.Tensor) -> torch.Tensor:
    """Nearest decoded INT8 level per weight; ties resolve to the lower index."""
    rows, groups, group_size = grouped.shape
    last = levels.shape[-1] - 1
    level_rows = levels.reshape(rows * groups, last + 1).contiguous()
    target = (grouped / channel_scales.view(-1, 1, 1)).reshape(rows * groups, group_size).contiguous()
    position = torch.searchsorted(level_rows, target)
    low = (position - 1).clamp(0, last)
    high = position.clamp(0, last)
    low_distance = target.sub(torch.gather(level_rows, 1, low)).abs_()
    high_distance = target.sub(torch.gather(level_rows, 1, high)).abs_()
    return torch.where(high_distance < low_distance, high, low).to(torch.int32).reshape(rows, groups, group_size)


def pack_w4a8_codes(codes: torch.Tensor) -> torch.Tensor:
    """Unsigned 4-bit codes [N, K] -> int8 [N, K/2], even column in the low nibble."""
    if codes.ndim != 2 or codes.shape[1] % 2:
        raise ValueError("W4A8 codes must be a 2D matrix with an even column count.")
    return ((codes[:, 0::2] & 0xF) | ((codes[:, 1::2] & 0xF) << 4)).to(torch.int8).contiguous()


def unpack_w4a8_codes(packed: torch.Tensor) -> torch.Tensor:
    unsigned = packed.view(torch.uint8).to(torch.int32)
    return torch.stack((unsigned & 0xF, unsigned >> 4), dim=-1).reshape(packed.shape[0], -1)


def _quantize_rotated(rotated: torch.Tensor, codebook: torch.Tensor) -> W4A8Result:
    rows, columns = rotated.shape
    grouped = rotated.float().view(rows, columns // W4A8_GROUP_SIZE, W4A8_GROUP_SIZE)
    group_scale = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    codes = _assign_codes(grouped / group_scale, codebook)
    for _ in range(_ALS_ITERS):
        levels = codebook[codes]
        group_scale = (
            (grouped * levels).sum(-1, keepdim=True)
            / (levels * levels).sum(-1, keepdim=True).clamp(min=1e-8)
        ).clamp(min=1e-8)
        codes = _assign_codes(grouped / group_scale, codebook)
    shifted = codebook[codes] * group_scale
    channel_scales = (shifted.abs().amax(dim=(1, 2)) / 127.0).clamp(min=1e-8)
    group_scales = (group_scale.squeeze(-1) / channel_scales.unsqueeze(1)).float().to(torch.float8_e4m3fn)
    grid = (codebook.view(1, 1, -1) * group_scales.float().unsqueeze(-1)).round_().clamp_(-127, 127)
    codes = _assign_grid(grouped, grid, channel_scales).view(rows, columns)
    return W4A8Result(pack_w4a8_codes(codes), group_scales.contiguous(), channel_scales.float().contiguous(), codebook)


def quantize_w4a8(weights: torch.Tensor, device: str | torch.device = "cpu") -> W4A8Result:
    """Rotate and quantize a floating weight; tensors are returned on the CPU."""
    validate_w4a8_weights(weights)
    target = resolve_convrot_device(device, format_name=W4A8_FORMAT_NAME)
    weights = weights.to(device=target, dtype=torch.float32)
    rows, columns = weights.shape
    codebook = _decide_codebook(weights)
    block = max(1, QUANT_ROW_ELEM_BUDGET // columns)
    parts = []
    for start in range(0, rows, block):
        rotated = apply_hadamard_rotation(weights[start:start + block].contiguous(), CONVROT_GROUP_SIZE)
        parts.append(_quantize_rotated(rotated, codebook))
        del rotated
    return W4A8Result(
        torch.cat([part.packed_codes for part in parts]).cpu(),
        torch.cat([part.group_scales for part in parts]).cpu(),
        torch.cat([part.channel_scales for part in parts]).cpu(),
        codebook.float().cpu().contiguous(),
    )


def dequantize_w4a8_int8(result: W4A8Result) -> torch.Tensor:
    """The rotated INT8 grid the kernels consume: round(codebook[code] * s_rel)."""
    codes = unpack_w4a8_codes(result.packed_codes)
    rows, columns = codes.shape
    levels = result.codebook.float()[codes].view(rows, columns // W4A8_GROUP_SIZE, W4A8_GROUP_SIZE)
    scaled = levels * result.group_scales.float().unsqueeze(-1)
    return scaled.round_().clamp_(-127, 127).view(rows, columns)


def dequantize_w4a8(result: W4A8Result) -> torch.Tensor:
    """Float32 weights in the original (unrotated) basis."""
    rotated = dequantize_w4a8_int8(result) * result.channel_scales.float().unsqueeze(1)
    # The normalized Hadamard is symmetric and orthogonal, so applying it again inverts it.
    return apply_hadamard_rotation(rotated, CONVROT_GROUP_SIZE)
