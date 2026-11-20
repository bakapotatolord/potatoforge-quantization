"""W4A8 ConvRot weights in ComfyUI's ``asym_w4a8_int8`` storage contract.

kitchen's symmetric 4-bit codebook recipe on the shared grouped core
(``grouped_int8``): codes index a 16-entry Lloyd-Max codebook, kitchen's
frozen table unless the rotated weight is heavy-tailed, then a per-tensor fit.

- ``weight`` int8 [N, K/2]: nibble plane, even column in the low nibble.
- ``weight_s_rel`` float8_e4m3fn [N, K/16]; ``weight_s_channel`` float32 [N].
- ``weight_codebook`` float32 [16].
"""

from __future__ import annotations

from typing import Final, NamedTuple

import torch

from . import grouped_int8 as core
from .hadamard import CONVROT_GROUP_SIZE, apply_hadamard_rotation

W4A8_FORMAT_NAME: Final[str] = "W4A8"
W4A8_CODEBOOK_LEVELS: Final[int] = 16

# Lloyd-Max levels for a group-normalized Gaussian (kitchen's _FIXED_LUT, the
# streamed decode kernel's built-in table).
FIXED_W4A8_CODEBOOK: Final[tuple[float, ...]] = (
    -0.980602, -0.794529, -0.638165, -0.500986, -0.377321, -0.263187, -0.155210, -0.050720,
    0.052541, 0.156985, 0.265284, 0.379533, 0.502636, 0.638953, 0.794876, 0.980671,
)

_GATE_KURTOSIS: Final[float] = -0.1
_KURTOSIS_SAMPLE: Final[int] = 1 << 19
_FIT_ITERATIONS: Final[int] = 25
_FIT_SAMPLE: Final[int] = 300000
_CODEBOOK_SAMPLE_ELEMS: Final[int] = 1 << 22


class W4A8Result(NamedTuple):
    packed_codes: torch.Tensor
    group_scales: torch.Tensor
    channel_scales: torch.Tensor
    codebook: torch.Tensor


def default_w4a8_codebook(device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.tensor(FIXED_W4A8_CODEBOOK, dtype=torch.float32, device=device)


def _fit_codebook(normalized: torch.Tensor) -> torch.Tensor:
    samples = normalized.flatten()
    if samples.numel() > _FIT_SAMPLE:
        generator = torch.Generator(device=samples.device).manual_seed(0)
        indices = torch.randint(0, samples.numel(), (_FIT_SAMPLE,), device=samples.device, generator=generator)
        samples = samples[indices]
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
    """Kitchen's frozen table unless the sample is heavy-tailed. The sampling
    generator lives on the tensor's device, so a fitted table can differ between
    CPU and CUDA runs; real model layers stay on the frozen table."""
    x = normalized.flatten()
    if x.numel() > _KURTOSIS_SAMPLE:
        generator = torch.Generator(device=x.device).manual_seed(0)
        x = x[torch.randint(0, x.numel(), (_KURTOSIS_SAMPLE,), device=x.device, generator=generator)]
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
    groups = core.grouped(apply_hadamard_rotation(sample.contiguous(), CONVROT_GROUP_SIZE))
    return _codebook_for(groups / groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8))


def _assign_codes(normalized: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Nearest codebook index; the codebook is sorted, so one binary search."""
    last = codebook.numel() - 1
    position = torch.searchsorted(codebook, normalized.contiguous())
    low = (position - 1).clamp(0, last)
    high = position.clamp(0, last)
    low_distance = (normalized - codebook[low]).abs_()
    high_distance = (normalized - codebook[high]).abs_()
    return torch.where(high_distance < low_distance, high, low).to(torch.int32)


def pack_w4a8_codes(codes: torch.Tensor) -> torch.Tensor:
    """Unsigned 4-bit codes [N, K] -> int8 [N, K/2], even column in the low nibble."""
    if codes.ndim != 2 or codes.shape[1] % 2:
        raise ValueError("W4A8 codes must be a 2D matrix with an even column count.")
    return ((codes[:, 0::2] & 0xF) | ((codes[:, 1::2] & 0xF) << 4)).to(torch.int8).contiguous()


def unpack_w4a8_codes(packed: torch.Tensor) -> torch.Tensor:
    unsigned = packed.view(torch.uint8).to(torch.int32)
    return torch.stack((unsigned & 0xF, unsigned >> 4), dim=-1).reshape(packed.shape[0], -1)


def _quantize_rotated(rotated: torch.Tensor, codebook: torch.Tensor) -> tuple[torch.Tensor, ...]:
    groups = core.grouped(rotated)
    start = groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    group_scale, codes = core.refine_group_scales(groups, codebook, start, lambda x: _assign_codes(x, codebook))
    channel_scales, group_scales = core.stored_scales(codebook, codes, group_scale)
    grid = core.decoded_grid(codebook, group_scales)
    codes = core.assign_grid(groups, grid, core.grid_target(groups, channel_scales))
    return pack_w4a8_codes(codes.view(rotated.shape)), group_scales, channel_scales


def quantize_w4a8(weights: torch.Tensor, device: str | torch.device = "cpu") -> W4A8Result:
    """Rotate and quantize a floating weight; tensors are returned on the CPU."""
    prepared = core.prepare_weights(weights, device, W4A8_FORMAT_NAME)
    codebook = _decide_codebook(prepared)
    packed, group_scales, channel_scales = core.quantize_in_chunks(
        prepared, lambda rotated: _quantize_rotated(rotated, codebook)
    )
    return W4A8Result(packed, group_scales, channel_scales, codebook.cpu())


def dequantize_w4a8_int8(result: W4A8Result) -> torch.Tensor:
    """The rotated INT8 grid the kernels consume: round(codebook[code] * s_rel)."""
    return core.int8_grid(result.codebook.float()[unpack_w4a8_codes(result.packed_codes)], result.group_scales)


def dequantize_w4a8(result: W4A8Result) -> torch.Tensor:
    """Float32 weights in the original (unrotated) basis."""
    return core.unrotate(dequantize_w4a8_int8(result), result.channel_scales)
