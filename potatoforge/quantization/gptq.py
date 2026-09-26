"""Weight-independent block-diagonal GPTQ preparation and solving."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from .hadamard import CONVROT_GROUP_SIZE, apply_hadamard_rotation
from ..timing import timed_stage


DAMP_PERCENT = 0.01
GPTQ_BLOCKSIZE = 128


def rotate_gram(gram: torch.Tensor) -> torch.Tensor:
    """Apply the 256-wide ConvRot transform to both Gram axes."""
    rotated = apply_hadamard_rotation(gram, CONVROT_GROUP_SIZE)
    return apply_hadamard_rotation(rotated.T, CONVROT_GROUP_SIZE).T.contiguous()


def gptq_factor(gram: torch.Tensor, damp_percent: float = DAMP_PERCENT) -> torch.Tensor:
    hessian = (gram + gram.T) * 0.5
    hessian.diagonal().add_(damp_percent * hessian.diagonal().mean())
    try:
        lower = torch.linalg.cholesky(hessian)
        inverse = torch.cholesky_inverse(lower)
        return torch.linalg.cholesky(inverse, upper=True)
    except torch.linalg.LinAlgError as error:
        raise ValueError("Damped GPTQ Gram is not positive definite; inspect calibration.") from error


def gptq_quantize(
    rotated_weights: torch.Tensor,
    inverse_factor: torch.Tensor,
    quantize_column: Callable[[torch.Tensor], torch.Tensor],
    blocksize: int = GPTQ_BLOCKSIZE,
) -> torch.Tensor:
    """Solve columns in order with a caller-provided quantizer."""
    if blocksize < 1:
        raise ValueError("GPTQ solve blocksize must be positive.")
    weights = rotated_weights.clone()
    result = torch.empty_like(weights)
    columns = weights.shape[1]
    if inverse_factor.shape != (columns, columns):
        raise ValueError(f"GPTQ factor shape {tuple(inverse_factor.shape)} does not match {columns} columns.")
    for start in range(0, columns, blocksize):
        end = min(start + blocksize, columns)
        working = weights[:, start:end].clone()
        errors = torch.empty_like(working)
        factor_block = inverse_factor[start:end, start:end]
        for column in range(end - start):
            current = working[:, column]
            quantized = quantize_column(current)
            result[:, start + column] = quantized
            error = (current - quantized) / factor_block[column, column]
            errors[:, column] = error
            working[:, column:] -= error[:, None] * factor_block[column, column:][None, :]
        if end < columns:
            weights[:, end:] -= errors @ inverse_factor[start:end, end:]
    return result


def prepare_factors(
    gram_blocks: torch.Tensor,
    in_features: int,
    *,
    block_size: int = 512,
    timings: dict[str, float] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Prepare one logical input source for all its member weights."""
    if block_size < CONVROT_GROUP_SIZE or block_size % CONVROT_GROUP_SIZE:
        raise ValueError("GPTQ Gram block size must be divisible by 256.")
    if in_features % CONVROT_GROUP_SIZE:
        raise ValueError("GPTQ ConvRot input width must be divisible by 256.")
    expected_shape = ((in_features + block_size - 1) // block_size, block_size, block_size)
    if gram_blocks.dtype != torch.float32 or tuple(gram_blocks.shape) != expected_shape:
        raise ValueError(f"GPTQ Gram must be FP32 {expected_shape}; got {gram_blocks.dtype} {tuple(gram_blocks.shape)}.")
    factors = []
    for index in range(gram_blocks.shape[0]):
        width = min(block_size, in_features - index * block_size)
        if width % CONVROT_GROUP_SIZE:
            raise ValueError("GPTQ final Gram block must contain complete 256-wide ConvRot groups.")
        with timed_stage(timings, "gram_transform"):
            rotated = rotate_gram(gram_blocks[index, :width, :width])
        with timed_stage(timings, "hessian_factor"):
            factors.append(gptq_factor(rotated))
    return tuple(factors)


def solve_blocks(
    rotated_weights: torch.Tensor,
    factors: Sequence[torch.Tensor],
    quantize_column: Callable[[torch.Tensor], torch.Tensor],
    blocksize: int = GPTQ_BLOCKSIZE,
) -> torch.Tensor:
    if sum(factor.shape[0] for factor in factors) != rotated_weights.shape[1]:
        raise ValueError("GPTQ factors do not cover the weight input width.")
    result = torch.empty_like(rotated_weights)
    start = 0
    for factor in factors:
        end = start + factor.shape[0]
        result[:, start:end] = gptq_quantize(
            rotated_weights[:, start:end], factor, quantize_column, blocksize
        )
        start = end
    return result
