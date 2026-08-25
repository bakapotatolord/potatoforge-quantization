from typing import Final

import torch


CONVROT_GROUP_SIZE: Final[int] = 256


def build_normalized_hadamard_matrix(group_size: int) -> torch.Tensor:
    remaining_size = group_size

    while remaining_size % 4 == 0:
        remaining_size //= 4

    if group_size < 4 or remaining_size != 1:
        raise ValueError(
            "Hadamard group size must be a power of four"
        )

    base = torch.tensor(
        [
            [1.0, 1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    matrix = base
    current_size = 4

    while current_size < group_size:
        matrix = torch.kron(matrix, base)
        current_size *= 4

    return matrix / (group_size**0.5)


def apply_hadamard_rotation(weights: torch.Tensor, group_size: int, output_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if weights.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(
            "Hadamard weights must use torch.bfloat16, torch.float16, "
            "or torch.float32"
        )

    if output_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(
            "Hadamard output dtype must be torch.bfloat16 "
            "or torch.float32"
        )

    if weights.ndim != 2:
        raise ValueError("Hadamard weights must be a 2D matrix")

    if weights.shape[1] % group_size != 0:
        raise ValueError(
            "weight input features must be divisible by the Hadamard "
            "group size"
        )

    rotation_weights = weights.to(dtype=output_dtype)

    hadamard = build_normalized_hadamard_matrix(group_size).to(
        device=weights.device,
        dtype=output_dtype,
    )

    output_features, input_features = rotation_weights.shape
    grouped_weights = rotation_weights.reshape(
        output_features,
        input_features // group_size,
        group_size,
    )

    rotated_groups = grouped_weights @ hadamard.T

    return rotated_groups.reshape(
        output_features,
        input_features,
    )
