from typing import NamedTuple

import torch

from ..quantization.convrot_w4a4 import (
    dequantize_convrot_w4a4,
    quantize_convrot_w4a4,
)
from ..quantization.hadamard import CONVROT_GROUP_SIZE
from ..quantization.int6_rowwise import (
    dequantize_int6_convrot,
    dequantize_int6_rowwise,
    quantize_int6_convrot,
    quantize_int6_rowwise,
)
from ..quantization.int8_tensorwise import (
    dequantize_int8_convrot,
    dequantize_int8_tensorwise,
    quantize_int8_convrot,
    quantize_int8_tensorwise,
)


def relative_l2_error(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    if original.shape != reconstructed.shape:
        raise ValueError(
            "Original and reconstructed tensors must have matching shapes."
        )

    original_float = original.float()
    reconstructed_float = reconstructed.float()

    original_norm = torch.linalg.vector_norm(original_float)
    error_norm = torch.linalg.vector_norm(
        original_float - reconstructed_float
    )

    if original_norm == 0:
        if error_norm == 0:
            return 0.0

        return float("inf")

    return float((error_norm / original_norm).item())


class AllQuantizationComparison(NamedTuple):
    int8_relative_l2_error: float
    int6_relative_l2_error: float | None
    int8_convrot_relative_l2_error: float | None
    int6_convrot_relative_l2_error: float | None
    w4a4_relative_l2_error: float | None


def compare_all_reconstructions(
    weights: torch.Tensor,
) -> AllQuantizationComparison:
    int8_result = quantize_int8_tensorwise(weights)
    int8_reconstructed = dequantize_int8_tensorwise(int8_result)
    int8_error = relative_l2_error(weights, int8_reconstructed)

    del int8_result
    del int8_reconstructed

    if weights.shape[1] % 4 != 0:
        return AllQuantizationComparison(
            int8_relative_l2_error=int8_error,
            int6_relative_l2_error=None,
            int8_convrot_relative_l2_error=None,
            int6_convrot_relative_l2_error=None,
            w4a4_relative_l2_error=None,
        )

    int6_result = quantize_int6_rowwise(weights)
    int6_reconstructed = dequantize_int6_rowwise(int6_result)
    int6_error = relative_l2_error(weights, int6_reconstructed)

    del int6_result
    del int6_reconstructed

    if weights.shape[1] % CONVROT_GROUP_SIZE != 0:
        return AllQuantizationComparison(
            int8_relative_l2_error=int8_error,
            int6_relative_l2_error=int6_error,
            int8_convrot_relative_l2_error=None,
            int6_convrot_relative_l2_error=None,
            w4a4_relative_l2_error=None,
        )

    int8_convrot_result = quantize_int8_convrot(weights)
    int8_convrot_reconstructed = dequantize_int8_convrot(
        int8_convrot_result,
    )
    int8_convrot_error = relative_l2_error(
        weights,
        int8_convrot_reconstructed,
    )

    del int8_convrot_result
    del int8_convrot_reconstructed

    int6_convrot_result = quantize_int6_convrot(weights)
    int6_convrot_reconstructed = dequantize_int6_convrot(int6_convrot_result)
    int6_convrot_error = relative_l2_error(
        weights,
        int6_convrot_reconstructed,
    )

    del int6_convrot_result
    del int6_convrot_reconstructed

    w4a4_result = quantize_convrot_w4a4(weights)
    w4a4_reconstructed = dequantize_convrot_w4a4(w4a4_result)
    w4a4_error = relative_l2_error(weights, w4a4_reconstructed)

    return AllQuantizationComparison(
        int8_relative_l2_error=int8_error,
        int6_relative_l2_error=int6_error,
        int8_convrot_relative_l2_error=int8_convrot_error,
        int6_convrot_relative_l2_error=int6_convrot_error,
        w4a4_relative_l2_error=w4a4_error,
    )
