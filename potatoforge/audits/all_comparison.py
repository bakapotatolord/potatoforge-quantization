from typing import NamedTuple

import torch

from ..quantization.convrot_w4a4 import (
    dequantize_convrot_w4a4,
    quantize_convrot_w4a4,
    quantize_convrot_w4a4_mse,
)
from ..quantization.hadamard import CONVROT_GROUP_SIZE
from ..quantization.int6_rowwise import (
    dequantize_int6_convrot,
    dequantize_int6_convrot_packed,
    dequantize_int6_rowwise,
    quantize_int6_convrot,
    quantize_int6_convrot_packed,
    quantize_int6_rowwise,
)
from ..quantization.int8_tensorwise import (
    dequantize_int8_convrot,
    dequantize_int8_tensorwise,
    quantize_int8_convrot,
    quantize_int8_tensorwise,
)
def _relative_l2_error_from_float(
    original_float: torch.Tensor,
    reconstructed: torch.Tensor,
    source_l2: torch.Tensor,
) -> float:
    error_norm = torch.linalg.vector_norm(
        original_float - reconstructed.float()
    )

    if source_l2.item() == 0:
        if error_norm == 0:
            return 0.0

        return float("inf")

    return float((error_norm / source_l2).item())


def relative_l2_error(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    if original.shape != reconstructed.shape:
        raise ValueError(
            "Original and reconstructed tensors must have matching shapes."
        )

    original_float = original.float()
    source_l2 = torch.linalg.vector_norm(original_float)
    return _relative_l2_error_from_float(
        original_float,
        reconstructed,
        source_l2,
    )


class AllQuantizationComparison(NamedTuple):
    int8_relative_l2_error: float | None
    int6_relative_l2_error: float | None
    int8_convrot_relative_l2_error: float | None
    int6_convrot_relative_l2_error: float | None
    w4a4_relative_l2_error: float | None
    w4a4_mse_relative_l2_error: float | None


def _dequantize_int6_convrot_for_device(
    weights: torch.Tensor,
    device: str | torch.device,
) -> torch.Tensor:
    if torch.device(device).type != "cuda":
        return dequantize_int6_convrot(quantize_int6_convrot(weights))

    packed = quantize_int6_convrot_packed(weights, device=device)
    return dequantize_int6_convrot_packed(packed, device=device)


def compare_all_reconstructions(
    weights: torch.Tensor,
    *,
    original_float: torch.Tensor | None = None,
    source_l2: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
    include_plain_methods: bool = True,
) -> AllQuantizationComparison:
    if original_float is None:
        original_float = weights.float()
    if source_l2 is None:
        source_l2 = torch.linalg.vector_norm(original_float)

    int8_error = None
    if include_plain_methods:
        int8_result = quantize_int8_tensorwise(weights)
        int8_reconstructed = dequantize_int8_tensorwise(int8_result)
        int8_error = _relative_l2_error_from_float(
            original_float,
            int8_reconstructed,
            source_l2,
        )

        del int8_result
        del int8_reconstructed

    if weights.shape[1] % 4 != 0:
        return AllQuantizationComparison(
            int8_relative_l2_error=int8_error,
            int6_relative_l2_error=None,
            int8_convrot_relative_l2_error=None,
            int6_convrot_relative_l2_error=None,
            w4a4_relative_l2_error=None,
            w4a4_mse_relative_l2_error=None,
        )

    int6_error = None
    if include_plain_methods:
        int6_result = quantize_int6_rowwise(weights)
        int6_reconstructed = dequantize_int6_rowwise(int6_result)
        int6_error = _relative_l2_error_from_float(
            original_float,
            int6_reconstructed,
            source_l2,
        )

        del int6_result
        del int6_reconstructed

    if weights.shape[1] % CONVROT_GROUP_SIZE != 0:
        return AllQuantizationComparison(
            int8_relative_l2_error=int8_error,
            int6_relative_l2_error=int6_error,
            int8_convrot_relative_l2_error=None,
            int6_convrot_relative_l2_error=None,
            w4a4_relative_l2_error=None,
            w4a4_mse_relative_l2_error=None,
        )

    int8_convrot_result = quantize_int8_convrot(
        weights,
        device=device,
    )
    int8_convrot_reconstructed = dequantize_int8_convrot(
        int8_convrot_result,
    )
    int8_convrot_error = _relative_l2_error_from_float(
        original_float,
        int8_convrot_reconstructed,
        source_l2,
    )

    del int8_convrot_result
    del int8_convrot_reconstructed

    int6_convrot_reconstructed = _dequantize_int6_convrot_for_device(
        weights,
        device,
    )
    int6_convrot_error = _relative_l2_error_from_float(
        original_float,
        int6_convrot_reconstructed,
        source_l2,
    )

    del int6_convrot_reconstructed

    w4a4_result = quantize_convrot_w4a4(
        weights,
        device=device,
    )
    w4a4_reconstructed = dequantize_convrot_w4a4(w4a4_result)
    w4a4_error = _relative_l2_error_from_float(
        original_float,
        w4a4_reconstructed,
        source_l2,
    )
    w4a4_mse_result = quantize_convrot_w4a4_mse(
        weights,
        device=device,
    )
    w4a4_mse_reconstructed = dequantize_convrot_w4a4(w4a4_mse_result)
    w4a4_mse_error = _relative_l2_error_from_float(
        original_float,
        w4a4_mse_reconstructed,
        source_l2,
    )

    return AllQuantizationComparison(
        int8_relative_l2_error=int8_error,
        int6_relative_l2_error=int6_error,
        int8_convrot_relative_l2_error=int8_convrot_error,
        int6_convrot_relative_l2_error=int6_convrot_error,
        w4a4_relative_l2_error=w4a4_error,
        w4a4_mse_relative_l2_error=w4a4_mse_error,
    )
