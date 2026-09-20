from typing import NamedTuple

import torch

from ..quantization.convrot_w4a4 import (
    dequantize_convrot_w4a4,
    quantize_convrot_w4a4,
    quantize_convrot_w4a4_mse,
)
from ..quantization.hadamard import CONVROT_GROUP_SIZE
from ..quantization.int5_rowwise import (
    dequantize_int5_convrot,
    dequantize_int5_convrot_packed,
    quantize_int5_convrot,
    quantize_int5_convrot_mse,
    quantize_int5_convrot_mse_packed,
)
from ..quantization.int6_rowwise import (
    dequantize_int6_convrot,
    dequantize_int6_convrot_packed,
    dequantize_int6_rowwise,
    quantize_int6_convrot,
    quantize_int6_convrot_mse,
    quantize_int6_convrot_mse_packed,
    quantize_int6_convrot_packed,
    quantize_int6_rowwise,
)
from ..quantization.int7_rowwise import (
    dequantize_int7_convrot,
    dequantize_int7_convrot_packed,
    dequantize_int7_rowwise,
    quantize_int7_convrot,
    quantize_int7_convrot_mse,
    quantize_int7_convrot_mse_packed,
    quantize_int7_convrot_packed,
    quantize_int7_rowwise,
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
    original_float = original_float.to(device=reconstructed.device)
    source_l2 = source_l2.to(device=reconstructed.device)
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
    int7_relative_l2_error: float | None
    int8_convrot_relative_l2_error: float | None
    int5_convrot_relative_l2_error: float | None
    int5_convrot_mse_relative_l2_error: float | None
    int6_convrot_relative_l2_error: float | None
    int6_convrot_mse_relative_l2_error: float | None
    int7_convrot_relative_l2_error: float | None
    int7_convrot_mse_relative_l2_error: float | None
    w4a4_relative_l2_error: float | None
    w4a4_mse_relative_l2_error: float | None


def _dequantize_int6_convrot_for_device(
    weights: torch.Tensor,
    device: str | torch.device,
    *,
    mse: bool = False,
) -> torch.Tensor:
    if torch.device(device).type != "cuda":
        quantize = quantize_int6_convrot_mse if mse else quantize_int6_convrot
        return dequantize_int6_convrot(quantize(weights))

    quantize_packed = (
        quantize_int6_convrot_mse_packed
        if mse
        else quantize_int6_convrot_packed
    )
    packed = quantize_packed(weights, device=device)
    return dequantize_int6_convrot_packed(packed, device=device)


def _dequantize_int5_convrot_for_device(
    weights: torch.Tensor,
    device: str | torch.device,
    *,
    mse: bool = False,
) -> torch.Tensor:
    if torch.device(device).type != "cuda":
        quantize = quantize_int5_convrot_mse if mse else quantize_int5_convrot
        return dequantize_int5_convrot(quantize(weights))

    quantize_packed = (
        quantize_int5_convrot_mse_packed
        if mse
        else quantize_int5_convrot_packed
    )
    packed = quantize_packed(weights, device=device)
    return dequantize_int5_convrot_packed(packed)


def _dequantize_int7_convrot_for_device(
    weights: torch.Tensor,
    device: str | torch.device,
    *,
    mse: bool = False,
) -> torch.Tensor:
    if torch.device(device).type != "cuda":
        quantize = quantize_int7_convrot_mse if mse else quantize_int7_convrot
        return dequantize_int7_convrot(quantize(weights))

    quantize_packed = (
        quantize_int7_convrot_mse_packed
        if mse
        else quantize_int7_convrot_packed
    )
    packed = quantize_packed(weights, device=device)
    return dequantize_int7_convrot_packed(packed, device=device)


def compare_all_reconstructions(
    weights: torch.Tensor,
    *,
    original_float: torch.Tensor | None = None,
    source_l2: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
    include_plain_methods: bool = True,
    methods: frozenset[str] | None = None,
) -> AllQuantizationComparison:
    methods = (
        frozenset(
            (
                "int8",
                "int6",
                "int7",
                "int8_convrot",
                "int5_convrot",
                "int5_convrot_mse",
                "int6_convrot",
                "int6_convrot_mse",
                "int7_convrot",
                "int7_convrot_mse",
                "convrot_w4a4",
                "convrot_w4a4_mse",
            )
        )
        if methods is None
        else methods
    )
    if original_float is None:
        original_float = weights.float()
    if source_l2 is None:
        source_l2 = torch.linalg.vector_norm(original_float)

    int8_error = None
    if include_plain_methods and "int8" in methods:
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
            int7_relative_l2_error=None,
            int8_convrot_relative_l2_error=None,
            int5_convrot_relative_l2_error=None,
            int5_convrot_mse_relative_l2_error=None,
            int6_convrot_relative_l2_error=None,
            int6_convrot_mse_relative_l2_error=None,
            int7_convrot_relative_l2_error=None,
            int7_convrot_mse_relative_l2_error=None,
            w4a4_relative_l2_error=None,
            w4a4_mse_relative_l2_error=None,
        )

    int6_error = None
    if include_plain_methods and "int6" in methods:
        int6_result = quantize_int6_rowwise(weights)
        int6_reconstructed = dequantize_int6_rowwise(int6_result)
        int6_error = _relative_l2_error_from_float(
            original_float,
            int6_reconstructed,
            source_l2,
        )

        del int6_result
        del int6_reconstructed

    int7_error = None
    if include_plain_methods and "int7" in methods:
        int7_result = quantize_int7_rowwise(weights)
        int7_reconstructed = dequantize_int7_rowwise(int7_result)
        int7_error = _relative_l2_error_from_float(
            original_float,
            int7_reconstructed,
            source_l2,
        )

        del int7_result
        del int7_reconstructed

    if weights.shape[1] % CONVROT_GROUP_SIZE != 0:
        return AllQuantizationComparison(
            int8_relative_l2_error=int8_error,
            int6_relative_l2_error=int6_error,
            int7_relative_l2_error=int7_error,
            int8_convrot_relative_l2_error=None,
            int5_convrot_relative_l2_error=None,
            int5_convrot_mse_relative_l2_error=None,
            int6_convrot_relative_l2_error=None,
            int6_convrot_mse_relative_l2_error=None,
            int7_convrot_relative_l2_error=None,
            int7_convrot_mse_relative_l2_error=None,
            w4a4_relative_l2_error=None,
            w4a4_mse_relative_l2_error=None,
        )

    int8_convrot_error = None
    if "int8_convrot" in methods:
        int8_convrot_result = quantize_int8_convrot(weights, device=device)
        int8_convrot_reconstructed = dequantize_int8_convrot(int8_convrot_result)
        int8_convrot_error = _relative_l2_error_from_float(
            original_float, int8_convrot_reconstructed, source_l2
        )

    int5_convrot_error = None
    if "int5_convrot" in methods:
        int5_convrot_reconstructed = _dequantize_int5_convrot_for_device(
            weights,
            device,
        )
        int5_convrot_error = _relative_l2_error_from_float(
            original_float,
            int5_convrot_reconstructed,
            source_l2,
        )

    int5_convrot_mse_error = None
    if "int5_convrot_mse" in methods:
        int5_convrot_mse_reconstructed = _dequantize_int5_convrot_for_device(
            weights,
            device,
            mse=True,
        )
        int5_convrot_mse_error = _relative_l2_error_from_float(
            original_float,
            int5_convrot_mse_reconstructed,
            source_l2,
        )

    int6_convrot_error = None
    if "int6_convrot" in methods:
        int6_convrot_reconstructed = _dequantize_int6_convrot_for_device(weights, device)
        int6_convrot_error = _relative_l2_error_from_float(
            original_float, int6_convrot_reconstructed, source_l2
        )

    int6_convrot_mse_error = None
    if "int6_convrot_mse" in methods:
        int6_convrot_mse_reconstructed = _dequantize_int6_convrot_for_device(
            weights,
            device,
            mse=True,
        )
        int6_convrot_mse_error = _relative_l2_error_from_float(
            original_float,
            int6_convrot_mse_reconstructed,
            source_l2,
        )

    int7_convrot_error = None
    if "int7_convrot" in methods:
        int7_convrot_reconstructed = _dequantize_int7_convrot_for_device(
            weights,
            device,
        )
        int7_convrot_error = _relative_l2_error_from_float(
            original_float,
            int7_convrot_reconstructed,
            source_l2,
        )

    int7_convrot_mse_error = None
    if "int7_convrot_mse" in methods:
        int7_convrot_mse_reconstructed = _dequantize_int7_convrot_for_device(
            weights,
            device,
            mse=True,
        )
        int7_convrot_mse_error = _relative_l2_error_from_float(
            original_float,
            int7_convrot_mse_reconstructed,
            source_l2,
        )

    w4a4_error = None
    if "convrot_w4a4" in methods:
        w4a4_result = quantize_convrot_w4a4(weights, device=device)
        w4a4_reconstructed = dequantize_convrot_w4a4(w4a4_result)
        w4a4_error = _relative_l2_error_from_float(
            original_float, w4a4_reconstructed, source_l2
        )

    w4a4_mse_error = None
    if "convrot_w4a4_mse" in methods:
        w4a4_mse_result = quantize_convrot_w4a4_mse(weights, device=device)
        w4a4_mse_reconstructed = dequantize_convrot_w4a4(w4a4_mse_result)
        w4a4_mse_error = _relative_l2_error_from_float(
            original_float, w4a4_mse_reconstructed, source_l2
        )

    return AllQuantizationComparison(
        int8_relative_l2_error=int8_error,
        int6_relative_l2_error=int6_error,
        int7_relative_l2_error=int7_error,
        int8_convrot_relative_l2_error=int8_convrot_error,
        int5_convrot_relative_l2_error=int5_convrot_error,
        int5_convrot_mse_relative_l2_error=int5_convrot_mse_error,
        int6_convrot_relative_l2_error=int6_convrot_error,
        int6_convrot_mse_relative_l2_error=int6_convrot_mse_error,
        int7_convrot_relative_l2_error=int7_convrot_error,
        int7_convrot_mse_relative_l2_error=int7_convrot_mse_error,
        w4a4_relative_l2_error=w4a4_error,
        w4a4_mse_relative_l2_error=w4a4_mse_error,
    )
