"""Batched CUDA rowwise MSE scale search.

The per-candidate evaluators in ``w4a4_mse_native`` launch once per candidate
(34 launches per tensor), each with its own reduction. These functions score a
whole stage (19 coarse or 15 fine candidates) in one launch, one block per row
with the row held in registers. Their scores and multiplier arithmetic
reproduce the per-candidate CUDA path, so results are bit-identical to it.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache

import torch

from .w4a4_mse_native import _candidate_library_paths

_SYMBOL = "potatoforge_rowwise_mse_search"
_COARSE_STAGE = 0
_FINE_STAGE = 1
_AUTO_ROW_CACHE = -1
# The INT5/INT6 rowwise search clamps every candidate scale to this minimum.
ROWWISE_MSE_SCALE_FLOOR = 1e-30


@lru_cache(maxsize=1)
def _load_search_function() -> tuple[ctypes.CDLL, ctypes._CFuncPtr] | None:
    for library_path in _candidate_library_paths():
        if not library_path.is_file():
            continue
        try:
            library = ctypes.CDLL(str(library_path))
            function = getattr(library, _SYMBOL)
        except (AttributeError, OSError):
            continue
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        function.restype = ctypes.c_int
        return library, function
    return None


def _supported(weights: torch.Tensor, base_scale: torch.Tensor) -> bool:
    return (
        weights.device.type == "cuda"
        and weights.dtype == torch.float32
        and weights.ndim == 2
        and weights.is_contiguous()
        and weights.shape[0] > 0
        and weights.shape[1] > 0
        and base_scale.device == weights.device
        and base_scale.dtype == torch.float32
        and tuple(base_scale.shape) == (weights.shape[0], 1)
    )


def _run_stage(
    function: ctypes._CFuncPtr,
    weights: torch.Tensor,
    base_scale: torch.Tensor,
    best_mse: torch.Tensor,
    best_multiplier: torch.Tensor,
    *,
    qmin: int,
    qmax: int,
    scale_floor: float,
    stage: int,
    row_cache_mode: int,
) -> None:
    rows, columns = (int(value) for value in weights.shape)
    with torch.cuda.device(weights.device):
        stream = torch.cuda.current_stream(weights.device).cuda_stream
        status = function(
            ctypes.c_void_p(weights.data_ptr()),
            ctypes.c_void_p(base_scale.data_ptr()),
            ctypes.c_void_p(best_mse.data_ptr()),
            ctypes.c_void_p(best_multiplier.data_ptr()),
            ctypes.c_int(rows),
            ctypes.c_int(columns),
            ctypes.c_int(qmin),
            ctypes.c_int(qmax),
            ctypes.c_float(scale_floor),
            ctypes.c_int(stage),
            ctypes.c_int(row_cache_mode),
            ctypes.c_void_p(int(stream)),
        )
    if status != 0:
        raise RuntimeError(
            f"Fused rowwise MSE search failed for {rows}x{columns} "
            f"on {weights.device} with status {status}."
        )


def fused_rowwise_mse_coarse_search(
    weights: torch.Tensor,
    base_scale: torch.Tensor,
    *,
    qmin: int,
    qmax: int,
    scale_floor: float,
    row_cache_mode: int = _AUTO_ROW_CACHE,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return the coarse-grid [rows, 1] best MSE and multiplier, or None.

    None means the inputs or the native library are unsupported; callers then
    use the per-candidate search. ``scale_floor`` is the minimum candidate
    scale: ROWWISE_MSE_SCALE_FLOOR for INT5/INT6 and 1e-10 for W4A4.
    ``row_cache_mode`` -1 keeps rows of up to 16384 columns in registers; 0
    always reads rows from global memory.
    """
    if not _supported(weights, base_scale):
        return None
    loaded = _load_search_function()
    if loaded is None:
        return None
    scales = base_scale.contiguous()
    best_mse = torch.empty_like(scales)
    best_multiplier = torch.empty_like(scales)
    _run_stage(
        loaded[1], weights, scales, best_mse, best_multiplier,
        qmin=qmin, qmax=qmax, scale_floor=scale_floor,
        stage=_COARSE_STAGE, row_cache_mode=row_cache_mode,
    )
    return best_mse, best_multiplier


def fused_rowwise_mse_fine_search(
    weights: torch.Tensor,
    base_scale: torch.Tensor,
    best_mse: torch.Tensor,
    best_multiplier: torch.Tensor,
    *,
    qmin: int,
    qmax: int,
    scale_floor: float,
    row_cache_mode: int = _AUTO_ROW_CACHE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refine fused_rowwise_mse_coarse_search's result; returns new tensors."""
    loaded = _load_search_function()
    if loaded is None or not _supported(weights, base_scale):
        raise ValueError("Fused fine search needs the inputs its coarse search accepted.")
    for tensor in (best_mse, best_multiplier):
        if tensor.device != weights.device or tensor.dtype != torch.float32 or tuple(
            tensor.shape
        ) != (weights.shape[0], 1):
            raise ValueError("Fused fine search state must be CUDA FP32 [rows, 1].")
    refined_mse = best_mse.contiguous().clone()
    refined_multiplier = best_multiplier.contiguous().clone()
    _run_stage(
        loaded[1], weights, base_scale.contiguous(), refined_mse, refined_multiplier,
        qmin=qmin, qmax=qmax, scale_floor=scale_floor,
        stage=_FINE_STAGE, row_cache_mode=row_cache_mode,
    )
    return refined_mse, refined_multiplier
