from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path

import torch


_LIBRARY_ENVIRONMENT_VARIABLE = "POTATOFORGE_W4A4_MSE_LIBRARY"
_WINDOWS_LIBRARY_NAME = "potatoforge_w4a4_mse_fused.dll"
_POSIX_LIBRARY_NAME = "libpotatoforge_w4a4_mse_fused.so"


def _library_name() -> str:
    return (
        _WINDOWS_LIBRARY_NAME
        if os.name == "nt"
        else _POSIX_LIBRARY_NAME
    )


def _candidate_library_paths() -> tuple[Path, ...]:
    configured_path = os.environ.get(_LIBRARY_ENVIRONMENT_VARIABLE)
    if configured_path:
        return (Path(configured_path),)

    library_name = _library_name()
    package_path = Path(__file__).with_name(library_name)
    repository_build_path = (
        Path(__file__).resolve().parents[2] / "build" / library_name
    )
    return package_path, repository_build_path


def _configure_native_function(
    library: ctypes.CDLL,
) -> ctypes._CFuncPtr:
    function = library.potatoforge_w4a4_mse_candidate
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    function.restype = ctypes.c_int
    return function


class W4A4MSECandidateEvaluator:
    def __init__(
        self,
        library: ctypes.CDLL,
        function: ctypes._CFuncPtr,
        device: torch.device,
    ) -> None:
        self._library = library
        self._function = function
        self._device = device
        self._output: torch.Tensor | None = None

    def __call__(
        self,
        weights: torch.Tensor,
        weights_float: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        del weights_float
        if weights.device != self._device or weights.dtype != torch.float32:
            raise ValueError(
                "Fused evaluator expects contiguous CUDA FP32 weights."
            )
        if scales.device != self._device or scales.dtype != torch.float32:
            raise ValueError(
                "Fused evaluator expects contiguous CUDA FP32 scales."
            )
        if not weights.is_contiguous() or not scales.is_contiguous():
            raise ValueError("Fused evaluator inputs must be contiguous.")
        if weights.ndim != 2:
            raise ValueError("Fused evaluator weights must be 2D.")
        if scales.ndim != 2 or scales.shape[1] != 1:
            raise ValueError("Fused evaluator scales must have shape [rows, 1].")

        rows, columns = (int(value) for value in weights.shape)
        if scales.shape[0] != rows:
            raise ValueError("Fused evaluator scale rows must match weights.")

        if (
            self._output is None
            or self._output.shape != scales.shape
            or self._output.device != self._device
        ):
            self._output = torch.empty(
                scales.shape,
                dtype=torch.float32,
                device=self._device,
            )

        stream = torch.cuda.current_stream(self._device).cuda_stream
        status = self._function(
            ctypes.c_void_p(weights.data_ptr()),
            ctypes.c_void_p(scales.data_ptr()),
            ctypes.c_void_p(self._output.data_ptr()),
            ctypes.c_int(rows),
            ctypes.c_int(columns),
            ctypes.c_void_p(int(stream)),
        )
        if status != 0:
            raise RuntimeError(
                "Fused W4A4-MSE CUDA launch failed "
                f"for {rows}x{columns} on {self._device} with status {status}."
            )

        # The search copies this result through torch.where before the next
        # candidate overwrites the reusable buffer on the same CUDA stream.
        return self._output


@lru_cache(maxsize=8)
def load_w4a4_mse_candidate(
    device_index: int,
) -> W4A4MSECandidateEvaluator | None:
    device = torch.device("cuda", device_index)
    for library_path in _candidate_library_paths():
        if not library_path.is_file():
            continue
        try:
            library = ctypes.CDLL(str(library_path))
            function = _configure_native_function(library)
        except (AttributeError, OSError):
            continue
        return W4A4MSECandidateEvaluator(library, function, device)
    return None
