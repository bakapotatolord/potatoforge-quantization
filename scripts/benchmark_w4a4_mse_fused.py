"""Benchmark a fused CUDA W4A4-MSE candidate evaluator on one tensor.

This is a development benchmark only. It does not participate in the normal
quantization command or production quantizer dispatch.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from potatoforge.headers.source_header import read_source_model_header
from potatoforge.quantization.convrot_w4a4 import (
    _W4A4_MSE_COARSE_MULTIPLIERS,
    _W4A4_MSE_FINE_STEPS,
    _W4_SCALE_FLOOR,
    _calculate_w4a4_candidate_mse,
    _quantize_w4_codes,
    _run_w4a4_mse_coarse_search,
    _run_w4a4_mse_fine_search,
    _select_w4_absmax_scale,
    _w4a4_mse_zero_row_state,
    ConvRotW4A4Result,
    pack_signed_int4_row_major,
    unpack_signed_int4_row_major,
)
from potatoforge.quantization.hadamard import (
    CONVROT_GROUP_SIZE,
    apply_hadamard_rotation,
    cached_cuda_hadamard_matrix,
)
from potatoforge.quantization.w4a4_mse_native import (
    W4A4MSECandidateEvaluator,
)
from potatoforge.source_payloads import stream_bf16_source_tensors


SHAPE = (16384, 6144)
SOURCE_DTYPE = torch.bfloat16


@dataclass
class Samples:
    values: list[float]

    @property
    def median(self) -> float:
        return statistics.median(self.values)


@dataclass
class CandidateMeasurement:
    samples: Samples
    peak_allocated: int
    peak_reserved: int


@dataclass
class SearchMeasurement:
    coarse: Samples
    fine: Samples
    combined: Samples


@dataclass
class SearchResult:
    coarse_multiplier: torch.Tensor
    fine_multiplier: torch.Tensor
    result: ConvRotW4A4Result


CandidateEvaluator = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]


def _torch_candidate_mse(
    weights: torch.Tensor,
    weights_float: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    return _calculate_w4a4_candidate_mse(weights, weights_float, scales)


def _random_source_tensor(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(
        SHAPE,
        generator=generator,
        dtype=torch.float32,
    ).to(SOURCE_DTYPE)


def _model_source_tensor(model: Path, tensor_name: str) -> torch.Tensor:
    header = read_source_model_header(model)
    descriptor = header.tensors.get(tensor_name)
    if descriptor is None:
        raise ValueError(f"Tensor {tensor_name!r} was not found in {model}.")
    _, source = next(
        stream_bf16_source_tensors(
            model,
            [(tensor_name, descriptor)],
        )
    )
    if source.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got {source.shape}.")
    return source


def _build_native_library(
    nvcc: str,
) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    temporary_directory = tempfile.TemporaryDirectory(
        prefix="potatoforge-w4a4-mse-",
        ignore_cleanup_errors=True,
    )
    output = Path(temporary_directory.name) / (
        "potatoforge_w4a4_mse_fused.dll"
        if os.name == "nt"
        else "libpotatoforge_w4a4_mse_fused.so"
    )
    source = PROJECT_ROOT / "potatoforge" / "quantization" / "w4a4_mse_fused.cu"
    command = [nvcc, "-O3", "--shared"]
    command.append("-Xcompiler=/MD" if os.name == "nt" else "-Xcompiler=-fPIC")
    command.extend([str(source), "-o", str(output)])
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        temporary_directory.cleanup()
        raise RuntimeError(
            "nvcc failed while building the fused evaluator:\n"
            f"{' '.join(command)}\n{completed.stdout}{completed.stderr}"
        )
    return temporary_directory, output


def _load_native_function(
    library_path: Path,
) -> tuple[ctypes.CDLL, ctypes._CFuncPtr]:
    library = ctypes.CDLL(str(library_path))
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
    return library, function


def _sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _timed_cuda(
    device: torch.device,
    operation: Callable[[], torch.Tensor | ConvRotW4A4Result],
) -> tuple[torch.Tensor | ConvRotW4A4Result, float]:
    _sync(device)
    started_at = time.perf_counter()
    result = operation()
    _sync(device)
    return result, time.perf_counter() - started_at


def _mib(byte_count: int) -> float:
    return byte_count / 1024**2


def _prepare_rotated(
    source: torch.Tensor,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    bool,
]:
    device_source = source.to(device=device)
    rotated = apply_hadamard_rotation(
        device_source,
        group_size=CONVROT_GROUP_SIZE,
        output_dtype=torch.float32,
        hadamard=cached_cuda_hadamard_matrix(
            CONVROT_GROUP_SIZE,
            device,
            torch.float32,
        ),
    )
    base_scale = _select_w4_absmax_scale(rotated)
    weights_float, zero_rows, all_zero = _w4a4_mse_zero_row_state(rotated)
    return device_source, rotated, base_scale, weights_float, zero_rows, all_zero


def _run_fused_coarse_search(
    evaluator: CandidateEvaluator,
    weights: torch.Tensor,
    weights_float: torch.Tensor,
    base_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        best_mse = torch.full_like(base_scale, float("inf"))
        best_multiplier = torch.ones_like(base_scale)
        for multiplier in _W4A4_MSE_COARSE_MULTIPLIERS:
            candidate_multiplier = torch.full_like(best_multiplier, multiplier)
            mse = evaluator(
                weights,
                weights_float,
                base_scale * multiplier,
            )
            better = mse < best_mse
            best_mse = torch.where(better, mse, best_mse)
            best_multiplier = torch.where(
                better,
                candidate_multiplier,
                best_multiplier,
            )
        return best_mse, best_multiplier


def _run_fused_fine_search(
    evaluator: CandidateEvaluator,
    weights: torch.Tensor,
    weights_float: torch.Tensor,
    base_scale: torch.Tensor,
    best_mse: torch.Tensor,
    best_multiplier: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        lower = (best_multiplier - 0.05).clamp_min(0.10)
        upper = (best_multiplier + 0.05).clamp_max(1.0)
        for step in range(1, _W4A4_MSE_FINE_STEPS):
            fraction = step / _W4A4_MSE_FINE_STEPS
            candidate_multiplier = lower + (upper - lower) * fraction
            mse = evaluator(
                weights,
                weights_float,
                base_scale * candidate_multiplier,
            )
            better = mse < best_mse
            best_mse = torch.where(better, mse, best_mse)
            best_multiplier = torch.where(
                better,
                candidate_multiplier,
                best_multiplier,
            )
        return best_mse, best_multiplier


def _run_coarse_search(
    evaluator: CandidateEvaluator,
    weights: torch.Tensor,
    weights_float: torch.Tensor,
    base_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if evaluator is _torch_candidate_mse:
        return _run_w4a4_mse_coarse_search(
            weights,
            weights_float,
            base_scale,
        )
    return _run_fused_coarse_search(
        evaluator,
        weights,
        weights_float,
        base_scale,
    )


def _run_fine_search(
    evaluator: CandidateEvaluator,
    weights: torch.Tensor,
    weights_float: torch.Tensor,
    base_scale: torch.Tensor,
    best_mse: torch.Tensor,
    best_multiplier: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if evaluator is _torch_candidate_mse:
        return _run_w4a4_mse_fine_search(
            weights,
            weights_float,
            base_scale,
            best_mse,
            best_multiplier,
        )
    return _run_fused_fine_search(
        evaluator,
        weights,
        weights_float,
        base_scale,
        best_mse,
        best_multiplier,
    )


def _search_result(
    evaluator: CandidateEvaluator,
    weights: torch.Tensor,
    weights_float: torch.Tensor,
    base_scale: torch.Tensor,
    zero_rows: torch.Tensor,
    all_zero: bool,
) -> SearchResult:
    if all_zero:
        coarse_multiplier = torch.ones_like(base_scale)
        fine_multiplier = torch.ones_like(base_scale)
    else:
        coarse_mse, coarse_multiplier = _run_coarse_search(
            evaluator,
            weights,
            weights_float,
            base_scale,
        )
        _, fine_multiplier = _run_fine_search(
            evaluator,
            weights,
            weights_float,
            base_scale,
            coarse_mse,
            coarse_multiplier,
        )
    scales = torch.where(
        zero_rows,
        base_scale,
        (base_scale * fine_multiplier).clamp_min(_W4_SCALE_FLOOR),
    )
    codes, stored_scales = _quantize_w4_codes(weights, scales)
    return SearchResult(
        coarse_multiplier=coarse_multiplier,
        fine_multiplier=fine_multiplier,
        result=ConvRotW4A4Result(
            packed_codes=pack_signed_int4_row_major(codes).cpu(),
            scales=stored_scales.squeeze(dim=1).cpu(),
        ),
    )


def _measure_candidate(
    evaluator: CandidateEvaluator,
    weights: torch.Tensor,
    weights_float: torch.Tensor,
    scales: torch.Tensor,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> CandidateMeasurement:
    for _ in range(warmup):
        result = evaluator(weights, weights_float, scales)
        _sync(device)
        del result

    gc.collect()
    torch.cuda.empty_cache()
    _sync(device)
    torch.cuda.reset_peak_memory_stats(device)
    values: list[float] = []
    for _ in range(iterations):
        result, elapsed = _timed_cuda(
            device,
            lambda: evaluator(weights, weights_float, scales),
        )
        values.append(elapsed)
        del result
    return CandidateMeasurement(
        samples=Samples(values),
        peak_allocated=torch.cuda.max_memory_allocated(device),
        peak_reserved=torch.cuda.max_memory_reserved(device),
    )


def _measure_search(
    evaluator: CandidateEvaluator,
    weights: torch.Tensor,
    weights_float: torch.Tensor,
    base_scale: torch.Tensor,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> SearchMeasurement:
    def run_once_measured() -> dict[str, float]:
        _sync(device)
        started_at = time.perf_counter()
        best_mse, best_multiplier = _run_coarse_search(
            evaluator,
            weights,
            weights_float,
            base_scale,
        )
        _sync(device)
        coarse_finished_at = time.perf_counter()
        _run_fine_search(
            evaluator,
            weights,
            weights_float,
            base_scale,
            best_mse,
            best_multiplier,
        )
        _sync(device)
        finished_at = time.perf_counter()
        return {
            "coarse": coarse_finished_at - started_at,
            "fine": finished_at - coarse_finished_at,
            "combined": finished_at - started_at,
        }

    for _ in range(warmup):
        run_once_measured()
    gc.collect()
    torch.cuda.empty_cache()
    _sync(device)

    measurements = {stage: [] for stage in ("coarse", "fine", "combined")}
    for _ in range(iterations):
        result = run_once_measured()
        for stage, value in result.items():
            measurements[stage].append(value)
    return SearchMeasurement(
        coarse=Samples(measurements["coarse"]),
        fine=Samples(measurements["fine"]),
        combined=Samples(measurements["combined"]),
    )


def _timed_stage(
    timings: dict[str, float],
    stage: str,
    device: torch.device,
    operation: Callable[[], torch.Tensor | tuple[torch.Tensor, ...] | ConvRotW4A4Result],
):
    _sync(device)
    started_at = time.perf_counter()
    result = operation()
    _sync(device)
    timings[stage] = time.perf_counter() - started_at
    return result


def _run_fused_complete(
    source: torch.Tensor,
    device: torch.device,
    evaluator: CandidateEvaluator,
) -> tuple[ConvRotW4A4Result, dict[str, float]]:
    timings: dict[str, float] = {}
    device_source = _timed_stage(
        timings,
        "prepare",
        device,
        lambda: source.to(device=device),
    )
    rotated = _timed_stage(
        timings,
        "rotation",
        device,
        lambda: apply_hadamard_rotation(
            device_source,
            group_size=CONVROT_GROUP_SIZE,
            output_dtype=torch.float32,
            hadamard=cached_cuda_hadamard_matrix(
                CONVROT_GROUP_SIZE,
                device,
                torch.float32,
            ),
        ),
    )
    base_scale = _timed_stage(
        timings,
        "scale_init",
        device,
        lambda: _select_w4_absmax_scale(rotated),
    )
    weights_float, zero_rows, all_zero = _timed_stage(
        timings,
        "zero_row_check",
        device,
        lambda: _w4a4_mse_zero_row_state(rotated),
    )
    if all_zero:
        best_multiplier = None
    else:
        best_mse, best_multiplier = _timed_stage(
            timings,
            "coarse_search",
            device,
            lambda: _run_fused_coarse_search(
                evaluator,
                rotated,
                weights_float,
                base_scale,
            ),
        )
        _, best_multiplier = _timed_stage(
            timings,
            "fine_search",
            device,
            lambda: _run_fused_fine_search(
                evaluator,
                rotated,
                weights_float,
                base_scale,
                best_mse,
                best_multiplier,
            ),
        )
    def final_quantize() -> tuple[torch.Tensor, torch.Tensor]:
        if all_zero:
            scales = base_scale
        else:
            scales = torch.where(
                zero_rows,
                base_scale,
                (base_scale * best_multiplier).clamp_min(
                    _W4_SCALE_FLOOR,
                ),
            )
        return _quantize_w4_codes(rotated, scales)

    codes, stored_scales = _timed_stage(
        timings,
        "final_quantize",
        device,
        final_quantize,
    )
    packed_codes = _timed_stage(
        timings,
        "pack",
        device,
        lambda: pack_signed_int4_row_major(codes),
    )
    result = _timed_stage(
        timings,
        "finalize",
        device,
        lambda: ConvRotW4A4Result(
            packed_codes=packed_codes.cpu(),
            scales=stored_scales.squeeze(dim=1).cpu(),
        ),
    )
    del device_source, rotated, weights_float, zero_rows, codes, stored_scales
    return result, timings


def _measure_complete_baseline(
    source: torch.Tensor,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, Samples], Samples]:
    from potatoforge.quantization.convrot_w4a4 import quantize_convrot_w4a4_mse

    def run_once() -> tuple[float, dict[str, float]]:
        timings: dict[str, float] = {}
        _sync(device)
        started_at = time.perf_counter()
        result = quantize_convrot_w4a4_mse(
            source,
            device=device,
            internal_timings=timings,
        )
        _sync(device)
        elapsed = time.perf_counter() - started_at
        del result
        return elapsed, timings

    for _ in range(warmup):
        run_once()
    gc.collect()
    torch.cuda.empty_cache()
    _sync(device)
    stage_names = (
        "prepare",
        "rotation",
        "scale_init",
        "zero_row_check",
        "coarse_search",
        "fine_search",
        "final_quantize",
        "pack",
        "finalize",
    )
    measurements = {stage: [] for stage in stage_names}
    totals: list[float] = []
    for _ in range(iterations):
        total, timings = run_once()
        totals.append(total)
        for stage in stage_names:
            measurements[stage].append(timings.get(stage, 0.0))
    return {stage: Samples(values) for stage, values in measurements.items()}, Samples(totals)


def _measure_complete_fused(
    source: torch.Tensor,
    device: torch.device,
    evaluator: CandidateEvaluator,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, Samples], Samples]:
    for _ in range(warmup):
        result, _ = _run_fused_complete(source, device, evaluator)
        del result
    gc.collect()
    torch.cuda.empty_cache()
    _sync(device)
    stage_names = (
        "prepare",
        "rotation",
        "scale_init",
        "zero_row_check",
        "coarse_search",
        "fine_search",
        "final_quantize",
        "pack",
        "finalize",
    )
    measurements = {stage: [] for stage in stage_names}
    totals: list[float] = []
    for _ in range(iterations):
        _sync(device)
        started_at = time.perf_counter()
        result, timings = _run_fused_complete(source, device, evaluator)
        _sync(device)
        totals.append(time.perf_counter() - started_at)
        del result
        for stage in stage_names:
            measurements[stage].append(timings.get(stage, 0.0))
    return {stage: Samples(values) for stage, values in measurements.items()}, Samples(totals)


def _format_samples(samples: Samples) -> str:
    return f"median {samples.median:.4f} s"


def _speedup(baseline: Samples, candidate: Samples) -> float:
    return baseline.median / candidate.median if candidate.median else float("inf")


def _compare_candidate_outputs(
    reference: torch.Tensor,
    fused: torch.Tensor,
) -> dict[str, float]:
    difference = fused - reference
    denominator = reference.norm().item()
    exact_rows = (reference == fused).all(dim=1).float().mean().item()
    return {
        "max_abs": difference.abs().max().item(),
        "relative_l2": difference.norm().item() / denominator
        if denominator
        else 0.0,
        "exact_row_percent": exact_rows * 100.0,
    }


def _compare_search_results(
    reference: SearchResult,
    fused: SearchResult,
    rotated: torch.Tensor,
) -> dict[str, float | int]:
    coarse_matches = int(
        torch.count_nonzero(
            reference.coarse_multiplier == fused.coarse_multiplier,
        ).item()
    )
    fine_matches = int(
        torch.count_nonzero(
            reference.fine_multiplier == fused.fine_multiplier,
        ).item()
    )
    reference_codes = unpack_signed_int4_row_major(
        reference.result.packed_codes,
    )
    fused_codes = unpack_signed_int4_row_major(fused.result.packed_codes)
    code_difference = (
        fused_codes.to(torch.int16) - reference_codes.to(torch.int16)
    ).abs()
    scale_difference = (
        fused.result.scales.float() - reference.result.scales.float()
    ).abs()
    rotated_cpu = rotated.cpu()
    reference_reconstruction = (
        reference_codes.float() * reference.result.scales.unsqueeze(dim=1)
    )
    fused_reconstruction = fused_codes.float() * fused.result.scales.unsqueeze(dim=1)
    reference_mse = (rotated_cpu - reference_reconstruction).square().mean(dim=1)
    fused_mse = (rotated_cpu - fused_reconstruction).square().mean(dim=1)
    code_matches = int(torch.count_nonzero(reference_codes == fused_codes).item())
    code_count = reference_codes.numel()
    packed_matches = int(
        torch.count_nonzero(
            reference.result.packed_codes == fused.result.packed_codes,
        ).item()
    )
    packed_count = reference.result.packed_codes.numel()
    return {
        "coarse_winner_percent": coarse_matches / reference.coarse_multiplier.numel() * 100.0,
        "coarse_winner_matches": coarse_matches,
        "coarse_winner_count": reference.coarse_multiplier.numel(),
        "fine_winner_percent": fine_matches / reference.fine_multiplier.numel() * 100.0,
        "fine_winner_matches": fine_matches,
        "fine_winner_count": reference.fine_multiplier.numel(),
        "code_match_percent": code_matches / code_count * 100.0,
        "code_matches": code_matches,
        "code_count": code_count,
        "packed_match_percent": packed_matches / packed_count * 100.0,
        "packed_matches": packed_matches,
        "packed_count": packed_count,
        "max_code_difference": float(code_difference.max()),
        "max_scale_difference": float(scale_difference.max()),
        "max_final_mse_difference": float((fused_mse - reference_mse).abs().max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark one fused CUDA W4A4-MSE candidate tensor."
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model",
        type=Path,
        help="Read one tensor from this safetensors model instead of generating random data.",
    )
    parser.add_argument(
        "--tensor",
        default="blocks.4.mlp.gate.weight",
        help="Tensor name to read with --model.",
    )
    parser.add_argument(
        "--library",
        type=Path,
        help="Use an existing native library instead of compiling with nvcc.",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error("--warmup must be non-negative and --iterations >= 1")

    print("W4A4-MSE fused CUDA candidate benchmark")
    print("========================================")
    print(f"Shape: {SHAPE[0]} x {SHAPE[1]}")
    print(f"Warmup: {args.warmup}; measured iterations: {args.iterations}")
    print()

    if not torch.cuda.is_available():
        print("CUDA unavailable; benchmark not run.")
        return 2

    device = torch.device("cuda:0")
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    try:
        library_path = args.library
        if library_path is None:
            nvcc = shutil.which("nvcc")
            if nvcc is None:
                raise RuntimeError("nvcc was not found on PATH.")
            temporary_directory, library_path = _build_native_library(nvcc)
        native_library, native_function = _load_native_function(library_path)
        fused_evaluator = W4A4MSECandidateEvaluator(
            native_library,
            native_function,
            device,
        )

        source = (
            _model_source_tensor(args.model, args.tensor)
            if args.model is not None
            else _random_source_tensor(args.seed)
        )
        if tuple(source.shape) != SHAPE:
            raise ValueError(
                f"Representative tensor must have shape {SHAPE}; "
                f"got {tuple(source.shape)}."
            )
        if source.dtype not in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ):
            raise ValueError(f"Unsupported source dtype: {source.dtype}.")
        print(
            f"Source: {args.tensor if args.model is not None else 'deterministic random tensor'}"
        )
        print(f"Source dtype: {source.dtype}")
        print(f"Source size: {_mib(source.numel() * source.element_size()):.1f} MiB")
        with torch.no_grad():
            (
                device_source,
                rotated,
                base_scale,
                weights_float,
                zero_rows,
                all_zero,
            ) = _prepare_rotated(
                source,
                device,
            )
            candidate_scales = base_scale * 0.73
            print()

            reference_candidate = _torch_candidate_mse(
                rotated,
                weights_float,
                candidate_scales,
            )
            fused_candidate = fused_evaluator(
                rotated,
                weights_float,
                candidate_scales,
            )
            _sync(device)
            candidate_comparison = _compare_candidate_outputs(
                reference_candidate,
                fused_candidate,
            )
            del reference_candidate, fused_candidate

            torch_candidate = _measure_candidate(
                _torch_candidate_mse,
                rotated,
                weights_float,
                candidate_scales,
                device,
                args.warmup,
                args.iterations,
            )
            fused_candidate_measurement = _measure_candidate(
                fused_evaluator,
                rotated,
                weights_float,
                candidate_scales,
                device,
                args.warmup,
                args.iterations,
            )
            torch_search = _measure_search(
                _torch_candidate_mse,
                rotated,
                weights_float,
                base_scale,
                device,
                args.warmup,
                args.iterations,
            )
            fused_search = _measure_search(
                fused_evaluator,
                rotated,
                weights_float,
                base_scale,
                device,
                args.warmup,
                args.iterations,
            )
            torch_result = _search_result(
                _torch_candidate_mse,
                rotated,
                weights_float,
                base_scale,
                zero_rows,
                all_zero,
            )
            fused_result = _search_result(
                fused_evaluator,
                rotated,
                weights_float,
                base_scale,
                zero_rows,
                all_zero,
            )
            _sync(device)

            baseline_stages, baseline_total = _measure_complete_baseline(
                source,
                device,
                args.warmup,
                args.iterations,
            )
            fused_stages, fused_total = _measure_complete_fused(
                source,
                device,
                fused_evaluator,
                args.warmup,
                args.iterations,
            )
            search_comparison = _compare_search_results(
                torch_result,
                fused_result,
                rotated,
            )
            del device_source, rotated, base_scale, weights_float

        print("One candidate")
        print("----------------------------------------")
        print(f"Torch:  {_format_samples(torch_candidate.samples)}")
        print(f"Fused:  {_format_samples(fused_candidate_measurement.samples)}")
        print(f"Speedup: {_speedup(torch_candidate.samples, fused_candidate_measurement.samples):.2f}x")
        print()

        print("Candidate peak CUDA memory")
        print("----------------------------------------")
        print(
            f"Torch: allocated {_mib(torch_candidate.peak_allocated):.1f} MiB; "
            f"reserved {_mib(torch_candidate.peak_reserved):.1f} MiB"
        )
        print(
            f"Fused: allocated {_mib(fused_candidate_measurement.peak_allocated):.1f} MiB; "
            f"reserved {_mib(fused_candidate_measurement.peak_reserved):.1f} MiB"
        )
        print()

        print("34-candidate search")
        print("----------------------------------------")
        for label, measurement in (("Torch", torch_search), ("Fused", fused_search)):
            print(label)
            print(f"  coarse:   {_format_samples(measurement.coarse)}")
            print(f"  fine:     {_format_samples(measurement.fine)}")
            print(f"  combined: {_format_samples(measurement.combined)}")
        print(
            f"Combined speedup: "
            f"{_speedup(torch_search.combined, fused_search.combined):.2f}x"
        )
        print()

        print("Complete one-tensor W4A4-MSE path")
        print("----------------------------------------")
        print(f"Torch total: {_format_samples(baseline_total)}")
        print(f"Fused total: {_format_samples(fused_total)}")
        print(f"Speedup: {_speedup(baseline_total, fused_total):.2f}x")
        print()
        for stage in baseline_stages:
            print(
                f"{stage:<16} Torch {_format_samples(baseline_stages[stage])}; "
                f"Fused {_format_samples(fused_stages[stage])}"
            )
        print()

        print("Correctness")
        print("----------------------------------------")
        print(f"Candidate max absolute MSE difference: {candidate_comparison['max_abs']:.8e}")
        print(f"Candidate relative L2 difference:       {candidate_comparison['relative_l2']:.8e}")
        print(f"Candidate exact tensor match:             {candidate_comparison['exact_row_percent']:.1f}%")
        print(
            f"Coarse winner match:                      "
            f"{search_comparison['coarse_winner_percent']:.5f}% "
            f"({search_comparison['coarse_winner_matches']:,}/"
            f"{search_comparison['coarse_winner_count']:,})"
        )
        print(
            f"Fine winner match:                        "
            f"{search_comparison['fine_winner_percent']:.5f}% "
            f"({search_comparison['fine_winner_matches']:,}/"
            f"{search_comparison['fine_winner_count']:,})"
        )
        print(
            f"W4 code match:                            "
            f"{search_comparison['code_match_percent']:.8f}% "
            f"({search_comparison['code_matches']:,}/"
            f"{search_comparison['code_count']:,})"
        )
        print(
            f"Packed byte match:                         "
            f"{search_comparison['packed_match_percent']:.8f}% "
            f"({search_comparison['packed_matches']:,}/"
            f"{search_comparison['packed_count']:,})"
        )
        print(f"Maximum W4 code difference:               {search_comparison['max_code_difference']:.0f}")
        print(f"Maximum scale difference:                 {search_comparison['max_scale_difference']:.8e}")
        print(f"Maximum final row MSE difference:         {search_comparison['max_final_mse_difference']:.8e}")
        print()

        combined_speedup = _speedup(torch_search.combined, fused_search.combined)
        print("Decision")
        print("----------------------------------------")
        if combined_speedup >= 2.0:
            print("Promising: the fused evaluator meets the approximately 2x integration threshold.")
        elif combined_speedup >= 1.2:
            print("Marginal: the fused evaluator is faster, but below the approximately 2x threshold.")
        else:
            print("Not promising yet: the fused evaluator is below the approximately 20% speedup threshold.")
        print("Only this one representative tensor was benchmarked; no full-model loop was run.")
        return 0
    finally:
        del temporary_directory


if __name__ == "__main__":
    raise SystemExit(main())
