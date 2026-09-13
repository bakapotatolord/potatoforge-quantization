"""Benchmark the existing ConvRot operation on CPU and CUDA.

This is an experimental benchmark. It does not participate in the normal
quantization command or move production tensors to CUDA.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from potatoforge.quantization.hadamard import (
    CONVROT_GROUP_SIZE,
    apply_hadamard_rotation,
    build_normalized_hadamard_matrix,
)
from potatoforge.quantization.int8_tensorwise import (
    quantize_int8_rows,
)


SHAPES = ((16384, 6144), (6144, 16384))
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class Samples:
    values: list[float]

    @property
    def median(self) -> float:
        return statistics.median(self.values)

    @property
    def mean(self) -> float:
        return statistics.mean(self.values)

    @property
    def minimum(self) -> float:
        return min(self.values)

    @property
    def maximum(self) -> float:
        return max(self.values)


def _samples(values: list[float]) -> Samples:
    if not values:
        raise ValueError("Benchmark produced no samples.")
    return Samples(values)


def _output_dtype(source_dtype: torch.dtype) -> torch.dtype:
    if source_dtype in (torch.float16, torch.float32):
        return torch.float32
    return torch.bfloat16


def _cached_rotation(
    weights: torch.Tensor,
    hadamard: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """The existing apply_hadamard_rotation body with a supplied matrix."""

    rotation_weights = weights.to(dtype=output_dtype)
    output_features, input_features = rotation_weights.shape
    grouped_weights = rotation_weights.reshape(
        output_features,
        input_features // CONVROT_GROUP_SIZE,
        CONVROT_GROUP_SIZE,
    )
    rotated_groups = grouped_weights @ hadamard.T
    return rotated_groups.reshape(output_features, input_features)


def _source_tensor(
    shape: tuple[int, int],
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    source = torch.randn(shape, generator=generator, dtype=torch.float32)
    return source if dtype == torch.float32 else source.to(dtype)


def _measure_cpu_rotation(
    source: torch.Tensor,
    output_dtype: torch.dtype,
    warmup: int,
    iterations: int,
) -> Samples:
    for _ in range(warmup):
        rotated = apply_hadamard_rotation(
            source,
            group_size=CONVROT_GROUP_SIZE,
            output_dtype=output_dtype,
        )
        del rotated

    values: list[float] = []
    for _ in range(iterations):
        started_at = time.perf_counter()
        rotated = apply_hadamard_rotation(
            source,
            group_size=CONVROT_GROUP_SIZE,
            output_dtype=output_dtype,
        )
        values.append(time.perf_counter() - started_at)
        del rotated
    return _samples(values)


def _measure_cpu_cached_rotation(
    source: torch.Tensor,
    hadamard: torch.Tensor,
    output_dtype: torch.dtype,
    warmup: int,
    iterations: int,
) -> Samples:
    for _ in range(warmup):
        rotated = _cached_rotation(source, hadamard, output_dtype)
        del rotated

    values: list[float] = []
    for _ in range(iterations):
        started_at = time.perf_counter()
        rotated = _cached_rotation(source, hadamard, output_dtype)
        values.append(time.perf_counter() - started_at)
        del rotated
    return _samples(values)


def _measure_cpu_full_int8(
    source: torch.Tensor,
    output_dtype: torch.dtype,
    warmup: int,
    iterations: int,
) -> dict[str, Samples]:
    def run_once() -> dict[str, float]:
        started_at = time.perf_counter()
        rotated = apply_hadamard_rotation(
            source,
            group_size=CONVROT_GROUP_SIZE,
            output_dtype=output_dtype,
        )
        rotation = time.perf_counter() - started_at

        internal_timings: dict[str, float] = {}
        quantize_int8_rows(
            rotated,
            internal_timings=internal_timings,
        )
        total = time.perf_counter() - started_at
        del rotated
        return {
            "rotation": rotation,
            "scale": internal_timings.get("scale", 0.0),
            "quantize_values": internal_timings.get(
                "quantize_values",
                0.0,
            ),
            "finalize": internal_timings.get("finalize", 0.0),
            "total": total,
        }

    for _ in range(warmup):
        run_once()

    measurements = {stage: [] for stage in (
        "rotation",
        "scale",
        "quantize_values",
        "finalize",
        "total",
    )}
    for _ in range(iterations):
        result = run_once()
        for stage, value in result.items():
            measurements[stage].append(value)
    return {stage: _samples(values) for stage, values in measurements.items()}


def _cuda_timed(
    device: torch.device,
    action: Callable[[], torch.Tensor],
) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    result = action()
    torch.cuda.synchronize(device)
    return result, time.perf_counter() - started_at


def _measure_cuda_rotation(
    source: torch.Tensor,
    hadamard: torch.Tensor,
    output_dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, Samples], int, int]:
    def run_once() -> dict[str, float]:
        gpu_source, h2d = _cuda_timed(
            device,
            lambda: source.to(device=device),
        )
        rotated, rotation = _cuda_timed(
            device,
            lambda: _cached_rotation(
                gpu_source,
                hadamard,
                output_dtype,
            ),
        )
        _, d2h = _cuda_timed(device, lambda: rotated.cpu())
        del gpu_source, rotated
        return {
            "h2d": h2d,
            "rotation": rotation,
            "d2h": d2h,
            "total": h2d + rotation + d2h,
        }

    for _ in range(warmup):
        run_once()

    torch.cuda.reset_peak_memory_stats(device)
    measurements = {stage: [] for stage in ("h2d", "rotation", "d2h", "total")}
    for _ in range(iterations):
        result = run_once()
        for stage, value in result.items():
            measurements[stage].append(value)

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    return (
        {stage: _samples(values) for stage, values in measurements.items()},
        peak_allocated,
        peak_reserved,
    )


def _cuda_quantize_rows(
    rotated: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Mirror the existing non-float32 quantize_int8_rows operations.

    ConvRot emits BF16 for BF16 input and FP32 for FP16/FP32 input, so the
    production rowwise path does not take its float16 promotion branch here.
    CUDA synchronization is deliberately local to this benchmark.
    """

    torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    scales = (
        rotated.abs()
        .amax(dim=1, keepdim=True)
        .float()
        .div(127)
        .clamp_min(1e-30)
    )
    torch.cuda.synchronize(device)
    scale_seconds = time.perf_counter() - started_at

    torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    scale_math = scales.to(rotated.dtype)
    tiny = torch.finfo(rotated.dtype).tiny
    scale_math = torch.where(
        scale_math == 0,
        torch.full_like(scale_math, tiny),
        scale_math,
    )
    codes = torch.round(rotated / scale_math).clamp(-128, 127)
    torch.cuda.synchronize(device)
    quantize_seconds = time.perf_counter() - started_at

    torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    codes_int8 = codes.to(torch.int8)
    scales_float32 = scales.float()
    torch.cuda.synchronize(device)
    finalize_seconds = time.perf_counter() - started_at

    return codes_int8, scales_float32, {
        "scale": scale_seconds,
        "quantize_values": quantize_seconds,
        "finalize": finalize_seconds,
    }


def _measure_cuda_full_int8(
    source: torch.Tensor,
    hadamard: torch.Tensor,
    output_dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, Samples], int, int]:
    def run_once() -> dict[str, float]:
        gpu_source, h2d = _cuda_timed(
            device,
            lambda: source.to(device=device),
        )
        rotated, rotation = _cuda_timed(
            device,
            lambda: _cached_rotation(
                gpu_source,
                hadamard,
                output_dtype,
            ),
        )
        codes, scales, quantize_timings = _cuda_quantize_rows(
            rotated,
            device,
        )
        torch.cuda.synchronize(device)
        started_at = time.perf_counter()
        codes.cpu()
        scales.cpu()
        torch.cuda.synchronize(device)
        d2h = time.perf_counter() - started_at
        total = h2d + rotation + sum(quantize_timings.values()) + d2h
        del gpu_source, rotated, codes, scales
        return {
            "h2d": h2d,
            "rotation": rotation,
            "scale": quantize_timings["scale"],
            "quantize_values": quantize_timings["quantize_values"],
            "finalize": quantize_timings["finalize"],
            "d2h": d2h,
            "total": total,
        }

    for _ in range(warmup):
        run_once()

    torch.cuda.reset_peak_memory_stats(device)
    stage_names = (
        "h2d",
        "rotation",
        "scale",
        "quantize_values",
        "finalize",
        "d2h",
        "total",
    )
    measurements = {stage: [] for stage in stage_names}
    for _ in range(iterations):
        result = run_once()
        for stage, value in result.items():
            measurements[stage].append(value)

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    return (
        {stage: _samples(values) for stage, values in measurements.items()},
        peak_allocated,
        peak_reserved,
    )


def _validate_outputs(
    cpu_rotated: torch.Tensor,
    cpu_codes: torch.Tensor,
    cpu_scales: torch.Tensor,
    gpu_rotated: torch.Tensor,
    gpu_codes: torch.Tensor,
    gpu_scales: torch.Tensor,
) -> dict[str, float]:
    difference = gpu_rotated.float() - cpu_rotated.float()
    denominator = cpu_rotated.float().norm().item()
    relative_l2 = difference.norm().item() / denominator if denominator else 0.0
    code_difference = (gpu_codes.to(torch.int16) - cpu_codes.to(torch.int16)).abs()
    scale_difference = (gpu_scales.float() - cpu_scales.float()).abs()
    code_match_count = int((gpu_codes == cpu_codes).sum().item())
    code_count = gpu_codes.numel()
    return {
        "relative_l2": relative_l2,
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "code_match_percent": code_match_count / code_count * 100.0,
        "code_match_count": code_match_count,
        "code_count": code_count,
        "max_scale_difference": scale_difference.max().item(),
        "max_code_difference": code_difference.max().item(),
    }


def _format_stats(stats: Samples) -> str:
    return (
        f"median {stats.median:.4f} s | mean {stats.mean:.4f} s | "
        f"min {stats.minimum:.4f} s | max {stats.maximum:.4f} s"
    )


def _print_stage_stats(
    stages: dict[str, Samples],
    names: tuple[str, ...],
) -> None:
    for stage in names:
        print(f"{stage + ':':<20} {_format_stats(stages[stage])}")


def _mib(byte_count: int) -> float:
    return byte_count / 1024**2


def _run_shape(
    shape: tuple[int, int],
    source_dtype: torch.dtype,
    seed: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> None:
    output_dtype = _output_dtype(source_dtype)
    print(f"Shape: {shape[0]} x {shape[1]}")
    print(f"Source dtype: {source_dtype}")
    print(f"Internal rotation dtype: {output_dtype}")

    source = _source_tensor(shape, source_dtype, seed)
    print(f"Source size: {_mib(source.numel() * source.element_size()):.1f} MiB")

    setup_started = time.perf_counter()
    hadamard_base = build_normalized_hadamard_matrix(CONVROT_GROUP_SIZE)
    cpu_hadamard = hadamard_base.to(dtype=output_dtype)
    gpu_hadamard = cpu_hadamard.to(device=device)
    torch.cuda.synchronize(device)
    setup_seconds = time.perf_counter() - setup_started
    print(f"Hadamard setup (one-time): {setup_seconds:.4f} s")
    print()

    with torch.no_grad():
        cpu_rotation = _measure_cpu_rotation(
            source,
            output_dtype,
            warmup,
            iterations,
        )
        cpu_cached_rotation = _measure_cpu_cached_rotation(
            source,
            cpu_hadamard,
            output_dtype,
            warmup,
            iterations,
        )
        cpu_full = _measure_cpu_full_int8(
            source,
            output_dtype,
            warmup,
            iterations,
        )

        cpu_rotated = apply_hadamard_rotation(
            source,
            group_size=CONVROT_GROUP_SIZE,
            output_dtype=output_dtype,
        )
        cpu_result = quantize_int8_rows(cpu_rotated)

        gpu_rotated, _ = _cuda_timed(
            device,
            lambda: _cached_rotation(
                source.to(device=device),
                gpu_hadamard,
                output_dtype,
            ),
        )
        gpu_rotated_cpu = gpu_rotated.cpu()
        torch.cuda.synchronize(device)
        gpu_codes, gpu_scales, _ = _cuda_quantize_rows(
            gpu_rotated,
            device,
        )
        gpu_codes_cpu = gpu_codes.cpu()
        gpu_scales_cpu = gpu_scales.cpu()
        torch.cuda.synchronize(device)
        validation = _validate_outputs(
            cpu_rotated,
            cpu_result.codes,
            cpu_result.scales,
            gpu_rotated_cpu,
            gpu_codes_cpu,
            gpu_scales_cpu,
        )
        del (
            gpu_rotated,
            gpu_rotated_cpu,
            gpu_codes,
            gpu_scales,
            gpu_codes_cpu,
            gpu_scales_cpu,
            cpu_rotated,
            cpu_result,
        )

        cuda_rotation, rotation_peak, rotation_reserved = _measure_cuda_rotation(
            source,
            gpu_hadamard,
            output_dtype,
            device,
            warmup,
            iterations,
        )
        cuda_full, full_peak, full_reserved = _measure_cuda_full_int8(
            source,
            gpu_hadamard,
            output_dtype,
            device,
            warmup,
            iterations,
        )

    print("CPU")
    print("--------------------------------")
    print(f"rotation (existing): {_format_stats(cpu_rotation)}")
    print(f"rotation (cached):   {_format_stats(cpu_cached_rotation)}")
    _print_stage_stats(
        cpu_full,
        ("rotation", "scale", "quantize_values", "finalize", "total"),
    )
    print()

    print("CUDA rotation-only (cached Hadamard)")
    print("--------------------------------")
    _print_stage_stats(
        cuda_rotation,
        ("h2d", "rotation", "d2h", "total"),
    )
    print(
        f"peak CUDA allocated: {_mib(rotation_peak):.1f} MiB; "
        f"reserved: {_mib(rotation_reserved):.1f} MiB"
    )
    print(
        f"rotation speedup vs existing CPU: "
        f"{cpu_rotation.median / cuda_rotation['rotation'].median:.2f}x"
    )
    print(
        f"transfer-inclusive speedup: "
        f"{cpu_rotation.median / cuda_rotation['total'].median:.2f}x"
    )
    print()

    print("CUDA full INT8 path (cached Hadamard)")
    print("--------------------------------")
    _print_stage_stats(
        cuda_full,
        (
            "h2d",
            "rotation",
            "scale",
            "quantize_values",
            "finalize",
            "d2h",
            "total",
        ),
    )
    print(
        f"peak CUDA allocated: {_mib(full_peak):.1f} MiB; "
        f"reserved: {_mib(full_reserved):.1f} MiB"
    )
    print(
        f"full-path speedup vs current CPU total: "
        f"{cpu_full['total'].median / cuda_full['total'].median:.2f}x"
    )
    print()

    print("Numerical comparison: cached CUDA vs existing CPU")
    print("--------------------------------")
    print(f"relative L2:              {validation['relative_l2']:.8e}")
    print(f"max absolute difference:   {validation['max_abs']:.8e}")
    print(f"mean absolute difference:  {validation['mean_abs']:.8e}")
    print(
        f"exact INT8 code match:     "
        f"{validation['code_match_percent']:.9f}% "
        f"({validation['code_match_count']:,}/"
        f"{validation['code_count']:,})"
    )
    print(f"maximum scale difference:  {validation['max_scale_difference']:.8e}")
    print(f"maximum code difference:   {validation['max_code_difference']:.0f}")
    print()

    del source, cpu_hadamard, gpu_hadamard, hadamard_base
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark existing ConvRot CPU and CUDA paths."
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPES),
        default="bfloat16",
        help="Source dtype; ConvRot chooses its internal output dtype from it.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations < 1:
        parser.error("--warmup must be non-negative and --iterations >= 1")

    print("ConvRot CUDA benchmark")
    print("======================")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    print(f"Warmup: {args.warmup}; measured iterations: {args.iterations}")
    print()

    if not torch.cuda.is_available():
        print("CUDA unavailable; benchmark not run.")
        return 2

    device = torch.device("cuda:0")
    source_dtype = DTYPES[args.dtype]
    for index, shape in enumerate(SHAPES):
        _run_shape(
            shape,
            source_dtype,
            args.seed + index,
            args.warmup,
            args.iterations,
            device,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
