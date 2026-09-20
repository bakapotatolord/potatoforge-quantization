"""Small, unpacked INT6 rowwise reference for measurement experiments."""

from __future__ import annotations

from typing import Final, NamedTuple

import torch

from .hadamard import (
    CONVROT_GROUP_SIZE,
    apply_hadamard_rotation,
    cached_cuda_hadamard_matrix,
)
from .int6_packing import (
    Int6PackedResult,
    pack_int6_row_major,
    unpack_int6_row_major,
)
from ..timing import timed_internal_stage
from .rowwise import (
    dequantize_rowwise,
    quantize_rows_with_scale,
    quantize_symmetric_rows,
    resolve_convrot_device,
    run_rowwise_mse_coarse_search,
    run_rowwise_mse_fine_search,
    select_rowwise_absmax_scale,
    select_rowwise_mse_scale,
    validate_convrot_input_width,
    validate_rowwise_finite_weights,
    validate_rowwise_weight_metadata,
    validate_rowwise_weights,
)
from .w4a4_mse_native import load_int6_convrot_mse_candidate


INT6_QMAX: Final[int] = 31
INT6_QMIN: Final[int] = -31
INT6_FORMAT_NAME: Final[str] = "INT6"


class Int6RowwiseResult(NamedTuple):
    """Logical INT6 values stored in int8, plus one scale per output row."""

    codes: torch.Tensor
    scales: torch.Tensor


class Int6ConvRotPackedResult(NamedTuple):
    """Packed ConvRot INT6 values and row scales for serialization."""

    packed_codes: torch.Tensor
    scales: torch.Tensor
    original_shape: tuple[int, int]


def quantize_int6_rowwise(weights: torch.Tensor) -> Int6RowwiseResult:
    """Quantize weights to logical signed INT6 codes held in an int8 tensor."""

    validate_rowwise_weights(weights, format_name=INT6_FORMAT_NAME)
    result = quantize_symmetric_rows(
        weights,
        qmin=INT6_QMIN,
        qmax=INT6_QMAX,
        use_float32_math=True,
        zero_scale=1.0,
    )
    return Int6RowwiseResult(result.codes, result.scales)


def select_int6_mse_scale(weights: torch.Tensor) -> torch.Tensor:
    validate_rowwise_weights(weights, format_name=INT6_FORMAT_NAME)
    return select_rowwise_mse_scale(
        weights,
        qmin=INT6_QMIN,
        qmax=INT6_QMAX,
    )


def dequantize_int6_rowwise(result: Int6RowwiseResult) -> torch.Tensor:
    """Reconstruct float32 weights from an INT6 rowwise result."""

    return dequantize_rowwise(
        result.codes,
        result.scales,
        format_name=INT6_FORMAT_NAME,
        qmin=INT6_QMIN,
        qmax=INT6_QMAX,
    )


def quantize_int6_convrot(weights: torch.Tensor) -> Int6RowwiseResult:
    """Rotate then quantize W6 weights with Comfy's fixed ConvRot grouping."""

    rotated_weights = apply_hadamard_rotation(
        weights,
        group_size=CONVROT_GROUP_SIZE,
        output_dtype=torch.float32,
    )
    return quantize_int6_rowwise(rotated_weights)


def quantize_int6_convrot_mse(weights: torch.Tensor) -> Int6RowwiseResult:
    """Rotate then quantize W6 weights with rowwise MSE scale selection."""

    validate_rowwise_weights(weights, format_name=INT6_FORMAT_NAME)
    validate_convrot_input_width(
        weights,
        format_name=INT6_FORMAT_NAME,
        group_size=CONVROT_GROUP_SIZE,
    )
    rotated_weights = apply_hadamard_rotation(
        weights,
        group_size=CONVROT_GROUP_SIZE,
        output_dtype=torch.float32,
    )
    result = quantize_rows_with_scale(
        rotated_weights,
        select_int6_mse_scale(rotated_weights),
        qmin=INT6_QMIN,
        qmax=INT6_QMAX,
    )
    return Int6RowwiseResult(result.codes, result.scales)


def _quantize_int6_convrot_cuda(
    weights: torch.Tensor,
    device: torch.device,
    internal_timings: dict[str, float] | None,
) -> Int6ConvRotPackedResult:
    validate_rowwise_weight_metadata(
        weights,
        format_name=INT6_FORMAT_NAME,
        internal_timings=internal_timings,
    )
    with timed_internal_stage(
        internal_timings,
        "prepare",
        device,
    ):
        device_weights = weights.to(device=device)

    with timed_internal_stage(
        internal_timings,
        "validate_finite",
        device,
    ):
        validate_rowwise_finite_weights(
            device_weights,
            format_name=INT6_FORMAT_NAME,
        )

    with timed_internal_stage(
        internal_timings,
        "rotation",
        device,
    ):
        rotated_weights = apply_hadamard_rotation(
            device_weights,
            group_size=CONVROT_GROUP_SIZE,
            output_dtype=torch.float32,
            hadamard=cached_cuda_hadamard_matrix(
                CONVROT_GROUP_SIZE,
                device,
                torch.float32,
            ),
        )
    rowwise_result = quantize_symmetric_rows(
        rotated_weights,
        qmin=INT6_QMIN,
        qmax=INT6_QMAX,
        use_float32_math=True,
        zero_scale=1.0,
        internal_timings=internal_timings,
        _timing_device=device if internal_timings is not None else None,
    )
    with timed_internal_stage(
        internal_timings,
        "pack",
        device,
    ):
        packed_result = pack_int6_row_major(rowwise_result.codes)

    with timed_internal_stage(
        internal_timings,
        "finalize",
        device,
    ):
        return Int6ConvRotPackedResult(
            packed_codes=packed_result.packed_codes.cpu(),
            scales=rowwise_result.scales.cpu(),
            original_shape=packed_result.original_shape,
        )


def quantize_int6_convrot_packed(
    weights: torch.Tensor,
    *,
    device: str | torch.device = "cpu",
    internal_timings: dict[str, float] | None = None,
) -> Int6ConvRotPackedResult:
    """Quantize ConvRot INT6 and return only its serialization payload."""

    with timed_internal_stage(
        internal_timings,
        "resolve_device",
        None,
    ):
        target_device = resolve_convrot_device(
            device,
            format_name=INT6_FORMAT_NAME,
        )
    if target_device.type == "cuda":
        return _quantize_int6_convrot_cuda(
            weights,
            target_device,
            internal_timings,
        )

    rowwise_result = quantize_int6_convrot(weights)
    packed_result = pack_int6_row_major(rowwise_result.codes)
    return Int6ConvRotPackedResult(
        packed_codes=packed_result.packed_codes,
        scales=rowwise_result.scales,
        original_shape=packed_result.original_shape,
    )


def quantize_int6_convrot_mse_packed(
    weights: torch.Tensor,
    *,
    device: str | torch.device = "cpu",
    internal_timings: dict[str, float] | None = None,
) -> Int6ConvRotPackedResult:
    """Quantize ConvRot INT6 with MSE scales for serialization."""

    with timed_internal_stage(
        internal_timings,
        "resolve_device",
        None,
    ):
        target_device = resolve_convrot_device(
            device,
            format_name=INT6_FORMAT_NAME,
        )
    if target_device.type == "cuda":
        return _quantize_int6_convrot_mse_cuda(
            weights,
            target_device,
            internal_timings,
        )

    rowwise_result = quantize_int6_convrot_mse(weights)
    packed_result = pack_int6_row_major(rowwise_result.codes)
    return Int6ConvRotPackedResult(
        packed_codes=packed_result.packed_codes,
        scales=rowwise_result.scales,
        original_shape=packed_result.original_shape,
    )


def _quantize_int6_convrot_mse_cuda(
    weights: torch.Tensor,
    device: torch.device,
    internal_timings: dict[str, float] | None,
) -> Int6ConvRotPackedResult:
    validate_rowwise_weight_metadata(
        weights,
        format_name=INT6_FORMAT_NAME,
    )
    with timed_internal_stage(
        internal_timings,
        "prepare",
        device,
    ):
        device_weights = weights.to(device=device)
    validate_rowwise_finite_weights(
        device_weights,
        format_name=INT6_FORMAT_NAME,
    )
    with timed_internal_stage(
        internal_timings,
        "rotation",
        device,
    ):
        rotated_weights = apply_hadamard_rotation(
            device_weights,
            group_size=CONVROT_GROUP_SIZE,
            output_dtype=torch.float32,
            hadamard=cached_cuda_hadamard_matrix(
                CONVROT_GROUP_SIZE,
                device,
                torch.float32,
            ),
        )
    with timed_internal_stage(
        internal_timings,
        "scale_init",
        device,
    ):
        base_scale = select_rowwise_absmax_scale(
            rotated_weights,
            qmax=INT6_QMAX,
        )
    with timed_internal_stage(
        internal_timings,
        "zero_row_check",
        device,
    ):
        weights_float = rotated_weights.float()
        zero_rows = weights_float.abs().amax(dim=1, keepdim=True) == 0
        all_zero = bool(zero_rows.all())

    best_mse = base_scale
    best_multiplier = torch.ones_like(base_scale)
    if not all_zero:
        candidate_mse = load_int6_convrot_mse_candidate(
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        )
        with timed_internal_stage(
            internal_timings,
            "coarse_search",
            device,
        ):
            best_mse, best_multiplier = run_rowwise_mse_coarse_search(
                rotated_weights,
                weights_float,
                base_scale,
                qmin=INT6_QMIN,
                qmax=INT6_QMAX,
                candidate_mse=candidate_mse,
            )
        with timed_internal_stage(
            internal_timings,
            "fine_search",
            device,
        ):
            best_mse, best_multiplier = run_rowwise_mse_fine_search(
                rotated_weights,
                weights_float,
                base_scale,
                best_mse,
                best_multiplier,
                qmin=INT6_QMIN,
                qmax=INT6_QMAX,
                candidate_mse=candidate_mse,
            )

    with timed_internal_stage(
        internal_timings,
        "final_quantize",
        device,
    ):
        scales = torch.where(
            zero_rows,
            base_scale,
            base_scale * best_multiplier,
        ).clamp_min(1e-30)
        rowwise_result = quantize_rows_with_scale(
            rotated_weights,
            scales,
            qmin=INT6_QMIN,
            qmax=INT6_QMAX,
        )
    with timed_internal_stage(
        internal_timings,
        "pack",
        device,
    ):
        packed_result = pack_int6_row_major(rowwise_result.codes)
    with timed_internal_stage(
        internal_timings,
        "finalize",
        device,
    ):
        return Int6ConvRotPackedResult(
            packed_codes=packed_result.packed_codes.cpu(),
            scales=rowwise_result.scales.cpu(),
            original_shape=packed_result.original_shape,
        )


def dequantize_int6_convrot(result: Int6RowwiseResult) -> torch.Tensor:
    """Reconstruct ConvRot W6 weights in their original Linear space."""

    return apply_hadamard_rotation(
        dequantize_int6_rowwise(result),
        group_size=CONVROT_GROUP_SIZE,
        output_dtype=torch.float32,
    )


def dequantize_int6_convrot_packed(
    result: Int6ConvRotPackedResult,
    *,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Reconstruct ConvRot W6 weights from a packed serialization result."""

    packed_codes = result.packed_codes
    scales = result.scales
    if device is not None:
        packed_codes = packed_codes.to(device=device)
        scales = scales.to(device=device)
    codes = unpack_int6_row_major(
        Int6PackedResult(packed_codes, result.original_shape),
    )
    return dequantize_int6_convrot(Int6RowwiseResult(codes, scales))
