"""Shared rowwise and ConvRot quantization for the INT5, INT6 and INT7 formats.

Each format module defines one ``RowwiseFormat`` and re-exports its methods
under the historical ``quantize_intN_*`` names.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch

from ..timing import timed_internal_stage
from .hadamard import (
    CONVROT_GROUP_SIZE,
    apply_hadamard_rotation,
    cached_cuda_hadamard_matrix,
)
from .mse_search_native import (
    ROWWISE_MSE_SCALE_FLOOR,
    fused_rowwise_mse_coarse_search,
    fused_rowwise_mse_fine_search,
)
from .rowwise import (
    dequantize_rowwise,
    quantize_rows_with_scale,
    quantize_symmetric_rows,
    resolve_convrot_device,
    run_rowwise_mse_coarse_search,
    run_rowwise_mse_fine_search,
    select_rowwise_mse_scale,
    validate_convrot_input_width,
    validate_rowwise_finite_weights,
    validate_rowwise_weight_metadata,
    validate_rowwise_weights,
)


class RowwiseResult(NamedTuple):
    """Logical signed codes stored in int8, plus one scale per output row."""

    codes: torch.Tensor
    scales: torch.Tensor


class ConvRotPackedResult(NamedTuple):
    """Packed ConvRot codes and row scales for serialization."""

    packed_codes: torch.Tensor
    scales: torch.Tensor
    original_shape: tuple[int, int]


MseCandidate = Callable[[torch.Tensor, torch.device], Any]


def native_mse_candidate(load: Callable[[int], Any]) -> MseCandidate:
    """Adapt a per-device native kernel loader to the candidate interface."""

    def candidate(rotated: torch.Tensor, device: torch.device) -> Any:
        index = device.index if device.index is not None else torch.cuda.current_device()
        return load(index)

    return candidate


@dataclass(frozen=True)
class RowwiseFormat:
    name: str
    qmin: int
    qmax: int
    # Lowest code the packed layout can hold; INT5 stores -16 as well.
    storage_qmin: int
    pack: Callable[[torch.Tensor], Any]
    unpack: Callable[[torch.Tensor, tuple[int, int]], torch.Tensor]
    mse_candidate: MseCandidate
    # Try the fused CUDA scale search before the per-candidate kernel.
    fused_mse_search: bool = False

    # Rowwise.

    def quantize_rowwise(self, weights: torch.Tensor) -> RowwiseResult:
        validate_rowwise_weights(weights, format_name=self.name)
        result = quantize_symmetric_rows(
            weights,
            qmin=self.qmin,
            qmax=self.qmax,
            use_float32_math=True,
            zero_scale=1.0,
        )
        return RowwiseResult(result.codes, result.scales)

    def select_mse_scale(self, weights: torch.Tensor) -> torch.Tensor:
        validate_rowwise_weights(weights, format_name=self.name)
        return select_rowwise_mse_scale(weights, qmin=self.qmin, qmax=self.qmax)

    def dequantize_rowwise(self, result: RowwiseResult) -> torch.Tensor:
        return dequantize_rowwise(
            result.codes,
            result.scales,
            format_name=self.name,
            qmin=self.storage_qmin,
            qmax=self.qmax,
        )

    # ConvRot on CPU.

    def _rotate_validated(self, weights: torch.Tensor) -> torch.Tensor:
        validate_rowwise_weights(weights, format_name=self.name)
        validate_convrot_input_width(
            weights,
            format_name=self.name,
            group_size=CONVROT_GROUP_SIZE,
        )
        return apply_hadamard_rotation(
            weights,
            group_size=CONVROT_GROUP_SIZE,
            output_dtype=torch.float32,
        )

    def quantize_convrot(self, weights: torch.Tensor) -> RowwiseResult:
        """Rotate with Comfy's fixed ConvRot grouping, then quantize rowwise."""

        return self.quantize_rowwise(self._rotate_validated(weights))

    def quantize_convrot_mse(self, weights: torch.Tensor) -> RowwiseResult:
        """Rotate, then quantize with rowwise MSE scale selection."""

        rotated = self._rotate_validated(weights)
        result = quantize_rows_with_scale(
            rotated,
            self.select_mse_scale(rotated),
            qmin=self.qmin,
            qmax=self.qmax,
        )
        return RowwiseResult(result.codes, result.scales)

    def dequantize_convrot(self, result: RowwiseResult) -> torch.Tensor:
        """Reconstruct ConvRot weights in their original Linear space."""

        return apply_hadamard_rotation(
            self.dequantize_rowwise(result),
            group_size=CONVROT_GROUP_SIZE,
            output_dtype=torch.float32,
        )

    def dequantize_convrot_packed(
        self,
        result: ConvRotPackedResult,
        *,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Reconstruct ConvRot weights from a packed serialization result."""

        packed_codes, scales = result.packed_codes, result.scales
        if device is not None:
            packed_codes = packed_codes.to(device=device)
            scales = scales.to(device=device)
        codes = self.unpack(packed_codes, result.original_shape)
        return self.dequantize_convrot(RowwiseResult(codes, scales))

    # Packed serialization entry points.

    def quantize_convrot_packed(
        self,
        weights: torch.Tensor,
        *,
        device: str | torch.device = "cpu",
        internal_timings: dict[str, float] | None = None,
    ) -> ConvRotPackedResult:
        """Quantize ConvRot and return only its serialization payload."""

        return self._packed(weights, device, internal_timings, mse=False)

    def quantize_convrot_mse_packed(
        self,
        weights: torch.Tensor,
        *,
        device: str | torch.device = "cpu",
        internal_timings: dict[str, float] | None = None,
    ) -> ConvRotPackedResult:
        """Quantize ConvRot with MSE scales for serialization."""

        return self._packed(weights, device, internal_timings, mse=True)

    def _packed(
        self,
        weights: torch.Tensor,
        device: str | torch.device,
        internal_timings: dict[str, float] | None,
        *,
        mse: bool,
    ) -> ConvRotPackedResult:
        with timed_internal_stage(internal_timings, "resolve_device", None):
            target_device = resolve_convrot_device(device, format_name=self.name)
        if target_device.type == "cuda":
            quantize_cuda = self._convrot_mse_cuda if mse else self._convrot_cuda
            return quantize_cuda(weights, target_device, internal_timings)
        result = (self.quantize_convrot_mse if mse else self.quantize_convrot)(weights)
        packed = self.pack(result.codes)
        return ConvRotPackedResult(packed.packed_codes, result.scales, packed.original_shape)

    # ConvRot on CUDA.

    def _rotate_on_device(
        self,
        weights: torch.Tensor,
        device: torch.device,
        internal_timings: dict[str, float] | None,
        *,
        time_validation: bool,
    ) -> torch.Tensor:
        validate_rowwise_weight_metadata(
            weights,
            format_name=self.name,
            internal_timings=internal_timings if time_validation else None,
        )
        validate_convrot_input_width(
            weights,
            format_name=self.name,
            group_size=CONVROT_GROUP_SIZE,
        )
        with timed_internal_stage(internal_timings, "prepare", device):
            device_weights = weights.to(device=device)
        with timed_internal_stage(
            internal_timings if time_validation else None, "validate_finite", device
        ):
            validate_rowwise_finite_weights(device_weights, format_name=self.name)
        with timed_internal_stage(internal_timings, "rotation", device):
            return apply_hadamard_rotation(
                device_weights,
                group_size=CONVROT_GROUP_SIZE,
                output_dtype=torch.float32,
                hadamard=cached_cuda_hadamard_matrix(
                    CONVROT_GROUP_SIZE,
                    device,
                    torch.float32,
                ),
            )

    def _finish_cuda(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        device: torch.device,
        internal_timings: dict[str, float] | None,
    ) -> ConvRotPackedResult:
        with timed_internal_stage(internal_timings, "pack", device):
            packed = self.pack(codes)
        with timed_internal_stage(internal_timings, "finalize", device):
            return ConvRotPackedResult(
                packed.packed_codes.cpu(),
                scales.cpu(),
                packed.original_shape,
            )

    def _convrot_cuda(
        self,
        weights: torch.Tensor,
        device: torch.device,
        internal_timings: dict[str, float] | None,
    ) -> ConvRotPackedResult:
        rotated = self._rotate_on_device(
            weights, device, internal_timings, time_validation=True
        )
        result = quantize_symmetric_rows(
            rotated,
            qmin=self.qmin,
            qmax=self.qmax,
            use_float32_math=True,
            zero_scale=1.0,
            internal_timings=internal_timings,
            _timing_device=device if internal_timings is not None else None,
        )
        return self._finish_cuda(result.codes, result.scales, device, internal_timings)

    def _convrot_mse_cuda(
        self,
        weights: torch.Tensor,
        device: torch.device,
        internal_timings: dict[str, float] | None,
    ) -> ConvRotPackedResult:
        rotated = self._rotate_on_device(
            weights, device, internal_timings, time_validation=False
        )
        weights_float = rotated.float()
        with timed_internal_stage(internal_timings, "scale_init", device):
            # rowwise.select_rowwise_absmax_scale, keeping max_abs for the
            # zero-row check instead of scanning the weights a second time.
            max_abs = weights_float.abs().amax(dim=1, keepdim=True)
            zero_rows = max_abs == 0
            base_scale = torch.where(
                zero_rows,
                torch.ones_like(max_abs),
                (max_abs / self.qmax).clamp_min(1e-30),
            )
        with timed_internal_stage(internal_timings, "zero_row_check", device):
            all_zero = bool(zero_rows.all())

        best_multiplier = torch.ones_like(base_scale)
        if not all_zero:
            best_multiplier = self._mse_multiplier(
                rotated, weights_float, base_scale, device, internal_timings
            )

        with timed_internal_stage(internal_timings, "final_quantize", device):
            scales = torch.where(
                zero_rows,
                base_scale,
                base_scale * best_multiplier,
            ).clamp_min(1e-30)
            result = quantize_rows_with_scale(
                rotated,
                scales,
                qmin=self.qmin,
                qmax=self.qmax,
            )
        return self._finish_cuda(result.codes, result.scales, device, internal_timings)

    def _mse_multiplier(
        self,
        rotated: torch.Tensor,
        weights_float: torch.Tensor,
        base_scale: torch.Tensor,
        device: torch.device,
        internal_timings: dict[str, float] | None,
    ) -> torch.Tensor:
        bounds = {"qmin": self.qmin, "qmax": self.qmax}
        if self.fused_mse_search:
            with timed_internal_stage(internal_timings, "coarse_search", device):
                coarse = fused_rowwise_mse_coarse_search(
                    rotated, base_scale, **bounds, scale_floor=ROWWISE_MSE_SCALE_FLOOR
                )
            if coarse is not None:
                with timed_internal_stage(internal_timings, "fine_search", device):
                    _, multiplier = fused_rowwise_mse_fine_search(
                        rotated,
                        base_scale,
                        *coarse,
                        **bounds,
                        scale_floor=ROWWISE_MSE_SCALE_FLOOR,
                    )
                return multiplier

        candidate_mse = self.mse_candidate(rotated, device)
        with timed_internal_stage(internal_timings, "coarse_search", device):
            best_mse, multiplier = run_rowwise_mse_coarse_search(
                rotated, weights_float, base_scale, **bounds, candidate_mse=candidate_mse
            )
        with timed_internal_stage(internal_timings, "fine_search", device):
            _, multiplier = run_rowwise_mse_fine_search(
                rotated,
                weights_float,
                base_scale,
                best_mse,
                multiplier,
                **bounds,
                candidate_mse=candidate_mse,
            )
        return multiplier
