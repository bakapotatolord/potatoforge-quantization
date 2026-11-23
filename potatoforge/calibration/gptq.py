"""Read versioned model-wide GPTQ Gram calibration artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


FORMAT = "potatoforge_gptq_calibration"
VERSION = 1


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"GPTQ calibration {label} must be a positive integer.")
    return value


@dataclass(frozen=True)
class GramSource:
    source_id: str
    tensor_key: str
    in_features: int
    row_count: int
    invocation_count: int
    block_size: int
    num_blocks: int
    last_block_size: int
    members: tuple[str, ...]
    member_shapes: tuple[tuple[int, int], ...]


class GPTQCalibration:
    def __init__(self, metadata_path: str | Path) -> None:
        self.metadata_path = Path(metadata_path)
        self.tensors_path = self.metadata_path.with_suffix(".safetensors")
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read GPTQ calibration JSON '{self.metadata_path}': {error}") from error
        if not isinstance(metadata, dict):
            raise ValueError(f"GPTQ calibration JSON '{self.metadata_path}' must contain an object.")
        if metadata.get("format") != FORMAT or metadata.get("version") != VERSION:
            raise ValueError(f"GPTQ calibration '{self.metadata_path}' format/version must be potatoforge_gptq_calibration v1.")
        if metadata.get("finalized") is not True:
            raise ValueError(f"GPTQ calibration '{self.metadata_path}' is not finalized.")
        if metadata.get("activation_basis") != "logical_linear_input":
            raise ValueError("GPTQ calibration activation_basis must be logical_linear_input.")
        block_size = metadata.get("gram_block_size")
        if metadata.get("gram_mode") != "block_diag" or type(block_size) is not int or block_size < 256 or block_size % 256:
            raise ValueError("GPTQ calibration requires block_diag Gram blocks with a block size divisible by 256.")
        prompt_count = _positive_int(metadata.get("completed_prompt_generations"), "completed_prompt_generations")
        evaluation_count = _positive_int(metadata.get("evaluation_count"), "evaluation_count")
        prompt_evaluations = metadata.get("prompt_evaluation_counts")
        if (
            not isinstance(prompt_evaluations, list)
            or len(prompt_evaluations) != prompt_count
            or any(type(count) is not int or count < 1 for count in prompt_evaluations)
            or sum(prompt_evaluations) != evaluation_count
        ):
            raise ValueError("GPTQ calibration prompt evaluation counts do not match its totals.")
        if not isinstance(metadata.get("session_id"), str) or not metadata["session_id"]:
            raise ValueError("GPTQ calibration needs a session_id.")
        if not isinstance(metadata.get("model_class"), str) or not metadata["model_class"]:
            raise ValueError("GPTQ calibration needs a model_class.")
        entries = metadata.get("sources")
        if not isinstance(entries, list) or not entries:
            raise ValueError("GPTQ calibration needs input sources.")

        self.metadata = metadata
        self.sources: dict[str, GramSource] = {}
        self.weight_to_source: dict[str, GramSource] = {}
        self._factor_cache: dict[str, tuple[torch.Tensor, ...]] = {}
        self._factor_device: torch.device | None = None
        tensor_keys: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("GPTQ calibration source entries must be objects.")
            source_id = entry.get("source_id")
            tensor_key = entry.get("tensor_key")
            if not isinstance(source_id, str) or not source_id or source_id in self.sources:
                raise ValueError(f"GPTQ calibration has invalid or duplicate source_id {source_id!r}.")
            if not isinstance(tensor_key, str) or not tensor_key or tensor_key in tensor_keys:
                raise ValueError(f"GPTQ calibration source '{source_id}' has an invalid tensor_key.")
            width = _positive_int(entry.get("in_features"), f"source '{source_id}' in_features")
            rows = _positive_int(entry.get("row_count"), f"source '{source_id}' row_count")
            invocations = _positive_int(entry.get("invocation_count"), f"source '{source_id}' invocation_count")
            block_count = _positive_int(entry.get("num_blocks"), f"source '{source_id}' num_blocks")
            last_size = _positive_int(entry.get("last_block_size"), f"source '{source_id}' last_block_size")
            if block_count != (width + block_size - 1) // block_size or last_size != width - (block_count - 1) * block_size:
                raise ValueError(f"GPTQ calibration source '{source_id}' block layout does not cover its width.")
            if width % 256 or last_size % 256:
                raise ValueError(f"GPTQ calibration source '{source_id}' needs complete 256-wide ConvRot groups.")
            members = entry.get("members")
            shapes = entry.get("member_shapes")
            if not isinstance(members, list) or not members or not isinstance(shapes, list) or len(shapes) != len(members):
                raise ValueError(f"GPTQ calibration source '{source_id}' needs aligned members and shapes.")
            validated_shapes = []
            source_members: set[str] = set()
            for member, shape in zip(members, shapes):
                if not isinstance(member, str) or not member.endswith(".weight") or member in self.weight_to_source or member in source_members:
                    raise ValueError(f"GPTQ calibration source '{source_id}' has an invalid or duplicate member {member!r}.")
                source_members.add(member)
                if (
                    not isinstance(shape, list) or len(shape) != 2
                    or type(shape[0]) is not int or shape[0] < 1
                    or type(shape[1]) is not int or shape[1] != width
                ):
                    raise ValueError(f"GPTQ calibration member '{member}' has an invalid weight shape.")
                validated_shapes.append((shape[0], shape[1]))
            source = GramSource(source_id, tensor_key, width, rows, invocations, block_size, block_count, last_size,
                                tuple(members), tuple(validated_shapes))
            self.sources[source_id] = source
            tensor_keys.add(tensor_key)
            for member in members:
                self.weight_to_source[member] = source

        with safe_open(str(self.tensors_path), framework="pt", device="cpu") as artifact:
            if set(artifact.keys()) != tensor_keys:
                raise ValueError(f"GPTQ calibration '{self.tensors_path}' Safetensors keys do not match source tensor_key values.")
            for source in self.sources.values():
                slice_info = artifact.get_slice(source.tensor_key)
                expected = [source.num_blocks, source.block_size, source.block_size]
                if slice_info.get_shape() != expected:
                    raise ValueError(
                        f"GPTQ calibration '{self.tensors_path}' source '{source.source_id}' "
                        f"expected Gram shape {expected}, got {slice_info.get_shape()}."
                    )
                if slice_info.get_dtype() != "F32":
                    raise ValueError(f"GPTQ calibration '{self.tensors_path}' source '{source.source_id}' Gram must be FP32; got {slice_info.get_dtype()}.")

    def source_for_weight(self, weight_name: str) -> GramSource:
        try:
            return self.weight_to_source[weight_name]
        except KeyError as error:
            raise ValueError(f"GPTQ calibration '{self.metadata_path}' has no Gram for '{weight_name}'.") from error

    def validate_model(self, tensors: dict[str, Any], model_metadata: dict[str, str] | None = None) -> None:
        model_class = (model_metadata or {}).get("diffusion_model_class") or (model_metadata or {}).get("model_class")
        if model_class and model_class != self.metadata["model_class"]:
            raise ValueError(
                f"GPTQ calibration '{self.metadata_path}' model_class {self.metadata['model_class']!r} "
                f"does not match source model {model_class!r}."
            )
        for source in self.sources.values():
            for name, expected_shape in zip(source.members, source.member_shapes):
                descriptor = tensors.get(name)
                if descriptor is None or tuple(descriptor["shape"]) != expected_shape:
                    actual_shape = None if descriptor is None else tuple(descriptor["shape"])
                    raise ValueError(
                        f"GPTQ calibration '{self.metadata_path}' source '{source.source_id}' member '{name}' "
                        f"has a missing or mismatched model weight; expected shape {expected_shape}, got {actual_shape}."
                    )

    def load_blocks(self, source: GramSource) -> torch.Tensor:
        with safe_open(str(self.tensors_path), framework="pt", device="cpu") as artifact:
            blocks = artifact.get_tensor(source.tensor_key)
        if not bool(torch.isfinite(blocks).all().item()):
            raise ValueError(f"GPTQ calibration '{self.tensors_path}' source '{source.source_id}' contains nonfinite Gram blocks.")
        for index, block in enumerate(blocks):
            width = source.last_block_size if index == source.num_blocks - 1 else source.block_size
            actual = block[:width, :width]
            if not torch.allclose(actual, actual.T, rtol=1e-4, atol=1e-3):
                raise ValueError(f"GPTQ calibration '{self.tensors_path}' source '{source.source_id}' block {index} is not symmetric.")
        if source.last_block_size < source.block_size and (
            bool(torch.count_nonzero(blocks[-1, source.last_block_size:, :]).item())
            or bool(torch.count_nonzero(blocks[-1, :, source.last_block_size:]).item())
        ):
            raise ValueError(f"GPTQ calibration '{self.tensors_path}' source '{source.source_id}' final block padding must be zero.")
        return blocks

    def factors_for_weight(
        self, weight_name: str, timings: dict[str, float] | None = None,
        device: str | torch.device = "cpu",
    ) -> tuple[torch.Tensor, ...]:
        from ..quantization.gptq import prepare_factors
        from ..timing import timed_stage

        source = self.source_for_weight(weight_name)
        target_device = torch.device(device)
        if self._factor_device != target_device:
            self._factor_cache.clear()
            self._factor_device = target_device
        factors = self._factor_cache.pop(source.source_id, None)
        if factors is None:
            with timed_stage(timings, "gram_load"):
                blocks = self.load_blocks(source)
            try:
                factors = prepare_factors(blocks, source.in_features, block_size=source.block_size, timings=timings)
            except ValueError as error:
                raise ValueError(
                    f"GPTQ calibration '{self.metadata_path}' source '{source.source_id}' "
                    f"for '{weight_name}': {error}"
                ) from error
            del blocks
            if target_device.type != "cpu":
                with timed_stage(timings, "hessian_transfer"):
                    factors = tuple(factor.to(target_device) for factor in factors)
        # ponytail: cache only two prepared sources; raise the limit if measured
        # full-model interleaving shows repeated factorization.
        self._factor_cache[source.source_id] = factors
        if len(self._factor_cache) > 2:
            del self._factor_cache[next(iter(self._factor_cache))]
        return factors
