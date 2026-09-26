"""Research/regression tool for full-Gram and block-diagonal GPTQ comparisons.

Artifact contract: JSON has matching weight_name, in_features,
activation_basis="logical_linear_input", and row_count > 0;
safetensors has FP32 gram/H_sum or native gram_blocks.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import struct
import sys
import time
from pathlib import Path
from typing import NamedTuple

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from potatoforge.quantization.convrot_w4a4 import (
    _quantize_w4_codes,
    _select_w4_absmax_scale,
    quantize_convrot_w4a4,
    quantize_convrot_w4a4_mse,
    select_w4a4_mse_scale,
    unpack_signed_int4_row_major,
)
from potatoforge.quantization.hadamard import (
    CONVROT_GROUP_SIZE,
    apply_hadamard_rotation,
)
from potatoforge.quantization.convrot_w4a4_gptq import (
    rotate_gram,
    gptq_factor,
    gptq_quantize,
    gptq_quantize_blocks,
)


# Explicit Krea2 shared-input groups verified in comfy/ldm/krea2/model.py.
SIBLING_TARGETS = {
    "blocks.4.attn.wq.weight": (
        "blocks.4.attn.wk.weight", "blocks.4.attn.wv.weight", "blocks.4.attn.gate.weight",
    ),
    "blocks.14.attn.wq.weight": (
        "blocks.14.attn.wk.weight", "blocks.14.attn.wv.weight", "blocks.14.attn.gate.weight",
    ),
    "blocks.24.attn.wq.weight": (
        "blocks.24.attn.wk.weight", "blocks.24.attn.wv.weight", "blocks.24.attn.gate.weight",
    ),
    "blocks.14.mlp.gate.weight": ("blocks.14.mlp.up.weight",),
}


class GramBlocks(NamedTuple):
    blocks: torch.Tensor
    in_features: int
    block_size: int
    last_block_size: int


def validate_gram_target(source: str, target: str) -> None:
    if target != source and target not in SIBLING_TARGETS.get(source, ()):
        raise ValueError(f"Gram reuse is not verified for {source} -> {target}")


def output_energy(weights: torch.Tensor, gram: torch.Tensor) -> float:
    return float((weights * (weights @ gram)).sum().item())


def gptq_quantize_block_diagonal(
    rotated_weights: torch.Tensor,
    rotated_train_gram: torch.Tensor,
    scales: torch.Tensor,
    damp_percent: float,
    blocksize: int,
    hessian_block_size: int,
) -> torch.Tensor:
    """Run the existing GPTQ solver independently in each Hessian block."""
    columns = rotated_weights.shape[1]
    if hessian_block_size <= 0:
        raise ValueError("Hessian block size must be positive")
    if rotated_train_gram.shape != (columns, columns):
        raise ValueError("Rotated training Gram dimensions must match weight in_features")
    factors = tuple(
        gptq_factor(rotated_train_gram[start:min(start + hessian_block_size, columns),
                                       start:min(start + hessian_block_size, columns)], damp_percent)
        for start in range(0, columns, hessian_block_size)
    )
    return gptq_quantize_blocks(rotated_weights, factors, scales, blocksize)


def gptq_quantize_saved_blocks(
    rotated_weights: torch.Tensor,
    rotated_blocks: tuple[torch.Tensor, ...],
    scales: torch.Tensor,
    damp_percent: float,
    blocksize: int,
) -> torch.Tensor:
    factors = tuple(gptq_factor(gram, damp_percent) for gram in rotated_blocks)
    return gptq_quantize_blocks(rotated_weights, factors, scales, blocksize)


def load_weight(path: Path, tensor_name: str, in_features: int = 6144) -> torch.Tensor:
    # An ordinary read avoids a Windows mmap access violation on sparse checkpoints.
    with path.open("rb") as source:
        header_length_bytes = source.read(8)
        if len(header_length_bytes) != 8:
            raise ValueError("Invalid safetensors header")
        header_length = struct.unpack("<Q", header_length_bytes)[0]
        if not 2 <= header_length <= 100_000_000:
            raise ValueError("Invalid safetensors header size")
        header = json.loads(source.read(header_length))
        entry = header[tensor_name]
        shape = entry["shape"]
        offsets = entry["data_offsets"]
        if entry["dtype"] != "BF16" or len(shape) != 2 or shape[1] != in_features:
            raise ValueError(f"Selected weight must be BF16 [out_features, {in_features}]")
        byte_count = shape[0] * shape[1] * 2
        if offsets[1] - offsets[0] != byte_count:
            raise ValueError("Selected tensor byte count does not match its shape")
        source.seek(8 + header_length + offsets[0])
        raw = bytearray(byte_count)
        if source.readinto(raw) != byte_count:
            raise ValueError("Selected tensor data is incomplete")
    return torch.frombuffer(raw, dtype=torch.bfloat16).reshape(shape).clone()


def load_gram(json_path: Path, safetensors_path: Path, tensor_name: str) -> tuple[torch.Tensor | GramBlocks, int]:
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if metadata.get("activation_basis") != "logical_linear_input":
        raise ValueError("Gram activation_basis must be logical_linear_input")
    if metadata.get("weight_name") != tensor_name:
        raise ValueError(f"Gram weight_name does not match {tensor_name}")
    in_features = metadata.get("in_features")
    if type(in_features) is not int or in_features <= 0:
        raise ValueError("Gram in_features must be a positive integer")
    row_count = metadata.get("row_count")
    if type(row_count) is not int or row_count <= 0:
        raise ValueError("Gram row_count must be a positive integer")
    for key in ("tensor", "tensor_name"):
        if key in metadata and metadata[key] != tensor_name:
            raise ValueError(f"Gram {key} does not match --tensor")
    mode = metadata.get("gram_mode", "full")
    if mode not in ("full", "block_diag"):
        raise ValueError(f"Unsupported gram_mode: {mode}")
    if metadata.get("dtype", "float32") != "float32":
        raise ValueError("Gram dtype metadata must be float32")
    with safe_open(str(safetensors_path), framework="pt", device="cpu") as artifact:
        if mode == "block_diag":
            if set(artifact.keys()) != {"gram_blocks"}:
                raise ValueError("Block-diagonal artifact must contain gram_blocks only")
            gram = artifact.get_tensor("gram_blocks")
        else:
            gram = artifact.get_tensor("gram" if "gram" in artifact.keys() else "H_sum")
    if mode == "full":
        if in_features != 6144 or gram.dtype != torch.float32 or gram.shape != (6144, 6144):
            raise ValueError("Full Gram must be FP32 [6144, 6144]")
        if not torch.isfinite(gram).all():
            raise ValueError("Gram contains nonfinite values")
        return gram, row_count
    block_size = metadata.get("gram_block_size")
    num_blocks = metadata.get("num_blocks")
    last_size = metadata.get("last_block_size")
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("gram_block_size must be a positive integer")
    if type(num_blocks) is not int or num_blocks <= 0 or num_blocks != math.ceil(in_features / block_size):
        raise ValueError("num_blocks does not match in_features and gram_block_size")
    if type(last_size) is not int or last_size != in_features - (num_blocks - 1) * block_size:
        raise ValueError("last_block_size does not cover the final features exactly")
    if gram.dtype != torch.float32 or gram.shape != (num_blocks, block_size, block_size):
        raise ValueError("gram_blocks shape or dtype does not match metadata")
    if not torch.isfinite(gram).all():
        raise ValueError("gram_blocks contains nonfinite values")
    if last_size < block_size and (
        torch.count_nonzero(gram[-1, last_size:, :]).item() or
        torch.count_nonzero(gram[-1, :, last_size:]).item()
    ):
        raise ValueError("Final gram_blocks padding must be zero")
    return GramBlocks(gram, in_features, block_size, last_size), row_count


def load_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, int]:
    target = args.target_tensor or args.tensor
    validate_gram_target(args.tensor, target)
    gram, row_count = load_gram(args.gram_json, args.gram_safetensors, args.tensor)
    if isinstance(gram, GramBlocks):
        raise ValueError("Block-diagonal artifacts require --gram-folder for leave-one-out scoring")
    weights = load_weight(args.model, target)
    return weights, gram, row_count


def score_candidates(
    reference: torch.Tensor,
    candidates: list[torch.Tensor],
    rotated_gram: torch.Tensor,
) -> list[tuple[float, float]]:
    reference_energy = output_energy(reference, rotated_gram)
    if not math.isfinite(reference_energy) or reference_energy <= 0:
        raise ValueError("Reference output energy must be positive and finite")
    scores = []
    for candidate in candidates:
        error_energy = output_energy(reference - candidate, rotated_gram)
        if not math.isfinite(error_energy) or error_energy < -1e-5 * reference_energy:
            raise ValueError("Invalid output error energy")
        error_energy = max(0.0, error_energy)
        scores.append((error_energy, error_energy / reference_energy))
    return scores


def rotate_saved_blocks(gram: GramBlocks) -> tuple[torch.Tensor, ...]:
    if gram.in_features % CONVROT_GROUP_SIZE or gram.block_size % CONVROT_GROUP_SIZE:
        raise ValueError("ConvRot requires in_features and gram_block_size divisible by 256")
    rotated = []
    for index in range(gram.blocks.shape[0]):
        width = gram.last_block_size if index == gram.blocks.shape[0] - 1 else gram.block_size
        if width % CONVROT_GROUP_SIZE:
            raise ValueError("Final Gram block must contain complete 256-wide ConvRot groups")
        rotated.append(rotate_gram(gram.blocks[index, :width, :width]))
    return tuple(rotated)


def score_block_candidates(
    reference: torch.Tensor,
    candidates: list[torch.Tensor],
    rotated_blocks: tuple[torch.Tensor, ...],
) -> list[float]:
    if sum(block.shape[0] for block in rotated_blocks) != reference.shape[1]:
        raise ValueError("Rotated Gram blocks do not cover every weight feature")
    reference_energy = 0.0
    errors = [reference - candidate for candidate in candidates]
    error_energies = [0.0] * len(candidates)
    start = 0
    for block in rotated_blocks:
        end = start + block.shape[0]
        reference_energy += output_energy(reference[:, start:end], block)
        for index, error in enumerate(errors):
            error_energies[index] += output_energy(error[:, start:end], block)
        start = end
    if not math.isfinite(reference_energy) or reference_energy <= 0:
        raise ValueError("Block-diagonal reference output energy must be positive and finite")
    if any(not math.isfinite(energy) or energy < -1e-5 * reference_energy for energy in error_energies):
        raise ValueError("Invalid block-diagonal output error energy")
    return [max(0.0, energy) / reference_energy for energy in error_energies]


def discover_gram_pairs(folder: Path, tensor_name: str) -> list[tuple[str, torch.Tensor | GramBlocks, int]]:
    if not folder.is_dir():
        raise ValueError(f"Gram folder does not exist: {folder}")
    json_files = {path.stem: path for path in folder.glob("gptq_poc_*.json") if path.is_file()}
    tensor_files = {path.stem: path for path in folder.glob("gptq_poc_*.safetensors") if path.is_file()}
    if json_files.keys() != tensor_files.keys():
        missing = sorted(json_files.keys() ^ tensor_files.keys())
        raise ValueError(f"GPTQ POC artifacts need same-stem JSON/Safetensors pairs: {', '.join(missing)}")
    if len(json_files) != 4:
        raise ValueError(f"Expected exactly four GPTQ POC Gram pairs in {folder}; found {len(json_files)}")
    pairs = []
    expected_format = None
    for stem in sorted(json_files):
        try:
            gram, rows = load_gram(json_files[stem], tensor_files[stem], tensor_name)
        except (KeyError, ValueError) as error:
            raise ValueError(f"Invalid GPTQ POC artifact {stem}: {error}") from error
        artifact_format = ((gram.in_features, gram.block_size, gram.last_block_size)
                           if isinstance(gram, GramBlocks) else (gram.shape[0],))
        if expected_format is None:
            expected_format = artifact_format
        elif artifact_format != expected_format:
            raise ValueError("GPTQ POC Grams must share mode, in_features, and block dimensions")
        pairs.append((stem, gram, rows))
    return pairs


def fold_training_gram(
    pairs: list[tuple[str, torch.Tensor | GramBlocks, int]], held_out: int,
) -> tuple[torch.Tensor | GramBlocks, int]:
    first = pairs[0][1]
    raw_sum = torch.zeros_like(first.blocks if isinstance(first, GramBlocks) else first, dtype=torch.float64)
    row_sum = 0
    for index, (_, gram, rows) in enumerate(pairs):
        if index != held_out:
            raw_sum.add_(gram.blocks if isinstance(gram, GramBlocks) else gram)
            row_sum += rows
    if isinstance(first, GramBlocks):
        return GramBlocks(raw_sum.float(), first.in_features, first.block_size, first.last_block_size), row_sum
    return raw_sum.float(), row_sum


def run_saved_block_leave_one_out(
    args: argparse.Namespace,
    pairs: list[tuple[str, GramBlocks, int]],
) -> tuple[list[float], float, float, float, float, int]:
    source = args.tensor
    target = args.target_tensor or source
    validate_gram_target(source, target)
    artifact = pairs[0][1]
    weights = load_weight(args.model, target, artifact.in_features)
    rotated_weights = apply_hadamard_rotation(weights, CONVROT_GROUP_SIZE)
    packed = quantize_convrot_w4a4_mse(weights, device="cpu")
    baseline = unpack_signed_int4_row_major(packed.packed_codes).float() * packed.scales[:, None]
    scales = select_w4a4_mse_scale(rotated_weights)
    print(f"Tensor: {target}; dimensions: {tuple(weights.shape)}; block size: "
          f"{artifact.block_size}; blocks: {artifact.blocks.shape[0]}", flush=True)
    results = []
    for held_out, (validation_stem, validation_gram, validation_rows) in enumerate(pairs):
        train_gram, train_rows = fold_training_gram(pairs, held_out)
        print(f"Fold {held_out + 1}: training rows {train_rows}; "
              f"held out {validation_stem} ({validation_rows} rows)", flush=True)
        rotated_train = rotate_saved_blocks(train_gram)
        quantized = gptq_quantize_saved_blocks(
            rotated_weights, rotated_train, scales, args.damp_percent, args.gptq_blocksize
        )
        rotated_validation = rotate_saved_blocks(validation_gram)
        mse_sse, gptq_sse = score_block_candidates(
            rotated_weights, [baseline, quantized], rotated_validation
        )
        if mse_sse == 0:
            raise ValueError(f"Fold {held_out + 1} has zero MSE baseline error")
        improvement = 100 * (1 - gptq_sse / mse_sse)
        results.append((held_out + 1, train_rows, validation_rows, mse_sse, gptq_sse, improvement))
        print(f"  block_diag_{artifact.block_size} held-out improvement: "
              f"{improvement:.1f}%", flush=True)
    label = f"block_diag_{artifact.block_size}_relative_output_sse"
    print(f"fold train_rows validation_rows mse_held_out_{label} "
          f"mse_gptq_held_out_{label} improvement_percent")
    for fold, train_rows, validation_rows, mse_sse, gptq_sse, improvement in results:
        print(f"{fold} {train_rows} {validation_rows} {mse_sse:.8g} "
              f"{gptq_sse:.8g} {improvement:.1f}")
    gains = [result[-1] for result in results]
    mean, median = statistics.mean(gains), statistics.median(gains)
    minimum, maximum = min(gains), max(gains)
    positive = sum(gain > 0 for gain in gains)
    print(f"Held-out block_diag_{artifact.block_size} MSE+GPTQ improvement over MSE: "
          f"mean {mean:.1f}%, median {median:.1f}%, min {minimum:.1f}%, "
          f"max {maximum:.1f}%, positive folds {positive}/4")
    return gains, mean, median, minimum, maximum, positive


def run_leave_one_out(
    args: argparse.Namespace,
    pairs: list[tuple[str, torch.Tensor, int]] | None = None,
):
    source = args.tensor
    target = args.target_tensor or source
    validate_gram_target(source, target)
    if pairs is None:
        pairs = discover_gram_pairs(args.gram_folder, source)
        print("Discovered GPTQ POC Grams (sorted):", flush=True)
        for stem, _, rows in pairs:
            print(f"  {stem}: {rows} rows", flush=True)
    if isinstance(pairs[0][1], GramBlocks):
        if args.compare_bd256 or args.compare_bd_sweep:
            raise ValueError("Full-Gram comparisons require full-Gram artifacts")
        return run_saved_block_leave_one_out(args, pairs)
    print("Loading weight and preparing ConvRot W4A4 MSE baseline...", flush=True)
    weights = load_weight(args.model, target)
    if target != source:
        print(f"Gram source: {source}\nTarget weight: {target}\n"
              "Reusing Gram because the verified projections share a logical input and dimensions match.",
              flush=True)
    rotated_weights = apply_hadamard_rotation(weights, CONVROT_GROUP_SIZE)
    packed = quantize_convrot_w4a4_mse(weights, device="cpu")
    codes = unpack_signed_int4_row_major(packed.packed_codes)
    baseline = codes.float() * packed.scales[:, None]
    scales = select_w4a4_mse_scale(rotated_weights)
    fold_results = []
    hessian_block_sizes = (256, 512, 1024) if args.compare_bd_sweep else (256,) if args.compare_bd256 else ()

    for held_out, (validation_stem, validation_gram, validation_rows) in enumerate(pairs):
        fold = held_out + 1
        training_stems = [stem for index, (stem, _, _) in enumerate(pairs) if index != held_out]
        train_gram, train_rows = fold_training_gram(pairs, held_out)
        print(f"Fold {fold}: training on {', '.join(training_stems)}; holding out {validation_stem}", flush=True)
        rotated_train = rotate_gram(train_gram)
        del train_gram
        inverse_factor = gptq_factor(rotated_train, args.damp_percent)
        quantized = gptq_quantize(rotated_weights, inverse_factor, scales, args.gptq_blocksize)
        del inverse_factor
        candidates = [baseline, quantized]
        for hessian_block_size in hessian_block_sizes:
            candidates.append(gptq_quantize_block_diagonal(
                rotated_weights, rotated_train, scales, args.damp_percent,
                args.gptq_blocksize, hessian_block_size,
            ))
        train_scores = score_candidates(rotated_weights, candidates, rotated_train)
        rotated_validation = rotate_gram(validation_gram)
        validation_scores = score_candidates(rotated_weights, candidates, rotated_validation)
        mse_validation = validation_scores[0][1]
        if mse_validation == 0:
            raise ValueError(f"Fold {fold} has zero MSE baseline error")
        improvement = 100 * (1 - validation_scores[1][1] / mse_validation)
        result = (
            fold, ",".join(training_stems), validation_stem, train_rows, validation_rows,
            train_scores[0][1], train_scores[1][1], mse_validation,
            validation_scores[1][1], improvement,
        )
        if hessian_block_sizes:
            block_results = []
            for index, size in enumerate(hessian_block_sizes, 2):
                bd_gain = 100 * (1 - validation_scores[index][1] / mse_validation)
                retained = 100 * bd_gain / improvement if improvement > 0 else None
                block_results.append((size, train_scores[index][1], validation_scores[index][1], bd_gain, retained))
            result += (block_results,)
            print(f"Fold {fold} held-out gain: full {improvement:.1f}%; " + ", ".join(
                f"BD{size} {gain:.1f}% (retained {retained:.1f}%)" if retained is not None
                else f"BD{size} {gain:.1f}% (retained n/a)"
                for size, _, _, gain, retained in block_results
            ), flush=True)
        else:
            print(f"Fold {fold} held-out improvement: {improvement:.1f}%", flush=True)
        fold_results.append(result)
        del rotated_train, quantized, rotated_validation, candidates

    if hessian_block_sizes:
        print("fold validation_artifact train_rows validation_rows mse_held_out_relative_output_sse "
              "full_held_out_relative_output_sse " + " ".join(
                  f"bd{size}_held_out_relative_output_sse" for size in hessian_block_sizes
              ) + " full_gain_percent " + " ".join(
                  f"bd{size}_gain_percent bd{size}_retained_percent" for size in hessian_block_sizes
              ))
        for result in fold_results:
            fold, _, validation, train_rows, validation_rows, _, _, mse_val, full_val, full_gain, blocks = result
            print(f"{fold} {validation} {train_rows} {validation_rows} {mse_val:.8g} "
                  f"{full_val:.8g} " + " ".join(f"{block[2]:.8g}" for block in blocks) +
                  f" {full_gain:.1f} " + " ".join(
                      f"{gain:.1f} {retained:.1f}" if retained is not None else f"{gain:.1f} n/a"
                      for _, _, _, gain, retained in blocks
                  ))
        full_gains = [result[9] for result in fold_results]
        bd_gains = {size: [result[10][index][3] for result in fold_results]
                    for index, size in enumerate(hessian_block_sizes)}
        retained_gains = {size: [result[10][index][4] for result in fold_results
                                 if result[10][index][4] is not None]
                          for index, size in enumerate(hessian_block_sizes)}
        print(f"Mean held-out gain: full {statistics.mean(full_gains):.1f}%")
        for size in hessian_block_sizes:
            retained = retained_gains[size]
            stats = (f"mean {statistics.mean(retained):.1f}%, "
                     f"median {statistics.median(retained):.1f}%, "
                     f"min {min(retained):.1f}%, max {max(retained):.1f}%") if retained else "n/a"
            print(f"BD{size}: mean gain {statistics.mean(bd_gains[size]):.1f}%; "
                  f"retained {stats} ({len(retained)}/4 defined); "
                  f"positive folds {sum(gain > 0 for gain in bd_gains[size])}/4")
        return full_gains, bd_gains, retained_gains

    print("fold training_artifacts validation_artifact train_rows validation_rows "
          "mse_train_relative_output_sse mse_gptq_train_relative_output_sse "
          "mse_validation_relative_output_sse mse_gptq_validation_relative_output_sse "
          "held_out_improvement_percent")
    for fold, training, validation, train_rows, validation_rows, mse_train, gptq_train, mse_val, gptq_val, improvement in fold_results:
        print(f"{fold} {training} {validation} {train_rows} {validation_rows} "
              f"{mse_train:.8g} {gptq_train:.8g} {mse_val:.8g} {gptq_val:.8g} {improvement:.1f}")
    improvements = [result[-1] for result in fold_results]
    print("Held-out MSE+GPTQ improvement over MSE:")
    for fold, improvement in enumerate(improvements, 1):
        print(f"  Fold {fold}: {improvement:.1f}%")
    print(f"  Mean: {statistics.mean(improvements):.1f}%")
    print(f"  Median: {statistics.median(improvements):.1f}%")
    print(f"  Min: {min(improvements):.1f}%")
    print(f"  Max: {max(improvements):.1f}%")
    print(f"  All four folds improve over MSE: {'yes' if all(value > 0 for value in improvements) else 'no'}")
    return (
        improvements, statistics.mean(improvements), statistics.median(improvements),
        min(improvements), max(improvements), sum(value > 0 for value in improvements),
    )


def run_gram_root(args: argparse.Namespace) -> None:
    if not args.gram_root.is_dir():
        raise ValueError(f"Gram root does not exist: {args.gram_root}")
    folders = sorted(path for path in args.gram_root.iterdir() if path.is_dir())
    if not folders:
        raise ValueError(f"Gram root has no tensor subfolders: {args.gram_root}")
    summaries = []
    for folder in folders:
        json_files = sorted(folder.glob("gptq_poc_*.json"))
        if not json_files:
            raise ValueError(f"No GPTQ POC Gram JSON files in {folder}")
        metadata = json.loads(json_files[0].read_text(encoding="utf-8"))
        source = metadata.get("weight_name")
        if not isinstance(source, str) or not source:
            raise ValueError(f"Missing weight_name in {json_files[0]}")
        if args.sibling_targets and source not in SIBLING_TARGETS:
            continue
        pairs = discover_gram_pairs(folder, source)
        if isinstance(pairs[0][1], GramBlocks):
            raise ValueError(f"Use --gram-folder for block-diagonal artifacts: {folder}")
        print(f"\nGram source: {source} ({folder.name})", flush=True)
        print("Discovered GPTQ POC Grams (sorted):", flush=True)
        for stem, _, rows in pairs:
            print(f"  {stem}: {rows} rows", flush=True)
        folder_args = argparse.Namespace(**vars(args))
        folder_args.gram_folder = folder
        folder_args.tensor = source
        if args.all_targets:
            targets = (source, *SIBLING_TARGETS.get(source, ()))
        else:
            targets = SIBLING_TARGETS[source] if args.sibling_targets else (source,)
        for target in targets:
            folder_args.target_tensor = target
            print(f"\nTensor: {target}", flush=True)
            summaries.append((target, source, *run_leave_one_out(folder_args, pairs)))
    if not summaries:
        raise ValueError(f"No verified sibling Gram sources found in {args.gram_root}")
    if args.compare_bd256 or args.compare_bd_sweep:
        sizes = (256, 512, 1024) if args.compare_bd_sweep else (256,)
        print("\ntensor full_gain " + " ".join(f"bd{size}_gain" for size in sizes) +
              " " + " ".join(f"bd{size}_retained" for size in sizes))
        families: dict[str, list[dict[int, list[float]]]] = {}
        for target, _, full_gains, bd_gains, retained in summaries:
            print(f"{target} {statistics.mean(full_gains):.1f}% " +
                  " ".join(f"{statistics.mean(bd_gains[size]):.1f}%" for size in sizes) +
                  " " + " ".join(
                      f"{statistics.mean(retained[size]):.1f}%" if retained[size] else "n/a"
                      for size in sizes
                  ))
            parts = target.split(".")
            family = f"block_{parts[1]}_attention" if parts[2] == "attn" else f"block_{parts[1]}_mlp"
            if target == "blocks.14.attn.wo.weight":
                family = "block_14_attention_wo"
            families.setdefault(family, []).append(retained)
        print("\nfamily tensor_count " + " ".join(f"bd{size}_mean_retained" for size in sizes))
        for family, members in families.items():
            print(f"{family} {len(members)} " + " ".join(
                f"{statistics.mean(values):.1f}%" if (values := [gain for member in members for gain in member[size]]) else "n/a"
                for size in sizes
            ))
        return
    print("\ntensor gram_source fold_1_gain fold_2_gain fold_3_gain fold_4_gain "
          "mean_held_out_improvement median_held_out_improvement "
          "min_improvement max_improvement positive_folds")
    for target, source, gains, mean, median, minimum, maximum, positive in summaries:
        print(f"{target} {source} " + " ".join(f"{gain:.1f}%" for gain in gains) +
              f" {mean:.1f}% {median:.1f}% {minimum:.1f}% {maximum:.1f}% {positive}/4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="BF16 safetensors checkpoint file")
    parser.add_argument("--tensor", help="Gram source weight key (inferred with --gram-root)")
    parser.add_argument("--target-tensor", help="Explicit verified sibling weight to quantize with the source Gram")
    parser.add_argument("--gram-json", type=Path)
    parser.add_argument("--gram-safetensors", type=Path)
    parser.add_argument("--gram-folder", type=Path, help="Run four-fold leave-one-out on direct GPTQ POC pairs")
    parser.add_argument("--gram-root", type=Path, help="Run leave-one-out for each immediate tensor subfolder")
    parser.add_argument("--sibling-targets", action="store_true", help="With --gram-root, test the ten verified Krea2 sibling targets")
    parser.add_argument("--all-targets", action="store_true", help="With --gram-root, test source and verified sibling targets")
    parser.add_argument("--compare-bd256", action="store_true", help="Compare full GPTQ with independent 256-wide Gram blocks in leave-one-out mode")
    parser.add_argument("--compare-bd-sweep", action="store_true", help="Compare full GPTQ with independent 256, 512, and 1024-wide Gram blocks")
    parser.add_argument(
        "--validation-gram", nargs=2, action="append", type=Path,
        metavar=("JSON", "SAFETENSORS"), default=[],
        help="Held-out Gram pair; repeat for each validation prompt",
    )
    parser.add_argument("--damp-percent", type=float, default=0.01)
    parser.add_argument("--gptq-blocksize", type=int, default=128)
    args = parser.parse_args()
    if not math.isfinite(args.damp_percent) or args.damp_percent <= 0:
        parser.error("--damp-percent must be finite and positive")
    if args.gptq_blocksize <= 0:
        parser.error("--gptq-blocksize must be positive")
    if args.sibling_targets and not args.gram_root:
        parser.error("--sibling-targets requires --gram-root")
    if args.all_targets and not args.gram_root:
        parser.error("--all-targets requires --gram-root")
    if args.all_targets and args.sibling_targets:
        parser.error("--all-targets and --sibling-targets are mutually exclusive")
    if args.compare_bd256 and not (args.gram_root or args.gram_folder):
        parser.error("--compare-bd256 requires --gram-root or --gram-folder")
    if args.compare_bd_sweep and not (args.gram_root or args.gram_folder):
        parser.error("--compare-bd-sweep requires --gram-root or --gram-folder")
    if args.compare_bd_sweep and args.compare_bd256:
        parser.error("--compare-bd-sweep and --compare-bd256 are mutually exclusive")
    if args.gram_root:
        if args.tensor or args.target_tensor or args.gram_folder or args.gram_json or args.gram_safetensors or args.validation_gram:
            parser.error("--gram-root cannot be combined with tensor overrides or individual Gram arguments")
        with torch.no_grad():
            run_gram_root(args)
        return
    if not args.tensor:
        parser.error("--tensor is required without --gram-root")
    if args.gram_folder:
        if args.gram_json or args.gram_safetensors or args.validation_gram:
            parser.error("--gram-folder cannot be combined with individual Gram arguments")
        with torch.no_grad():
            run_leave_one_out(args)
        return
    if not args.gram_json or not args.gram_safetensors:
        parser.error("--gram-json and --gram-safetensors are required without --gram-folder")

    with torch.no_grad():
        print("Loading selected weight and full Gram...", flush=True)
        weights, gram, row_count = load_inputs(args)
        target = args.target_tensor or args.tensor
        print(f"Loaded {target}; calibration rows: {row_count}", flush=True)
        if target != args.tensor:
            print(f"Gram source: {args.tensor}\nTarget weight: {target}\n"
                  "Reusing Gram because the verified projections share a logical input and dimensions match.",
                  flush=True)
        reference = weights.float()
        rotation_start = time.perf_counter()
        print("Rotating weight and full Gram...", flush=True)
        rotated_weights = apply_hadamard_rotation(weights, CONVROT_GROUP_SIZE)
        rotated_gram = rotate_gram(gram)
        rotation_time = time.perf_counter() - rotation_start
        print("Scoring reference output energy...", flush=True)
        reference_energy = output_energy(rotated_weights, rotated_gram)
        if not math.isfinite(reference_energy) or reference_energy <= 0:
            raise ValueError("Reference output energy must be positive and finite")
        print(f"Calibration rows: {row_count}; full Gram: {tuple(gram.shape)}")

        results: list[tuple[str, float, float, float, float, torch.Tensor]] = []

        def record(name: str, quantized: torch.Tensor, seconds: float) -> None:
            error = rotated_weights - quantized
            weighted_sse = output_energy(error, rotated_gram)
            if not math.isfinite(weighted_sse) or weighted_sse < -1e-5 * reference_energy:
                raise ValueError(f"Invalid output error energy for {name}")
            weighted_sse = max(0.0, weighted_sse)
            results.append((name, float(error.square().mean()), weighted_sse, weighted_sse / reference_energy, seconds, quantized))

        for name, quantizer in (
            ("convrot_w4a4", quantize_convrot_w4a4),
            ("convrot_w4a4_mse", quantize_convrot_w4a4_mse),
        ):
            print(f"Quantizing {name}...", flush=True)
            started = time.perf_counter()
            packed = quantizer(weights, device="cpu")
            seconds = time.perf_counter() - started
            codes = unpack_signed_int4_row_major(packed.packed_codes)
            quantized = codes.float() * packed.scales[:, None]
            record(name, quantized, seconds)
            print(f"Scored {name}", flush=True)
            if name == "convrot_w4a4_mse":
                logical_error = reference - apply_hadamard_rotation(quantized, CONVROT_GROUP_SIZE)
                logical_ratio = output_energy(logical_error, gram) / output_energy(reference, gram)
                if not math.isclose(logical_ratio, results[-1][3], rel_tol=2e-3, abs_tol=1e-6):
                    raise ValueError("ConvRot output metric differs between logical and rotated bases")
                print(f"Basis invariance check: logical={logical_ratio:.8g}, rotated={results[-1][3]:.8g}")

        print("Factoring damped full Gram for GPTQ...", flush=True)
        started = time.perf_counter()
        inverse_factor = gptq_factor(rotated_gram, args.damp_percent)
        factor_time = time.perf_counter() - started
        for name, scale_selector in (
            ("convrot_w4a4_gptq", _select_w4_absmax_scale),
            ("convrot_w4a4_mse_gptq", select_w4a4_mse_scale),
        ):
            print(f"Quantizing {name}...", flush=True)
            started = time.perf_counter()
            scales = scale_selector(rotated_weights)
            quantized = gptq_quantize(rotated_weights, inverse_factor, scales, args.gptq_blocksize)
            seconds = rotation_time + factor_time + time.perf_counter() - started
            record(name, quantized, seconds)
            print(f"Scored {name}", flush=True)

        print("method                         weight_mse  activation_weighted_output_sse  relative_output_sse  quantization_time_s")
        for name, mse, sse, relative, seconds, _ in results:
            print(f"{name:30} {mse:10.6g} {sse:32.6g} {relative:20.6g} {seconds:20.2f}")
        baseline = results[1][3]
        if baseline == 0:
            print("GPTQ improvement over ConvRot W4A4 MSE: undefined (zero baseline error)")
        else:
            for name, _, _, relative, _, _ in results[2:]:
                print(f"{name} improvement over ConvRot W4A4 MSE: {(1 - relative / baseline) * 100:.1f}%")

        if args.validation_gram:
            candidates = [result[5] for result in results]
            validation_scores: list[list[float]] = []
            combined_gram = torch.zeros_like(gram, dtype=torch.float64)
            combined_rows = 0
            for index, (json_path, safetensors_path) in enumerate(args.validation_gram, 1):
                print(f"Loading and scoring validation {index}...", flush=True)
                validation_gram, validation_rows = load_gram(json_path, safetensors_path, args.tensor)
                combined_gram.add_(validation_gram)
                combined_rows += validation_rows
                scores = score_candidates(rotated_weights, candidates, rotate_gram(validation_gram))
                validation_scores.append([relative for _, relative in scores])
                print(f"Validation {index} rows: {validation_rows}", flush=True)
            print("Scoring combined held-out Gram...", flush=True)
            combined_scores = [
                relative for _, relative in score_candidates(
                    rotated_weights, candidates, rotate_gram(combined_gram.float())
                )
            ]
            print(f"Combined validation rows: {combined_rows}")
            labels = ["train"] + [f"val_{index}" for index in range(1, len(validation_scores) + 1)] + ["combined"]
            print("method " + " ".join(f"{label}_relative_output_sse" for label in labels))
            for method_index, result in enumerate(results):
                ratios = [result[3]] + [scores[method_index] for scores in validation_scores] + [combined_scores[method_index]]
                print(f"{result[0]:30} " + " ".join(f"{ratio:.8g}" for ratio in ratios))
            for method_index in (2, 3):
                print(f"{results[method_index][0]} improvement over convrot_w4a4_mse:")
                for label, ratios in zip(
                    labels,
                    [[result[3] for result in results]] + validation_scores + [combined_scores],
                ):
                    improvement = "undefined (zero baseline error)" if ratios[1] == 0 else f"{(1 - ratios[method_index] / ratios[1]) * 100:.1f}%"
                    print(f"  {label}: {improvement}")


if __name__ == "__main__":
    main()
