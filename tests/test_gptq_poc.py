import argparse
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from scripts.gptq_poc import (
    GramBlocks,
    discover_gram_pairs,
    fold_training_gram,
    gptq_factor,
    gptq_quantize,
    gptq_quantize_block_diagonal,
    load_gram,
    load_weight,
    output_energy,
    rotate_gram,
    rotate_saved_blocks,
    run_gram_root,
    run_saved_block_leave_one_out,
    score_block_candidates,
    score_candidates,
    validate_gram_target,
)
from potatoforge.quantization.convrot_w4a4 import (
    _quantize_w4_codes,
    select_w4a4_mse_scale,
)
from potatoforge.quantization.hadamard import (
    CONVROT_GROUP_SIZE,
    apply_hadamard_rotation,
)


class TestGptqPoc(unittest.TestCase):
    def test_root_uses_metadata_names_in_sorted_subfolders(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for folder_name, tensor_name in (
                ("z_folder", "blocks.4.attn.wq.weight"),
                ("a_folder", "blocks.14.mlp.gate.weight"),
            ):
                folder = root / folder_name
                folder.mkdir()
                (folder / "gptq_poc_one.json").write_text(
                    f'{{"weight_name":"{tensor_name}"}}', encoding="utf-8"
                )
            calls = []

            def record(args: argparse.Namespace, pairs: object) -> tuple[list[float], float, float, float, float, int]:
                calls.append((args.gram_folder.name, args.tensor, args.target_tensor))
                return [1.0] * 4, 1.0, 1.0, 1.0, 1.0, 4

            with patch("scripts.gptq_poc.discover_gram_pairs", return_value=[("sample", torch.eye(1), 1)]), patch(
                "scripts.gptq_poc.run_leave_one_out", side_effect=record
            ), redirect_stdout(io.StringIO()):
                run_gram_root(argparse.Namespace(
                    gram_root=root, sibling_targets=True, all_targets=False,
                    compare_bd256=False, compare_bd_sweep=False,
                ))
            self.assertEqual(
                [(folder, target) for folder, _, target in calls],
                [
                    ("a_folder", "blocks.14.mlp.up.weight"),
                    ("z_folder", "blocks.4.attn.wk.weight"),
                    ("z_folder", "blocks.4.attn.wv.weight"),
                    ("z_folder", "blocks.4.attn.gate.weight"),
                ],
            )
            self.assertTrue(all(source.endswith("gate.weight") if folder == "a_folder" else source.endswith("wq.weight") for folder, source, _ in calls))

            calls.clear()
            with patch("scripts.gptq_poc.discover_gram_pairs", return_value=[("sample", torch.eye(1), 1)]), patch(
                "scripts.gptq_poc.run_leave_one_out", side_effect=record
            ), redirect_stdout(io.StringIO()):
                run_gram_root(argparse.Namespace(
                    gram_root=root, sibling_targets=False, all_targets=True,
                    compare_bd256=False, compare_bd_sweep=False,
                ))
            self.assertEqual(
                [target for _, _, target in calls],
                ["blocks.14.mlp.gate.weight", "blocks.14.mlp.up.weight",
                 "blocks.4.attn.wq.weight", "blocks.4.attn.wk.weight",
                 "blocks.4.attn.wv.weight", "blocks.4.attn.gate.weight"],
            )

    def test_gram_reuse_requires_verified_source_target_pair(self) -> None:
        validate_gram_target("blocks.4.attn.wq.weight", "blocks.4.attn.wk.weight")
        validate_gram_target("blocks.14.mlp.gate.weight", "blocks.14.mlp.up.weight")
        with self.assertRaisesRegex(ValueError, "not verified"):
            validate_gram_target("blocks.4.attn.wq.weight", "blocks.24.attn.wk.weight")

    def test_each_fold_sums_only_other_raw_grams_and_rows(self) -> None:
        pairs = [
            (str(index), torch.eye(2) * value, rows)
            for index, (value, rows) in enumerate(((1, 3), (2, 5), (4, 7), (8, 11)))
        ]
        for held_out in range(4):
            gram, rows = fold_training_gram(pairs, held_out)
            self.assertTrue(torch.equal(gram, torch.eye(2) * (15 - 2**held_out)))
            self.assertEqual(rows, 26 - pairs[held_out][2])

    def test_native_block_artifact_contract_and_raw_fold_sum(self) -> None:
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            blocks = torch.zeros(2, 512, 512)
            blocks[0] = torch.eye(512)
            blocks[1, :256, :256] = torch.eye(256) * 2
            metadata = {
                "weight_name": "blocks.14.mlp.down.weight", "in_features": 768,
                "row_count": 3, "dtype": "float32",
                "activation_basis": "logical_linear_input", "gram_mode": "block_diag",
                "gram_block_size": 512, "num_blocks": 2, "last_block_size": 256,
            }
            json_path = folder / "gptq_poc_one.json"
            tensor_path = folder / "gptq_poc_one.safetensors"
            json_path.write_text(json.dumps(metadata), encoding="utf-8")
            save_file({"gram_blocks": blocks}, tensor_path)
            gram, rows = load_gram(json_path, tensor_path, metadata["weight_name"])
            self.assertIsInstance(gram, GramBlocks)
            self.assertEqual(rows, 3)
            self.assertEqual([block.shape[0] for block in rotate_saved_blocks(gram)], [512, 256])
            pairs = [(str(index), gram, rows) for index in range(4)]
            train, train_rows = fold_training_gram(pairs, 0)
            self.assertEqual(train_rows, 9)
            self.assertTrue(torch.equal(train.blocks, blocks * 3))
            metadata["last_block_size"] = 255
            json_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "last_block_size"):
                load_gram(json_path, tensor_path, metadata["weight_name"])

    def test_native_block_metric_sums_held_out_blocks(self) -> None:
        reference = torch.ones(1, 768)
        candidate = reference.clone()
        candidate[:, 0] = 0
        candidate[:, 512] = 0
        blocks = (torch.eye(512), torch.eye(256) * 3)
        ratio = score_block_candidates(reference, [candidate], blocks)[0]
        self.assertAlmostEqual(ratio, 4 / (512 + 256 * 3))

    def test_native_blocks_run_four_held_out_folds(self) -> None:
        torch.manual_seed(19)
        weights = torch.randn(2, 768).to(torch.bfloat16)
        pairs = []
        for index in range(4):
            blocks = torch.zeros(2, 512, 512)
            blocks[0] = torch.eye(512) * (index + 1)
            blocks[1, :256, :256] = torch.eye(256) * (index + 1)
            pairs.append((str(index), GramBlocks(blocks, 768, 512, 256), index + 1))
        args = argparse.Namespace(
            tensor="blocks.14.mlp.down.weight", target_tensor=None, model=Path("unused"),
            damp_percent=0.01, gptq_blocksize=128,
        )
        output = io.StringIO()
        with patch("scripts.gptq_poc.load_weight", return_value=weights), torch.no_grad(), redirect_stdout(output):
            gains, *_ = run_saved_block_leave_one_out(args, pairs)
        self.assertEqual(len(gains), 4)
        self.assertIn("block_diag_512_relative_output_sse", output.getvalue())
        self.assertIn("positive folds", output.getvalue())

    def test_folder_requires_four_same_stem_pairs(self) -> None:
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "gptq_poc_orphan.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "same-stem"):
                discover_gram_pairs(folder, "selected.weight")
            (folder / "gptq_poc_orphan.safetensors").touch()
            with self.assertRaisesRegex(ValueError, "exactly four"):
                discover_gram_pairs(folder, "selected.weight")

    def test_combined_raw_gram_scores_fixed_candidate(self) -> None:
        reference = torch.tensor([[1.0, 1.0]])
        candidate = torch.tensor([[0.0, 1.0]])
        first = torch.diag(torch.tensor([10.0, 1.0]))
        second = torch.eye(2)
        first_ratio = score_candidates(reference, [candidate], first)[0][1]
        second_ratio = score_candidates(reference, [candidate], second)[0][1]
        combined_ratio = score_candidates(reference, [candidate], first + second)[0][1]
        self.assertAlmostEqual(combined_ratio, 11 / 13)
        self.assertNotAlmostEqual(combined_ratio, (first_ratio + second_ratio) / 2)
        self.assertTrue(torch.equal(candidate, torch.tensor([[0.0, 1.0]])))

    def test_gram_rejects_different_weight_name(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "gram.json"
            path.write_text('{"activation_basis":"logical_linear_input","weight_name":"other.weight","in_features":6144,"row_count":3}')
            with self.assertRaisesRegex(ValueError, "weight_name"):
                load_gram(path, Path(directory) / "unused.safetensors", "selected.weight")

    def test_selected_bf16_weight_reads_without_mmap(self) -> None:
        weights = torch.arange(6144, dtype=torch.float32).to(torch.bfloat16)[None, :]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_file({"selected.weight": weights}, path)
            self.assertTrue(torch.equal(load_weight(path, "selected.weight"), weights))

    def test_identity_gram_keeps_the_existing_mse_quantizer(self) -> None:
        torch.manual_seed(7)
        weights = torch.randn(2, 256)
        scales = select_w4a4_mse_scale(weights)
        codes, stored_scales = _quantize_w4_codes(weights, scales)
        expected = codes.float() * stored_scales
        actual = gptq_quantize(weights, gptq_factor(torch.eye(256), 0.01), scales, 37)
        self.assertTrue(torch.equal(actual, expected))

    def test_correlated_gram_feedback_reduces_output_error(self) -> None:
        weights = torch.tensor([[0.49, 0.49]])
        gram = torch.tensor([[1.0, 0.9], [0.9, 1.0]])
        scales = torch.ones(1, 1)
        quantized = gptq_quantize(weights, gptq_factor(gram, 0.01), scales, 1)
        self.assertTrue(torch.equal(quantized, torch.tensor([[0.0, 1.0]])))
        self.assertLess(output_energy(weights - quantized, gram), output_energy(weights, gram))

    def test_bd256_discards_cross_group_compensation_but_scores_on_full_gram(self) -> None:
        weights = torch.zeros(1, 512)
        weights[0, 0] = weights[0, 256] = 0.49
        gram = torch.eye(512)
        gram[0, 256] = gram[256, 0] = 0.9
        scales = torch.ones(1, 1)
        full = gptq_quantize(weights, gptq_factor(gram, 0.01), scales, 128)
        bd256 = gptq_quantize_block_diagonal(weights, gram, scales, 0.01, 128, 256)
        bd512 = gptq_quantize_block_diagonal(weights, gram, scales, 0.01, 128, 512)
        self.assertEqual(full[0, 256].item(), 1.0)
        self.assertEqual(bd256[0, 256].item(), 0.0)
        self.assertTrue(torch.equal(full, bd512))
        full_sse, bd256_sse = score_candidates(weights, [full, bd256], gram)
        self.assertLess(full_sse[1], bd256_sse[1])

    def test_block_diagonal_quantizes_final_partial_block(self) -> None:
        weights = torch.zeros(1, 600)
        weights[0, 599] = 0.7
        gram = torch.eye(600)
        quantized = gptq_quantize_block_diagonal(weights, gram, torch.ones(1, 1), 0.01, 128, 512)
        self.assertEqual(quantized.shape, weights.shape)
        self.assertEqual(quantized[0, 599].item(), 1.0)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            gptq_quantize_block_diagonal(weights, torch.eye(512), torch.ones(1, 1), 0.01, 128, 512)

    def test_full_gram_metric_is_rotation_invariant(self) -> None:
        torch.manual_seed(11)
        sample_x = torch.randn(35, 256)
        gram = sample_x.T @ sample_x
        reference = torch.randn(3, 256)
        quantized = reference + 0.1 * torch.randn_like(reference)
        logical_error = reference - quantized
        rotated_error = apply_hadamard_rotation(logical_error, CONVROT_GROUP_SIZE)
        rotated_gram = rotate_gram(gram)
        exact = (sample_x @ logical_error.T).square().sum().item()
        self.assertAlmostEqual(output_energy(logical_error, gram), exact, delta=0.001)
        self.assertAlmostEqual(output_energy(rotated_error, rotated_gram), exact, delta=0.001)


if __name__ == "__main__":
    unittest.main()
