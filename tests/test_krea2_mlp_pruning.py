import json
import unittest
from math import prod
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file
from typer.testing import CliRunner

from potatoforge.cli import app
from potatoforge.headers.source_header import SourceModelHeader, read_source_model_header
from potatoforge.krea2_mlp_pruning import (
    Krea2MLPPrunePlan,
    Krea2MLPPruneResult,
    apply_krea2_mlp_plan,
    build_krea2_mlp_weight_stats,
    build_krea2_mlp_prune_plan,
    krea2_mlp_plan_sha256,
    aggregate_krea2_mlp_activation_stats,
    discover_krea2_mlp_activation_stats,
    load_krea2_mlp_activation_stats,
    load_krea2_mlp_weight_stats,
    load_krea2_mlp_cascade_audit,
    render_krea2_mlp_cascade_audit,
    select_krea2_mlp_plan,
    validate_krea2_mlp_plan,
    prune_down,
    prune_gate_or_up,
    prune_krea2_mlp_checkpoint,
    validate_prune_request,
)


def _descriptor(shape: tuple[int, ...], dtype: str = "BF16") -> dict[str, object]:
    return {
        "dtype": dtype,
        "shape": list(shape),
        "data_offsets": [0, prod(shape) * {"BF16": 2, "F16": 2, "F32": 4}.get(dtype, 1)],
    }


def _header(
    *,
    block_count: int = 2,
    mlp_width: int = 512,
    hidden_width: int = 4,
    dtype: str = "BF16",
) -> SourceModelHeader:
    tensors: dict[str, dict[str, object]] = {}
    for block_index in range(block_count):
        tensors[f"blocks.{block_index}.mlp.gate.weight"] = _descriptor(
            (mlp_width, hidden_width), dtype
        )
        tensors[f"blocks.{block_index}.mlp.up.weight"] = _descriptor(
            (mlp_width, hidden_width), dtype
        )
        tensors[f"blocks.{block_index}.mlp.down.weight"] = _descriptor(
            (hidden_width, mlp_width), dtype
        )
    return SourceModelHeader(tensors=tensors, metadata={"creator": "test"})  # type: ignore[arg-type]


def _saved_tensor(shape: tuple[int, ...], offset: int) -> torch.Tensor:
    return torch.arange(
        offset,
        offset + prod(shape),
        dtype=torch.float32,
    ).reshape(shape).to(torch.bfloat16)


def _write_activation_pair(
    root: Path,
    stem: str,
    sum_squared_activation: torch.Tensor,
    observation_count: torch.Tensor,
    *,
    base_plan_hash: object = "none",
    source_model_basename: str = "quantized.safetensors",
    source_model_fingerprint: str | None = None,
    metric: str | None = None,
    metadata_overrides: dict[str, object] | None = None,
) -> Path:
    metadata: dict[str, object] = {
        "format": "potatoforge_krea2_mlp_activation_stats",
        "version": "1",
        "architecture": "krea2",
        "block_count": str(sum_squared_activation.shape[0]),
        "main_hidden_width": "1",
        "original_mlp_width": str(sum_squared_activation.shape[1]),
        "base_plan_hash": base_plan_hash,
        "source_model_basename": source_model_basename,
    }
    if source_model_fingerprint is not None:
        metadata["source_model_fingerprint"] = source_model_fingerprint
    if metric is not None:
        metadata["metric"] = metric
    if metadata_overrides:
        metadata.update(metadata_overrides)
    tensor_path = root / f"{stem}.safetensors"
    save_file(
        {
            "sum_squared_activation": sum_squared_activation,
            "observation_count": observation_count,
        },
        str(tensor_path),
    )
    json_path = root / f"{stem}.json"
    json_path.write_text(json.dumps(metadata), encoding="utf-8")
    return json_path


class TestKrea2MLPPruning(unittest.TestCase):
    def test_activation_stats_directory_sums_raw_totals_before_mean(self) -> None:
        metadata = {
            "format": "potatoforge_krea2_mlp_activation_stats",
            "version": "1",
            "architecture": "krea2",
            "block_count": "1",
            "main_hidden_width": "1",
            "original_mlp_width": "1",
            "base_plan_hash": "none",
            "source_model_basename": "quantized.safetensors",
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name, total, count in (("a", 4, 1), ("b", 18, 9)):
                save_file(
                    {
                        "sum_squared_activation": torch.tensor([[total]], dtype=torch.float32),
                        "observation_count": torch.tensor([count], dtype=torch.int64),
                    },
                    str(root / f"{name}.safetensors"),
                )
                (root / f"{name}.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )

            stats = aggregate_krea2_mlp_activation_stats(root)

            self.assertEqual(stats.sum_squared_activation.tolist(), [[22.0]])
            self.assertEqual(stats.observation_count.tolist(), [10.0])
            self.assertAlmostEqual(
                float(
                    (
                        stats.sum_squared_activation
                        / stats.observation_count[:, None]
                    )[0, 0]
                ),
                2.2,
            )

    def test_directory_and_manual_merge_produce_identical_plan(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.safetensors"
            weight_stats = root / "weight-stats.safetensors"
            calibration = root / "calibration"
            calibration.mkdir()
            tensors = {}
            for block_index in range(2):
                tensors[f"blocks.{block_index}.mlp.gate.weight"] = torch.zeros(
                    (4, 1), dtype=torch.float32
                )
                tensors[f"blocks.{block_index}.mlp.up.weight"] = torch.zeros(
                    (4, 1), dtype=torch.float32
                )
                tensors[f"blocks.{block_index}.mlp.down.weight"] = torch.ones(
                    (1, 4), dtype=torch.float32
                )
            save_file(tensors, str(source))
            build_krea2_mlp_weight_stats(source, weight_stats)
            first_sum = torch.tensor([[4, 1, 2, 3], [3, 2, 1, 4]], dtype=torch.float32)
            second_sum = torch.tensor([[8, 6, 4, 2], [6, 4, 2, 8]], dtype=torch.float32)
            first_count = torch.tensor([1, 2], dtype=torch.int64)
            second_count = torch.tensor([3, 4], dtype=torch.int64)
            _write_activation_pair(
                calibration,
                "z-calibration",
                first_sum,
                first_count,
                source_model_fingerprint="header-a",
            )
            _write_activation_pair(
                calibration,
                "a-calibration",
                second_sum,
                second_count,
                source_model_fingerprint="header-a",
            )
            merged = _write_activation_pair(
                root,
                "merged-calibration",
                first_sum + second_sum,
                first_count + second_count,
                source_model_fingerprint="header-a",
            )
            (calibration / "unrelated.json").write_text(
                json.dumps({"format": "not_krea2_activation_stats"}), encoding="utf-8"
            )
            (calibration / "notes.txt").write_text("ignore me", encoding="utf-8")

            self.assertEqual(
                [path.name for path in discover_krea2_mlp_activation_stats(calibration)],
                ["a-calibration.json", "z-calibration.json"],
            )
            discovered = aggregate_krea2_mlp_activation_stats(calibration)
            reverse = aggregate_krea2_mlp_activation_stats(
                tuple(reversed(discover_krea2_mlp_activation_stats(calibration)))
            )
            directory_plan = root / "directory-plan.json"
            merged_plan = root / "merged-plan.json"
            select_krea2_mlp_plan(
                calibration,
                weight_stats,
                directory_plan,
                add_channels_per_block=1,
            )
            select_krea2_mlp_plan(
                merged,
                weight_stats,
                merged_plan,
                add_channels_per_block=1,
            )
            cli_plan = root / "cli-plan.json"
            cli_result = CliRunner().invoke(
                app,
                [
                    "prune",
                    "krea2-mlp",
                    "select",
                    "--activation-stats",
                    str(calibration),
                    "--weight-stats",
                    str(weight_stats),
                    "--add-channels-per-block",
                    "1",
                    "--output-plan",
                    str(cli_plan),
                ],
            )
            self.assertEqual(cli_result.exit_code, 0, cli_result.stdout)
            self.assertIn("Calibration artifacts:       2", cli_result.stdout)
            self.assertIn("per-file means are never averaged", cli_result.stdout)

            self.assertEqual(
                json.loads(directory_plan.read_text(encoding="utf-8")),
                json.loads(merged_plan.read_text(encoding="utf-8")),
            )
            self.assertEqual(discovered.observation_count.tolist(), [4.0, 6.0])
            self.assertTrue(torch.equal(discovered.sum_squared_activation, reverse.sum_squared_activation))
            self.assertTrue(torch.equal(discovered.observation_count, reverse.observation_count))

    def test_directory_rejects_base_model_and_fingerprint_mismatches(self) -> None:
        cases = (
            ("base_plan_hash", {"base_plan_hash": "wrong"}, "base_plan_hash"),
            (
                "source_model_basename",
                {"source_model_basename": "other.safetensors"},
                "source_model_basename",
            ),
            (
                "source_model_fingerprint",
                {"source_model_fingerprint": "header-b"},
                "source_model_fingerprint",
            ),
        )
        for _, overrides, expected_message in cases:
            with self.subTest(expected_message=expected_message), TemporaryDirectory() as directory:
                root = Path(directory)
                base_kwargs = {"source_model_fingerprint": "header-a"}
                _write_activation_pair(
                    root,
                    "a",
                    torch.ones((1, 2), dtype=torch.float32),
                    torch.ones(1, dtype=torch.int64),
                    **base_kwargs,
                )
                changed_kwargs = dict(base_kwargs)
                changed_kwargs.update(overrides)
                _write_activation_pair(
                    root,
                    "b",
                    torch.ones((1, 2), dtype=torch.float32),
                    torch.ones(1, dtype=torch.int64),
                    **changed_kwargs,
                )
                with self.assertRaisesRegex(ValueError, expected_message):
                    aggregate_krea2_mlp_activation_stats(root)

    def test_directory_rejects_shape_mismatch_and_missing_pair(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_activation_pair(
                root,
                "a",
                torch.ones((1, 2), dtype=torch.float32),
                torch.ones(1, dtype=torch.int64),
            )
            _write_activation_pair(
                root,
                "b",
                torch.ones((1, 3), dtype=torch.float32),
                torch.ones(1, dtype=torch.int64),
                metadata_overrides={"original_mlp_width": "2"},
            )
            with self.assertRaisesRegex(ValueError, "shape"):
                aggregate_krea2_mlp_activation_stats(root)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "missing.json").write_text(
                json.dumps(
                    {
                        "format": "potatoforge_krea2_mlp_activation_stats",
                        "version": "1",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FileNotFoundError, "matching"):
                aggregate_krea2_mlp_activation_stats(root)

    def test_directory_ignores_unrelated_files_and_rejects_duplicate_paths(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            valid = _write_activation_pair(
                root,
                "valid",
                torch.ones((1, 2), dtype=torch.float32),
                torch.ones(1, dtype=torch.int64),
            )
            (root / "unrelated.json").write_text(
                json.dumps({"format": "other"}), encoding="utf-8"
            )
            save_file({"other": torch.ones(1)}, str(root / "unrelated.safetensors"))
            self.assertEqual(
                aggregate_krea2_mlp_activation_stats(root).sum_squared_activation.tolist(),
                [[1.0, 1.0]],
            )
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                aggregate_krea2_mlp_activation_stats((valid, valid))

    def test_activation_stats_reject_nonfinite_sum(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "activation-stats.safetensors"
            save_file(
                {
                    "sum_squared_activation": torch.tensor([[float("nan")]], dtype=torch.float32),
                    "observation_count": torch.ones(1, dtype=torch.int64),
                },
                str(path),
                metadata={
                    "format": "potatoforge_krea2_mlp_activation_stats",
                    "version": "1",
                    "architecture": "krea2",
                    "block_count": "1",
                    "main_hidden_width": "1",
                    "original_mlp_width": "1",
                    "base_plan_hash": "null",
                    "source_model_basename": "quantized.safetensors",
                },
            )
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_krea2_mlp_activation_stats(path)

    def test_activation_stats_normalise_null_base_plan_hash(self) -> None:
        with TemporaryDirectory() as directory:
            path = _write_activation_pair(
                Path(directory),
                "activation",
                torch.ones((1, 1), dtype=torch.float32),
                torch.ones(1, dtype=torch.int64),
                base_plan_hash=None,
            )
            self.assertEqual(load_krea2_mlp_activation_stats(path).base_plan_hash, "none")

    def test_activation_stats_reject_zero_observation_count(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "activation-stats.safetensors"
            save_file(
                {
                    "sum_squared_activation": torch.ones((2, 8), dtype=torch.float32),
                    "observation_count": torch.tensor([1, 0], dtype=torch.int64),
                },
                str(path),
                metadata={
                    "format": "potatoforge_krea2_mlp_activation_stats",
                    "version": "1",
                    "architecture": "krea2",
                    "block_count": "2",
                    "main_hidden_width": "1",
                    "original_mlp_width": "8",
                    "base_plan_hash": "none",
                    "source_model_basename": "quantized.safetensors",
                },
            )

            with self.assertRaisesRegex(ValueError, "positive finite"):
                load_krea2_mlp_activation_stats(path)

    def test_plan_validator_rejects_duplicate_and_missing_blocks(self) -> None:
        plan = {
            "format": "potatoforge_krea2_mlp_prune_plan",
            "version": 1,
            "architecture": "krea2",
            "original_mlp_width": 8,
            "main_hidden_width": 1,
            "block_count": 2,
            "removed_per_block": 2,
            "new_mlp_width": 6,
            "blocks": {"0": [1, 1], "1": [2, 3]},
        }
        with self.assertRaisesRegex(ValueError, "unique and sorted"):
            validate_krea2_mlp_plan(plan)

        plan["blocks"] = {"0": [1, 2]}
        with self.assertRaisesRegex(ValueError, "block count"):
            validate_krea2_mlp_plan(plan)

    def test_cascade_audit_summary_reads_numeric_acceptance_metrics(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cascade-audit.json"
            path.write_text(
                json.dumps(
                    {
                        "candidate_plan_hash": "abc",
                        "forward_count": 28,
                        "relative_sse": 0.25,
                        "relative_rmse": 0.5,
                        "mean_per_forward_relative_sse": 0.01,
                        "max_per_forward_relative_sse": 0.03,
                    }
                ),
                encoding="utf-8",
            )

            audit = load_krea2_mlp_cascade_audit(path)

            self.assertEqual(audit.candidate_plan_hash, "abc")
            self.assertEqual(audit.forward_count, 28)
            self.assertIn("Relative SSE:                 0.25", render_krea2_mlp_cascade_audit(audit))

    def test_plan_hash_ignores_json_key_order(self) -> None:
        first = {
            "format": "potatoforge_krea2_mlp_prune_plan",
            "version": 1,
            "architecture": "krea2",
            "original_mlp_width": 8,
            "main_hidden_width": 1,
            "block_count": 1,
            "removed_per_block": 2,
            "new_mlp_width": 6,
            "blocks": {"0": [1, 4]},
        }
        second = json.loads(json.dumps(first, indent=2))
        second["blocks"] = {"0": second["blocks"]["0"]}
        second = {key: second[key] for key in reversed(list(second))}
        self.assertEqual(krea2_mlp_plan_sha256(first), krea2_mlp_plan_sha256(second))

    def test_apply_plan_uses_each_blocks_original_channel_ids(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.safetensors"
            plan_path = root / "plan.json"
            output = root / "pruned.safetensors"
            tensors = {}
            for block_index in range(2):
                base = block_index * 100
                tensors[f"blocks.{block_index}.mlp.gate.weight"] = torch.arange(
                    base,
                    base + 8,
                    dtype=torch.float32,
                ).reshape(8, 1)
                tensors[f"blocks.{block_index}.mlp.up.weight"] = torch.arange(
                    base + 20,
                    base + 28,
                    dtype=torch.float32,
                ).reshape(8, 1)
                tensors[f"blocks.{block_index}.mlp.down.weight"] = torch.arange(
                    base + 40,
                    base + 48,
                    dtype=torch.float32,
                ).reshape(1, 8)
            save_file(tensors, str(source))
            plan_path.write_text(
                json.dumps(
                    {
                        "format": "potatoforge_krea2_mlp_prune_plan",
                        "version": 1,
                        "architecture": "krea2",
                        "original_mlp_width": 8,
                        "main_hidden_width": 1,
                        "block_count": 2,
                        "block_indices": [0, 1],
                        "removed_per_block": 2,
                        "new_mlp_width": 6,
                        "stage": 1,
                        "blocks": {"0": [1, 4], "1": [0, 6]},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "divisible"):
                apply_krea2_mlp_plan(source, plan_path, output)
            self.assertFalse(output.exists())

            apply_krea2_mlp_plan(source, plan_path, output, group_size=2)
            actual = load_file(str(output), device="cpu")
            self.assertTrue(
                torch.equal(
                    actual["blocks.0.mlp.gate.weight"].flatten(),
                    torch.tensor([0, 2, 3, 5, 6, 7], dtype=torch.float32),
                )
            )
            self.assertTrue(
                torch.equal(
                    actual["blocks.1.mlp.down.weight"].flatten(),
                    torch.tensor([141, 142, 143, 144, 145, 147], dtype=torch.float32),
                )
            )

    def test_selector_ranks_each_block_and_writes_original_ids(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.safetensors"
            weight_stats = root / "weight-stats.safetensors"
            activation_stats = root / "activation-stats.safetensors"
            plan_path = root / "stage1-plan.json"
            tensors = {}
            for block_index in range(2):
                tensors[f"blocks.{block_index}.mlp.gate.weight"] = torch.zeros(
                    (8, 1), dtype=torch.float32
                )
                tensors[f"blocks.{block_index}.mlp.up.weight"] = torch.zeros(
                    (8, 1), dtype=torch.float32
                )
                tensors[f"blocks.{block_index}.mlp.down.weight"] = torch.ones(
                    (1, 8), dtype=torch.float32
                )
            save_file(tensors, str(source))
            build_krea2_mlp_weight_stats(source, weight_stats)
            save_file(
                {
                    "sum_squared_activation": torch.tensor(
                        [[5, 1, 2, 3, 4, 6, 7, 8], [2, 1, 3, 4, 5, 6, 7, 8]],
                        dtype=torch.float32,
                    ),
                    "observation_count": torch.ones(2, dtype=torch.int64),
                },
                str(activation_stats),
                metadata={
                    "format": "potatoforge_krea2_mlp_activation_stats",
                    "version": "1",
                    "architecture": "krea2",
                    "block_count": "2",
                    "main_hidden_width": "1",
                    "original_mlp_width": "8",
                    "base_plan_hash": "none",
                    "source_model_basename": "quantized.safetensors",
                },
            )

            select_krea2_mlp_plan(
                activation_stats,
                weight_stats,
                plan_path,
                add_channels_per_block=2,
            )
            plan = json.loads(plan_path.read_text(encoding="utf-8"))

            self.assertEqual(plan["blocks"], {"0": [1, 2], "1": [0, 1]})
            self.assertEqual(plan["removed_per_block"], 2)
            self.assertEqual(plan["new_mlp_width"], 6)
            self.assertEqual(plan["selection"]["parent_plan_sha256"], "none")

    def test_selector_rejects_stage_lineage_hash_mismatch(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.safetensors"
            weight_stats = root / "weight-stats.safetensors"
            activation_stats = root / "activation-stats.safetensors"
            base_plan = root / "stage1-plan.json"
            output_plan = root / "stage2-plan.json"
            tensors = {
                "blocks.0.mlp.gate.weight": torch.zeros((4, 1), dtype=torch.float32),
                "blocks.0.mlp.up.weight": torch.zeros((4, 1), dtype=torch.float32),
                "blocks.0.mlp.down.weight": torch.ones((1, 4), dtype=torch.float32),
            }
            save_file(tensors, str(source))
            build_krea2_mlp_weight_stats(source, weight_stats)
            base_document = {
                "format": "potatoforge_krea2_mlp_prune_plan",
                "version": 1,
                "architecture": "krea2",
                "original_mlp_width": 4,
                "main_hidden_width": 1,
                "block_count": 1,
                "block_indices": [0],
                "removed_per_block": 2,
                "new_mlp_width": 2,
                "stage": 1,
                "blocks": {"0": [0, 3]},
            }
            base_plan.write_text(json.dumps(base_document), encoding="utf-8")
            save_file(
                {
                    "sum_squared_activation": torch.ones((1, 4), dtype=torch.float32),
                    "observation_count": torch.ones(1, dtype=torch.int64),
                },
                str(activation_stats),
                metadata={
                    "format": "potatoforge_krea2_mlp_activation_stats",
                    "version": "1",
                    "architecture": "krea2",
                    "block_count": "1",
                    "main_hidden_width": "1",
                    "original_mlp_width": "4",
                    "base_plan_hash": "wrong",
                    "source_model_basename": "quantized.safetensors",
                },
            )

            with self.assertRaisesRegex(ValueError, "base_plan_hash"):
                select_krea2_mlp_plan(
                    activation_stats,
                    weight_stats,
                    output_plan,
                    base_plan_path=base_plan,
                    add_channels_per_block=1,
                )

    def test_weight_stats_store_exact_down_column_norms(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.safetensors"
            output = root / "weight-stats.safetensors"
            tensors = {
                "blocks.0.mlp.gate.weight": torch.zeros((4, 2), dtype=torch.bfloat16),
                "blocks.0.mlp.up.weight": torch.zeros((4, 2), dtype=torch.bfloat16),
                "blocks.0.mlp.down.weight": torch.tensor(
                    [[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32
                ),
            }
            save_file(tensors, str(source))

            build_krea2_mlp_weight_stats(source, output)
            stats = load_krea2_mlp_weight_stats(output)

            self.assertTrue(
                torch.equal(
                    stats.down_column_norm_sq,
                    torch.tensor([[26, 40, 58, 80]], dtype=torch.float32),
                )
            )

    def test_gate_up_and_down_remove_the_same_interior_range(self) -> None:
        gate = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        down = torch.arange(32, dtype=torch.float32).reshape(4, 8)

        expected_rows = torch.tensor(
            [0, 1, 5, 6, 7],
            dtype=torch.float32,
        )
        expected_gate_first_column = torch.tensor(
            [0, 4, 20, 24, 28],
            dtype=torch.float32,
        )
        self.assertTrue(
            torch.equal(
                prune_gate_or_up(gate, 2, 5)[:, 0],
                expected_gate_first_column,
            )
        )
        self.assertTrue(
            torch.equal(
                prune_down(down, 2, 5)[0],
                expected_rows,
            )
        )

    def test_gate_up_and_down_support_prefix_and_suffix_ranges(self) -> None:
        tensor = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        self.assertEqual(
            prune_gate_or_up(tensor, 0, 2).shape,
            (6, 4),
        )
        self.assertEqual(
            prune_gate_or_up(tensor, 6, 8).shape,
            (6, 4),
        )

    def test_plan_discovers_blocks_and_validates_shapes(self) -> None:
        plan = build_krea2_mlp_prune_plan(_header(), 128, 256)

        self.assertEqual(plan.block_indices, (0, 1))
        self.assertEqual(plan.hidden_width, 4)
        self.assertEqual(plan.original_mlp_width, 512)
        self.assertEqual(plan.end_channel, 384)
        self.assertEqual(plan.new_mlp_width, 256)

    def test_plan_rejects_missing_member_and_shape_mismatches(self) -> None:
        missing = _header()
        missing.tensors.pop("blocks.1.mlp.down.weight")
        with self.assertRaisesRegex(ValueError, "missing down"):
            build_krea2_mlp_prune_plan(missing, 128, 256)

        mismatched = _header()
        mismatched.tensors["blocks.1.mlp.up.weight"] = _descriptor((256, 4))
        with self.assertRaisesRegex(ValueError, "gate/up shapes must match"):
            build_krea2_mlp_prune_plan(mismatched, 128, 256)

        bad_down = _header()
        bad_down.tensors["blocks.0.mlp.down.weight"] = _descriptor((3, 512))
        with self.assertRaisesRegex(ValueError, "down shape"):
            build_krea2_mlp_prune_plan(bad_down, 128, 256)

    def test_range_validation_rejects_invalid_requests(self) -> None:
        invalid = (
            (-1, 256, "non-negative"),
            (0, 0, "greater than zero"),
            (512, 256, "less than the MLP width"),
            (256, 512, "exceeds the MLP width"),
            (0, 512, "at least one"),
            (0, 1, "divisible"),
        )
        for start, count, message in invalid:
            with self.subTest(start=start, count=count):
                with self.assertRaisesRegex(ValueError, message):
                    validate_prune_request(512, start, count)

    def test_source_format_rejects_integer_and_quantized_target_weights(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            integer_source = root / "integer.safetensors"
            integer_tensors = {
                name: torch.zeros(tuple(descriptor["shape"]), dtype=torch.uint8)
                for name, descriptor in _header(block_count=1, mlp_width=256).tensors.items()
            }
            save_file(integer_tensors, str(integer_source))
            with self.assertRaisesRegex(ValueError, "Prune before quantization"):
                prune_krea2_mlp_checkpoint(
                    integer_source,
                    root / "integer-output.safetensors",
                    0,
                    256,
                )

            quantized_source = root / "quantized.safetensors"
            quantized_tensors = {
                name: torch.zeros(tuple(descriptor["shape"]), dtype=torch.bfloat16)
                for name, descriptor in _header(block_count=1, mlp_width=256).tensors.items()
            }
            quantized_tensors["blocks.0.mlp.gate.weight_scale"] = torch.ones(
                (256, 1), dtype=torch.float32
            )
            save_file(quantized_tensors, str(quantized_source))
            with self.assertRaisesRegex(ValueError, "Prune before quantization"):
                prune_krea2_mlp_checkpoint(
                    quantized_source,
                    root / "quantized-output.safetensors",
                    0,
                    256,
                )

            collision_source = root / "collision.safetensors"
            collision_tensors = {
                name: torch.zeros(tuple(descriptor["shape"]), dtype=torch.bfloat16)
                for name, descriptor in _header(block_count=1, mlp_width=512).tensors.items()
            }
            collision_output = root / "collision-output.safetensors"
            save_file(
                collision_tensors,
                str(collision_source),
                metadata={"potatoforge.pruning": "old"},
            )
            with self.assertRaisesRegex(ValueError, "provenance keys"):
                prune_krea2_mlp_checkpoint(
                    collision_source,
                    collision_output,
                    0,
                    256,
                )
            self.assertFalse(collision_output.exists())

    def test_integration_streams_target_tensors_and_preserves_other_data(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.safetensors"
            output = root / "pruned.safetensors"
            tensors: dict[str, torch.Tensor] = {}
            for block_index in range(2):
                tensors[f"blocks.{block_index}.mlp.gate.weight"] = _saved_tensor(
                    (512, 4), block_index * 8192
                )
                tensors[f"blocks.{block_index}.mlp.up.weight"] = _saved_tensor(
                    (512, 4), block_index * 8192 + 4096
                )
                tensors[f"blocks.{block_index}.mlp.down.weight"] = _saved_tensor(
                    (4, 512), block_index * 8192 + 6144
                )
            tensors["unrelated.tensor"] = _saved_tensor((3, 2), 10000)
            save_file(tensors, str(source), metadata={"creator": "test"})

            result = prune_krea2_mlp_checkpoint(source, output, 64, 256)
            actual = load_file(str(output), device="cpu")

            self.assertEqual(result.target_tensors_modified, 6)
            self.assertEqual(result.other_tensors_copied, 1)
            self.assertEqual(result.plan.new_mlp_width, 256)
            self.assertFalse(output.with_name("pruned.safetensors.partial").exists())
            self.assertEqual(tuple(actual["blocks.0.mlp.gate.weight"].shape), (256, 4))
            self.assertEqual(tuple(actual["blocks.0.mlp.down.weight"].shape), (4, 256))
            self.assertEqual(actual["blocks.0.mlp.gate.weight"].dtype, torch.bfloat16)
            self.assertTrue(
                torch.equal(
                    actual["blocks.0.mlp.gate.weight"],
                    torch.cat((tensors["blocks.0.mlp.gate.weight"][:64], tensors["blocks.0.mlp.gate.weight"][320:]), dim=0),
                )
            )
            self.assertTrue(torch.equal(actual["unrelated.tensor"], tensors["unrelated.tensor"]))

            output_header = read_source_model_header(output)
            self.assertEqual(output_header.metadata["creator"], "test")
            self.assertEqual(output_header.metadata["potatoforge.pruning"], "krea2_mlp_contiguous_v1")
            config = json.loads(output_header.metadata["potatoforge.pruning_config"])
            self.assertEqual(config["block_count"], 2)
            self.assertEqual(config["remove_channels"], 256)

    def test_cli_uses_prune_group_and_required_options(self) -> None:
        plan = Krea2MLPPrunePlan(
            block_indices=(0, 1),
            hidden_width=4,
            original_mlp_width=512,
            start_channel=128,
            remove_channels=256,
            end_channel=384,
            new_mlp_width=256,
            estimated_saving_bytes=6144,
        )
        fake_result = Krea2MLPPruneResult(
            output_path=Path("output.safetensors"),
            plan=plan,
            target_tensors_modified=6,
            other_tensors_copied=1,
            output_bytes=123,
        )
        with patch(
            "potatoforge.cli.prune_krea2_mlp_checkpoint",
            return_value=fake_result,
        ) as prune_mock:
            result = CliRunner().invoke(
                app,
                [
                    "prune",
                    "krea2-mlp",
                    "--model",
                    "source.safetensors",
                    "--output",
                    "output.safetensors",
                    "--start-channel",
                    "128",
                    "--remove-channels",
                    "256",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        prune_mock.assert_called_once()
        self.assertEqual(
            prune_mock.call_args.args,
            (Path("source.safetensors"), Path("output.safetensors"), 128, 256),
        )
        self.assertIn("Pruning completed.", result.output)
        self.assertIn("Target tensors modified: 6", result.output)


if __name__ == "__main__":
    unittest.main()
