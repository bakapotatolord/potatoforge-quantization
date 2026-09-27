import json
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from potatoforge.headers.source_header import SourceModelHeader, read_source_model_header
from potatoforge.krea2_block_pruning import (
    add_krea2_block_to_plan,
    apply_krea2_block_plan,
    krea2_block_old_to_new,
    rank_krea2_blocks,
    remap_krea2_block_profile,
    summarize_krea2_block_audits,
)
from potatoforge.krea2_combined_pruning import (
    _logical_block_pruned_header,
    apply_krea2_combined_prune,
)
from potatoforge.krea2_mlp_pruning import (
    apply_krea2_mlp_plan,
    krea2_mlp_plan_sha256,
    remap_krea2_mlp_plan_blocks,
)
from potatoforge.profiles import load_profile, resolve_profile


def calibration(root, stem, *, removed=(), inputs=(100, 100, 100), residuals=(1, 5, 2)):
    (root / f"{stem}.json").write_text(json.dumps({"format": "potatoforge_krea2_block_calibration", "version": 1, "architecture": "krea2", "block_count": 3, "base_plan_hash": "none", "removed_blocks": list(removed), "active_blocks": [i for i in range(3) if i not in removed], "metric": "relative_residual_energy", "metric_version": 1}), encoding="utf-8")
    save_file({"sum_input_sq": torch.tensor(inputs, dtype=torch.float64), "sum_residual_sq": torch.tensor(residuals, dtype=torch.float64), "observation_count": torch.tensor((10, 10, 10), dtype=torch.int64)}, str(root / f"{stem}.safetensors"))


def audit(root, stem, candidate, mean, forwards):
    (root / f"{stem}.json").write_text(json.dumps({"format": "potatoforge_krea2_block_audit", "version": 2, "architecture": "krea2", "block_count": 3, "base_removed_blocks": [], "candidate_block": candidate, "candidate_removed_blocks": [candidate], "base_plan_hash": "none", "forward_count": forwards, "mean_relative_sse": mean, "total_relative_sse": mean, "max_relative_sse": mean}), encoding="utf-8")


def block_plan(root, removed, block_count=4):
    path = root / "plan.json"
    path.write_text(json.dumps({"format": "potatoforge_krea2_block_prune_plan", "version": 1, "architecture": "krea2", "block_count": block_count, "removed_blocks": list(removed)}), encoding="utf-8")
    return path


def checkpoint(root, block_count=4):
    tensors = {"outside.weight": torch.tensor([99]), "txtfusion.layerwise_blocks.0.weight": torch.tensor([98])}
    for block in range(block_count):
        tensors[f"blocks.{block}.a.weight"] = torch.tensor([[block, block + 1]], dtype=torch.bfloat16)
        tensors[f"blocks.{block}.mlp.unusual.weight"] = torch.tensor([[block]], dtype=torch.float16)
    path = root / "source.safetensors"
    save_file(tensors, str(path), metadata={"foo": "bar"})
    return path, tensors


class TestKrea2BlockPruning(unittest.TestCase):
    def test_raw_sum_rank_and_plan(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            calibration(root, "a", removed=(1,), residuals=(1, 5, 3))
            calibration(root, "b", removed=(1,), inputs=(900, 900, 900), residuals=(18, 50, 4))
            result = rank_krea2_blocks(root, root / "candidates.json", top_k=5)
            self.assertEqual(result["top_k"], [2, 0])
            self.assertAlmostEqual(result["ranking"][1]["score"], 0.019)
            first = add_krea2_block_to_plan(root / "first.json", block=1, block_count=3)
            second = add_krea2_block_to_plan(root / "second.json", block=2, base_plan_path=root / "first.json")
            self.assertEqual(first["removed_blocks"], [1])
            self.assertEqual(second["removed_blocks"], [1, 2])

    def test_audit_weighted_mean(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audit(root, "one", 1, 0.01, 1)
            audit(root, "two", 1, 0.03, 3)
            audit(root, "three", 2, 0.02, 1)
            result = summarize_krea2_block_audits(root, root / "summary.json")
            self.assertEqual([item["candidate_block"] for item in result["ranking"]], [2, 1])
            self.assertAlmostEqual(result["ranking"][1]["mean_relative_sse"], 0.025)

    def test_rejects_incompatible_calibration_and_invalid_plan_add(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            calibration(root, "a")
            calibration(root, "b", removed=(1,))
            with self.assertRaisesRegex(ValueError, "removed_blocks"):
                rank_krea2_blocks(root, root / "out.json")
            with self.assertRaisesRegex(ValueError, "within the original"):
                add_krea2_block_to_plan(root / "bad.json", block=3, block_count=3)

    def test_apply_plan_rewrites_blocks_preserves_payloads_metadata_and_manifest(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, tensors = checkpoint(root)
            result = apply_krea2_block_plan(source, block_plan(root, [1]), root / "pruned.safetensors")
            output = load_file(str(result.output_path))
            self.assertEqual(set(output), {"blocks.0.a.weight", "blocks.0.mlp.unusual.weight", "blocks.1.a.weight", "blocks.1.mlp.unusual.weight", "blocks.2.a.weight", "blocks.2.mlp.unusual.weight", "outside.weight", "txtfusion.layerwise_blocks.0.weight"})
            self.assertTrue(torch.equal(output["blocks.1.a.weight"], tensors["blocks.2.a.weight"]))
            self.assertEqual(output["blocks.1.mlp.unusual.weight"].dtype, torch.float16)
            self.assertTrue(torch.equal(output["outside.weight"], tensors["outside.weight"]))
            with safe_open(str(result.output_path), framework="pt", device="cpu") as handle:
                self.assertEqual(handle.metadata()["foo"], "bar")
                self.assertEqual(handle.metadata()["potatoforge.block_pruning_remaining_block_count"], "3")
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["old_to_new"], {"0": 0, "1": None, "2": 1, "3": 2})
            self.assertEqual(result.removed_tensor_count, 2)

    def test_apply_plan_uses_one_pass_original_ids_and_rejects_wrong_topology(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = checkpoint(root, block_count=6)
            plan = block_plan(root, [1, 4], block_count=6)
            self.assertEqual(krea2_block_old_to_new(plan), {0: 0, 1: None, 2: 1, 3: 2, 4: None, 5: 3})
            result = apply_krea2_block_plan(source, plan, root / "pruned.safetensors")
            self.assertEqual(sorted({int(name.split(".")[1]) for name in load_file(str(result.output_path)) if name.startswith("blocks.")}), [0, 1, 2, 3])
            with self.assertRaisesRegex(ValueError, "original block topology"):
                apply_krea2_block_plan(result.output_path, plan, root / "again.safetensors")

    def test_apply_plan_accepts_documented_original_block_count_fields(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = checkpoint(root)
            plan = root / "plan.json"
            plan.write_text(json.dumps({"format": "potatoforge_krea2_block_prune_plan", "version": 1, "architecture": "krea2", "original_block_count": 4, "remaining_block_count": 3, "removed_blocks": [2]}), encoding="utf-8")
            result = apply_krea2_block_plan(source, plan, root / "pruned.safetensors")
            self.assertEqual(result.removed_blocks, (2,))

    def test_remap_profile_keeps_rule_order_and_surviving_assignments(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "profile.json"
            profile_path.write_text(json.dumps({"format_version": 1, "profile_id": "mixed", "default": "keep", "rules": [
                {"action": "int8", "prefix": "blocks.0.", "suffixes": ["a.weight"]},
                {"action": "int8", "prefix": "blocks.1.", "suffixes": ["a.weight"]},
                {"action": "int8", "prefix": "blocks.2.", "suffixes": ["a.weight"]},
                {"action": "int8", "prefix": "blocks.3.", "suffixes": ["a.weight"]},
                {"action": "int8", "prefix": "blocks.", "suffixes": ["b.weight"]},
                {"action": "int8", "prefix": "conditioner.", "suffixes": ["weight"]}
            ]}), encoding="utf-8")
            plan = block_plan(root, [1])
            source, _ = checkpoint(root)
            pruned = apply_krea2_block_plan(source, plan, root / "pruned.safetensors")
            result = remap_krea2_block_profile(
                profile_path, plan, root / "remapped.json", source_model_path=source,
                pruned_model_path=pruned.output_path,
            )
            self.assertEqual((result.removed_rules, result.rewritten_rules, result.unchanged_rules), (1, 3, 2))
            self.assertEqual(result.verified_assignments, 8)
            original, remapped = load_profile(profile_path), load_profile(result.output_path)
            mapping = krea2_block_old_to_new(plan)
            for old_id, new_id in mapping.items():
                if new_id is not None:
                    self.assertEqual(resolve_profile(original, f"blocks.{old_id}.a.weight"), resolve_profile(remapped, f"blocks.{new_id}.a.weight"))
            self.assertEqual(resolve_profile(remapped, "blocks.1.b.weight"), "int8")
            self.assertEqual(resolve_profile(remapped, "conditioner.x.weight"), "int8")

    def test_dry_run_does_not_write_outputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = checkpoint(root)
            result = apply_krea2_block_plan(source, block_plan(root, [0]), root / "pruned.safetensors", dry_run=True)
            self.assertTrue(result.dry_run)
            self.assertFalse(result.output_path.exists())

    def test_combined_prune_remaps_mlp_and_profile_from_one_mapping(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.safetensors"
            tensors = {
                "conditioner.weight": torch.tensor([1]),
                "txtfusion.layerwise_blocks.0.weight": torch.tensor([2]),
            }
            for block in range(6):
                tensors[f"blocks.{block}.a.weight"] = torch.tensor([block], dtype=torch.float32)
                for component, shape in (("gate", (512, 4)), ("up", (512, 4)), ("down", (4, 512))):
                    tensors[f"blocks.{block}.mlp.{component}.weight"] = (
                        torch.arange(512 * 4, dtype=torch.float32)
                        .reshape(shape)
                        .to(torch.bfloat16) + block
                    )
            save_file(tensors, str(source), metadata={"foo": "bar"})
            plan = block_plan(root, [1, 4], block_count=6)
            mlp_plan = root / "mlp.json"
            mlp_plan.write_text(json.dumps({
                "format": "potatoforge_krea2_mlp_prune_plan", "version": 1, "architecture": "krea2",
                "original_mlp_width": 512, "main_hidden_width": 4, "block_count": 6,
                "block_indices": list(range(6)), "removed_per_block": 256, "new_mlp_width": 256,
                "blocks": {str(index): list(range(index, index + 256)) for index in range(6)},
            }), encoding="utf-8")
            profile = root / "profile.json"
            profile.write_text(json.dumps({"format_version": 1, "profile_id": "mixed", "default": "keep", "rules": [
                *({"action": "int8", "prefix": f"blocks.{index}.", "suffixes": ["weight"]} for index in range(6)),
                {"action": "int8", "prefix": "conditioner.", "suffixes": ["weight"]},
                {"action": "int8", "prefix": "txtfusion.layerwise_blocks.", "suffixes": ["weight"]},
            ]}), encoding="utf-8")
            output = root / "final.safetensors"
            dry_result = apply_krea2_combined_prune(source, plan, mlp_plan, profile, output, dry_run=True)
            self.assertTrue(dry_result.dry_run)
            self.assertFalse(any(path.exists() for path in (
                output, dry_result.remapped_mlp_plan_path, dry_result.output_profile_path,
                dry_result.manifest_path, output.with_name("final.safetensors.partial"),
            )))

            reference_block = root / "reference-block.safetensors"
            block_result = apply_krea2_block_plan(source, plan, reference_block)
            mapping = krea2_block_old_to_new(plan)
            remapped_plan = remap_krea2_mlp_plan_blocks(json.loads(mlp_plan.read_text(encoding="utf-8")), mapping)
            reference_plan = root / "reference-mlp.json"
            reference_plan.write_text(json.dumps(remapped_plan), encoding="utf-8")
            reference_output = root / "reference-final.safetensors"
            apply_krea2_mlp_plan(block_result.output_path, reference_plan, reference_output)

            result = apply_krea2_combined_prune(source, plan, mlp_plan, profile, output)
            self.assertEqual(result.verified_assignments, len(tensors) - 8)
            remapped_mlp = json.loads(result.remapped_mlp_plan_path.read_text(encoding="utf-8"))
            self.assertEqual(remapped_mlp["block_indices"], [0, 1, 2, 3])
            self.assertEqual(remapped_mlp["blocks"]["1"], list(range(2, 258)))
            self.assertEqual(remapped_mlp["blocks"]["3"], list(range(5, 261)))
            final = load_file(str(output))
            self.assertEqual(tuple(final["blocks.1.mlp.gate.weight"].shape), (256, 4))
            self.assertTrue(torch.equal(final["blocks.1.a.weight"], tensors["blocks.2.a.weight"]))
            self.assertEqual(output.read_bytes(), reference_output.read_bytes())
            self.assertEqual(
                read_source_model_header(output).metadata,
                read_source_model_header(reference_output).metadata,
            )
            metadata = read_source_model_header(output).metadata
            self.assertEqual(metadata["foo"], "bar")
            self.assertEqual(metadata["potatoforge.block_pruning_removed_original_blocks"], "[1, 4]")
            self.assertEqual(
                json.loads(metadata["potatoforge.pruning_config"])["plan_sha256"],
                krea2_mlp_plan_sha256(remapped_plan),
            )
            remapped_profile = load_profile(result.output_profile_path)
            self.assertEqual(resolve_profile(remapped_profile, "blocks.1.a.weight"), "int8")
            self.assertEqual(resolve_profile(remapped_profile, "txtfusion.layerwise_blocks.0.weight"), "int8")
            self.assertFalse((root / "final.safetensors.combined-block.partial").exists())
            self.assertFalse((root / "final.safetensors.combined-block-manifest.partial").exists())

            published_paths = (
                output, result.remapped_mlp_plan_path, result.output_profile_path, result.manifest_path,
            )
            published_contents = tuple(path.read_bytes() for path in published_paths)
            with self.assertRaises(FileExistsError):
                apply_krea2_combined_prune(source, plan, mlp_plan, profile, output)
            self.assertEqual(tuple(path.read_bytes() for path in published_paths), published_contents)

            restore_collision = result.remapped_mlp_plan_path.with_name(
                f"{result.remapped_mlp_plan_path.name}.restore"
            )
            restore_collision.write_bytes(b"preserve this existing file")
            with self.assertRaisesRegex(FileExistsError, "Restore output already exists"):
                apply_krea2_combined_prune(source, plan, mlp_plan, profile, output, overwrite=True)
            self.assertEqual(tuple(path.read_bytes() for path in published_paths), published_contents)
            self.assertEqual(restore_collision.read_bytes(), b"preserve this existing file")
            self.assertFalse(output.with_name("final.safetensors.restore").exists())
            self.assertFalse(any(path.with_name(f"{path.name}.partial").exists() for path in published_paths))

            restore_collision.unlink()
            apply_krea2_combined_prune(source, plan, mlp_plan, profile, output, overwrite=True)
            self.assertEqual(tuple(path.read_bytes() for path in published_paths), published_contents)
            self.assertFalse(any(path.with_name(f"{path.name}.restore").exists() for path in published_paths))

    def test_logical_block_header_rejects_duplicate_mapped_names(self):
        descriptor = {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
        header = SourceModelHeader(
            tensors={"blocks.0.a.weight": descriptor, "blocks.1.a.weight": descriptor},
            metadata={},
        )
        plan = {
            "format": "potatoforge_krea2_block_prune_plan",
            "version": 1,
            "architecture": "krea2",
            "block_count": 2,
            "removed_blocks": [],
        }

        with self.assertRaisesRegex(ValueError, "Duplicate output tensor name"):
            _logical_block_pruned_header(header, plan, {0: 0, 1: 0})
