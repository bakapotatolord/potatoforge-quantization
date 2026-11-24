import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from potatoforge.lora.lora_extract import (
    ExtractionOptions,
    SplitRule,
    extract_lora,
    summarize,
)
from potatoforge.lora.lora_merge import AdapterMergeInput, merge_bf16_adapters
from potatoforge.quantization.hadamard import apply_hadamard_rotation


def options(**overrides) -> ExtractionOptions:
    values = {
        "ranks": (16,),
        "dtype": "fp32",
        "device": "cpu",
        "key_prefix": "diffusion_model.",
    }
    values.update(overrides)
    return ExtractionOptions(**values)


def lora_delta(lora: dict[str, torch.Tensor], target: str) -> torch.Tensor:
    up = lora[f"diffusion_model.{target}.lora_up.weight"]
    down = lora[f"diffusion_model.{target}.lora_down.weight"]
    alpha = lora[f"diffusion_model.{target}.alpha"].item()
    return (alpha / down.shape[0]) * (up @ down)


def save_int8_convrot(
    path: Path,
    weights: dict[str, torch.Tensor],
    group_size: int,
    prefix: str = "",
) -> dict[str, torch.Tensor]:
    """Write a ComfyUI-style INT8 ConvRot checkpoint; return what it dequantizes to."""
    tensors: dict[str, torch.Tensor] = {}
    dequantized: dict[str, torch.Tensor] = {}
    config = torch.tensor(
        list(
            json.dumps(
                {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": group_size,
                }
            ).encode()
        ),
        dtype=torch.uint8,
    )

    for name, weight in weights.items():
        rotated = apply_hadamard_rotation(weight, group_size)
        scale = rotated.abs().amax(dim=1, keepdim=True) / 127.0
        codes = (rotated / scale).round().clamp(-127, 127).to(torch.int8)
        tensors[f"{prefix}{name}.weight"] = codes
        tensors[f"{prefix}{name}.weight_scale"] = scale
        tensors[f"{prefix}{name}.comfy_quant"] = config
        dequantized[name] = apply_hadamard_rotation(
            codes.float() * scale,
            group_size,
        )

    save_file(tensors, str(path))
    return dequantized


class TestExtractLora(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.dir = Path(self._dir.name)
        torch.manual_seed(1)

    def test_full_rank_extraction_merges_back_to_tuned(self) -> None:
        base = {"blk.proj.weight": torch.randn(12, 10), "blk.norm.weight": torch.randn(12)}
        tuned = {
            "blk.proj.weight": base["blk.proj.weight"] + 0.1 * torch.randn(12, 10),
            "blk.norm.weight": base["blk.norm.weight"],
        }
        save_file(base, str(self.dir / "base.safetensors"))
        save_file(tuned, str(self.dir / "tuned.safetensors"))

        result = extract_lora(
            self.dir / "base.safetensors",
            self.dir / "tuned.safetensors",
            self.dir / "lora.safetensors",
            options(ranks=(10,)),
        )

        self.assertEqual(set(result.skipped), {"blk.norm.weight"})
        merge_bf16_adapters(
            self.dir / "base.safetensors",
            self.dir / "merged.safetensors",
            [AdapterMergeInput(self.dir / "lora.safetensors", 1.0)],
        )
        merged = load_file(str(self.dir / "merged.safetensors"))
        torch.testing.assert_close(
            merged["blk.proj.weight"], tuned["blk.proj.weight"], atol=1e-4, rtol=1e-4
        )

    def test_strength_scales_the_merged_delta(self) -> None:
        base = torch.randn(8, 8)
        tuned = base + torch.randn(8, 8)
        save_file({"a.weight": base}, str(self.dir / "base.safetensors"))
        save_file({"a.weight": tuned}, str(self.dir / "tuned.safetensors"))
        extract_lora(
            self.dir / "base.safetensors",
            self.dir / "tuned.safetensors",
            self.dir / "lora.safetensors",
            options(ranks=(8,)),
        )
        merge_bf16_adapters(
            self.dir / "base.safetensors",
            self.dir / "merged.safetensors",
            [AdapterMergeInput(self.dir / "lora.safetensors", 0.5)],
        )
        merged = load_file(str(self.dir / "merged.safetensors"))["a.weight"]
        torch.testing.assert_close(merged, base + 0.5 * (tuned - base), atol=1e-4, rtol=1e-4)

    def test_truncated_rank_is_the_best_low_rank_fit(self) -> None:
        left, right = torch.randn(20, 3), torch.randn(3, 18)
        base = torch.randn(20, 18)
        tuned = base + left @ right + 0.001 * torch.randn(20, 18)
        save_file({"a.weight": base}, str(self.dir / "base.safetensors"))
        save_file({"a.weight": tuned}, str(self.dir / "tuned.safetensors"))
        result = extract_lora(
            self.dir / "base.safetensors",
            self.dir / "tuned.safetensors",
            self.dir / "lora.safetensors",
            options(ranks=(3,)),
        )
        self.assertLess(result.tensors[0].residual["3"], 0.01)
        lora = load_file(str(self.dir / "lora.safetensors"))
        self.assertEqual(lora["diffusion_model.a.lora_down.weight"].shape, (3, 18))
        self.assertEqual(lora["diffusion_model.a.alpha"].item(), 3.0)

    def test_multiple_ranks_share_one_pass_and_truncate(self) -> None:
        base = torch.randn(24, 24)
        tuned = base + torch.randn(24, 24)
        save_file({"a.weight": base}, str(self.dir / "base.safetensors"))
        save_file({"a.weight": tuned}, str(self.dir / "tuned.safetensors"))
        result = extract_lora(
            self.dir / "base.safetensors",
            self.dir / "tuned.safetensors",
            str(self.dir / "lora_r{rank}.safetensors"),
            options(ranks=(4, 12)),
        )
        self.assertEqual(set(result.outputs), {"4", "12"})
        self.assertTrue((self.dir / "lora_r4.safetensors").is_file())
        self.assertLess(result.tensors[0].residual["12"], result.tensors[0].residual["4"])
        summary = summarize(result)
        self.assertGreater(summary["12"]["energy_captured"], summary["4"]["energy_captured"])

    def test_multiple_ranks_require_placeholder(self) -> None:
        save_file({"a.weight": torch.randn(4, 4)}, str(self.dir / "base.safetensors"))
        save_file({"a.weight": torch.randn(4, 4)}, str(self.dir / "tuned.safetensors"))
        with self.assertRaisesRegex(ValueError, r"\{rank\}"):
            extract_lora(
                self.dir / "base.safetensors",
                self.dir / "tuned.safetensors",
                self.dir / "lora.safetensors",
                options(ranks=(2, 4)),
            )

    def test_energy_mode_picks_a_small_rank_for_a_low_rank_delta(self) -> None:
        base = torch.randn(32, 32)
        tuned = base + torch.randn(32, 2) @ torch.randn(2, 32)
        save_file({"a.weight": base}, str(self.dir / "base.safetensors"))
        save_file({"a.weight": tuned}, str(self.dir / "tuned.safetensors"))
        result = extract_lora(
            self.dir / "base.safetensors",
            self.dir / "tuned.safetensors",
            self.dir / "lora.safetensors",
            options(ranks=(16,), energy=0.999),
        )
        self.assertEqual(result.tensors[0].rank["e99.9"], 2)
        lora = load_file(str(self.dir / "lora.safetensors"))
        self.assertEqual(lora["diffusion_model.a.alpha"].item(), 2.0)

    def test_split_fused_writes_separate_parts(self) -> None:
        base = torch.randn(12, 6)
        tuned = base + torch.randn(12, 6)
        save_file({"l.proj.qkv.weight": base}, str(self.dir / "base.safetensors"))
        save_file({"l.proj.qkv.weight": tuned}, str(self.dir / "tuned.safetensors"))
        extract_lora(
            self.dir / "base.safetensors",
            self.dir / "tuned.safetensors",
            self.dir / "lora.safetensors",
            options(ranks=(4,), split=(SplitRule("qkv", ("q", "k", "v")),)),
        )
        lora = load_file(str(self.dir / "lora.safetensors"))
        self.assertNotIn("diffusion_model.l.proj.qkv.alpha", lora)
        rebuilt = torch.cat(
            [lora_delta(lora, f"l.proj.{part}") for part in "qkv"]
        )
        # rank 4 of 4x6 blocks is full rank, so the split rebuild is exact
        torch.testing.assert_close(rebuilt, tuned - base, atol=1e-4, rtol=1e-4)

    def test_prefix_is_detected_on_either_side(self) -> None:
        base = torch.randn(6, 6)
        tuned = base + torch.randn(6, 6)
        save_file({"layers.0.o.weight": base}, str(self.dir / "base.safetensors"))
        save_file(
            {"model.diffusion_model.layers.0.o.weight": tuned},
            str(self.dir / "tuned.safetensors"),
        )
        result = extract_lora(
            self.dir / "base.safetensors",
            self.dir / "tuned.safetensors",
            self.dir / "lora_a.safetensors",
            options(ranks=(6,)),
        )
        self.assertEqual(result.tensors[0].name, "layers.0.o")
        swapped = extract_lora(
            self.dir / "tuned.safetensors",
            self.dir / "base.safetensors",
            self.dir / "lora_b.safetensors",
            options(ranks=(6,)),
        )
        self.assertEqual(swapped.tensors[0].name, "layers.0.o")

    def test_int8_convrot_inputs_are_dequantized_and_unrotated(self) -> None:
        base_weights = {"layers.0.proj.o": torch.randn(8, 32)}
        tuned_weights = {"layers.0.proj.o": base_weights["layers.0.proj.o"] + 0.3 * torch.randn(8, 32)}
        base_deq = save_int8_convrot(self.dir / "base.safetensors", base_weights, 16)
        tuned_deq = save_int8_convrot(
            self.dir / "tuned.safetensors", tuned_weights, 16, prefix="model.diffusion_model."
        )
        extract_lora(
            self.dir / "base.safetensors",
            self.dir / "tuned.safetensors",
            self.dir / "lora.safetensors",
            options(ranks=(8,)),
        )
        lora = load_file(str(self.dir / "lora.safetensors"))
        torch.testing.assert_close(
            lora_delta(lora, "layers.0.proj.o"),
            tuned_deq["layers.0.proj.o"] - base_deq["layers.0.proj.o"],
            atol=1e-4,
            rtol=1e-4,
        )
        self.assertNotIn("diffusion_model.layers.0.proj.o.weight_scale.lora_up.weight", lora)

    def test_mixed_float_and_int8_inputs(self) -> None:
        weights = {"layers.0.proj.o": torch.randn(8, 32)}
        base_deq = save_int8_convrot(self.dir / "base.safetensors", weights, 16)
        tuned = weights["layers.0.proj.o"] + torch.randn(8, 32)
        save_file({"layers.0.proj.o.weight": tuned}, str(self.dir / "tuned.safetensors"))
        extract_lora(
            self.dir / "base.safetensors",
            self.dir / "tuned.safetensors",
            self.dir / "lora.safetensors",
            options(ranks=(8,)),
        )
        lora = load_file(str(self.dir / "lora.safetensors"))
        torch.testing.assert_close(
            lora_delta(lora, "layers.0.proj.o"),
            tuned - base_deq["layers.0.proj.o"],
            atol=1e-4,
            rtol=1e-4,
        )

    def test_unsupported_quantization_format_fails_loudly(self) -> None:
        config = torch.tensor(list(json.dumps({"format": "mystery"}).encode()), dtype=torch.uint8)
        save_file(
            {
                "a.weight": torch.zeros(4, 16, dtype=torch.int8),
                "a.weight_scale": torch.ones(4, 1),
                "a.comfy_quant": config,
            },
            str(self.dir / "base.safetensors"),
        )
        save_file({"a.weight": torch.randn(4, 16)}, str(self.dir / "tuned.safetensors"))
        with self.assertRaisesRegex(ValueError, "mystery"):
            extract_lora(
                self.dir / "base.safetensors",
                self.dir / "tuned.safetensors",
                self.dir / "lora.safetensors",
                options(),
            )

    def test_unchanged_mismatched_and_unmatched_tensors_are_reported(self) -> None:
        same = torch.randn(6, 6)
        changed = torch.randn(6, 6)
        save_file(
            {
                "same.weight": same,
                "changed.weight": changed,
                "shape.weight": torch.randn(4, 4),
                "only_base.weight": torch.randn(2, 2),
            },
            str(self.dir / "base.safetensors"),
        )
        save_file(
            {
                "same.weight": same,
                "changed.weight": changed + torch.randn(6, 6),
                "shape.weight": torch.randn(5, 4),
                "only_tuned.weight": torch.randn(2, 2),
            },
            str(self.dir / "tuned.safetensors"),
        )
        result = extract_lora(
            self.dir / "base.safetensors",
            self.dir / "tuned.safetensors",
            self.dir / "lora.safetensors",
            options(ranks=(6,)),
        )
        self.assertIn("unchanged", result.skipped["same.weight"])
        self.assertIn("shape mismatch", result.skipped["shape.weight"])
        self.assertEqual(result.unmatched_base, ["only_base.weight"])
        self.assertEqual(result.unmatched_tuned, ["only_tuned.weight"])
        self.assertEqual([t.name for t in result.tensors], ["changed"])

    def test_non_float_metadata_tensors_are_skipped(self) -> None:
        meta = torch.tensor([1, 2, 3], dtype=torch.uint8)
        base = {"a.weight": torch.randn(6, 6), "a.config": meta}
        tuned = {"a.weight": base["a.weight"] + torch.randn(6, 6), "a.config": meta}
        save_file(base, str(self.dir / "base.safetensors"))
        save_file(tuned, str(self.dir / "tuned.safetensors"))
        result = extract_lora(
            self.dir / "base.safetensors",
            self.dir / "tuned.safetensors",
            self.dir / "lora.safetensors",
            options(ranks=(6,)),
        )
        self.assertIn("not a float", result.skipped["a.config"])

    def test_include_and_exclude_filters(self) -> None:
        base = {"a.weight": torch.randn(6, 6), "b.weight": torch.randn(6, 6)}
        tuned = {k: v + torch.randn(6, 6) for k, v in base.items()}
        save_file(base, str(self.dir / "base.safetensors"))
        save_file(tuned, str(self.dir / "tuned.safetensors"))
        result = extract_lora(
            self.dir / "base.safetensors",
            self.dir / "tuned.safetensors",
            self.dir / "lora.safetensors",
            options(ranks=(6,), exclude=r"^b\."),
        )
        self.assertEqual([t.name for t in result.tensors], ["a"])

    def test_diffs_for_changed_vectors_merge_back(self) -> None:
        base = {"n.weight": torch.randn(8), "m.weight": torch.randn(8, 8)}
        tuned = {"n.weight": base["n.weight"] + 0.5, "m.weight": base["m.weight"] + torch.randn(8, 8)}
        save_file(base, str(self.dir / "base.safetensors"))
        save_file(tuned, str(self.dir / "tuned.safetensors"))
        extract_lora(
            self.dir / "base.safetensors",
            self.dir / "tuned.safetensors",
            self.dir / "lora.safetensors",
            options(ranks=(8,), include_diffs=True),
        )
        merge_bf16_adapters(
            self.dir / "base.safetensors",
            self.dir / "merged.safetensors",
            [AdapterMergeInput(self.dir / "lora.safetensors", 1.0)],
        )
        merged = load_file(str(self.dir / "merged.safetensors"))
        torch.testing.assert_close(merged["n.weight"], tuned["n.weight"], atol=1e-5, rtol=1e-5)

    def test_refuses_to_overwrite_and_leaves_no_partial_file(self) -> None:
        save_file({"a.weight": torch.randn(4, 4)}, str(self.dir / "base.safetensors"))
        save_file({"a.weight": torch.randn(4, 4)}, str(self.dir / "tuned.safetensors"))
        (self.dir / "lora.safetensors").write_bytes(b"x")
        with self.assertRaises(FileExistsError):
            extract_lora(
                self.dir / "base.safetensors",
                self.dir / "tuned.safetensors",
                self.dir / "lora.safetensors",
                options(),
            )
        extract_lora(
            self.dir / "base.safetensors",
            self.dir / "tuned.safetensors",
            self.dir / "fresh.safetensors",
            options(ranks=(4,)),
        )
        self.assertFalse(list(self.dir.glob("*.partial")))

    def test_identical_checkpoints_have_nothing_to_extract(self) -> None:
        weights = {"a.weight": torch.randn(4, 4)}
        save_file(weights, str(self.dir / "base.safetensors"))
        save_file(weights, str(self.dir / "tuned.safetensors"))
        with self.assertRaisesRegex(ValueError, "nothing to extract"):
            extract_lora(
                self.dir / "base.safetensors",
                self.dir / "tuned.safetensors",
                self.dir / "lora.safetensors",
                options(),
            )

    def test_output_metadata_is_written(self) -> None:
        save_file({"a.weight": torch.randn(4, 4)}, str(self.dir / "base.safetensors"))
        save_file({"a.weight": torch.randn(4, 4)}, str(self.dir / "tuned.safetensors"))
        extract_lora(
            self.dir / "base.safetensors",
            self.dir / "tuned.safetensors",
            self.dir / "lora.safetensors",
            options(ranks=(2,)),
        )
        with safe_open(str(self.dir / "lora.safetensors"), "pt") as handle:
            metadata = handle.metadata()
        self.assertEqual(metadata["rank"], "2")
        self.assertEqual(metadata["base"], "base.safetensors")


if __name__ == "__main__":
    unittest.main()
