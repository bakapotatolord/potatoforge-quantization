import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from potatoforge.converter import convert_model_from_profile, estimate_output_bytes
from potatoforge.headers.source_header import read_source_model_header
from potatoforge.patch_planning import build_patch_metadata, build_patch_plan
from potatoforge.patching import execute_patch_plan
from potatoforge.planning import W4A8_MARKER
from potatoforge.quantization import w4a8

REFERENCE = Path(__file__).parent / "data" / "w4a8_kitchen_reference.pt"


class TestW4A8Quantizer(unittest.TestCase):
    def test_matches_comfy_kitchen_reference_bytes(self):
        reference = torch.load(REFERENCE, weights_only=True)
        self.assertEqual(set(reference), {"gaussian", "heavy"})
        for name, expected in reference.items():
            with self.subTest(case=name):
                actual = w4a8.quantize_w4a8(expected["weights"])
                self.assertTrue(torch.equal(actual.packed_codes, expected["packed"]))
                self.assertTrue(torch.equal(actual.group_scales.view(torch.uint8), expected["s_rel"]))
                self.assertTrue(torch.equal(actual.channel_scales, expected["s_channel"]))
                self.assertTrue(torch.equal(actual.codebook, expected["codebook"]))
                self.assertTrue(torch.equal(w4a8.dequantize_w4a8_int8(actual).to(torch.int8), expected["int8"]))
                self.assertTrue(torch.equal(w4a8.dequantize_w4a8(actual), expected["dequantized"]))
        # the Gaussian case keeps kitchen's frozen table; the heavy-tailed one is fitted
        self.assertTrue(torch.equal(reference["gaussian"]["codebook"], w4a8.default_w4a8_codebook()))
        self.assertFalse(torch.equal(reference["heavy"]["codebook"], w4a8.default_w4a8_codebook()))

    def test_storage_contract_and_quality(self):
        weights = torch.randn(64, 1024, generator=torch.Generator().manual_seed(3)) * 0.02
        result = w4a8.quantize_w4a8(weights.to(torch.bfloat16))
        self.assertEqual((result.packed_codes.dtype, tuple(result.packed_codes.shape)), (torch.int8, (64, 512)))
        self.assertEqual((result.group_scales.dtype, tuple(result.group_scales.shape)), (torch.float8_e4m3fn, (64, 64)))
        self.assertEqual((result.channel_scales.dtype, tuple(result.channel_scales.shape)), (torch.float32, (64,)))
        self.assertEqual((result.codebook.dtype, tuple(result.codebook.shape)), (torch.float32, (16,)))
        self.assertTrue(all(t.device.type == "cpu" for t in result))
        restored = w4a8.dequantize_w4a8(result)
        relative = ((restored - weights).norm() / weights.norm()).item()
        self.assertLess(relative, 0.1)

    def test_chunking_does_not_change_the_result(self):
        weights = torch.randn(40, 512, generator=torch.Generator().manual_seed(5))
        whole = w4a8.quantize_w4a8(weights)
        with patch.object(w4a8, "QUANT_ROW_ELEM_BUDGET", 512 * 3):
            chunked = w4a8.quantize_w4a8(weights)
        for first, second in zip(whole, chunked):
            self.assertTrue(torch.equal(first, second))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_matches_cpu_reconstruction(self):
        weights = torch.randn(96, 768, generator=torch.Generator().manual_seed(9)) * 0.02
        cpu = w4a8.quantize_w4a8(weights)
        cuda = w4a8.quantize_w4a8(weights, device="cuda")
        self.assertTrue(all(t.device.type == "cpu" for t in cuda))
        self.assertTrue(torch.equal(cpu.codebook, cuda.codebook))
        cpu_restored, cuda_restored = w4a8.dequantize_w4a8(cpu), w4a8.dequantize_w4a8(cuda)
        self.assertLess(((cpu_restored - cuda_restored).norm() / cpu_restored.norm()).item(), 1e-3)

    def test_nibble_order_round_trip(self):
        codes = torch.arange(32, dtype=torch.int32).remainder(16).reshape(2, 16)
        packed = w4a8.pack_w4a8_codes(codes)
        self.assertEqual(packed.view(torch.uint8)[0].tolist(), [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE])
        self.assertTrue(torch.equal(w4a8.unpack_w4a8_codes(packed), codes))

    def test_zero_weights_and_invalid_inputs(self):
        zero = w4a8.quantize_w4a8(torch.zeros(2, 256))
        self.assertTrue(torch.equal(w4a8.dequantize_w4a8(zero), torch.zeros(2, 256)))
        for weights in (
            torch.ones(256),
            torch.ones(2, 128),
            torch.empty(0, 256),
            torch.full((2, 256), float("nan")),
            torch.ones(2, 256, dtype=torch.float64),
        ):
            with self.subTest(shape=tuple(weights.shape), dtype=weights.dtype), self.assertRaises(ValueError):
                w4a8.quantize_w4a8(weights)


class TestW4A8Conversion(unittest.TestCase):
    def _source(self, directory: Path) -> tuple[Path, dict[str, torch.Tensor]]:
        generator = torch.Generator().manual_seed(21)
        tensors = {
            "blocks.0.attn.wq.weight": (torch.randn(32, 512, generator=generator) * 0.02).to(torch.bfloat16),
            "blocks.0.attn.wq.bias": torch.randn(32, generator=generator).to(torch.bfloat16),
            "blocks.0.norm.weight": torch.ones(512, dtype=torch.bfloat16),
        }
        source = directory / "source.safetensors"
        save_file(tensors, str(source))
        return source, tensors

    def _check_layer(self, loaded: dict[str, torch.Tensor], weights: torch.Tensor, prefix: str) -> None:
        expected = w4a8.quantize_w4a8(weights)
        self.assertEqual(loaded[f"{prefix}.weight"].dtype, torch.int8)
        self.assertTrue(torch.equal(loaded[f"{prefix}.weight"], expected.packed_codes))
        self.assertEqual(loaded[f"{prefix}.weight_s_rel"].dtype, torch.float8_e4m3fn)
        self.assertTrue(torch.equal(loaded[f"{prefix}.weight_s_rel"].view(torch.uint8), expected.group_scales.view(torch.uint8)))
        self.assertTrue(torch.equal(loaded[f"{prefix}.weight_s_channel"], expected.channel_scales))
        self.assertTrue(torch.equal(loaded[f"{prefix}.weight_codebook"], expected.codebook))
        self.assertEqual(json.loads(bytes(loaded[f"{prefix}.comfy_quant"].tolist())), W4A8_MARKER)
        self.assertNotIn(f"{prefix}.weight_scale", loaded)

    def test_profile_conversion_writes_comfy_asym_w4a8_layers(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, tensors = self._source(root)
            profile = root / "w4a8.json"
            profile.write_text(json.dumps({
                "format_version": 1,
                "profile_id": "test-w4a8",
                "default": "keep",
                "rules": [{"action": "w4a8", "prefix": "", "suffixes": [".attn.wq.weight"]}],
            }), encoding="utf-8")
            output = root / "output.safetensors"
            estimated = estimate_output_bytes(source, profile)
            convert_model_from_profile(source, output, profile, on_entry_started=None)
            self.assertEqual(estimated, output.stat().st_size)
            loaded = load_file(str(output))
            self._check_layer(loaded, tensors["blocks.0.attn.wq.weight"], "blocks.0.attn.wq")
            self.assertTrue(torch.equal(loaded["blocks.0.attn.wq.bias"], tensors["blocks.0.attn.wq.bias"]))
            self.assertTrue(torch.equal(loaded["blocks.0.norm.weight"], tensors["blocks.0.norm.weight"]))
            metadata = read_source_model_header(output).metadata
            self.assertEqual(metadata["potatoforge.quantization"], "w4a8")

    def test_patch_round_trip_leaves_source_untouched(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, tensors = self._source(root)
            original = source.read_bytes()
            plan = build_patch_plan(
                read_source_model_header(source).tensors, "blocks.0.attn.wq.weight", "w4a8", "test"
            )
            output = root / "patch.safetensors"
            execute_patch_plan(source, output, plan, build_patch_metadata(plan, source))
            loaded = load_file(str(output))
            self.assertEqual(len(loaded), 5)
            self._check_layer(loaded, tensors["blocks.0.attn.wq.weight"], "blocks.0.attn.wq")
            self.assertEqual(source.read_bytes(), original)

    def test_plan_rejects_unsupported_widths(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "narrow.safetensors"
            save_file({"blocks.0.weight": torch.ones(4, 128, dtype=torch.bfloat16)}, str(source))
            with self.assertRaisesRegex(ValueError, "divisible by 256"):
                build_patch_plan(read_source_model_header(source).tensors, "blocks.0.weight", "w4a8", "test")


if __name__ == "__main__":
    unittest.main()
