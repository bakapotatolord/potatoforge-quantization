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
from potatoforge.planning import W5A8_MARKER
from potatoforge.quantization import grouped_int8, w4a8, w5a8, w6a8

REFERENCE = Path(__file__).parent / "data" / "w5a8_kitchen_reference.pt"


class TestW5A8Quantizer(unittest.TestCase):
    def test_matches_comfy_kitchen_reference_bytes(self):
        reference = torch.load(REFERENCE, weights_only=True)
        self.assertEqual(set(reference), {"gaussian", "scaled"})
        for name, expected in reference.items():
            with self.subTest(case=name):
                actual = w5a8.quantize_w5a8(expected["weights"])
                self.assertTrue(torch.equal(actual.packed_codes, expected["packed"]))
                self.assertTrue(torch.equal(actual.group_scales.view(torch.uint8), expected["s_rel"]))
                self.assertTrue(torch.equal(actual.channel_scales, expected["s_channel"]))
                self.assertTrue(torch.equal(w5a8.dequantize_w5a8(actual), expected["dequantized"]))

    def test_storage_contract_and_quality(self):
        weights = torch.randn(64, 1024, generator=torch.Generator().manual_seed(3)) * 0.02
        result = w5a8.quantize_w5a8(weights.to(torch.bfloat16))
        self.assertEqual((result.packed_codes.dtype, tuple(result.packed_codes.shape)), (torch.int8, (64, 640)))
        self.assertEqual((result.group_scales.dtype, tuple(result.group_scales.shape)), (torch.float8_e4m3fn, (64, 64)))
        self.assertEqual((result.channel_scales.dtype, tuple(result.channel_scales.shape)), (torch.float32, (64,)))
        codes = w5a8.unpack_w5a8_codes(result.packed_codes)
        self.assertGreaterEqual(int(codes.min()), 1)
        self.assertLessEqual(int(codes.max()), 31)
        relative = lambda value: ((value - weights).norm() / weights.norm()).item()
        five = relative(w5a8.dequantize_w5a8(result))
        four = relative(w4a8.dequantize_w4a8(w4a8.quantize_w4a8(weights)))
        six = relative(w6a8.dequantize_w6a8(w6a8.quantize_w6a8(weights)))
        self.assertLess(six, five)
        self.assertLess(five, four)

    def test_chunking_does_not_change_the_result(self):
        weights = torch.randn(40, 512, generator=torch.Generator().manual_seed(5))
        whole = w5a8.quantize_w5a8(weights)
        with patch.object(grouped_int8, "QUANT_ROW_ELEM_BUDGET", 512 * 3):
            chunked = w5a8.quantize_w5a8(weights)
        for first, second in zip(whole, chunked):
            as_bytes = lambda t: t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t
            self.assertTrue(torch.equal(as_bytes(first), as_bytes(second)))

    def test_plane_layout_round_trip(self):
        codes = torch.arange(128, dtype=torch.int32).reshape(2, 64) % 32
        packed = w5a8.pack_w5a8_codes(codes)
        self.assertEqual(tuple(packed.shape), (2, 40))
        raw = packed.view(torch.uint8).to(torch.int32)
        self.assertTrue(torch.equal(raw[:, :32], w4a8.pack_w4a8_codes(codes & 0xF).view(torch.uint8).to(torch.int32)))
        for column in range(64):
            self.assertTrue(torch.equal((raw[:, 32 + column // 8] >> (column % 8)) & 1, codes[:, column] >> 4))
        self.assertTrue(torch.equal(w5a8.unpack_w5a8_codes(packed), codes))

    def test_zero_weights_and_invalid_inputs(self):
        zero = w5a8.quantize_w5a8(torch.zeros(2, 256))
        self.assertTrue(torch.equal(w5a8.dequantize_w5a8(zero), torch.zeros(2, 256)))
        for weights in (torch.ones(256), torch.ones(2, 128), torch.ones(2, 288), torch.full((2, 256), float("inf"))):
            with self.subTest(shape=tuple(weights.shape)), self.assertRaisesRegex(ValueError, "^W5A8 "):
                w5a8.quantize_w5a8(weights)


class TestW5A8Conversion(unittest.TestCase):
    def test_profile_conversion_writes_comfy_w5a8_layers(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            weights = (torch.randn(32, 512, generator=torch.Generator().manual_seed(21)) * 0.02).to(torch.bfloat16)
            source = root / "source.safetensors"
            save_file({"blocks.0.attn.wq.weight": weights, "blocks.0.norm.weight": torch.ones(512, dtype=torch.bfloat16)},
                      str(source))
            profile = root / "w5a8.json"
            profile.write_text(json.dumps({
                "format_version": 1,
                "profile_id": "test-w5a8",
                "default": "keep",
                "rules": [{"action": "w5a8", "prefix": "", "suffixes": [".attn.wq.weight"]}],
            }), encoding="utf-8")
            output = root / "output.safetensors"
            estimated = estimate_output_bytes(source, profile)
            convert_model_from_profile(source, output, profile, on_entry_started=None)
            self.assertEqual(estimated, output.stat().st_size)
            loaded = load_file(str(output))
            expected = w5a8.quantize_w5a8(weights)
            prefix = "blocks.0.attn.wq"
            self.assertEqual(set(loaded), {f"{prefix}.weight", f"{prefix}.weight_s_rel", f"{prefix}.comfy_quant",
                                           f"{prefix}.weight_s_channel", "blocks.0.norm.weight"})
            self.assertTrue(torch.equal(loaded[f"{prefix}.weight"], expected.packed_codes))
            self.assertTrue(torch.equal(loaded[f"{prefix}.weight_s_rel"].view(torch.uint8),
                                        expected.group_scales.view(torch.uint8)))
            self.assertTrue(torch.equal(loaded[f"{prefix}.weight_s_channel"], expected.channel_scales))
            self.assertEqual(json.loads(bytes(loaded[f"{prefix}.comfy_quant"].tolist())), W5A8_MARKER)
            self.assertEqual(W5A8_MARKER["format"], "w5a8_int8")
            self.assertEqual(read_source_model_header(output).metadata["potatoforge.quantization"], "w5a8")

    def test_patch_plan_width_validation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.safetensors"
            save_file({"blocks.0.weight": torch.randn(8, 256).to(torch.bfloat16),
                       "blocks.1.weight": torch.ones(4, 128, dtype=torch.bfloat16)}, str(source))
            header = read_source_model_header(source).tensors
            plan = build_patch_plan(header, "blocks.0.weight", "w5a8", "test")
            output = root / "patch.safetensors"
            execute_patch_plan(source, output, plan, build_patch_metadata(plan, source))
            self.assertEqual(len(load_file(str(output))), 4)
            with self.assertRaisesRegex(ValueError, "divisible by 256"):
                build_patch_plan(header, "blocks.1.weight", "w5a8", "test")


if __name__ == "__main__":
    unittest.main()
