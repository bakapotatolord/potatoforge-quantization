from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open

from potatoforge.application.commands import quantize_model
from potatoforge.calibration.gptq import GPTQCalibration
from potatoforge.converter import convert_model_from_profile
from potatoforge.quantization.convrot_w4a4 import (
    quantize_convrot_w4a4_mse,
    unpack_signed_int4_row_major,
)
from potatoforge.quantization.convrot_w4a4_gptq import (
    gptq_quantize,
    prepare_factors,
    quantize_convrot_w4a4_mse_gptq,
    rotate_gram,
)
from scripts.gptq_poc import gptq_quantize_saved_blocks
from potatoforge.quantization.hadamard import apply_hadamard_rotation
from potatoforge.quantization.convrot_w4a4 import select_w4a4_mse_scale


class GPTQV1Tests(unittest.TestCase):
    def _artifact(self, directory: Path) -> Path:
        path = directory / "calibration.json"
        # A 256-wide final partial BD512 block. Keep its padding exactly zero.
        blocks = torch.zeros(1, 512, 512, dtype=torch.float32)
        blocks[0, :256, :256] = torch.eye(256) * 8
        save_file({"gram.0000": blocks}, str(path.with_suffix(".safetensors")))
        path.write_text(json.dumps({
            "format": "potatoforge_gptq_calibration", "version": 1,
            "activation_basis": "logical_linear_input", "gram_mode": "block_diag",
            "gram_block_size": 512, "session_id": "synthetic", "model_class": "synthetic.Model",
            "finalized": True,
            "completed_prompt_generations": 2, "evaluation_count": 2,
            "prompt_evaluation_counts": [1, 1],
            "sources": [{
                "source_id": "shared.input", "tensor_key": "gram.0000",
                "in_features": 256, "row_count": 8, "invocation_count": 2, "num_blocks": 1,
                "last_block_size": 256,
                "members": ["a.weight", "b.weight"],
                "member_shapes": [[2, 256], [2, 256]],
            }],
        }), encoding="utf-8")
        return path

    def test_shared_source_partial_block_and_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            artifact = GPTQCalibration(self._artifact(directory))
            self.assertIs(artifact.source_for_weight("a.weight"), artifact.source_for_weight("b.weight"))
            self.assertIs(artifact.factors_for_weight("a.weight"), artifact.factors_for_weight("b.weight"))
            torch.manual_seed(5)
            weights = (torch.randn(2, 256) * 0.1).bfloat16()
            factors = artifact.factors_for_weight("a.weight")
            result = quantize_convrot_w4a4_mse_gptq(weights, factors)
            rotated = apply_hadamard_rotation(weights, 256)
            scales = select_w4a4_mse_scale(rotated)
            expected = gptq_quantize(rotated, factors[0], scales)
            actual = unpack_signed_int4_row_major(result.packed_codes).float() * result.scales[:, None]
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
            self.assertEqual(result.packed_codes.shape, (2, 128))
            self.assertEqual(result.scales.shape, (2,))
            self.assertTrue(torch.equal(result.scales, quantize_convrot_w4a4_mse(weights).scales))
            self.assertTrue(torch.equal(result.packed_codes, quantize_convrot_w4a4_mse_gptq(weights, factors).packed_codes))
            with self.assertRaisesRegex(ValueError, "no Gram"):
                artifact.source_for_weight("missing.weight")

    def test_conversion_uses_existing_runtime_payload_and_rejects_missing_gram(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            calibration = self._artifact(directory)
            torch.manual_seed(7)
            source = directory / "source.safetensors"
            save_file({
                "a.weight": torch.randn(2, 256).bfloat16(),
                "b.weight": torch.randn(2, 256).bfloat16(),
            }, str(source))
            profile = directory / "profile.json"
            profile.write_text(json.dumps({
                "format_version": 1, "profile_id": "gptq-test",
                "default": "convrot_w4a4_mse_gptq", "rules": [],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "--gptq-calibration"):
                convert_model_from_profile(source, directory / "missing.safetensors", profile)
            output = directory / "output.safetensors"
            convert_model_from_profile(source, output, profile, gptq_calibration=calibration, on_entry_started=None)
            tensors = load_file(str(output))
            self.assertEqual(tensors["a.weight"].shape, (2, 128))
            self.assertEqual(tensors["b.weight"].shape, (2, 128))
            self.assertEqual(tensors["a.weight_scale"].shape, (2,))
            self.assertIn("a.comfy_quant", tensors)

    def test_rejects_wrong_member_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            calibration = GPTQCalibration(self._artifact(Path(temporary)))
            with self.assertRaisesRegex(ValueError, "mismatched model weight"):
                calibration.validate_model({"a.weight": {"shape": [3, 256]}})

    def test_rejects_incompatible_metadata_and_tensor_layout(self):
        changes = (
            ({"version": 2}, "format/version"),
            ({"finalized": False}, "not finalized"),
            ({"activation_basis": "rotated"}, "activation_basis"),
            ({"gram_mode": "full"}, "block_diag"),
            ({"gram_block_size": 384}, "block size"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = self._artifact(Path(temporary))
            original = json.loads(path.read_text(encoding="utf-8"))
            for change, message in changes:
                with self.subTest(change=change):
                    path.write_text(json.dumps({**original, **change}), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        GPTQCalibration(path)
            missing_state = dict(original)
            del missing_state["finalized"]
            path.write_text(json.dumps(missing_state), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not finalized"):
                GPTQCalibration(path)
            for source_change, message in (
                ({"row_count": 0}, "row_count"),
                ({"last_block_size": 255}, "block layout"),
                ({"members": ["a.weight", "a.weight"]}, "duplicate member"),
            ):
                with self.subTest(source_change=source_change):
                    metadata = json.loads(json.dumps(original))
                    metadata["sources"][0].update(source_change)
                    path.write_text(json.dumps(metadata), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        GPTQCalibration(path)
            path.write_text(json.dumps(original), encoding="utf-8")
            bad = torch.zeros(1, 512, 512)
            bad[0, 256, 0] = 1
            save_file({"gram.0000": bad}, str(path.with_suffix(".safetensors")))
            calibration = GPTQCalibration(path)
            with self.assertRaisesRegex(ValueError, "padding"):
                calibration.factors_for_weight("a.weight")
            save_file({"wrong": torch.zeros(1, 512, 512)}, str(path.with_suffix(".safetensors")))
            with self.assertRaisesRegex(ValueError, "Safetensors keys"):
                GPTQCalibration(path)
            save_file({"gram.0000": torch.zeros(1, 512, 512, dtype=torch.float16)},
                      str(path.with_suffix(".safetensors")))
            with self.assertRaisesRegex(ValueError, "FP32"):
                GPTQCalibration(path)

    def test_valid_nondefault_block_size_is_consumed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._artifact(Path(temporary))
            metadata = json.loads(path.read_text(encoding="utf-8"))
            metadata["gram_block_size"] = 1024
            path.write_text(json.dumps(metadata), encoding="utf-8")
            blocks = torch.zeros(1, 1024, 1024)
            blocks[0, :256, :256] = torch.eye(256)
            save_file({"gram.0000": blocks}, str(path.with_suffix(".safetensors")))
            calibration = GPTQCalibration(path)
            self.assertEqual(calibration.source_for_weight("a.weight").block_size, 1024)
            self.assertEqual(calibration.factors_for_weight("a.weight")[0].shape, (256, 256))

    def test_shared_bd512_factors_match_poc_and_are_prepared_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._artifact(Path(temporary))
            metadata = json.loads(path.read_text(encoding="utf-8"))
            metadata["sources"][0].update({
                "in_features": 768, "num_blocks": 2, "last_block_size": 256,
                "member_shapes": [[2, 768], [2, 768]],
            })
            path.write_text(json.dumps(metadata), encoding="utf-8")
            blocks = torch.zeros(2, 512, 512)
            blocks[0] = torch.eye(512) * 3
            blocks[1, :256, :256] = torch.eye(256) * 7
            save_file({"gram.0000": blocks}, str(path.with_suffix(".safetensors")))
            calibration = GPTQCalibration(path)
            with patch("potatoforge.quantization.gptq.prepare_factors", wraps=prepare_factors) as prepared:
                factors = calibration.factors_for_weight("a.weight")
                self.assertIs(factors, calibration.factors_for_weight("b.weight"))
                self.assertEqual(prepared.call_count, 1)
            torch.manual_seed(19)
            weights = torch.randn(2, 768).bfloat16()
            rotated = apply_hadamard_rotation(weights, 256)
            scales = select_w4a4_mse_scale(rotated)
            poc = gptq_quantize_saved_blocks(
                rotated, (rotate_gram(blocks[0]), rotate_gram(blocks[1, :256, :256])),
                scales, 0.01, 128,
            )
            output = quantize_convrot_w4a4_mse_gptq(weights, factors)
            actual = unpack_signed_int4_row_major(output.packed_codes).float() * output.scales[:, None]
            self.assertTrue(torch.equal(actual, poc))

    def test_config_and_runtime_payload_use_existing_w4a4_format(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            calibration = self._artifact(directory)
            source = directory / "source.safetensors"
            save_file({"a.weight": torch.zeros(2, 256), "b.weight": torch.zeros(2, 256)}, str(source))
            outputs = {}
            for action in ("convrot_w4a4", "convrot_w4a4_mse", "convrot_w4a4_mse_gptq"):
                profile = directory / f"{action}.json"
                output = directory / f"{action}.safetensors"
                profile.write_text(json.dumps({
                    "format_version": 1, "profile_id": action, "default": action, "rules": [],
                }), encoding="utf-8")
                if action.endswith("gptq"):
                    config = directory / "quantize.toml"
                    config.write_text(
                        "format_version = 1\n[paths]\n"
                        f'source = "{source.as_posix()}"\nprofile = "{profile.as_posix()}"\n'
                        f'quantized_output = "{output.as_posix()}"\n'
                        f'gptq_calibration = "{calibration.as_posix()}"\n',
                        encoding="utf-8",
                    )
                    quantize_model(None, None, profile=None, config=config)
                else:
                    convert_model_from_profile(source, output, profile, on_entry_started=None)
                with safe_open(str(output), framework="pt", device="cpu") as artifact:
                    outputs[action] = (
                        {key: (artifact.get_slice(key).get_dtype(), artifact.get_slice(key).get_shape())
                         for key in artifact.keys()},
                        artifact.metadata(),
                        artifact.get_tensor("a.comfy_quant"),
                    )
            ordinary = outputs["convrot_w4a4"]
            for action in ("convrot_w4a4_mse", "convrot_w4a4_mse_gptq"):
                self.assertEqual(outputs[action][:2], ordinary[:2])
                self.assertTrue(torch.equal(outputs[action][2], ordinary[2]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_gptq_uses_existing_mse_scales_and_cpu_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            calibration = self._artifact(directory)
            artifact = GPTQCalibration(calibration)
            cuda_factors = artifact.factors_for_weight("a.weight", device="cuda")
            self.assertIs(cuda_factors, artifact.factors_for_weight("b.weight", device="cuda"))
            self.assertEqual(cuda_factors[0].device.type, "cuda")
            torch.manual_seed(11)
            weights = (torch.randn(2, 256) * 0.1).bfloat16()
            factors = artifact.factors_for_weight("a.weight")
            cpu = quantize_convrot_w4a4_mse_gptq(weights, factors, device="cpu")
            cuda = quantize_convrot_w4a4_mse_gptq(weights, factors, device="cuda")
            self.assertEqual(cuda.packed_codes.device.type, "cpu")
            self.assertEqual(cuda.scales.device.type, "cpu")
            self.assertTrue(torch.equal(cuda.scales, quantize_convrot_w4a4_mse(weights, device="cuda").scales))
            self.assertTrue(torch.equal(cpu.packed_codes, cuda.packed_codes))

            source = directory / "source.safetensors"
            profile = directory / "profile.json"
            output = directory / "cuda.safetensors"
            save_file({"a.weight": weights, "b.weight": weights.clone()}, str(source))
            profile.write_text(json.dumps({
                "format_version": 1, "profile_id": "gptq-cuda-test",
                "default": "convrot_w4a4_mse_gptq", "rules": [],
            }), encoding="utf-8")
            convert_model_from_profile(
                source, output, profile, gptq_calibration=calibration,
                device="cuda", on_entry_started=None,
            )
            self.assertTrue(torch.equal(load_file(str(output))["a.weight"], cuda.packed_codes))


if __name__ == "__main__":
    unittest.main()
