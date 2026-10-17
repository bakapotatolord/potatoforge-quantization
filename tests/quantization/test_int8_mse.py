import unittest
from unittest.mock import patch

import torch

from potatoforge.quantization import int8_tensorwise as int8
from potatoforge.quantization.hadamard import apply_hadamard_rotation


class TestInt8ConvRotMSE(unittest.TestCase):
    def _weights(self, dtype):
        weights = torch.randn((8, 512), generator=torch.Generator().manual_seed(1729))
        weights[0].zero_()
        weights[1, 0] = 20
        return weights.to(dtype)

    def test_mse_reduces_rotated_error_and_preserves_plain_quantization(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                weights = self._weights(dtype)
                original = weights.clone()
                rotated = apply_hadamard_rotation(
                    weights, 256,
                    output_dtype=torch.bfloat16 if dtype == torch.bfloat16 else torch.float32,
                )
                baseline_scale = rotated.abs().amax(1, keepdim=True).float().div(127).clamp_min(1e-30)
                baseline_codes = torch.round(rotated / baseline_scale.to(rotated.dtype)).clamp(-128, 127).to(torch.int8)
                plain = int8.quantize_int8_convrot(weights)
                self.assertTrue(torch.equal(plain.codes, baseline_codes))
                self.assertTrue(torch.equal(plain.scales, baseline_scale))

                result = int8.quantize_int8_convrot_mse(weights)
                plain_error = (plain.codes.float() * plain.scales - rotated.float()).square().mean(1)
                mse_error = (result.codes.float() * result.scales - rotated.float()).square().mean(1)
                self.assertTrue(bool((mse_error <= plain_error + 1e-10).all()))
                self.assertLess(mse_error.sum().item(), plain_error.sum().item())
                self.assertEqual(result.codes.shape, weights.shape)
                self.assertEqual(result.codes.dtype, torch.int8)
                self.assertEqual(result.scales.shape, (8, 1))
                self.assertEqual(result.scales.dtype, torch.float32)
                self.assertTrue(bool(torch.isfinite(result.scales).all()))
                self.assertTrue(bool((result.scales > 0).all()))
                self.assertEqual(torch.count_nonzero(result.codes[0]).item(), 0)
                self.assertTrue(torch.equal(weights, original))

    def test_cpu_default_and_timing(self):
        weights = self._weights(torch.bfloat16)
        timings = {}
        default = int8.quantize_int8_convrot_mse(weights)
        explicit = int8.quantize_int8_convrot_mse(weights, device="cpu", internal_timings=timings)
        self.assertTrue(torch.equal(default.codes, explicit.codes))
        self.assertTrue(torch.equal(default.scales, explicit.scales))
        self.assertEqual(set(timings), {"prepare", "rotation", "mse_scale", "final_quantize", "finalize"})
        self.assertTrue(all(value >= 0 for value in timings.values()))

    def test_rejects_invalid_inputs_and_devices(self):
        for weights, message in (
            (torch.ones(256), "2D"),
            (torch.ones((2, 256), dtype=torch.int8), "BF16, F16, or F32"),
            (torch.ones((2, 128)), "divisible by 256"),
            (torch.full((2, 256), float("nan")), "finite"),
            (torch.full((2, 256), float("inf")), "finite"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    int8.quantize_int8_convrot_mse(weights)
        for device in ("mps", "invalid"):
            with self.subTest(device=device):
                with self.assertRaisesRegex(ValueError, "device must be cpu or cuda"):
                    int8.quantize_int8_convrot_mse(torch.ones((1, 256)), device=device)
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA.*unavailable"):
                int8.quantize_int8_convrot_mse(torch.ones((1, 256)), device="cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_preserves_cpu_payload_and_optimizes_matching_arithmetic(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                weights = self._weights(dtype)
                timings = {}
                result = int8.quantize_int8_convrot_mse(weights, device="cuda", internal_timings=timings)
                plain = int8.quantize_int8_convrot(weights, device="cuda")
                rotated = apply_hadamard_rotation(
                    weights.cuda(), 256,
                    output_dtype=torch.bfloat16 if dtype == torch.bfloat16 else torch.float32,
                ).float().cpu()
                plain_error = (plain.codes.float() * plain.scales - rotated).square().mean(1)
                mse_error = (result.codes.float() * result.scales - rotated).square().mean(1)
                self.assertTrue(bool((mse_error <= plain_error + 1e-10).all()))
                self.assertEqual(result.codes.device.type, "cpu")
                self.assertEqual(result.scales.device.type, "cpu")
                self.assertEqual(set(timings), {"prepare", "rotation", "mse_scale", "final_quantize", "finalize"})

        with patch.object(torch.cuda, "synchronize") as synchronize:
            int8.quantize_int8_convrot_mse(weights, device="cuda")
        synchronize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
