import unittest

import torch

from potatoforge.quantization.convrot_w4a4 import _quantize_w4_codes
from potatoforge.quantization.int8_tensorwise import _quantize_int8_with_scale
from potatoforge.quantization.rowwise import quantize_symmetric_rows


class TestFinalCodeBuffers(unittest.TestCase):
    def test_fixed_scale_codes_preserve_rounding_clipping_and_inputs(self):
        values = torch.tensor([
            -200, -128.5, -127.5, -7.5, -6.5, -1.5, -0.5, 0,
            0.5, 1.5, 6.5, 7.5, 126.5, 127.5, 128.5, 200,
        ])
        encoders = (
            (_quantize_int8_with_scale, [-128, -128, -128, -8, -6, -2, 0, 0, 0, 2, 6, 8, 126, 127, 127, 127]),
            (_quantize_w4_codes, [-7, -7, -7, -7, -6, -2, 0, 0, 0, 2, 6, 7, 7, 7, 7, 7]),
        )
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                scales = torch.tensor([1, 0.5, 2, 1], device=device)[:, None]
                weights = (values.to(device).repeat(4, 1) * scales).to(dtype)
                weights[-1].zero_()
                weights = weights.t().contiguous().t().requires_grad_()
                saved_weights = weights.detach().clone()
                saved_scales = scales.clone()
                for encode, row in encoders:
                    with self.subTest(device=device, dtype=dtype, encoder=encode.__name__):
                        expected = torch.tensor(row, dtype=torch.int8, device=device).repeat(4, 1)
                        expected[-1].zero_()
                        codes, stored_scales = encode(weights, scales)
                        self.assertEqual(codes.dtype, torch.int8)
                        self.assertEqual(stored_scales.dtype, torch.float32)
                        self.assertTrue(torch.equal(codes, expected))
                        self.assertTrue(torch.equal(stored_scales, saved_scales))
                        self.assertTrue(torch.equal(weights, saved_weights))
                        self.assertTrue(torch.equal(scales, saved_scales))

    def test_symmetric_codes_preserve_both_arithmetic_branches_and_inputs(self):
        values = torch.tensor([
            -127, -126.5, -7.5, -6.5, -1.5, -0.5, 0, 0.5,
            1.5, 6.5, 7.5, 126.5, 127, 1, -1, 2,
        ])
        expected_row = [-127, -126, -8, -6, -2, 0, 0, 0, 2, 6, 8, 126, 127, 1, -1, 2]
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                scales = torch.tensor([1, 0.5, 2, 1], device=device)[:, None]
                weights = (values.to(device).repeat(4, 1) * scales).to(dtype)
                weights[-1].zero_()
                weights = weights.t().contiguous().t().requires_grad_()
                saved = weights.detach().clone()
                expected_codes = torch.tensor(expected_row, dtype=torch.int8, device=device).repeat(4, 1)
                expected_codes[-1].zero_()
                for use_float32, zero_scale in ((False, None), (True, None), (True, 1.0)):
                    with self.subTest(device=device, dtype=dtype, fp32=use_float32, zero_scale=zero_scale):
                        expected_scales = scales.clone()
                        expected_scales[-1] = 1e-30 if zero_scale is None else zero_scale
                        result = quantize_symmetric_rows(
                            weights, use_float32_math=use_float32, zero_scale=zero_scale,
                        )
                        self.assertEqual(result.codes.dtype, torch.int8)
                        self.assertEqual(result.scales.dtype, torch.float32)
                        self.assertTrue(torch.equal(result.codes, expected_codes))
                        self.assertTrue(torch.equal(result.scales, expected_scales))
                        self.assertTrue(torch.equal(weights, saved))
