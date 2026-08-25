import unittest

import torch

from potatoforge.quantization.int6_rowwise import (
    Int6RowwiseResult,
    dequantize_int6_convrot,
    dequantize_int6_rowwise,
    quantize_int6_convrot,
    quantize_int6_rowwise,
)


class TestInt6Rowwise(unittest.TestCase):
    def test_accepts_supported_float_dtypes(self) -> None:
        values = [[1.0, -0.5, 0.25]]

        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                result = quantize_int6_rowwise(
                    torch.tensor(values, dtype=dtype)
                )

                self.assertEqual(result.codes.dtype, torch.int8)
                self.assertEqual(result.scales.dtype, torch.float32)

    def test_rowwise_scales_and_logical_codes(self) -> None:
        weights = torch.tensor([[1.0, -0.5, 0.0], [0.25, -0.125, 0.125]])

        result = quantize_int6_rowwise(weights)
        reconstructed = dequantize_int6_rowwise(result)

        self.assertEqual(result.codes.dtype, torch.int8)
        self.assertEqual(tuple(result.scales.shape), (2, 1))
        self.assertAlmostEqual(float(result.scales[0, 0]), 1.0 / 31.0, places=7)
        self.assertAlmostEqual(float(result.scales[1, 0]), 0.25 / 31.0, places=7)
        self.assertTrue(bool(((result.codes >= -31) & (result.codes <= 31)).all()))
        self.assertEqual(reconstructed.dtype, torch.float32)

    def test_rounding_error_is_bounded_by_half_a_row_scale(self) -> None:
        weights = torch.tensor(
            [[-1.0, -0.33, 0.2], [0.01, -0.02, 0.015]], dtype=torch.float32
        )
        result = quantize_int6_rowwise(weights)

        error = (dequantize_int6_rowwise(result) - weights).abs()
        self.assertTrue(bool((error <= result.scales * 0.5 + 1e-7).all()))

    def test_zero_rows_use_a_safe_unit_scale(self) -> None:
        weights = torch.tensor([[0.0, 0.0, 0.0], [1.0, -1.0, 0.0]])

        result = quantize_int6_rowwise(weights)

        self.assertEqual(float(result.scales[0, 0]), 1.0)
        self.assertTrue(bool((result.codes[0] == 0).all()))

    def test_convrot_round_trip_restores_the_original_weight_space(self) -> None:
        weights = torch.zeros((1, 256), dtype=torch.bfloat16)
        weights[0, 0] = 70.0

        result = quantize_int6_convrot(weights)
        reconstructed = dequantize_int6_convrot(result)

        self.assertEqual(result.codes.shape, weights.shape)
        self.assertEqual(result.scales.shape, (1, 1))
        self.assertTrue(torch.allclose(reconstructed, weights.float(), atol=0.01))

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            quantize_int6_rowwise(torch.ones(3))
        with self.assertRaises(ValueError):
            quantize_int6_rowwise(torch.ones((2, 2), dtype=torch.int8))
        with self.assertRaises(ValueError):
            quantize_int6_rowwise(torch.tensor([[float("nan")]]))

        invalid_codes = Int6RowwiseResult(
            torch.tensor([[32]], dtype=torch.int8),
            torch.tensor([[1.0]], dtype=torch.float32),
        )
        with self.assertRaises(ValueError):
            dequantize_int6_rowwise(invalid_codes)


if __name__ == "__main__":
    unittest.main()
