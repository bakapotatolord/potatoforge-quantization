import unittest

import torch

from potatoforge.quantization.int5_rowwise import (
    Int5RowwiseResult,
    dequantize_int5_convrot,
    dequantize_int5_convrot_packed,
    dequantize_int5_rowwise,
    quantize_int5_convrot,
    quantize_int5_convrot_packed,
    quantize_int5_rowwise,
)


class TestInt5Rowwise(unittest.TestCase):
    def test_accepts_supported_float_dtypes(self) -> None:
        values = [[1.0, -0.5, 0.25]]

        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                result = quantize_int5_rowwise(torch.tensor(values, dtype=dtype))

                self.assertEqual(result.codes.dtype, torch.int8)
                self.assertEqual(result.scales.dtype, torch.float32)

    def test_rowwise_result_has_valid_scales_and_codes(self) -> None:
        weights = torch.tensor(
            [[1.0, -0.5, 0.0], [0.25, -0.125, 0.125]],
        )

        result = quantize_int5_rowwise(weights)
        reconstructed = dequantize_int5_rowwise(result)

        self.assertEqual(result.codes.dtype, torch.int8)
        self.assertEqual(tuple(result.codes.shape), tuple(weights.shape))
        self.assertEqual(tuple(result.scales.shape), (2, 1))
        self.assertTrue(bool(torch.isfinite(result.scales).all()))
        self.assertTrue(bool((result.scales > 0).all()))
        self.assertTrue(bool(((result.codes >= -15) & (result.codes <= 15)).all()))
        self.assertEqual(reconstructed.dtype, torch.float32)

    def test_zero_rows_use_a_safe_unit_scale(self) -> None:
        result = quantize_int5_rowwise(torch.tensor([[0.0, 0.0], [1.0, -1.0]]))

        self.assertEqual(float(result.scales[0, 0]), 1.0)
        self.assertTrue(bool((result.codes[0] == 0).all()))

    def test_packed_convrot_reconstruction_matches_logical_result(self) -> None:
        generator = torch.Generator().manual_seed(1234)
        weights = torch.randn((2, 256), generator=generator)

        logical = dequantize_int5_convrot(quantize_int5_convrot(weights))
        packed = dequantize_int5_convrot_packed(
            quantize_int5_convrot_packed(weights),
        )

        self.assertTrue(torch.equal(packed, logical))

    def test_convrot_rejects_ineligible_width_and_nonfinite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by 256"):
            quantize_int5_convrot(torch.ones((1, 4)))

        with self.assertRaisesRegex(ValueError, "finite"):
            quantize_int5_convrot(torch.full((1, 256), float("nan")))

    def test_rejects_malformed_logical_results(self) -> None:
        with self.assertRaises(ValueError):
            quantize_int5_rowwise(torch.ones(3))
        with self.assertRaises(ValueError):
            quantize_int5_rowwise(torch.ones((2, 2), dtype=torch.int8))

        invalid = Int5RowwiseResult(
            torch.tensor([[-16]], dtype=torch.int8),
            torch.tensor([[-1.0]], dtype=torch.float32),
        )
        with self.assertRaises(ValueError):
            dequantize_int5_rowwise(invalid)

        invalid_codes = Int5RowwiseResult(
            torch.tensor([[16]], dtype=torch.int8),
            torch.tensor([[1.0]], dtype=torch.float32),
        )
        with self.assertRaises(ValueError):
            dequantize_int5_rowwise(invalid_codes)


if __name__ == "__main__":
    unittest.main()
