import unittest

import torch

from potatoforge.quantization import (
    dequantize_int8_tensorwise,
    quantize_int8_tensorwise,
    quantize_int8_convrot,
    quantize_int8_tensorwise,
    dequantize_int8_convrot
)


class TestTorchInt8Tensorwise(unittest.TestCase):
    def test_quantizes_bfloat16_rows(self) -> None:
        weights = torch.tensor(
            [
                [1.0, 0.25, -1.0],
                [0.01, -0.02, 0.02],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.bfloat16,
        )

        result = quantize_int8_tensorwise(weights)

        self.assertEqual(result.codes.dtype, torch.int8)
        self.assertEqual(result.codes.shape, (3, 3))

        self.assertEqual(result.scales.dtype, torch.float32)
        self.assertEqual(result.scales.shape, (3, 1))

        self.assertEqual(
            result.codes.tolist(),
            [
                [127, 32, -127],
                [64, -127, 127],
                [0, 0, 0],
            ],
        )

        self.assertGreater(result.scales[2, 0].item(), 0)

    def test_quantizes_float16_rows(self) -> None:
        weights = torch.tensor(
            [
                [1.0, 0.25, -1.0],
                [0.01, -0.02, 0.02],
            ],
            dtype=torch.float16,
        )

        result = quantize_int8_tensorwise(weights)

        self.assertEqual(result.codes.dtype, torch.int8)
        self.assertEqual(result.scales.dtype, torch.float32)
        self.assertEqual(
            result.codes.tolist(),
            [
                [127, 32, -127],
                [63, -127, 127],
            ],
        )

    def test_rejects_non_bfloat16_weights(self) -> None:
        weights = torch.tensor(
            [[1.0]],
            dtype=torch.float32,
        )

        with self.assertRaises(ValueError):
            quantize_int8_tensorwise(weights)

    def test_rejects_non_matrix_weights(self) -> None:
        weights = torch.tensor(
            [1.0],
            dtype=torch.bfloat16,
        )

        with self.assertRaises(ValueError):
            quantize_int8_tensorwise(weights)

    def test_dequantizes_rows_with_their_scales(self) -> None:
        weights = torch.tensor(
            [
                [1.0, 0.25, -1.0],
                [0.01, -0.02, 0.02],
            ],
            dtype=torch.bfloat16,
        )

        result = quantize_int8_tensorwise(weights)
        reconstructed = dequantize_int8_tensorwise(result)

        self.assertEqual(reconstructed.dtype, torch.float32)
        self.assertEqual(reconstructed.shape, weights.shape)

        errors = (reconstructed - weights.float()).abs()

        self.assertTrue(
            torch.all(errors <= result.scales + 1e-7)
        )

    def test_quantizes_convrot_int8_weights(self) -> None:
        weights = torch.zeros(
            (1, 256),
            dtype=torch.bfloat16,
        )
        weights[0, 0] = 70.0

        result = quantize_int8_convrot(weights)

        self.assertEqual(result.codes.dtype, torch.int8)
        self.assertEqual(tuple(result.codes.shape), (1, 256))

        self.assertEqual(result.scales.dtype, torch.float32)
        self.assertEqual(tuple(result.scales.shape), (1, 1))
        self.assertAlmostEqual(
            result.scales[0, 0].item(),
            4.375 / 127,
        )

        self.assertEqual(
            result.codes[0, :4].tolist(),
            [127, 127, 127, -127],
        )

    def test_quantizes_float16_convrot_int8_weights(self) -> None:
        weights = torch.zeros(
            (1, 256),
            dtype=torch.float16,
        )
        weights[0, 0] = 70.0

        result = quantize_int8_convrot(weights)

        self.assertEqual(result.codes.dtype, torch.int8)
        self.assertEqual(result.scales.dtype, torch.float32)
        self.assertAlmostEqual(
            result.scales[0, 0].item(),
            4.375 / 127,
        )
        self.assertEqual(
            result.codes[0, :4].tolist(),
            [127, 127, 127, -127],
        )

    def test_dequantizes_convrot_int8_weights(self) -> None:
        weights = torch.zeros(
            (1, 256),
            dtype=torch.bfloat16,
        )
        weights[0, 0] = 70.0

        result = quantize_int8_convrot(weights)
        dequantized = dequantize_int8_convrot(result)

        self.assertEqual(dequantized.dtype, torch.float32)
        self.assertEqual(tuple(dequantized.shape), (1, 256))
        self.assertTrue(
            torch.allclose(
                dequantized,
                weights.float(),
                rtol=0.0,
                atol=1e-4,
            )
        )
