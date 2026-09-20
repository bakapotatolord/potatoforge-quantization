import unittest
from unittest.mock import patch

import torch

from potatoforge.quantization.int6_rowwise import (
    INT6_QMAX,
    INT6_QMIN,
    Int6RowwiseResult,
    dequantize_int6_convrot,
    dequantize_int6_convrot_packed,
    dequantize_int6_rowwise,
    quantize_int6_convrot,
    quantize_int6_convrot_mse,
    quantize_int6_convrot_packed,
    quantize_int6_convrot_mse_packed,
    quantize_int6_rowwise,
    select_int6_mse_scale,
)
from potatoforge.quantization.rowwise import (
    calculate_rowwise_candidate_mse,
    quantize_rows_with_scale,
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

    def test_packed_convrot_reconstruction_matches_logical_result(self) -> None:
        weights = torch.randn((2, 256), dtype=torch.float32)

        logical = dequantize_int6_convrot(quantize_int6_convrot(weights))
        packed = dequantize_int6_convrot_packed(
            quantize_int6_convrot_packed(weights),
        )

        self.assertTrue(torch.equal(packed, logical))

    def test_mse_convrot_uses_the_same_packed_contract(self) -> None:
        generator = torch.Generator().manual_seed(4321)
        weights = torch.randn((2, 256), generator=generator)

        logical = quantize_int6_convrot_mse(weights)
        packed = quantize_int6_convrot_mse_packed(weights)

        self.assertEqual(logical.codes.dtype, torch.int8)
        self.assertEqual(logical.scales.dtype, torch.float32)
        self.assertEqual(logical.scales.shape, (2, 1))
        self.assertTrue(bool(((logical.codes >= -31) & (logical.codes <= 31)).all()))
        self.assertTrue(torch.equal(
            dequantize_int6_convrot(logical),
            dequantize_int6_convrot_packed(packed),
        ))

    def test_mse_candidate_matches_scale_aware_quantization(self) -> None:
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                generator = torch.Generator().manual_seed(7654)
                weights = torch.randn((2, 256), generator=generator, dtype=dtype)
                weights_float = weights.float()
                scales = quantize_int6_rowwise(weights).scales * 0.73

                observed = calculate_rowwise_candidate_mse(
                    weights,
                    weights_float,
                    scales,
                    qmin=INT6_QMIN,
                    qmax=INT6_QMAX,
                )
                result = quantize_rows_with_scale(
                    weights,
                    scales,
                    qmin=INT6_QMIN,
                    qmax=INT6_QMAX,
                )
                expected = (
                    weights_float - result.codes.float() * result.scales
                ).square().mean(dim=1, keepdim=True)

                self.assertTrue(torch.equal(observed, expected))

    def test_mse_search_uses_the_34_candidate_schedule(self) -> None:
        generator = torch.Generator().manual_seed(9876)
        weights = torch.randn((1, 256), generator=generator)
        base_scale = quantize_int6_rowwise(weights).scales
        evaluated_multipliers: list[float] = []
        evaluated_mse: list[float] = []

        def record_candidate(
            candidate_weights: torch.Tensor,
            candidate_weights_float: torch.Tensor,
            scales: torch.Tensor,
            *,
            qmin: int,
            qmax: int,
        ) -> torch.Tensor:
            evaluated_multipliers.append((scales / base_scale).item())
            mse = calculate_rowwise_candidate_mse(
                candidate_weights,
                candidate_weights_float,
                scales,
                qmin=qmin,
                qmax=qmax,
            )
            evaluated_mse.append(mse.item())
            return mse

        with patch(
            "potatoforge.quantization.rowwise."
            "calculate_rowwise_candidate_mse",
            side_effect=record_candidate,
        ):
            select_int6_mse_scale(weights)

        expected_coarse = [index / 20 for index in range(2, 21)]
        self.assertTrue(torch.allclose(
            torch.tensor(evaluated_multipliers[:19]),
            torch.tensor(expected_coarse),
            rtol=0,
            atol=1e-7,
        ))
        best_coarse = evaluated_multipliers[
            evaluated_mse[:19].index(min(evaluated_mse[:19]))
        ]
        lower = max(best_coarse - 0.05, 0.10)
        upper = min(best_coarse + 0.05, 1.0)
        expected_fine = [
            lower + (upper - lower) * step / 16
            for step in range(1, 16)
        ]
        self.assertEqual(len(evaluated_multipliers), 34)
        self.assertTrue(torch.allclose(
            torch.tensor(evaluated_multipliers[19:]),
            torch.tensor(expected_fine),
            rtol=0,
            atol=1e-7,
        ))

    def test_mse_reconstruction_is_no_worse_than_baseline(self) -> None:
        for seed, rows in ((41, 3), (43, 2), (47, 1)):
            with self.subTest(seed=seed):
                generator = torch.Generator().manual_seed(seed)
                weights = torch.randn((rows, 256), generator=generator)
                baseline = quantize_int6_convrot(weights)
                optimized = quantize_int6_convrot_mse(weights)

                baseline_error = (
                    weights - dequantize_int6_convrot(baseline)
                ).square().mean(dim=1)
                optimized_error = (
                    weights - dequantize_int6_convrot(optimized)
                ).square().mean(dim=1)

                self.assertTrue(torch.all(optimized_error <= baseline_error + 1e-7))

    def test_mse_zero_rows_keep_the_baseline_unit_scale(self) -> None:
        weights = torch.zeros((2, 256), dtype=torch.float32)
        weights[1, 0] = 1.0

        baseline = quantize_int6_convrot(weights)
        optimized = quantize_int6_convrot_mse(weights)

        self.assertEqual(float(optimized.scales[0, 0]), float(baseline.scales[0, 0]))
        self.assertTrue(bool((optimized.codes[0] == 0).all()))

    def test_finite_subnormal_rows_use_the_comfy_scale_floor(self) -> None:
        weights = torch.zeros((1, 256), dtype=torch.float32)
        weights[0, 0] = 1e-43

        for result in (
            quantize_int6_convrot(weights),
            quantize_int6_convrot_mse(weights),
        ):
            self.assertTrue(torch.equal(result.scales, torch.full_like(result.scales, 1e-30)))

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            quantize_int6_rowwise(torch.ones(3))
        with self.assertRaises(ValueError):
            quantize_int6_rowwise(torch.ones((2, 2), dtype=torch.int8))
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "INT6 weights must contain only finite values",
                ):
                    quantize_int6_rowwise(torch.tensor([[value]]))

        invalid_codes = Int6RowwiseResult(
            torch.tensor([[32]], dtype=torch.int8),
            torch.tensor([[1.0]], dtype=torch.float32),
        )
        with self.assertRaises(ValueError):
            dequantize_int6_rowwise(invalid_codes)


if __name__ == "__main__":
    unittest.main()
