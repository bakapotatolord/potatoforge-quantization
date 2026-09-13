import unittest
from unittest.mock import patch

import torch

from potatoforge.quantization.convrot_w4a4 import (
    _calculate_w4a4_candidate_mse,
    _quantize_w4_rowwise_with_scale,
    _run_w4a4_mse_coarse_search,
    dequantize_convrot_w4a4,
    dequantize_w4_rowwise,
    quantize_convrot_w4a4,
    quantize_convrot_w4a4_mse,
    quantize_w4_rowwise,
    _quantize_w4_codes,
    select_w4a4_mse_scale,
)


class TestW4A4MseScale(unittest.TestCase):
    def test_candidate_mse_matches_the_rowwise_quantizer(self) -> None:
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(101)
                weights = torch.randn((3, 18), dtype=dtype)
                base_scale = quantize_w4_rowwise(weights).scales
                scales = base_scale * torch.tensor(
                    [[0.61], [0.83], [1.0]],
                    dtype=torch.float32,
                )

                fast_mse = _calculate_w4a4_candidate_mse(
                    weights,
                    weights.float(),
                    scales,
                )
                result = _quantize_w4_rowwise_with_scale(weights, scales)
                expected_mse = (
                    weights.float() - dequantize_w4_rowwise(result)
                ).square().mean(dim=1, keepdim=True)

                self.assertTrue(
                    torch.allclose(fast_mse, expected_mse, rtol=0, atol=1e-8)
                )

    def test_candidate_mse_matches_the_direct_reference_expression(self) -> None:
        torch.manual_seed(109)
        weights = torch.randn((3, 256), dtype=torch.bfloat16)
        weights_float = weights.float()
        scales = quantize_w4_rowwise(weights).scales * 0.73
        codes, stored_scales = _quantize_w4_codes(weights, scales)
        reference = (
            weights_float - codes.float() * stored_scales
        ).square().mean(dim=1, keepdim=True)

        optimized = _calculate_w4a4_candidate_mse(
            weights,
            weights_float,
            scales,
        )

        self.assertTrue(torch.equal(optimized, reference))

    def test_mse_search_uses_the_lightweight_candidate_budget(self) -> None:
        torch.manual_seed(53)
        weights = torch.randn((1, 64), dtype=torch.bfloat16)
        base_scale = quantize_w4_rowwise(weights).scales
        evaluated_multipliers: list[float] = []
        evaluated_mse: list[float] = []

        def record_candidate(
            candidate_weights: torch.Tensor,
            candidate_weights_float: torch.Tensor,
            scales: torch.Tensor,
        ) -> torch.Tensor:
            evaluated_multipliers.append((scales / base_scale).item())
            mse = _calculate_w4a4_candidate_mse(
                candidate_weights,
                candidate_weights_float,
                scales,
            )
            evaluated_mse.append(mse.item())
            return mse

        with patch(
            "potatoforge.quantization.convrot_w4a4."
            "_calculate_w4a4_candidate_mse",
            side_effect=record_candidate,
        ):
            select_w4a4_mse_scale(weights)

        expected_coarse = [index / 20 for index in range(2, 21)]
        self.assertTrue(
            torch.allclose(
                torch.tensor(evaluated_multipliers[:19]),
                torch.tensor(expected_coarse),
                rtol=0,
                atol=1e-7,
            )
        )

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
        self.assertTrue(
            torch.allclose(
                torch.tensor(evaluated_multipliers[19:]),
                torch.tensor(expected_fine),
                rtol=0,
                atol=1e-7,
            )
        )

    def test_mse_search_accepts_an_explicit_candidate_evaluator(self) -> None:
        torch.manual_seed(59)
        weights = torch.randn((2, 256), dtype=torch.float32)
        weights_float = weights.float()
        base_scale = quantize_w4_rowwise(weights).scales

        expected = _run_w4a4_mse_coarse_search(
            weights,
            weights_float,
            base_scale,
        )
        observed = _run_w4a4_mse_coarse_search(
            weights,
            weights_float,
            base_scale,
            _calculate_w4a4_candidate_mse,
        )

        self.assertTrue(torch.equal(expected[0], observed[0]))
        self.assertTrue(torch.equal(expected[1], observed[1]))

    def test_convrot_mse_is_no_worse_with_the_same_w4a4_storage(self) -> None:
        for seed, rows in ((41, 3), (43, 2), (47, 1)):
            with self.subTest(seed=seed):
                torch.manual_seed(seed)
                weights = torch.randn((rows, 256), dtype=torch.bfloat16)

                baseline = quantize_convrot_w4a4(weights)
                optimized = quantize_convrot_w4a4_mse(weights)

                baseline_error = (
                    weights.float() - dequantize_convrot_w4a4(baseline)
                ).square().mean(dim=1)
                optimized_error = (
                    weights.float() - dequantize_convrot_w4a4(optimized)
                ).square().mean(dim=1)

                self.assertEqual(optimized.packed_codes.shape, baseline.packed_codes.shape)
                self.assertEqual(optimized.scales.shape, baseline.scales.shape)
                self.assertTrue(
                    torch.all(optimized_error <= baseline_error + 1e-6)
                )

    def test_zero_rows_keep_a_finite_baseline_scale(self) -> None:
        weights = torch.zeros((2, 10), dtype=torch.bfloat16)
        weights[1, 0] = 1.0

        baseline = quantize_w4_rowwise(weights)
        optimized_scales = select_w4a4_mse_scale(weights)

        self.assertTrue(torch.isfinite(optimized_scales).all())
        self.assertTrue(torch.equal(optimized_scales[0], baseline.scales[0]))


if __name__ == "__main__":
    unittest.main()
