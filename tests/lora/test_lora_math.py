import unittest

import torch

from potatoforge.lora.lora_math import calculate_additive_tensor_delta, calculate_linear_lora_delta, merge_tensor_contributions


def make_factors() -> tuple[torch.Tensor, torch.Tensor]:
    down = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=torch.bfloat16,
    )
    up = torch.tensor(
        [
            [5.0, 6.0],
            [7.0, 8.0],
        ],
        dtype=torch.bfloat16,
    )

    return down, up


class TestLinearLoraMath(unittest.TestCase):
    def test_calculates_delta_without_alpha(self) -> None:
        down, up = make_factors()

        result = calculate_linear_lora_delta(
            down,
            up,
            strength=1.0,
        )

        expected = torch.tensor(
            [
                [23.0, 34.0],
                [31.0, 46.0],
            ],
            dtype=torch.float32,
        )

        torch.testing.assert_close(result, expected)
        self.assertEqual(result.dtype, torch.float32)

    def test_applies_alpha_divided_by_rank(self) -> None:
        down, up = make_factors()

        result = calculate_linear_lora_delta(
            down,
            up,
            strength=1.0,
            alpha=4.0,
        )

        expected = torch.tensor(
            [
                [46.0, 68.0],
                [62.0, 92.0],
            ],
            dtype=torch.float32,
        )

        torch.testing.assert_close(result, expected)

    def test_applies_non_unit_strength(self) -> None:
        down, up = make_factors()

        result = calculate_linear_lora_delta(
            down,
            up,
            strength=0.5,
        )

        expected = torch.tensor(
            [
                [11.5, 17.0],
                [15.5, 23.0],
            ],
            dtype=torch.float32,
        )

        torch.testing.assert_close(result, expected)


class TestTensorContributionMerge(unittest.TestCase):
    def test_scales_additive_delta_in_fp32(self) -> None:
        delta = torch.tensor(
            [[2.0, 4.0]],
            dtype=torch.bfloat16,
        )

        result = calculate_additive_tensor_delta(
            delta,
            strength=0.5,
        )

        expected = torch.tensor(
            [[1.0, 2.0]],
            dtype=torch.float32,
        )

        torch.testing.assert_close(result, expected)
        self.assertEqual(result.dtype, torch.float32)

    def test_merges_lora_and_additive_contributions(self) -> None:
        down, up = make_factors()

        lora_delta = calculate_linear_lora_delta(
            down,
            up,
            strength=1.0,
        )
        additive_delta = calculate_additive_tensor_delta(
            torch.tensor(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
                dtype=torch.float32,
            ),
            strength=1.0,
        )

        base = torch.ones(
            (2, 2),
            dtype=torch.bfloat16,
        )

        result = merge_tensor_contributions(
            base,
            [lora_delta, additive_delta],
        )

        expected = torch.tensor(
            [
                [25.0, 37.0],
                [35.0, 51.0],
            ],
            dtype=torch.float32,
        )

        self.assertEqual(result.dtype, torch.bfloat16)
        torch.testing.assert_close(result.float(), expected)

    def test_rejects_mismatched_contribution_shape(self) -> None:
        base = torch.ones(
            (2, 2),
            dtype=torch.bfloat16,
        )
        contribution = torch.ones(
            (1, 2),
            dtype=torch.float32,
        )

        with self.assertRaisesRegex(
            ValueError,
            "Contribution shape must match base tensor",
        ):
            merge_tensor_contributions(
                base,
                [contribution],
            )

if __name__ == "__main__":
    unittest.main()