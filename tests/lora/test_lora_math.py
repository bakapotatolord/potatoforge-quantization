import unittest

import torch

from potatoforge.lora.lora_math import (
    calculate_additive_tensor_delta,
    calculate_linear_lora_delta,
    merge_tensor_contributions,
    reconstruct_lokr_direct,
)


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


class TestDirectLoKrMath(unittest.TestCase):
    def test_reconstructs_in_direct_w1_w2_orientation(self) -> None:
        w1 = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )
        w2 = torch.tensor(
            [
                [5.0, 6.0],
                [7.0, 8.0],
            ]
        )

        result = reconstruct_lokr_direct(
            w1,
            w2,
            target_shape=(4, 4),
        )

        expected = torch.tensor(
            [
                [5.0, 6.0, 10.0, 12.0],
                [7.0, 8.0, 14.0, 16.0],
                [15.0, 18.0, 20.0, 24.0],
                [21.0, 24.0, 28.0, 32.0],
            ]
        )

        torch.testing.assert_close(result, expected)

    def test_applies_strength_after_reconstruction(self) -> None:
        w1 = torch.ones((2, 2))
        w2 = torch.ones((2, 2))

        delta = reconstruct_lokr_direct(
            w1,
            w2,
            target_shape=(4, 4),
        )

        merged = torch.zeros((4, 4)) + 0.5 * delta

        torch.testing.assert_close(merged, torch.full((4, 4), 0.5))
        torch.testing.assert_close(
            torch.zeros((4, 4)) + 0.0 * delta,
            torch.zeros((4, 4)),
        )

    def test_accepts_krea_like_factor_shapes(self) -> None:
        delta = reconstruct_lokr_direct(
            torch.ones((4, 4)),
            torch.ones((3, 5)),
            target_shape=(12, 20),
        )

        self.assertEqual(delta.shape, (12, 20))

    def test_rejects_non_matrix_factors(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "only 2D lokr_w1/lokr_w2",
        ):
            reconstruct_lokr_direct(
                torch.ones((2, 2, 1)),
                torch.ones((2, 2)),
                target_shape=(4, 4),
            )

    def test_rejects_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "w1=.*w2=.*produced=.*target",
        ):
            reconstruct_lokr_direct(
                torch.ones((2, 2)),
                torch.ones((2, 3)),
                target_shape=(4, 5),
            )


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
