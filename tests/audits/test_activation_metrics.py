import unittest

import torch

from potatoforge.audits.activation_metrics import (
    aggregate_diagonal_relative_sse,
    aggregate_observed_relative_sse,
    bias_free_output_energy,
    diagonal_reference_output_energy,
    evaluation_relative_sse,
    global_channel_contribution,
    row_relative_channel_sse,
    reduce_evaluation_scores,
    sampled_diagonal_approximation_ratio,
    sampled_diagonal_sse,
    sampled_direction_error,
    sampled_output_error,
    sampled_per_row_relative_sse,
    sampled_relative_sse,
)


class TestActivationMetrics(unittest.TestCase):
    def test_bias_free_output_energy_removes_matched_bias(self) -> None:
        torch.testing.assert_close(
            bias_free_output_energy(
                torch.tensor([[3.0, -3.0]]),
                torch.tensor([[5.0, 5.0]]),
                torch.tensor([2]),
                torch.tensor([1.0, -1.0]),
            ),
            torch.tensor([[1.0, 1.0]]),
        )

        with self.assertRaisesRegex(ValueError, "meaningful negatives"):
            bias_free_output_energy(
                torch.ones((1, 1)) * 2,
                torch.zeros((1, 1)),
                torch.ones(1, dtype=torch.int64),
                torch.ones(1),
            )

    def test_observed_metrics_and_reducers(self) -> None:
        error = torch.tensor([[1.0, 3.0], [2.0, 8.0]])
        output_energy = torch.tensor([[10.0, 10.0], [2.0, 8.0]])

        per_evaluation = evaluation_relative_sse(error, output_energy)

        torch.testing.assert_close(per_evaluation, torch.tensor([0.2, 1.0]))
        self.assertAlmostEqual(
            aggregate_observed_relative_sse(error, output_energy),
            14.0 / 30.0,
        )
        self.assertAlmostEqual(
            reduce_evaluation_scores(per_evaluation, "mean"),
            0.6,
        )
        self.assertEqual(reduce_evaluation_scores(per_evaluation, "max"), 1.0)
        self.assertAlmostEqual(
            reduce_evaluation_scores(per_evaluation, "p95"),
            0.96,
        )
        self.assertEqual(
            reduce_evaluation_scores(per_evaluation, "cvar20"),
            1.0,
        )
        self.assertAlmostEqual(
            reduce_evaluation_scores(
                per_evaluation,
                "global_ratio",
                weights=output_energy.sum(dim=1),
            ),
            14.0 / 30.0,
        )

    def test_diagonal_denominator_uses_reference_weights(self) -> None:
        weights = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        input_energy = torch.tensor([[2.0, 5.0], [1.0, 3.0]])
        error = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        expected_reference_energy = torch.tensor(
            [[22.0, 98.0], [13.0, 57.0]]
        )

        torch.testing.assert_close(
            diagonal_reference_output_energy(weights, input_energy),
            expected_reference_energy,
        )
        self.assertAlmostEqual(
            aggregate_diagonal_relative_sse(error, weights, input_energy),
            10.0 / 190.0,
        )

    def test_channel_metrics_keep_global_and_row_relative_meanings_distinct(
        self,
    ) -> None:
        error = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        output_energy = torch.tensor([[10.0, 1.0, 0.0], [10.0, 3.0, 0.0]])

        torch.testing.assert_close(
            global_channel_contribution(error, output_energy),
            torch.tensor([5.0 / 24.0, 7.0 / 24.0, 9.0 / 24.0]),
        )
        torch.testing.assert_close(
            row_relative_channel_sse(
                error,
                output_energy,
                floor_fraction=0.5,
            ),
            torch.tensor([0.25, 7.0 / 6.0, 1.5]),
        )

        with self.assertRaisesRegex(ValueError, "zero"):
            row_relative_channel_sse(error, output_energy)

    def test_tail_and_denominator_edge_cases_are_explicit(self) -> None:
        scores = torch.tensor([1.0, 2.0, 100.0])
        self.assertEqual(reduce_evaluation_scores(scores, "cvar10"), 100.0)
        self.assertEqual(
            reduce_evaluation_scores(scores, "cvar20"),
            100.0,
        )
        self.assertEqual(
            evaluation_relative_sse(
                torch.zeros((2, 1)),
                torch.zeros((2, 1)),
            ).tolist(),
            [0.0, 0.0],
        )
        with self.assertRaisesRegex(ValueError, "zero"):
            evaluation_relative_sse(
                torch.ones((1, 1)),
                torch.zeros((1, 1)),
            )
        self.assertEqual(
            evaluation_relative_sse(
                torch.ones((1, 1)),
                torch.zeros((1, 1)),
                absolute_floor=2.0,
            ).item(),
            0.5,
        )

    def test_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            evaluation_relative_sse(torch.ones((2, 1)), torch.ones((1, 1)))
        with self.assertRaises(ValueError):
            reduce_evaluation_scores(torch.tensor([1.0]), "mean", weights=[1.0])
        with self.assertRaises(ValueError):
            reduce_evaluation_scores(torch.tensor([1.0]), "global_ratio")
        with self.assertRaises(ValueError):
            reduce_evaluation_scores(torch.tensor([1.0, -1.0]), "mean")
        with self.assertRaises(ValueError):
            diagonal_reference_output_energy(
                torch.ones((2, 2)),
                torch.ones((1, 3)),
            )

    def test_sampled_metrics_use_exact_cross_feature_math(self) -> None:
        sample_x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        reference = torch.eye(2)
        candidate = 2.0 * reference

        torch.testing.assert_close(
            sampled_output_error(sample_x, reference, candidate),
            sample_x,
        )
        self.assertEqual(sampled_relative_sse(sample_x, reference, candidate), 1.0)
        torch.testing.assert_close(
            sampled_per_row_relative_sse(sample_x, reference, candidate),
            torch.ones(2),
        )
        torch.testing.assert_close(
            sampled_direction_error(sample_x, reference, candidate),
            torch.zeros(2),
        )

    def test_sampled_diagonal_approximation_reports_cross_terms(self) -> None:
        sample_x = torch.tensor([[1.0, 2.0]])
        reference = torch.zeros((1, 2))
        candidate = torch.ones((1, 2))

        self.assertEqual(sampled_diagonal_sse(sample_x, reference, candidate), 5.0)
        self.assertAlmostEqual(
            sampled_diagonal_approximation_ratio(
                sample_x,
                reference,
                candidate,
            ),
            9.0 / 5.0,
        )

        both_zero = sampled_direction_error(
            torch.zeros((1, 2)),
            torch.zeros((1, 2)),
            torch.zeros((1, 2)),
        )
        torch.testing.assert_close(both_zero, torch.zeros(1))
        one_zero = sampled_direction_error(
            torch.ones((1, 2)),
            torch.zeros((1, 2)),
            torch.ones((1, 2)),
        )
        torch.testing.assert_close(one_zero, torch.ones(1))

    def test_sampled_metrics_reject_invalid_shapes(self) -> None:
        with self.assertRaises(ValueError):
            sampled_output_error(
                torch.ones((1, 3)),
                torch.ones((2, 2)),
                torch.ones((2, 2)),
            )
        with self.assertRaises(ValueError):
            sampled_relative_sse(
                torch.ones((1, 2)),
                torch.ones((2, 2)),
                torch.ones((1, 2)),
            )


if __name__ == "__main__":
    unittest.main()
