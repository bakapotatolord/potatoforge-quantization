import unittest

import torch

from potatoforge.audits.activation_error import activation_weighted_error


class TestActivationWeightedError(unittest.TestCase):
    def test_matches_the_manual_example(self) -> None:
        reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        candidate = torch.tensor([[2.0, 2.0], [3.0, 6.0]])

        result = activation_weighted_error(
            reference,
            candidate,
            torch.tensor([10.0, 3.0]),
        )

        self.assertEqual(result.activation_error, 22.0)
        torch.testing.assert_close(
            result.weight_error_per_input_sum,
            torch.tensor([1.0, 4.0]),
        )
        self.assertEqual(result.activation_energy_sum, 13.0)
        self.assertEqual(result.input_features, 2)

    def test_zero_activation_channel_contributes_nothing(self) -> None:
        result = activation_weighted_error(
            torch.zeros((2, 2)),
            torch.tensor([[1.0, 2.0], [0.0, 3.0]]),
            torch.tensor([100.0, 0.0]),
        )

        self.assertEqual(result.activation_error, 100.0)

    def test_exact_reconstruction_has_zero_error(self) -> None:
        reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        result = activation_weighted_error(
            reference,
            reference.clone(),
            torch.tensor([10.0, 3.0]),
        )

        self.assertEqual(result.activation_error, 0.0)

    def test_rejects_invalid_shapes(self) -> None:
        with self.assertRaises(ValueError):
            activation_weighted_error(
                torch.ones(2),
                torch.ones(2),
                torch.ones(2),
            )
        with self.assertRaises(ValueError):
            activation_weighted_error(
                torch.ones((2, 2)),
                torch.ones((2, 3)),
                torch.ones(2),
            )
        with self.assertRaises(ValueError):
            activation_weighted_error(
                torch.ones((2, 2)),
                torch.ones((2, 2)),
                torch.ones((3,)),
            )
        with self.assertRaises(ValueError):
            activation_weighted_error(
                torch.ones((2, 2)),
                torch.ones((2, 2)),
                torch.ones((1, 2)),
            )

    def test_rejects_non_finite_or_negative_activation_energy(self) -> None:
        reference = torch.ones((2, 2))

        with self.assertRaises(ValueError):
            activation_weighted_error(
                reference,
                torch.tensor([[float("nan"), 1.0], [1.0, 1.0]]),
                torch.ones(2),
            )
        with self.assertRaises(ValueError):
            activation_weighted_error(
                reference,
                reference,
                torch.tensor([1.0, -1.0]),
            )

    def test_uses_float32_for_the_metric(self) -> None:
        result = activation_weighted_error(
            torch.tensor([[1.0, 2.0]], dtype=torch.float16),
            torch.tensor([[2.0, 4.0]], dtype=torch.float16),
            torch.tensor([3.0, 5.0], dtype=torch.float64),
        )

        self.assertEqual(result.weight_error_per_input_sum.dtype, torch.float32)
        self.assertEqual(result.activation_error, 23.0)
