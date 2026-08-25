import unittest

import torch

from potatoforge.audits.all_comparison import relative_l2_error


class TestRelativeL2Error(unittest.TestCase):
    def test_returns_zero_for_an_exact_reconstruction(self) -> None:
        original = torch.tensor([3.0, 4.0])
        reconstructed = torch.tensor([3.0, 4.0])

        error = relative_l2_error(original, reconstructed)

        self.assertEqual(error, 0.0)

    def test_returns_one_when_reconstruction_is_zero(self) -> None:
        original = torch.tensor([3.0, 4.0])
        reconstructed = torch.tensor([0.0, 0.0])

        error = relative_l2_error(original, reconstructed)

        self.assertAlmostEqual(error, 1.0)

    def test_rejects_different_shapes(self) -> None:
        original = torch.tensor([1.0, 2.0])
        reconstructed = torch.tensor([[1.0, 2.0]])

        with self.assertRaises(ValueError):
            relative_l2_error(original, reconstructed)
