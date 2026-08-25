import unittest

import torch

from potatoforge.quantization.hadamard import (
    apply_hadamard_rotation,
    build_normalized_hadamard_matrix,
)

class TestHadamard(unittest.TestCase):
    def test_builds_normalized_hadamard_matrix(self) -> None:
        matrix = build_normalized_hadamard_matrix(4)

        expected = torch.tensor(
            [
                [0.5, 0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
        )

        self.assertEqual(matrix.dtype, torch.float32)
        self.assertTrue(torch.equal(matrix, expected))

        identity = torch.eye(4, dtype=torch.float32)
        self.assertTrue(torch.equal(matrix @ matrix.T, identity))

    def test_rotates_with_comfy_regular_hadamard(self) -> None:
        weights = torch.tensor(
            [[0.0, 0.0, 0.0, 70.0]],
            dtype=torch.bfloat16,
        )

        rotated = apply_hadamard_rotation(
            weights,
            group_size=4,
        )

        expected = torch.tensor(
            [[-35.0, 35.0, 35.0, 35.0]],
            dtype=torch.float32,
        )

        self.assertTrue(torch.equal(rotated, expected))