import math
import unittest

import torch

from potatoforge.audits.all_comparison import compare_all_reconstructions, l2_norm

PLAIN_FIELDS = ("int8_relative_l2_error", "int6_relative_l2_error", "int7_relative_l2_error")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class TestCudaResidentComparison(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(4242)
        weights = torch.randn((96, 1024), generator=generator) * 0.02
        weights[3].zero_()
        weights[5, 7] = 0.5
        self.weights = weights.to(torch.bfloat16)

    def test_matches_cpu_run(self) -> None:
        cpu = compare_all_reconstructions(self.weights, device="cpu")
        cuda = compare_all_reconstructions(self.weights, device="cuda")
        for field in cpu._fields:
            expected, actual = getattr(cpu, field), getattr(cuda, field)
            with self.subTest(field=field):
                self.assertIsInstance(actual, float)
                self.assertTrue(math.isfinite(actual))
                # Plain codes are identical, so only the FP64 sum order differs;
                # GPU kernels may round a few near-tie ConvRot values differently.
                tolerance = 1e-12 if field in PLAIN_FIELDS else 1e-3
                self.assertAlmostEqual(actual, expected, delta=tolerance * expected + 1e-15)

    def test_precomputed_cpu_original_and_norm_give_identical_results(self) -> None:
        original = self.weights.float()
        derived = compare_all_reconstructions(self.weights, device="cuda")
        supplied = compare_all_reconstructions(
            self.weights,
            original_float=original,
            source_l2=l2_norm(original),
            device="cuda",
        )
        self.assertEqual(derived, supplied)
        self.assertEqual(self.weights.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
