import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import torch

from potatoforge.quantization.convrot_w4a4 import (
    _calculate_w4a4_candidate_mse,
)
from potatoforge.quantization.w4a4_mse_native import (
    W4A4MSECandidateEvaluator,
    load_w4a4_mse_candidate,
)


class TestW4A4MseNative(unittest.TestCase):
    def setUp(self) -> None:
        load_w4a4_mse_candidate.cache_clear()

    def tearDown(self) -> None:
        load_w4a4_mse_candidate.cache_clear()

    def test_missing_native_library_falls_back(self) -> None:
        with TemporaryDirectory() as directory:
            library_path = Path(directory) / "candidate.dll"
            library_path.write_bytes(b"not a native library")
            with patch.dict(
                os.environ,
                {"POTATOFORGE_W4A4_MSE_LIBRARY": str(library_path)},
            ):
                with patch(
                    "potatoforge.quantization.w4a4_mse_native.ctypes.CDLL",
                    side_effect=OSError("not loadable"),
                ):
                    self.assertIsNone(load_w4a4_mse_candidate(0))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_loaded_native_candidate_matches_torch_candidate(self) -> None:
        evaluator = load_w4a4_mse_candidate(torch.cuda.current_device())
        if evaluator is None:
            self.skipTest("fused W4A4-MSE library is not built")

        torch.manual_seed(61)
        weights = torch.randn((2, 256), device="cuda", dtype=torch.float32)
        weights[0, 0] = 0
        weights[1, 1] = -0.5
        weights_float = weights.float()
        scales = torch.tensor(
            [[0.73], [0.125]],
            device="cuda",
            dtype=torch.float32,
        )

        expected = _calculate_w4a4_candidate_mse(
            weights,
            weights_float,
            scales,
        )
        observed = evaluator(weights, weights_float, scales)
        torch.cuda.synchronize()

        torch.testing.assert_close(observed, expected, rtol=0, atol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_native_invocation_failure_is_not_silently_replaced(self) -> None:
        function = Mock(return_value=7)
        evaluator = W4A4MSECandidateEvaluator(
            object(),
            function,
            torch.device("cuda", torch.cuda.current_device()),
        )
        weights = torch.ones((1, 256), device="cuda", dtype=torch.float32)
        scales = torch.ones((1, 1), device="cuda", dtype=torch.float32)

        with self.assertRaisesRegex(RuntimeError, "status 7"):
            evaluator(weights, weights, scales)


if __name__ == "__main__":
    unittest.main()
