import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from potatoforge.quantization import convrot_w4a4, int5_rowwise, int6_rowwise, rowwise_convrot
from potatoforge.quantization.mse_search_native import (
    ROWWISE_MSE_SCALE_FLOOR,
    _load_search_function,
    fused_rowwise_mse_coarse_search,
    fused_rowwise_mse_fine_search,
)
from potatoforge.quantization.rowwise import (
    run_rowwise_mse_coarse_search,
    run_rowwise_mse_fine_search,
    select_rowwise_absmax_scale,
)
from potatoforge.quantization.w4a4_mse_native import (
    load_int5_convrot_mse_candidate,
    load_int6_convrot_mse_candidate,
    load_w4a4_mse_candidate,
)


def fused_search(weights, base, **kwargs):
    coarse = fused_rowwise_mse_coarse_search(weights, base, **kwargs)
    if coarse is None:
        return None
    return fused_rowwise_mse_fine_search(weights, base, *coarse, **kwargs)[1]


def _rowwise_reference(weights, qmin, qmax, candidate):
    base = select_rowwise_absmax_scale(weights, qmax=qmax)
    best_mse, best = run_rowwise_mse_coarse_search(
        weights, weights, base, qmin=qmin, qmax=qmax, candidate_mse=candidate,
    )
    _, best = run_rowwise_mse_fine_search(
        weights, weights, base, best_mse, best, qmin=qmin, qmax=qmax, candidate_mse=candidate,
    )
    return base, best


def _w4a4_reference(weights, candidate):
    _, base = convrot_w4a4._w4_absmax_and_scale(weights)
    best_mse, best = convrot_w4a4._run_w4a4_mse_coarse_search(weights, weights, base, candidate)
    _, best = convrot_w4a4._run_w4a4_mse_fine_search(weights, weights, base, best_mse, best, candidate)
    return base, best


class TestFusedRowwiseMseSearch(unittest.TestCase):
    def setUp(self) -> None:
        _load_search_function.cache_clear()

    def tearDown(self) -> None:
        _load_search_function.cache_clear()

    def test_library_without_search_symbol_falls_back(self) -> None:
        with TemporaryDirectory() as directory:
            library_path = Path(directory) / "candidate.dll"
            library_path.write_bytes(b"old native library")
            with patch.dict(os.environ, {"POTATOFORGE_W4A4_MSE_LIBRARY": str(library_path)}):
                with patch(
                    "potatoforge.quantization.mse_search_native.ctypes.CDLL",
                    return_value=object(),
                ):
                    self.assertIsNone(_load_search_function())

    def test_unsupported_inputs_fall_back(self) -> None:
        weights = torch.randn(4, 256)
        base = torch.ones(4, 1)
        self.assertIsNone(
            fused_search(weights, base, qmin=-31, qmax=31, scale_floor=1e-30)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_matches_per_candidate_search_bit_for_bit(self) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        if _load_search_function() is None:
            self.skipTest("fused rowwise MSE search library is not built")
        evaluators = (
            load_int5_convrot_mse_candidate(device.index),
            load_int6_convrot_mse_candidate(device.index),
            load_w4a4_mse_candidate(device.index),
        )
        if any(evaluator is None for evaluator in evaluators):
            self.skipTest("per-candidate MSE evaluators are not built")
        int5_candidate, int6_candidate, w4a4_candidate = evaluators

        torch.manual_seed(83)
        normal = torch.randn(48, 768)
        normal[3] = 0
        normal[7, 11] = 25.0
        # Exact half-integer quotients exercise the exact-division fallback.
        halves = (torch.randint(-14, 15, (16, 1024)).float() + 0.5).clamp(-7, 7)
        halves[:, 0] = 7.0
        halves[1::2] *= 15.0 / 7.0
        # Wider than the register cache, so rows are read from global memory.
        wide = torch.distributions.StudentT(3.0).sample((4, 16640))

        for values in (normal, halves, wide):
            weights = values.to(device=device, dtype=torch.float32).contiguous()
            cases = (
                (_rowwise_reference(weights, -15, 15, int5_candidate), -15, 15, ROWWISE_MSE_SCALE_FLOOR),
                (_rowwise_reference(weights, -31, 31, int6_candidate), -31, 31, ROWWISE_MSE_SCALE_FLOOR),
                (_w4a4_reference(weights, w4a4_candidate), -7, 7, convrot_w4a4._W4_SCALE_FLOOR),
            )
            for (base, expected), qmin, qmax, floor in cases:
                for row_cache_mode in (-1, 0):
                    with self.subTest(shape=tuple(weights.shape), qmax=qmax, mode=row_cache_mode):
                        observed = fused_search(
                            weights, base, qmin=qmin, qmax=qmax,
                            scale_floor=floor, row_cache_mode=row_cache_mode,
                        )
                        self.assertIsNotNone(observed)
                        self.assertTrue(torch.equal(observed, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_quantizers_match_per_candidate_path(self) -> None:
        if _load_search_function() is None:
            self.skipTest("fused rowwise MSE search library is not built")
        torch.manual_seed(89)
        weights = (torch.randn(64, 512) * 0.02).to(torch.bfloat16)
        weights[2] = 0
        cases = (
            (rowwise_convrot, int5_rowwise.quantize_int5_convrot_mse_packed),
            (rowwise_convrot, int6_rowwise.quantize_int6_convrot_mse_packed),
            (convrot_w4a4, convrot_w4a4.quantize_convrot_w4a4_mse),
        )
        for module, quantize in cases:
            with self.subTest(module=module.__name__):
                fused = quantize(weights, device="cuda")
                with patch.object(module, "fused_rowwise_mse_coarse_search", return_value=None):
                    reference = quantize(weights, device="cuda")
                self.assertTrue(torch.equal(fused[0], reference[0]))
                self.assertTrue(torch.equal(fused[1], reference[1]))


if __name__ == "__main__":
    unittest.main()
