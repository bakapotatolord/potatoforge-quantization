import gc
import unittest
from unittest.mock import Mock, patch
import weakref

import torch

from potatoforge.quantization import int7_rowwise as int7
from potatoforge.quantization import int8_tensorwise as int8
from potatoforge.quantization import w4a4_mse_native as native
from potatoforge.quantization.rowwise import calculate_rowwise_candidate_mse


class TestExactMseNative(unittest.TestCase):
    def setUp(self):
        native._load_squared_error_function.cache_clear()

    def tearDown(self):
        native._load_squared_error_function.cache_clear()

    def assert_bits_equal(self, expected, actual):
        self.assertEqual(expected.dtype, actual.dtype)
        self.assertEqual(expected.shape, actual.shape)
        self.assertTrue(torch.equal(expected.contiguous().view(torch.uint8), actual.contiguous().view(torch.uint8)))

    def test_cpu_weights_do_not_load_cuda_library(self):
        with patch.object(native, "_load_squared_error_function") as load:
            self.assertIsNone(native.create_exact_mse_candidate(torch.ones((2, 256)), bits=8))
        load.assert_not_called()

    def test_missing_library_or_new_exports_falls_back_and_is_cached(self):
        for library in (None, object()):
            with self.subTest(library=library):
                native._load_squared_error_function.cache_clear()
                path = Mock()
                path.is_file.return_value = True
                with patch.object(native, "_candidate_library_paths", return_value=(path,)), patch.object(
                    native.ctypes, "CDLL", return_value=library,
                    side_effect=OSError("missing") if library is None else None,
                ) as load:
                    for bits in (7, 8):
                        self.assertIsNone(native._load_squared_error_function(bits))
                        self.assertIsNone(native._load_squared_error_function(bits))
                self.assertEqual(load.call_count, 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_unsupported_cuda_weights_fall_back_before_loading(self):
        weights = torch.ones((2, 256), device="cuda")
        cases = (weights.half(), weights[:, ::2], weights.unsqueeze(0),
                 weights.clone().requires_grad_(), weights[:0], weights[:, :0],
                 torch.empty((65536, 1), device="cuda"))
        with patch.object(native, "_load_squared_error_function") as load:
            for case in cases:
                self.assertIsNone(native.create_exact_mse_candidate(case, bits=7))
        load.assert_not_called()

    def evaluator(self, weights, bits):
        evaluator = native.create_exact_mse_candidate(weights, bits=bits)
        if evaluator is None:
            self.skipTest("exact INT7/INT8 MSE CUDA exports are not built")
        return evaluator

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_scores_are_bit_identical_and_reuse_search_local_scratch(self):
        generator = torch.Generator().manual_seed(2718)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for bits in (7, 8):
                for columns in (1, 257, 512, 6144, 8192, 16384):
                    with self.subTest(bits=bits, columns=columns):
                        weights = torch.randn((8, columns), generator=generator).cuda()
                        weights[0].zero_()
                        weights[1].mul_(1e-20)
                        weights[2] = -weights[2].abs()
                        weights[3, 0] = 100
                        weights[4] = (torch.arange(columns, device="cuda").float() % 520 - 260.5) * 0.125
                        saved = weights.clone()
                        evaluate = self.evaluator(weights, bits)
                        other = self.evaluator(weights, bits)
                        self.assertIsNot(evaluate, other)
                        self.assertIsNone(evaluate._output)
                        for scale in (0.125, 1e-36, 0.73):
                            scales = torch.full((8, 1), scale, device="cuda")
                            saved_scales = scales.clone()
                            if bits == 8:
                                expected = int8._calculate_int8_candidate_mse(weights, weights, scales)
                            else:
                                expected = calculate_rowwise_candidate_mse(weights, weights, scales, qmin=-63, qmax=63)
                            scratch = evaluate._output
                            actual = evaluate(weights, weights, scales)
                            self.assert_bits_equal(expected, actual)
                            if scratch is not None:
                                self.assertIs(evaluate._output, scratch)
                            self.assertIsNone(other._output)
                            self.assert_bits_equal(saved_scales, scales)
                        self.assert_bits_equal(saved, weights)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_quantizer_payloads_match_torch_and_release_scratch(self):
        source = torch.randn((8, 512), generator=torch.Generator().manual_seed(1729))
        source[0].zero_()
        source[1, 0] = 20
        for bits, module, quantize in (
            (7, int7, int7.quantize_int7_convrot_mse_packed),
            (8, int8, int8.quantize_int8_convrot_mse),
        ):
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                values = source.to(dtype)
                for case in (values, values.t().contiguous().t(), values.cuda(), torch.zeros_like(values)):
                    with self.subTest(bits=bits, dtype=dtype, shape=case.shape):
                        saved = case.clone()
                        with patch.object(module, "create_exact_mse_candidate", return_value=None):
                            expected = quantize(case, device="cuda")
                        evaluators = []
                        def record(weights, **kwargs):
                            evaluate = self.evaluator(weights, kwargs["bits"])
                            evaluators.append(weakref.ref(evaluate))
                            return evaluate
                        with patch.object(module, "create_exact_mse_candidate", side_effect=record):
                            actual = quantize(case, device="cuda")
                        for old, new in zip(expected, actual, strict=True):
                            if isinstance(old, torch.Tensor):
                                self.assert_bits_equal(old, new)
                                self.assertEqual(new.device.type, "cpu")
                            else:
                                self.assertEqual(old, new)
                        self.assert_bits_equal(saved, case)
                        gc.collect()
                        self.assertTrue(all(reference() is None for reference in evaluators))

    @unittest.skipUnless(torch.cuda.device_count() > 1, "Multiple CUDA devices unavailable")
    def test_native_launch_selects_target_device_and_restores_current_device(self):
        original_device = torch.cuda.current_device()
        target = torch.device("cuda", (original_device + 1) % torch.cuda.device_count())
        weights = torch.randn((8, 512), device=target)
        scales = torch.full((8, 1), 0.125, device=target)
        for bits in (7, 8):
            with self.subTest(bits=bits):
                evaluate = self.evaluator(weights, bits)
                expected = (int8._calculate_int8_candidate_mse(weights, weights, scales) if bits == 8 else
                            calculate_rowwise_candidate_mse(weights, weights, scales, qmin=-63, qmax=63))
                self.assertEqual(torch.cuda.current_device(), original_device)
                actual = evaluate(weights, weights, scales)
                self.assert_bits_equal(expected, actual)
                self.assertEqual(torch.cuda.current_device(), original_device)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_failed_exact_launch_is_not_silently_replaced(self):
        device = torch.device("cuda", torch.cuda.current_device())
        weights = torch.ones((2, 256), device=device)
        scales = torch.ones((2, 1), device=device)
        evaluate = native.MSECandidateEvaluator(object(), Mock(return_value=7), device, "INT8 ConvRot", squared_error=True)
        with self.assertRaisesRegex(RuntimeError, "status 7"):
            evaluate(weights, weights, scales)


if __name__ == "__main__":
    unittest.main()
