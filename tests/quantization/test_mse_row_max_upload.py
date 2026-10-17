import unittest
import weakref
from unittest.mock import Mock, patch

import torch

from potatoforge.quantization import convrot_w4a4 as w4
from potatoforge.quantization import int8_tensorwise as int8
from potatoforge.quantization import rowwise


class TestMseRowMaxAndUpload(unittest.TestCase):
    def test_shared_zero_rows_use_one_maximum_and_skip_search(self):
        devices = ("cpu", "cuda:0") if torch.cuda.is_available() else ("cpu",)
        original_amax = torch.Tensor.amax
        for device in devices:
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                with self.subTest(device=device, dtype=dtype):
                    weights = torch.zeros((2, 256), dtype=dtype, device=device)
                    maxima = []

                    def record_maximum(values, *args, **kwargs):
                        maxima.append(values.shape)
                        return original_amax(values, *args, **kwargs)

                    candidate = Mock(side_effect=AssertionError("Zero matrix must skip search"))
                    with patch.object(torch.Tensor, "amax", new=record_maximum):
                        scales = rowwise.select_rowwise_mse_scale(
                            weights, qmin=-128, qmax=127, candidate_mse=candidate,
                        )
                    self.assertEqual(maxima, [weights.shape])
                    self.assertTrue(torch.equal(scales, torch.ones((2, 1), device=device)))
                    candidate.assert_not_called()

    def test_w4_zero_rows_use_one_maximum_and_keep_scale_floor(self):
        devices = ("cpu", "cuda:0") if torch.cuda.is_available() else ("cpu",)
        original_amax = torch.Tensor.amax
        for device in devices:
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                selectors = [w4.select_w4a4_mse_scale]
                if device == "cuda:0" and dtype == torch.float32:
                    selectors.append(lambda values: w4.select_w4a4_mse_scale_cuda(values, torch.device(device)))
                for select in selectors:
                    with self.subTest(device=device, dtype=dtype, select=select):
                        weights = torch.zeros((2, 256), dtype=dtype, device=device)
                        maxima = []

                        def record_maximum(values, *args, **kwargs):
                            maxima.append(values.shape)
                            return original_amax(values, *args, **kwargs)

                        with patch.object(torch.Tensor, "amax", new=record_maximum):
                            scales = select(weights)
                        self.assertEqual(maxima, [weights.shape])
                        self.assertTrue(torch.equal(scales, torch.full((2, 1), 1e-10, device=device)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_owned_upload_is_released_but_caller_cuda_input_stays_alive(self):
        cases = (
            (int8, int8.quantize_int8_convrot, "quantize_symmetric_rows"),
            (int8, int8.quantize_int8_convrot_mse, "validate_rowwise_finite_weights"),
            (w4, w4.quantize_convrot_w4a4, "_select_w4_absmax_scale"),
            (w4, w4.quantize_convrot_w4a4_mse, "select_w4a4_mse_scale_cuda"),
        )
        for module, quantize, phase_name in cases:
            for source_device in ("cpu", "cuda:0"):
                with self.subTest(quantizer=quantize.__name__, source_device=source_device), torch.inference_mode():
                    source = torch.arange(512, dtype=torch.float32).reshape(2, 256).to(device=source_device, dtype=torch.bfloat16)
                    saved = source.clone()
                    uploaded = []
                    rotate = module.apply_hadamard_rotation
                    phase = getattr(module, phase_name)

                    def record_rotation(values, *args, **kwargs):
                        uploaded.append(weakref.ref(values))
                        return rotate(values, *args, **kwargs)

                    def observe_phase(*args, **kwargs):
                        if source_device == "cpu":
                            self.assertIsNone(uploaded[-1]())
                        else:
                            self.assertIs(uploaded[-1](), source)
                        return phase(*args, **kwargs)

                    with patch.object(module, "apply_hadamard_rotation", new=record_rotation), patch.object(module, phase_name, new=observe_phase):
                        result = quantize(source, device="cuda:0")
                    self.assertTrue(all(tensor.device.type == "cpu" for tensor in result))
                    self.assertTrue(torch.equal(source, saved))
