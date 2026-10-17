from dataclasses import fields
import unittest
from unittest.mock import patch

import torch

from potatoforge.audits import activation_measurements as activation
from potatoforge.calibration import LayerCalibration
from potatoforge.quantization import convrot_w4a4 as w4
from potatoforge.quantization import int8_tensorwise as int8


CASES = (
    ("int8_convrot", "quantize_int8_convrot", int8.quantize_int8_convrot, int8.dequantize_int8_convrot),
    ("int8_convrot_mse", "quantize_int8_convrot_mse", int8.quantize_int8_convrot_mse, int8.dequantize_int8_convrot),
    ("convrot_w4a4", "quantize_convrot_w4a4", w4.quantize_convrot_w4a4, w4.dequantize_convrot_w4a4),
    ("convrot_w4a4_mse", "quantize_convrot_w4a4_mse", w4.quantize_convrot_w4a4_mse, w4.dequantize_convrot_w4a4),
)


class TestCudaResidentActivation(unittest.TestCase):
    def assert_tensor_exact(self, expected, actual):
        self.assertEqual(expected.dtype, actual.dtype)
        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(expected.device, actual.device)
        self.assertTrue(torch.equal(
            expected.contiguous().view(torch.uint8),
            actual.contiguous().view(torch.uint8),
        ))

    def weights(self):
        values = torch.randn((4, 1024), generator=torch.Generator().manual_seed(1729))
        values[0].zero_()
        values[1].mul_(1e-32)
        values[2] = -values[2].abs()
        values[3, 0] = 100
        return values

    def test_default_cpu_payload_and_opt_in_device_results_match(self):
        devices = ("cpu", "cuda:0") if torch.cuda.is_available() else ("cpu",)
        with torch.inference_mode():
            for device in devices:
                for dtype in (torch.bfloat16, torch.float16, torch.float32):
                    source = self.weights().to(dtype)[:, ::2]
                    for values in (source, torch.zeros_like(source)):
                        saved = values.clone()
                        for method, _, quantize, decode in CASES:
                            with self.subTest(device=device, dtype=dtype, method=method):
                                expected = quantize(values, device=device)
                                actual = quantize(values, device=device, return_on_device=True)
                                self.assertTrue(all(value.device.type == "cpu" for value in expected))
                                self.assertTrue(all(value.device == torch.device(device) for value in actual))
                                for first, second in zip(expected, actual):
                                    self.assert_tensor_exact(first, second.cpu())
                                transferred = type(expected)(*(value.to(device) for value in expected))
                                self.assert_tensor_exact(decode(transferred), decode(actual))
                                self.assert_tensor_exact(saved, values)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_timing_stages_and_native_fallback_are_preserved(self):
        with torch.inference_mode():
            values = self.weights()[:, :512].contiguous()
            for _, _, quantize, _ in CASES:
                before, after = {}, {}
                expected = quantize(values, device="cuda:0", internal_timings=before)
                actual = quantize(values, device="cuda:0", internal_timings=after, return_on_device=True)
                self.assertEqual(before.keys(), after.keys())
                for first, second in zip(expected, actual):
                    self.assert_tensor_exact(first, second.cpu())
            with patch.object(w4, "load_w4a4_mse_candidate", return_value=None):
                expected = w4.quantize_convrot_w4a4_mse(values, device="cuda:0")
                actual = w4.quantize_convrot_w4a4_mse(values, device="cuda:0", return_on_device=True)
                for first, second in zip(expected, actual):
                    self.assert_tensor_exact(first, second.cpu())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_already_cuda_source_is_unchanged(self):
        with torch.inference_mode():
            source = self.weights()[:, :512].contiguous().to("cuda:0")
            saved = source.clone()
            for _, _, quantize, _ in CASES:
                actual = quantize(source, device="cuda:0", return_on_device=True)
                self.assertTrue(all(value.device == source.device for value in actual))
                self.assert_tensor_exact(saved, source)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_audits_keep_cuda_payloads_and_return_identical_cpu_metrics(self):
        name = "blocks.0.attn.wq.weight"
        values = self.weights()[:, :512].contiguous().bfloat16()
        descriptor = {"dtype": "BF16", "shape": list(values.shape), "data_offsets": [0, values.numel() * 2]}
        for mode in ("full", "cvar"):
            tensors = {"eval_sum_x2": torch.ones((2, 512))}
            if mode == "full":
                tensors.update(
                    sample_x=torch.randn((2, 2, 512), generator=torch.Generator().manual_seed(42)),
                    sample_x_valid=torch.full((2,), 2, dtype=torch.int64),
                )
            layer = LayerCalibration(
                tensor_name=name, input_features=512, output_features=4,
                sample_count=4, invocation_count=2, evaluation_count=2,
                collection_mode=mode, tensor_keys={key: key for key in tensors},
                tensor_store=tensors,
            )
            for method, symbol, quantize, _ in CASES:
                with self.subTest(mode=mode, method=method), torch.inference_mode():
                    def round_trip(*args, **kwargs):
                        kwargs.pop("return_on_device", None)
                        result = quantize(*args, **kwargs)
                        self.assertTrue(all(value.device.type == "cpu" for value in result))
                        return type(result)(*(value.to("cuda:0") for value in result))

                    def observe_device_result(*args, **kwargs):
                        self.assertTrue(kwargs["return_on_device"])
                        result = quantize(*args, **kwargs)
                        self.assertTrue(all(value.device == torch.device("cuda:0") for value in result))
                        return result

                    with patch.object(activation, symbol, new=round_trip):
                        expected = activation.measure_activation_candidate(name, descriptor, values, layer, method, device="cuda:0")
                    with patch.object(activation, symbol, new=observe_device_result):
                        actual = activation.measure_activation_candidate(name, descriptor, values, layer, method, device="cuda:0")
                    self.assertTrue(expected.available and actual.available)
                    for field in fields(expected):
                        first, second = getattr(expected, field.name), getattr(actual, field.name)
                        if isinstance(first, torch.Tensor):
                            self.assertEqual(second.device.type, "cpu")
                            self.assert_tensor_exact(first, second)
                        else:
                            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
