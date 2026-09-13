import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from potatoforge.converter import convert_model
from potatoforge.planning import build_plan
from potatoforge.profiles import QuantizationProfile
from potatoforge.quantization.int8_tensorwise import (
    quantize_int8_convrot,
)
from potatoforge.source_payloads import stream_output_payloads
from potatoforge.headers.source_header import read_source_model_header
from potatoforge.timing import TimingCollector


class TestInt8ConvRotDevice(unittest.TestCase):
    def test_default_and_explicit_cpu_match(self) -> None:
        weights = torch.arange(256, dtype=torch.float32).reshape(1, 256)

        default = quantize_int8_convrot(weights)
        explicit = quantize_int8_convrot(weights, device="cpu")

        self.assertTrue(torch.equal(default.codes, explicit.codes))
        self.assertTrue(torch.equal(default.scales, explicit.scales))

    def test_cuda_unavailable_fails_clearly(self) -> None:
        weights = torch.zeros((1, 256), dtype=torch.bfloat16)

        with patch(
            "potatoforge.quantization.int8_tensorwise.torch.cuda.is_available",
            return_value=False,
        ):
            with self.assertRaisesRegex(ValueError, "CUDA.*unavailable"):
                quantize_int8_convrot(weights, device="cuda")

    def test_converter_rejects_unavailable_cuda_before_reading_source(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "potatoforge.converter.torch.cuda.is_available",
                return_value=False,
            ):
                with self.assertRaisesRegex(ValueError, "CUDA.*unavailable"):
                    convert_model(
                        root / "missing.safetensors",
                        root / "output.safetensors",
                        {"default": "keep", "rules": ()},
                        on_entry_started=None,
                        device="cuda",
                    )

    def test_unsupported_action_stays_on_cpu_with_cuda_selection(self) -> None:
        tensor = torch.arange(256, dtype=torch.float32).reshape(1, 256)
        profile: QuantizationProfile = {
            "default": "keep",
            "rules": (
                {
                    "action": "int8",
                    "prefix": "",
                    "suffixes": (".weight",),
                },
            ),
        }

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.safetensors"
            save_file({"blocks.0.weight": tensor}, str(source_path))
            header = read_source_model_header(source_path)
            entries = build_plan(header.tensors, profile)

            payloads = list(
                stream_output_payloads(
                    source_path,
                    entries,
                    device="cuda",
                )
            )

        self.assertEqual(len(payloads), 3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_returns_cpu_compact_result_and_records_stages(self) -> None:
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                weights = torch.arange(
                    512,
                    dtype=torch.float32,
                ).reshape(2, 256).to(dtype)
                timings: dict[str, float] = {}

                cpu = quantize_int8_convrot(weights)
                cuda = quantize_int8_convrot(
                    weights,
                    device="cuda",
                    internal_timings=timings,
                )

                self.assertEqual(cuda.codes.device.type, "cpu")
                self.assertEqual(cuda.scales.device.type, "cpu")
                self.assertEqual(cuda.codes.shape, cpu.codes.shape)
                self.assertEqual(cuda.scales.shape, cpu.scales.shape)
                self.assertLessEqual(
                    int(
                        (cuda.codes.to(torch.int16)
                        - cpu.codes.to(torch.int16)).abs().max()
                    ),
                    1,
                )
                self.assertTrue(torch.allclose(cuda.scales, cpu.scales, atol=1e-6))
                for stage in (
                    "prepare",
                    "rotation",
                    "scale",
                    "quantize_values",
                    "finalize",
                ):
                    self.assertIn(stage, timings)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_without_timing_does_not_synchronize(self) -> None:
        weights = torch.zeros((1, 256), dtype=torch.bfloat16)

        with patch(
            "potatoforge.quantization.int8_tensorwise.torch.cuda.synchronize"
        ) as synchronize:
            quantize_int8_convrot(weights, device="cuda")

        synchronize.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_conversion_preserves_serialized_contract(self) -> None:
        weights = torch.arange(512, dtype=torch.float32).reshape(2, 256)
        bias = torch.tensor([1.0, -1.0], dtype=torch.bfloat16)
        profile: QuantizationProfile = {
            "default": "keep",
            "rules": (
                {
                    "action": "int8_convrot",
                    "prefix": "",
                    "suffixes": (".weight",),
                },
            ),
        }

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.safetensors"
            cpu_path = root / "cpu.safetensors"
            cuda_path = root / "cuda.safetensors"
            save_file(
                {
                    "blocks.0.weight": weights,
                    "blocks.0.bias": bias,
                },
                str(source_path),
            )

            convert_model(
                source_path,
                cpu_path,
                profile,
                on_entry_started=None,
            )
            timing = TimingCollector()
            convert_model(
                source_path,
                cuda_path,
                profile,
                on_entry_started=None,
                device="cuda",
                timing=timing,
            )

            cpu_payload = load_file(str(cpu_path))
            cuda_payload = load_file(str(cuda_path))

        self.assertEqual(set(cpu_payload), set(cuda_payload))
        for name in cpu_payload:
            self.assertEqual(cpu_payload[name].shape, cuda_payload[name].shape)
            self.assertEqual(cpu_payload[name].dtype, cuda_payload[name].dtype)

        code_delta = (
            cpu_payload["blocks.0.weight"].to(torch.int16)
            - cuda_payload["blocks.0.weight"].to(torch.int16)
        ).abs()
        self.assertLessEqual(int(code_delta.max()), 1)
        self.assertTrue(
            torch.allclose(
                cpu_payload["blocks.0.weight_scale"],
                cuda_payload["blocks.0.weight_scale"],
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.equal(cpu_payload["blocks.0.bias"], cuda_payload["blocks.0.bias"])
        )
        self.assertIn("rotation", next(
            record.internal_stages
            for record in timing.records
            if record.action == "int8_convrot"
        ))


if __name__ == "__main__":
    unittest.main()
