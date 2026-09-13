from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from potatoforge.converter import convert_model
from potatoforge.profiles import QuantizationProfile
from potatoforge.quantization.convrot_w4a4 import (
    dequantize_convrot_w4a4,
    quantize_convrot_w4a4,
    quantize_convrot_w4a4_mse,
    unpack_signed_int4_row_major,
)
from potatoforge.quantization.int6_packing import (
    Int6PackedResult,
    unpack_int6_row_major,
)
from potatoforge.quantization.int6_rowwise import (
    quantize_int6_convrot,
    quantize_int6_convrot_packed,
)
from potatoforge.timing import TimingCollector


CUDA_STAGES = (
    "prepare",
    "rotation",
    "scale",
    "quantize_values",
    "pack",
    "finalize",
)
INT6_VALIDATION_STAGES = (
    "resolve_device",
    "validate_metadata",
    "validate_finite",
)
W4A4_MSE_CUDA_STAGES = (
    "prepare",
    "rotation",
    "scale_init",
    "zero_row_check",
    "coarse_search",
    "fine_search",
    "final_quantize",
    "pack",
    "finalize",
)


class TestConvRotDevice(unittest.TestCase):
    def test_explicit_cpu_paths_match_existing_defaults(self) -> None:
        weights = torch.arange(512, dtype=torch.float32).reshape(2, 256)

        w4_default = quantize_convrot_w4a4(weights)
        w4_cpu = quantize_convrot_w4a4(weights, device="cpu")
        self.assertTrue(torch.equal(w4_default.packed_codes, w4_cpu.packed_codes))
        self.assertTrue(torch.equal(w4_default.scales, w4_cpu.scales))

        int6_default = quantize_int6_convrot(weights)
        int6_cpu = quantize_int6_convrot_packed(weights, device="cpu")
        int6_cpu_codes = unpack_int6_row_major(
            Int6PackedResult(int6_cpu.packed_codes, int6_cpu.original_shape),
        )
        self.assertTrue(torch.equal(int6_default.codes, int6_cpu_codes))
        self.assertTrue(torch.equal(int6_default.scales, int6_cpu.scales))

        mse_default = quantize_convrot_w4a4_mse(weights)
        mse_cpu = quantize_convrot_w4a4_mse(weights, device="cpu")
        self.assertTrue(torch.equal(mse_default.packed_codes, mse_cpu.packed_codes))
        self.assertTrue(torch.equal(mse_default.scales, mse_cpu.scales))

    def test_w4a4_mse_cuda_unavailable_fails_clearly(self) -> None:
        weights = torch.zeros((1, 256), dtype=torch.bfloat16)

        with patch(
            "potatoforge.quantization.convrot_w4a4.torch.cuda.is_available",
            return_value=False,
        ):
            with self.assertRaisesRegex(ValueError, "CUDA.*unavailable"):
                quantize_convrot_w4a4_mse(weights, device="cuda")

    @patch("potatoforge.timing.perf_counter")
    def test_untimed_int6_path_does_not_read_clock(self, perf_counter) -> None:
        weights = torch.arange(512, dtype=torch.float32).reshape(2, 256)

        quantize_int6_convrot_packed(weights, device="cpu")

        perf_counter.assert_not_called()

    def test_cpu_int6_convrot_rejects_nonfinite_inputs(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                weights = torch.zeros((1, 256), dtype=torch.float32)
                weights[0, 0] = value
                with self.assertRaisesRegex(
                    ValueError,
                    "INT6 weights must contain only finite values",
                ):
                    quantize_int6_convrot_packed(weights, device="cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_w4a4_cuda_matches_cpu_and_returns_compact_cpu_result(self) -> None:
        weights = torch.arange(512, dtype=torch.float32).reshape(2, 256)

        cpu = quantize_convrot_w4a4(weights)
        cuda = quantize_convrot_w4a4(weights, device="cuda")

        self.assertEqual(cuda.packed_codes.device.type, "cpu")
        self.assertEqual(cuda.scales.device.type, "cpu")
        self.assertEqual(cuda.packed_codes.shape, cpu.packed_codes.shape)
        self.assertEqual(cuda.scales.shape, cpu.scales.shape)
        cpu_codes = unpack_signed_int4_row_major(cpu.packed_codes)
        cuda_codes = unpack_signed_int4_row_major(cuda.packed_codes)
        self.assertLessEqual(
            int((cuda_codes.to(torch.int16) - cpu_codes.to(torch.int16)).abs().max()),
            1,
        )
        self.assertTrue(torch.allclose(cuda.scales, cpu.scales, atol=1e-6))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_w4a4_mse_cuda_matches_cpu_and_records_stages(self) -> None:
        torch.manual_seed(71)
        weights = torch.randn((2, 256), dtype=torch.bfloat16)

        cpu = quantize_convrot_w4a4_mse(weights)
        timings: dict[str, float] = {}
        cuda = quantize_convrot_w4a4_mse(
            weights,
            device="cuda",
            internal_timings=timings,
        )
        untimed_cuda = quantize_convrot_w4a4_mse(
            weights,
            device="cuda",
        )

        self.assertEqual(cuda.packed_codes.device.type, "cpu")
        self.assertEqual(cuda.scales.device.type, "cpu")
        self.assertTrue(
            torch.equal(cuda.packed_codes, untimed_cuda.packed_codes)
        )
        self.assertTrue(torch.equal(cuda.scales, untimed_cuda.scales))
        cpu_codes = unpack_signed_int4_row_major(cpu.packed_codes)
        cuda_codes = unpack_signed_int4_row_major(cuda.packed_codes)
        self.assertLessEqual(
            int((cuda_codes.to(torch.int16) - cpu_codes.to(torch.int16)).abs().max()),
            1,
        )
        cpu_error = (
            weights.float() - dequantize_convrot_w4a4(cpu)
        ).square().mean()
        cuda_error = (
            weights.float() - dequantize_convrot_w4a4(cuda)
        ).square().mean()
        self.assertLessEqual(float(cuda_error), float(cpu_error) + 1e-3)
        for stage in W4A4_MSE_CUDA_STAGES:
            self.assertIn(stage, timings)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_int6_convrot_cuda_matches_cpu_and_returns_packed_cpu_result(
        self,
    ) -> None:
        weights = torch.arange(512, dtype=torch.float32).reshape(2, 256)

        cpu = quantize_int6_convrot(weights)
        cuda = quantize_int6_convrot_packed(weights, device="cuda")
        cuda_codes = unpack_int6_row_major(
            Int6PackedResult(cuda.packed_codes, cuda.original_shape),
        )

        self.assertEqual(cuda.packed_codes.device.type, "cpu")
        self.assertEqual(cuda.scales.device.type, "cpu")
        self.assertEqual(cuda.packed_codes.shape[1], 256 // 4 * 3)
        self.assertEqual(cuda.scales.shape, cpu.scales.shape)
        self.assertLessEqual(
            int((cuda_codes.to(torch.int16) - cpu.codes.to(torch.int16)).abs().max()),
            1,
        )
        self.assertTrue(torch.allclose(cuda.scales, cpu.scales, atol=1e-6))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_new_cuda_paths_record_their_internal_stages(self) -> None:
        weights = torch.arange(512, dtype=torch.float32).reshape(2, 256)

        for action in ("convrot_w4a4", "int6_convrot"):
            with self.subTest(action=action):
                timings: dict[str, float] = {}
                if action == "convrot_w4a4":
                    quantize_convrot_w4a4(
                        weights,
                        device="cuda",
                        internal_timings=timings,
                    )
                else:
                    quantize_int6_convrot_packed(
                        weights,
                        device="cuda",
                        internal_timings=timings,
                    )
                for stage in CUDA_STAGES:
                    self.assertIn(stage, timings)
                if action == "int6_convrot":
                    for stage in INT6_VALIDATION_STAGES:
                        self.assertIn(stage, timings)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_int6_cuda_validates_the_transferred_tensor(self) -> None:
        weights = torch.zeros((1, 256), dtype=torch.float32)
        observed_devices: list[str] = []
        original_isfinite = torch.isfinite

        def observe_isfinite(values: torch.Tensor) -> torch.Tensor:
            observed_devices.append(values.device.type)
            return original_isfinite(values)

        with patch(
            "potatoforge.quantization.int6_rowwise.torch.isfinite",
            side_effect=observe_isfinite,
        ):
            quantize_int6_convrot_packed(weights, device="cuda")

        self.assertEqual(observed_devices, ["cuda"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_int6_cuda_rejects_nonfinite_inputs(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                weights = torch.zeros((1, 256), dtype=torch.float32)
                weights[0, 0] = value
                with self.assertRaisesRegex(
                    ValueError,
                    "INT6 weights must contain only finite values",
                ):
                    quantize_int6_convrot_packed(weights, device="cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_int6_cuda_timing_preserves_valid_output(self) -> None:
        weights = torch.arange(512, dtype=torch.float32).reshape(2, 256)

        untimed = quantize_int6_convrot_packed(weights, device="cuda")
        timed = quantize_int6_convrot_packed(
            weights,
            device="cuda",
            internal_timings={},
        )

        self.assertTrue(torch.equal(untimed.packed_codes, timed.packed_codes))
        self.assertTrue(torch.equal(untimed.scales, timed.scales))
        self.assertEqual(untimed.original_shape, timed.original_shape)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_new_cuda_paths_do_not_add_profiling_synchronization(self) -> None:
        weights = torch.zeros((1, 256), dtype=torch.bfloat16)

        with patch(
            "potatoforge.quantization.int8_tensorwise.torch.cuda.synchronize"
        ) as synchronize:
            quantize_convrot_w4a4(weights, device="cuda")
            quantize_convrot_w4a4_mse(weights, device="cuda")
            quantize_int6_convrot_packed(weights, device="cuda")

        synchronize.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_conversion_preserves_serialized_contract(self) -> None:
        profile: QuantizationProfile = {
            "default": "keep",
            "rules": (
                {
                    "action": "convrot_w4a4",
                    "prefix": "w4.",
                    "suffixes": (".weight",),
                },
                {
                    "action": "int6_convrot",
                    "prefix": "int6.",
                    "suffixes": (".weight",),
                },
            ),
        }
        weights = torch.arange(512, dtype=torch.float32).reshape(2, 256)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.safetensors"
            cpu_path = root / "cpu.safetensors"
            cuda_path = root / "cuda.safetensors"
            save_file(
                {
                    "w4.weight": weights,
                    "int6.weight": weights.clone(),
                    "kept": torch.tensor([1.0], dtype=torch.float32),
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
        self.assertIn(
            "pack",
            next(
                record.internal_stages
                for record in timing.records
                if record.action == "convrot_w4a4"
            ),
        )
        self.assertIn(
            "pack",
            next(
                record.internal_stages
                for record in timing.records
                if record.action == "int6_convrot"
            ),
        )


if __name__ == "__main__":
    unittest.main()
