import unittest
from unittest.mock import patch

import torch

from potatoforge.quantization.int7_packing import (
    Int7PackedResult,
    pack_int7_row_major,
    unpack_int7_row_major,
)
from potatoforge.quantization.int7_rowwise import (
    INT7_QMAX,
    INT7_QMIN,
    Int7RowwiseResult,
    dequantize_int7_convrot,
    dequantize_int7_convrot_packed,
    dequantize_int7_rowwise,
    quantize_int7_convrot,
    quantize_int7_convrot_mse,
    quantize_int7_convrot_packed,
    quantize_int7_convrot_mse_packed,
    quantize_int7_rowwise,
    select_int7_mse_scale,
)
from potatoforge.quantization.rowwise import (
    calculate_rowwise_candidate_mse,
    quantize_rows_with_scale,
)


class TestInt7Packing(unittest.TestCase):
    def test_golden_byte_vectors(self) -> None:
        cases = (
            ([0] * 8, [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
            ([63] * 8, [0xBF, 0xDF, 0xEF, 0xF7, 0xFB, 0xFD, 0x7E]),
            ([-63] * 8, [0xC1, 0x60, 0x30, 0x18, 0x0C, 0x06, 0x83]),
            (
                [-63, 0, 63, 1, -1, 2, -2, 7],
                [0x41, 0xC0, 0x2F, 0xF0, 0x17, 0xF8, 0x0F],
            ),
        )

        for values, expected_bytes in cases:
            with self.subTest(values=values):
                result = pack_int7_row_major(
                    torch.tensor([values], dtype=torch.int8)
                )
                self.assertEqual(result.packed_codes.tolist(), [expected_bytes])
                self.assertTrue(
                    torch.equal(
                        unpack_int7_row_major(result),
                        torch.tensor([values], dtype=torch.int8),
                    )
                )

    def test_round_trips_random_groups_and_matrices(self) -> None:
        generator = torch.Generator().manual_seed(1234)
        groups = torch.randint(
            -63,
            64,
            (4096, 8),
            generator=generator,
            dtype=torch.int8,
        )
        matrix = torch.randint(
            -63,
            64,
            (7, 24),
            generator=generator,
            dtype=torch.int8,
        )

        group_result = pack_int7_row_major(groups)
        matrix_result = pack_int7_row_major(matrix)

        self.assertEqual(group_result.packed_codes.dtype, torch.uint8)
        self.assertTrue(torch.equal(unpack_int7_row_major(group_result), groups))
        self.assertEqual(tuple(matrix_result.packed_codes.shape), (7, 21))
        self.assertTrue(torch.equal(unpack_int7_row_major(matrix_result), matrix))

    def test_handles_zero_width_matrix(self) -> None:
        values = torch.empty((2, 0), dtype=torch.int8)
        result = pack_int7_row_major(values)
        self.assertEqual(tuple(result.packed_codes.shape), (2, 0))
        self.assertTrue(torch.equal(unpack_int7_row_major(result), values))

    def test_rejects_invalid_codes_and_packed_shapes(self) -> None:
        with self.assertRaises(ValueError):
            pack_int7_row_major(torch.full((1, 8), 64, dtype=torch.int8))
        with self.assertRaises(ValueError):
            pack_int7_row_major(torch.full((1, 8), -64, dtype=torch.int8))
        with self.assertRaises(ValueError):
            pack_int7_row_major(torch.ones((1, 8), dtype=torch.float32))
        with self.assertRaises(ValueError):
            pack_int7_row_major(torch.ones((1, 7), dtype=torch.int8))
        with self.assertRaises(ValueError):
            pack_int7_row_major(torch.ones(8, dtype=torch.int8))
        with self.assertRaises(ValueError):
            unpack_int7_row_major(
                Int7PackedResult(
                    packed_codes=torch.zeros((1, 7), dtype=torch.int8),
                    original_shape=(1, 8),
                )
            )
        with self.assertRaises(ValueError):
            unpack_int7_row_major(
                Int7PackedResult(
                    packed_codes=torch.zeros((1, 7), dtype=torch.uint8),
                    original_shape=(1, 9),
                )
            )


class TestInt7Rowwise(unittest.TestCase):
    def test_accepts_supported_float_dtypes(self) -> None:
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                result = quantize_int7_rowwise(
                    torch.tensor([[1.0, -0.5, 0.25]], dtype=dtype)
                )
                self.assertEqual(result.codes.dtype, torch.int8)
                self.assertEqual(result.scales.dtype, torch.float32)

    def test_rowwise_scales_limits_and_reconstruction(self) -> None:
        weights = torch.tensor([[1.0, -1.0, 0.0], [0.25, -0.125, 0.125]])
        result = quantize_int7_rowwise(weights)
        reconstructed = dequantize_int7_rowwise(result)

        self.assertAlmostEqual(float(result.scales[0, 0]), 1.0 / 63.0, places=7)
        self.assertAlmostEqual(float(result.scales[1, 0]), 0.25 / 63.0, places=7)
        self.assertTrue(bool(((result.codes >= -63) & (result.codes <= 63)).all()))
        self.assertEqual(reconstructed.dtype, torch.float32)

    def test_zero_rows_use_a_safe_unit_scale(self) -> None:
        result = quantize_int7_rowwise(torch.zeros((1, 3)))
        self.assertEqual(float(result.scales[0, 0]), 1.0)
        self.assertTrue(bool((result.codes == 0).all()))

    def test_convrot_round_trip_and_packed_parity(self) -> None:
        weights = torch.zeros((1, 256), dtype=torch.bfloat16)
        weights[0, 0] = 70.0

        logical = dequantize_int7_convrot(quantize_int7_convrot(weights))
        packed = dequantize_int7_convrot_packed(
            quantize_int7_convrot_packed(weights)
        )

        self.assertEqual(logical.shape, weights.shape)
        self.assertTrue(torch.equal(packed, logical))
        self.assertTrue(torch.allclose(logical, weights.float(), atol=0.01))

    def test_mse_convrot_uses_the_same_packed_contract(self) -> None:
        generator = torch.Generator().manual_seed(4321)
        weights = torch.randn((2, 256), generator=generator)

        logical = quantize_int7_convrot_mse(weights)
        packed = quantize_int7_convrot_mse_packed(weights)

        self.assertEqual(logical.codes.dtype, torch.int8)
        self.assertEqual(logical.scales.dtype, torch.float32)
        self.assertEqual(logical.scales.shape, (2, 1))
        self.assertTrue(bool(((logical.codes >= -63) & (logical.codes <= 63)).all()))
        self.assertEqual(packed.packed_codes.dtype, torch.uint8)
        self.assertEqual(tuple(packed.packed_codes.shape), (2, 224))
        self.assertEqual(packed.original_shape, (2, 256))
        self.assertTrue(torch.equal(
            dequantize_int7_convrot(logical),
            dequantize_int7_convrot_packed(packed),
        ))

    def test_mse_candidate_matches_scale_aware_quantization(self) -> None:
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                generator = torch.Generator().manual_seed(7654)
                weights = torch.randn((2, 256), generator=generator, dtype=dtype)
                weights_float = weights.float()
                scales = quantize_int7_rowwise(weights).scales * 0.73

                observed = calculate_rowwise_candidate_mse(
                    weights,
                    weights_float,
                    scales,
                    qmin=INT7_QMIN,
                    qmax=INT7_QMAX,
                )
                result = quantize_rows_with_scale(
                    weights,
                    scales,
                    qmin=INT7_QMIN,
                    qmax=INT7_QMAX,
                )
                expected = (
                    weights_float - result.codes.float() * result.scales
                ).square().mean(dim=1, keepdim=True)

                self.assertTrue(torch.equal(observed, expected))

    def test_mse_search_uses_the_34_candidate_schedule(self) -> None:
        generator = torch.Generator().manual_seed(9876)
        weights = torch.randn((1, 256), generator=generator)
        base_scale = quantize_int7_rowwise(weights).scales
        evaluated_multipliers: list[float] = []
        evaluated_mse: list[float] = []

        def record_candidate(
            candidate_weights: torch.Tensor,
            candidate_weights_float: torch.Tensor,
            scales: torch.Tensor,
            *,
            qmin: int,
            qmax: int,
        ) -> torch.Tensor:
            evaluated_multipliers.append((scales / base_scale).item())
            mse = calculate_rowwise_candidate_mse(
                candidate_weights,
                candidate_weights_float,
                scales,
                qmin=qmin,
                qmax=qmax,
            )
            evaluated_mse.append(mse.item())
            return mse

        with patch(
            "potatoforge.quantization.rowwise."
            "calculate_rowwise_candidate_mse",
            side_effect=record_candidate,
        ):
            select_int7_mse_scale(weights)

        expected_coarse = [index / 20 for index in range(2, 21)]
        self.assertTrue(torch.allclose(
            torch.tensor(evaluated_multipliers[:19]),
            torch.tensor(expected_coarse),
            rtol=0,
            atol=1e-7,
        ))
        best_coarse = evaluated_multipliers[
            evaluated_mse[:19].index(min(evaluated_mse[:19]))
        ]
        lower = max(best_coarse - 0.05, 0.10)
        upper = min(best_coarse + 0.05, 1.0)
        expected_fine = [
            lower + (upper - lower) * step / 16
            for step in range(1, 16)
        ]
        self.assertEqual(len(evaluated_multipliers), 34)
        self.assertTrue(torch.allclose(
            torch.tensor(evaluated_multipliers[19:]),
            torch.tensor(expected_fine),
            rtol=0,
            atol=1e-7,
        ))

    def test_mse_reconstruction_is_no_worse_than_baseline(self) -> None:
        for seed, rows in ((41, 3), (43, 2), (47, 1)):
            with self.subTest(seed=seed):
                generator = torch.Generator().manual_seed(seed)
                weights = torch.randn((rows, 256), generator=generator)
                baseline = quantize_int7_convrot(weights)
                optimized = quantize_int7_convrot_mse(weights)

                baseline_error = (
                    weights - dequantize_int7_convrot(baseline)
                ).square().mean(dim=1)
                optimized_error = (
                    weights - dequantize_int7_convrot(optimized)
                ).square().mean(dim=1)

                self.assertTrue(torch.all(optimized_error <= baseline_error + 1e-7))

    def test_mse_zero_rows_keep_the_baseline_unit_scale(self) -> None:
        weights = torch.zeros((2, 256), dtype=torch.float32)
        weights[1, 0] = 1.0

        baseline = quantize_int7_convrot(weights)
        optimized = quantize_int7_convrot_mse(weights)

        self.assertEqual(float(optimized.scales[0, 0]), float(baseline.scales[0, 0]))
        self.assertTrue(bool((optimized.codes[0] == 0).all()))

    def test_finite_subnormal_rows_use_the_comfy_scale_floor(self) -> None:
        weights = torch.zeros((1, 256), dtype=torch.float32)
        weights[0, 0] = 1e-43

        results = (
            quantize_int7_convrot(weights),
            quantize_int7_convrot_mse(weights),
        )

        for result in results:
            self.assertTrue(torch.equal(result.scales, torch.full_like(result.scales, 1e-30)))
            self.assertEqual(dequantize_int7_convrot(result).shape, weights.shape)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_convrot_cuda_matches_cpu_and_returns_packed_cpu_result(self) -> None:
        torch.manual_seed(17)
        weights = torch.randn((2, 256), dtype=torch.bfloat16)

        cpu = quantize_int7_convrot(weights)
        timings: dict[str, float] = {}
        cuda = quantize_int7_convrot_packed(
            weights,
            device="cuda",
            internal_timings=timings,
        )
        cuda_codes = unpack_int7_row_major(
            Int7PackedResult(cuda.packed_codes, cuda.original_shape)
        )

        self.assertEqual(cuda.packed_codes.device.type, "cpu")
        self.assertEqual(cuda.scales.device.type, "cpu")
        self.assertEqual(cuda.packed_codes.shape, (2, 224))
        self.assertEqual(cuda.scales.shape, cpu.scales.shape)
        self.assertLessEqual(
            int((cuda_codes.to(torch.int16) - cpu.codes.to(torch.int16)).abs().max()),
            1,
        )
        self.assertTrue(torch.allclose(cuda.scales, cpu.scales, atol=1e-5))
        for stage in (
            "resolve_device",
            "validate_metadata",
            "prepare",
            "validate_finite",
            "rotation",
            "scale",
            "quantize_values",
            "pack",
            "finalize",
        ):
            self.assertIn(stage, timings)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            quantize_int7_rowwise(torch.ones(3))
        with self.assertRaises(ValueError):
            quantize_int7_rowwise(torch.ones((2, 2), dtype=torch.int8))
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "INT7 weights must contain only finite values",
                ):
                    quantize_int7_rowwise(torch.tensor([[value]]))
        with self.assertRaisesRegex(ValueError, "divisible by 256"):
            quantize_int7_convrot(torch.ones((1, 128)))
        with self.assertRaisesRegex(ValueError, "divisible by 256"):
            quantize_int7_convrot_mse(torch.ones((1, 128)))
        with self.assertRaisesRegex(ValueError, "finite values"):
            quantize_int7_convrot_mse(
                torch.cat((torch.ones((1, 255)), torch.tensor([[float("nan")]])), dim=1)
            )
        with self.assertRaises(ValueError):
            dequantize_int7_rowwise(
                Int7RowwiseResult(
                    torch.tensor([[64]], dtype=torch.int8),
                    torch.tensor([[1.0]], dtype=torch.float32),
                )
            )


if __name__ == "__main__":
    unittest.main()
