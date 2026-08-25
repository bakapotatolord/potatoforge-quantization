import unittest

import torch

from potatoforge.quantization.convrot_w4a4 import (
    dequantize_convrot_w4a4,
    pack_signed_int4_row_major,
    quantize_convrot_w4a4,
    unpack_signed_int4_row_major,
    quantize_w4_rowwise,
    dequantize_w4_rowwise,
)

class TestConvrotW4A4(unittest.TestCase):
    def test_packs_adjacent_signed_int4_codes(self) -> None:
        codes = torch.tensor(
            [[-8, -7, -1, 0, 1, 7]],
            dtype=torch.int8,
        )

        packed = pack_signed_int4_row_major(codes)

        self.assertEqual(packed.dtype, torch.int8)
        self.assertEqual(tuple(packed.shape), (1, 3))

        self.assertEqual(
            packed.view(torch.uint8).tolist(),
            [[0x98, 0x0F, 0x71]],
        )

    def test_unpacks_packed_int4_codes(self) -> None:
        original_codes = torch.tensor(
            [[-8, -7, -1, 0, 1, 7]],
            dtype=torch.int8,
        )

        packed = pack_signed_int4_row_major(original_codes)
        unpacked = unpack_signed_int4_row_major(packed)

        self.assertEqual(unpacked.dtype, torch.int8)
        self.assertEqual(tuple(unpacked.shape), (1, 6))
        self.assertTrue(torch.equal(unpacked, original_codes))

    def test_rejects_odd_code_width(self) -> None:
        codes = torch.tensor([[0, 1, 2]], dtype=torch.int8)

        with self.assertRaisesRegex(ValueError, "even width"):
            pack_signed_int4_row_major(codes)

    def test_rejects_code_outside_signed_int4_range(self) -> None:
        codes = torch.tensor([[8, 0]], dtype=torch.int8)

        with self.assertRaisesRegex(ValueError, r"\[-8, 7\]"):
            pack_signed_int4_row_major(codes)

    def test_quantizes_a_bfloat16_row_to_w4(self) -> None:
        weights = torch.tensor(
            [[70.0, 31.0, -70.0, 0.0]],
            dtype=torch.bfloat16,
        )

        result = quantize_w4_rowwise(weights)

        self.assertEqual(result.packed_codes.dtype, torch.int8)
        self.assertEqual(tuple(result.packed_codes.shape), (1, 2))

        self.assertEqual(result.scales.dtype, torch.float32)
        self.assertEqual(tuple(result.scales.shape), (1, 1))
        self.assertEqual(result.scales.tolist(), [[10.0]])

        unpacked_codes = unpack_signed_int4_row_major(
            result.packed_codes,
        )
        self.assertEqual(
            unpacked_codes.tolist(),
            [[7, 3, -7, 0]],
        )

        self.assertEqual(
            result.packed_codes.view(torch.uint8).tolist(),
            [[0x37, 0x09]],
        )

    def test_quantizes_a_float16_row_to_w4(self) -> None:
        weights = torch.tensor(
            [[70.0, 31.0, -70.0, 0.0]],
            dtype=torch.float16,
        )

        result = quantize_w4_rowwise(weights)

        self.assertEqual(result.packed_codes.dtype, torch.int8)
        self.assertEqual(result.scales.dtype, torch.float32)
        self.assertEqual(result.scales.tolist(), [[10.0]])
        self.assertEqual(
            unpack_signed_int4_row_major(result.packed_codes).tolist(),
            [[7, 3, -7, 0]],
        )

    def test_dequantizes_w4_codes_to_float32(self) -> None:
        weights = torch.tensor(
            [[70.0, 31.0, -70.0, 0.0]],
            dtype=torch.bfloat16,
        )

        result = quantize_w4_rowwise(weights)
        dequantized = dequantize_w4_rowwise(result)

        self.assertEqual(dequantized.dtype, torch.float32)
        self.assertEqual(tuple(dequantized.shape), (1, 4))
        self.assertEqual(
            dequantized.tolist(),
            [[70.0, 30.0, -70.0, 0.0]],
        )



    def test_quantizes_a_convrot_w4a4_weight(self) -> None:
        weights = torch.zeros(
            (1, 256),
            dtype=torch.bfloat16,
        )
        weights[0, 0] = 70.0

        result = quantize_convrot_w4a4(weights)

        self.assertEqual(result.packed_codes.dtype, torch.int8)
        self.assertEqual(tuple(result.packed_codes.shape), (1, 128))

        self.assertEqual(result.scales.dtype, torch.float32)
        self.assertEqual(tuple(result.scales.shape), (1,))
        self.assertEqual(result.scales.tolist(), [0.625])

        packed_bytes = result.packed_codes.view(torch.uint8)

        self.assertEqual(
            packed_bytes[0, :2].tolist(),
            [0x77, 0x97],
        )

    def test_quantizes_a_float16_convrot_w4a4_weight(self) -> None:
        weights = torch.zeros(
            (1, 256),
            dtype=torch.float16,
        )
        weights[0, 0] = 70.0

        result = quantize_convrot_w4a4(weights)

        self.assertEqual(result.packed_codes.dtype, torch.int8)
        self.assertEqual(tuple(result.packed_codes.shape), (1, 128))
        self.assertEqual(result.scales.dtype, torch.float32)
        self.assertEqual(result.scales.tolist(), [0.625])

    def test_dequantizes_a_convrot_w4a4_weight(self) -> None:
        weights = torch.zeros(
            (1, 256),
            dtype=torch.bfloat16,
        )
        weights[0, 0] = 70.0

        result = quantize_convrot_w4a4(weights)
        dequantized = dequantize_convrot_w4a4(result)

        self.assertEqual(dequantized.dtype, torch.float32)
        self.assertEqual(tuple(dequantized.shape), (1, 256))
        self.assertTrue(
            torch.allclose(
                dequantized,
                weights.float(),
                rtol=0.0,
                atol=1e-5,
            )
        )
