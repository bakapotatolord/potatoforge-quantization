import unittest

import torch

from potatoforge.quantization.int6_packing import (
    Int6PackedResult,
    pack_int6_row_major,
    unpack_int6_row_major,
)


class TestInt6Packing(unittest.TestCase):
    def test_golden_byte_vectors(self) -> None:
        cases = (
            ([0, 0, 0, 0], [0x00, 0x00, 0x00]),
            ([31, 31, 31, 31], [0xDF, 0xF7, 0x7D]),
            ([-31, -31, -31, -31], [0x61, 0x18, 0x86]),
            ([-31, 0, 31, 1], [0x21, 0xF0, 0x05]),
            ([1, 2, 3, 4], [0x81, 0x30, 0x10]),
            ([-1, -2, -3, -4], [0xBF, 0xDF, 0xF3]),
        )

        for values, expected_bytes in cases:
            with self.subTest(values=values):
                result = pack_int6_row_major(
                    torch.tensor([values], dtype=torch.int8)
                )

                self.assertEqual(
                    result.packed_codes.tolist(),
                    [expected_bytes],
                )
                self.assertTrue(
                    torch.equal(
                        unpack_int6_row_major(result),
                        torch.tensor([values], dtype=torch.int8),
                    )
                )

    def test_round_trips_every_legal_single_value(self) -> None:
        for value in (-31, -30, -1, 0, 1, 30, 31):
            with self.subTest(value=value):
                values = torch.full((1, 4), value, dtype=torch.int8)
                result = pack_int6_row_major(values)

                self.assertTrue(
                    torch.equal(unpack_int6_row_major(result), values)
                )

    def test_decoder_defines_the_six_bit_minus_32_pattern(self) -> None:
        result = Int6PackedResult(
            packed_codes=torch.tensor([[0x20, 0x00, 0x00]], dtype=torch.uint8),
            original_shape=(1, 4),
        )

        self.assertTrue(
            torch.equal(
                unpack_int6_row_major(result),
                torch.tensor([[-32, 0, 0, 0]], dtype=torch.int8),
            )
        )

    def test_round_trips_random_groups_and_matrices(self) -> None:
        generator = torch.Generator().manual_seed(1234)
        groups = torch.randint(
            -31,
            32,
            (4096, 4),
            generator=generator,
            dtype=torch.int8,
        )
        matrix = torch.randint(
            -31,
            32,
            (7, 12),
            generator=generator,
            dtype=torch.int8,
        )

        group_result = pack_int6_row_major(groups)
        matrix_result = pack_int6_row_major(matrix)

        self.assertEqual(group_result.packed_codes.dtype, torch.uint8)
        self.assertTrue(
            torch.equal(unpack_int6_row_major(group_result), groups)
        )
        self.assertEqual(tuple(matrix_result.packed_codes.shape), (7, 9))
        self.assertTrue(
            torch.equal(unpack_int6_row_major(matrix_result), matrix)
        )

    def test_handles_zero_width_matrix(self) -> None:
        values = torch.empty((2, 0), dtype=torch.int8)
        result = pack_int6_row_major(values)

        self.assertEqual(tuple(result.packed_codes.shape), (2, 0))
        self.assertTrue(torch.equal(unpack_int6_row_major(result), values))

    def test_rejects_invalid_codes_and_shapes(self) -> None:
        with self.assertRaises(ValueError):
            pack_int6_row_major(torch.ones((1, 4), dtype=torch.int8) * 32)
        with self.assertRaises(ValueError):
            pack_int6_row_major(torch.ones((1, 4), dtype=torch.float32))
        with self.assertRaises(ValueError):
            pack_int6_row_major(torch.ones((1, 3), dtype=torch.int8))
        with self.assertRaises(ValueError):
            pack_int6_row_major(torch.ones(4, dtype=torch.int8))
        with self.assertRaises(ValueError):
            unpack_int6_row_major(
                Int6PackedResult(
                    packed_codes=torch.zeros((1, 3), dtype=torch.int8),
                    original_shape=(1, 4),
                )
            )


if __name__ == "__main__":
    unittest.main()
