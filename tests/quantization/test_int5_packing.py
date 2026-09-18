import unittest

import torch

from potatoforge.quantization.int5_packing import (
    Int5PackedResult,
    pack_int5_row_major,
    unpack_int5_row_major,
)


class TestInt5Packing(unittest.TestCase):
    def test_golden_byte_vector(self) -> None:
        values = torch.tensor(
            [[-16, 0, 15, 0, 0, 0, 0, 0]],
            dtype=torch.int8,
        )

        result = pack_int5_row_major(values)

        self.assertEqual(result.packed_codes.tolist(), [[0x10, 0x3C, 0, 0, 0]])
        self.assertTrue(torch.equal(unpack_int5_row_major(result), values))

    def test_round_trips_multiple_rows_and_complete_storage_range(self) -> None:
        values = torch.tensor(
            [
                [-16, -15, -1, 0, 1, 7, 14, 15],
                [15, 14, 7, 1, 0, -1, -15, -16],
            ],
            dtype=torch.int8,
        )

        result = pack_int5_row_major(values)

        self.assertEqual(tuple(result.packed_codes.shape), (2, 5))
        self.assertTrue(torch.equal(unpack_int5_row_major(result), values))

    def test_rejects_invalid_codes_and_shapes(self) -> None:
        with self.assertRaises(ValueError):
            pack_int5_row_major(torch.ones((1, 8), dtype=torch.int8) * 16)
        with self.assertRaises(ValueError):
            pack_int5_row_major(torch.ones((1, 8), dtype=torch.int8) * -17)
        with self.assertRaises(ValueError):
            pack_int5_row_major(torch.ones((1, 8), dtype=torch.float32))
        with self.assertRaises(ValueError):
            pack_int5_row_major(torch.ones((1, 7), dtype=torch.int8))
        with self.assertRaises(ValueError):
            pack_int5_row_major(torch.ones(8, dtype=torch.int8))
        with self.assertRaises(ValueError):
            unpack_int5_row_major(
                Int5PackedResult(
                    packed_codes=torch.zeros((1, 5), dtype=torch.int8),
                    original_shape=(1, 8),
                )
            )


if __name__ == "__main__":
    unittest.main()
