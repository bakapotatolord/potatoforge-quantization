import unittest

import torch

from potatoforge.quantization import convrot_w4a4 as w4
from potatoforge.quantization import int6_rowwise as i6
from potatoforge.quantization import int7_rowwise as i7
from potatoforge.quantization import int8_tensorwise as i8
from potatoforge.quantization.grouped_int8 import stored_scales
from potatoforge.quantization.rowwise import (
    divide_exactly,
    quantize_symmetric_rows,
    select_rowwise_absmax_scale,
)

PLAIN_FORMATS = (
    ("int8", i8.quantize_int8_tensorwise),
    ("int6", i6.quantize_int6_rowwise),
    ("int7", i7.quantize_int7_rowwise),
)


def _heavy_tailed(rows: int, columns: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    weights = torch.randn((rows, columns), generator=generator) * 0.02
    weights *= torch.rand((rows, columns), generator=generator).pow(-0.5)
    weights[1].zero_()
    weights[2, 3] = 0.75
    return weights


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class TestCudaCpuParity(unittest.TestCase):
    """CUDA scale division must round exactly like the CPU.

    ConvRot results can still differ in rare rows: the Hadamard rotation is
    a matrix multiply whose sums round in a device-specific order. These
    tests therefore compare scale arithmetic on identical inputs.
    """

    def test_divide_exactly_matches_cpu_where_scalar_division_does_not(self) -> None:
        values = torch.rand(200_000, generator=torch.Generator().manual_seed(3)) * 3
        for divisor in (7, 15, 31, 63, 127, 127.0):
            with self.subTest(divisor=divisor):
                cpu = values / divisor
                self.assertTrue(torch.equal(divide_exactly(values.cuda(), divisor).cpu(), cpu))
                self.assertTrue(torch.equal(divide_exactly(values, divisor), cpu))

    def test_plain_formats_match_for_every_float_dtype(self) -> None:
        weights = _heavy_tailed(96, 1024, 11)
        for name, quantize in PLAIN_FORMATS:
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                with self.subTest(format=name, dtype=dtype):
                    cpu = quantize(weights.to(dtype))
                    cuda = quantize(weights.to(dtype).cuda())
                    self.assertTrue(torch.equal(cpu.codes, cuda.codes.cpu()))
                    self.assertTrue(torch.equal(cpu.scales, cuda.scales.cpu()))

    def test_scale_helpers_match_on_identical_rotated_input(self) -> None:
        rotated = _heavy_tailed(256, 1024, 7)
        cases = (
            ("int8 rows", lambda x: quantize_symmetric_rows(x).scales),
            ("int5-int7 rows", lambda x: quantize_symmetric_rows(
                x, qmin=-31, qmax=31, use_float32_math=True, zero_scale=1.0,
            ).scales),
            ("absmax scale", lambda x: select_rowwise_absmax_scale(x, qmax=63)),
            ("w4a4 scale", lambda x: w4._w4_absmax_and_scale(x)[1]),
        )
        for name, scales in cases:
            with self.subTest(helper=name):
                self.assertTrue(torch.equal(scales(rotated), scales(rotated.cuda()).cpu()))

    def test_grouped_a8_channel_scales_match_on_identical_input(self) -> None:
        generator = torch.Generator().manual_seed(5)
        level_table = torch.linspace(-8, 7, 16)
        codes = torch.randint(0, 16, (64, 8, 16), generator=generator)
        group_scale = torch.rand((64, 8, 1), generator=generator) * 0.01 + 1e-4
        cpu, _ = stored_scales(level_table, codes, group_scale)
        cuda, _ = stored_scales(level_table.cuda(), codes.cuda(), group_scale.cuda())
        self.assertTrue(torch.equal(cpu, cuda.cpu()))


if __name__ == "__main__":
    unittest.main()
