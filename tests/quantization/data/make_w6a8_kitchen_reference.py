"""Regenerate w6a8_kitchen_reference.pt from comfy-kitchen's eager W6A8 quantizer.

Run with a Python environment that has comfy-kitchen (upstream e769274 or later):

    <comfy-kitchen>/.venv/Scripts/python.exe -I tests/quantization/data/make_w6a8_kitchen_reference.py

PotatoForge does not depend on comfy-kitchen; the test compares against this file.
"""

from pathlib import Path

import torch
from comfy_kitchen.backends.eager import w4a8_int8 as eager_w4a8

OUTPUT = Path(__file__).with_name("w6a8_kitchen_reference.pt")


def _cases() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(611)
    gaussian = torch.randn(48, 512, generator=generator) * 0.02
    gaussian[3].zero_()
    gaussian[9, :16] = 0.0
    # Rows spanning four decades exercise the fp8 group-scale search near the
    # bottom of e4m3's range and the INT8 grid clamp.
    scaled = torch.randn(32, 768, generator=generator)
    scaled = scaled * torch.logspace(-4, 0, 32).view(-1, 1)
    return {"gaussian": gaussian, "scaled": scaled}


def main() -> None:
    reference = {}
    for name, weights in _cases().items():
        packed, s_rel, s_channel, correction, codebook = eager_w4a8.quantize_w4a8_int8_weight(weights, bits=6)
        assert correction is None and codebook is None
        reference[name] = {
            "weights": weights.contiguous(),
            "packed": packed,
            "s_rel": s_rel.view(torch.uint8),
            "s_channel": s_channel,
            "int8": eager_w4a8._dequant_int4_grouped_to_int8(packed, s_rel, None, 16),
            "dequantized": eager_w4a8.dequantize_w4a8_int8_weight(
                packed, s_rel, s_channel, output_dtype=torch.float32
            ),
        }
    torch.save(reference, OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
