"""Regenerate w4a8_kitchen_reference.pt from comfy-kitchen's eager W4A8 quantizer.

Run with a Python environment that has comfy-kitchen (upstream e769274 or later):

    <comfy-kitchen>/.venv/Scripts/python.exe -I tests/quantization/data/make_w4a8_kitchen_reference.py

PotatoForge does not depend on comfy-kitchen; the test compares against this file.
"""

from pathlib import Path

import torch
from comfy_kitchen.backends.eager import w4a8_int8 as eager_w4a8
from comfy_kitchen.backends.eager.quantization import rotate_int8_convrot_weight

OUTPUT = Path(__file__).with_name("w4a8_kitchen_reference.pt")


def _cases() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(431)
    gaussian = torch.randn(48, 512, generator=generator) * 0.02
    gaussian[5].zero_()
    gaussian[7, :16] = 0.0
    # Heavy-tailed after rotation (the Hadamard is its own inverse), so kitchen
    # fits a per-tensor codebook instead of using the frozen table.
    sparse = (torch.rand(32, 768, generator=generator) < 0.15).float()
    sparse = sparse * torch.randn(32, 768, generator=generator) * 0.05
    heavy = rotate_int8_convrot_weight(sparse, 256)
    return {"gaussian": gaussian, "heavy": heavy}


def main() -> None:
    reference = {}
    for name, weights in _cases().items():
        packed, s_rel, s_channel, correction, codebook = eager_w4a8.quantize_w4a8_int8_weight(weights)
        assert correction is None and codebook is not None
        int8 = eager_w4a8._dequant_int4_grouped_to_int8(packed, s_rel, codebook, 16)
        reference[name] = {
            "weights": weights.contiguous(),
            "packed": packed,
            "s_rel": s_rel.view(torch.uint8),
            "s_channel": s_channel,
            "codebook": codebook,
            "int8": int8,
            "dequantized": eager_w4a8.dequantize_w4a8_int8_weight(
                packed, s_rel, s_channel, codebook=codebook, output_dtype=torch.float32
            ),
        }
    torch.save(reference, OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
