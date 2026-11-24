"""Regenerate w3a8_kitchen_reference.pt from comfy-kitchen's eager W3A8 quantizer.

Run with a Python environment that has comfy-kitchen (upstream e769274 or later):

    <comfy-kitchen>/.venv/Scripts/python.exe -I tests/quantization/data/make_w3a8_kitchen_reference.py

PotatoForge does not depend on comfy-kitchen; the test compares against this file.
"""

from pathlib import Path

import torch
from comfy_kitchen.backends.eager import w4a8_int8 as eager_w4a8

OUTPUT = Path(__file__).with_name("w3a8_kitchen_reference.pt")


def _cases() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(311)
    gaussian = torch.randn(48, 512, generator=generator) * 0.02
    gaussian[3].zero_()
    gaussian[9, :16] = 0.0
    scaled = torch.randn(32, 768, generator=generator)
    scaled = scaled * torch.logspace(-4, 0, 32).view(-1, 1)
    # Weights that are sparse in the rotated basis stay heavy-tailed after ConvRot and
    # trip the kurtosis gate, so the per-tensor codebook fit (not the frozen table)
    # is compared too. The normalized Hadamard is its own inverse.
    sparse = torch.randn(32, 512, generator=generator) * (torch.rand(32, 512, generator=generator) < 0.05)
    heavy = eager_w4a8.rotate_int8_convrot_weight(sparse.contiguous(), 256) * 0.02
    return {"gaussian": gaussian, "scaled": scaled, "heavy": heavy}


def main() -> None:
    reference = {}
    for name, weights in _cases().items():
        packed, s_rel, s_channel, correction, codebook = eager_w4a8.quantize_w4a8_int8_weight(weights, bits=3)
        assert correction is None and codebook is not None
        reference[name] = {
            "weights": weights.contiguous(),
            "packed": packed,
            "s_rel": s_rel.view(torch.uint8),
            "s_channel": s_channel,
            "codebook": codebook,
            "dequantized": eager_w4a8.dequantize_w4a8_int8_weight(
                packed, s_rel, s_channel, codebook=codebook, output_dtype=torch.float32
            ),
        }
    torch.save(reference, OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
