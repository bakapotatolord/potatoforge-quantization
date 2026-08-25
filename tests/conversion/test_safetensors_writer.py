import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import load_file, save_file

from potatoforge.planning import (
    TensorHeader,
    build_output_layout,
    build_plan,
)
from potatoforge.profiles import QuantizationProfile
from potatoforge.quantization import quantize_int8_tensorwise
from potatoforge.safetensors_writer import write_safetensors_file
from potatoforge.source_payloads import stream_output_payloads
from potatoforge.headers.source_header import read_source_model_header


INT8_PROFILE: QuantizationProfile = {
    "default": "keep",
    "rules": (
        {
            "action": "int8",
            "prefix": "",
            "suffixes": (".attn.wq.weight",),
        },
    ),
}


def raw_tensor_bytes(tensor: torch.Tensor) -> bytes:
    return bytes(
        tensor.view(torch.uint8)
        .flatten()
        .tolist()
    )


class TestSafetensorsWriter(unittest.TestCase):
    def test_writes_a_valid_planned_file(self) -> None:
        source_header: TensorHeader= {
            "model.diffusion_model.blocks.1.attn.wq.weight": {
                "dtype": "BF16",
                "shape": [2, 3],
                "data_offsets": [0, 12],
            },
            "model.diffusion_model.blocks.1.attn.wq.bias": {
                "dtype": "BF16",
                "shape": [2],
                "data_offsets": [12, 16],
            },
        }

        profile = INT8_PROFILE
        layout = build_output_layout(
            build_plan(source_header, profile)
        )

        weights = torch.tensor(
            [
                [1.0, 0.25, -1.0],
                [0.01, -0.02, 0.02],
            ],
            dtype=torch.bfloat16,
        )
        result = quantize_int8_tensorwise(weights)

        bias = torch.tensor(
            [1.0, -1.0],
            dtype=torch.bfloat16,
        )
        marker = json.dumps(
            {"format": "int8_tensorwise"}
        ).encode("utf-8")

        payloads = {
            "model.diffusion_model.blocks.1.attn.wq.weight": raw_tensor_bytes(
                result.codes
            ),
            "model.diffusion_model.blocks.1.attn.wq.weight_scale": raw_tensor_bytes(
                result.scales
            ),
            "model.diffusion_model.blocks.1.attn.wq.comfy_quant": marker,
            "model.diffusion_model.blocks.1.attn.wq.bias": raw_tensor_bytes(
                bias
            ),
        }

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "manual.safetensors"

            write_safetensors_file(
                output_path,
                layout,
                payloads.items(),
            )

            tensors = load_file(output_path)

        self.assertTrue(
            torch.equal(
                tensors["model.diffusion_model.blocks.1.attn.wq.weight"],
                result.codes,
            )
        )
        self.assertTrue(
            torch.equal(
                tensors[
                    "model.diffusion_model.blocks.1.attn.wq.weight_scale"
                ],
                result.scales,
            )
        )
        self.assertTrue(
            torch.equal(
                tensors["model.diffusion_model.blocks.1.attn.wq.bias"],
                bias,
            )
        )

        marker_text = bytes(
            tensors[
                "model.diffusion_model.blocks.1.attn.wq.comfy_quant"
            ].tolist()
        ).decode("utf-8")

        self.assertEqual(
            json.loads(marker_text),
            {"format": "int8_tensorwise"},
        )

    def test_writes_a_stream(self) -> None:
        block_zero_weights = torch.tensor(
            [
                [1.0, 0.25, -1.0],
                [0.01, -0.02, 0.02],
            ],
            dtype=torch.bfloat16,
        )

        block_one_weights = torch.tensor(
            [
                [0.05, -0.02, 0.09, -0.07],
                [1.5, 0.25, -1.0, 0.5],
            ],
            dtype=torch.bfloat16,
        )

        bias = torch.tensor(
            [1.0, -1.0],
            dtype=torch.bfloat16,
        )

        source_tensors = {
            "model.diffusion_model.blocks.1.attn.wq.weight": block_zero_weights,
            "model.diffusion_model.blocks.2.attn.wq.weight": block_one_weights,
            "model.diffusion_model.blocks.2.attn.wq.bias": bias,
        }

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source_test.safetensors"
            target_path = Path(directory) / "quantized_test.safetensors"

            save_file(source_tensors, str(source_path))

            source_model = read_source_model_header(source_path)

            profile = INT8_PROFILE
            plan_entries = build_plan(
                source_model.tensors,
                profile,
            )

            safetensors_layout = build_output_layout(plan_entries)

            write_safetensors_file(
                target_path,
                safetensors_layout,
                stream_output_payloads(source_path, plan_entries),
            )

            output_tensors = load_file(target_path)

        result_zero = quantize_int8_tensorwise(block_zero_weights)
        result_one = quantize_int8_tensorwise(block_one_weights)

        self.assertTrue(
            torch.equal(
                output_tensors[
                    "model.diffusion_model.blocks.1.attn.wq.weight"
                ],
                result_zero.codes,
            )
        )
        self.assertTrue(
            torch.equal(
                output_tensors[
                    "model.diffusion_model.blocks.1.attn.wq.weight_scale"
                ],
                result_zero.scales,
            )
        )
        self.assertTrue(
            torch.equal(
                output_tensors[
                    "model.diffusion_model.blocks.2.attn.wq.weight"
                ],
                result_one.codes,
            )
        )
        self.assertTrue(
            torch.equal(
                output_tensors[
                    "model.diffusion_model.blocks.2.attn.wq.weight_scale"
                ],
                result_one.scales,
            )
        )
        self.assertTrue(
            torch.equal(
                output_tensors[
                    "model.diffusion_model.blocks.2.attn.wq.bias"
                ],
                bias,
            )
        )

        for marker_name in (
            "model.diffusion_model.blocks.1.attn.wq.comfy_quant",
            "model.diffusion_model.blocks.2.attn.wq.comfy_quant",
        ):
            marker_text = bytes(
                output_tensors[marker_name].tolist()
            ).decode("utf-8")

            self.assertEqual(
                json.loads(marker_text),
                {"format": "int8_tensorwise"},
            )
