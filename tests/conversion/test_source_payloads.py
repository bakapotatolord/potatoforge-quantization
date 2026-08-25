from pathlib import Path
import unittest

import torch
from tempfile import TemporaryDirectory
from safetensors.torch import save_file

from potatoforge.planning import (
    CONVROT_W4A4_MARKER_PAYLOAD,
    INT6_CONVROT_MARKER_PAYLOAD,
    INT6_ROWWISE_MARKER_PAYLOAD,
    INT8_CONVROT_MARKER_PAYLOAD,
    INT8_MARKER_PAYLOAD,
    build_plan,
)
from potatoforge.profiles import QuantizationProfile
from potatoforge.headers.source_header import read_source_model_header
from potatoforge.source_payloads import tensor_to_raw_bytes, stream_output_payloads
from potatoforge.quantization import quantize_int8_tensorwise, quantize_int8_convrot
from potatoforge.quantization.int6_rowwise import (
    quantize_int6_convrot,
    quantize_int6_rowwise,
)
from potatoforge.quantization.int6_packing import pack_int6_row_major
from potatoforge.quantization.convrot_w4a4 import quantize_convrot_w4a4


class TestSourcePayloads(unittest.TestCase):
    def test_stream_source_payload_casts_kept_float32_to_bfloat16(self) -> None:
        tensor = torch.tensor(
            [
                [1.0, 0.25, -1.0, 0.0],
                [0.01, -0.02, 0.02, 0.0],
            ],
            dtype=torch.float32,
        )

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "test_tensor.safetensors"

            save_file({"test_tensor": tensor}, str(source_path))

            header = read_source_model_header(source_path)
            profile: QuantizationProfile = {
                "default": "keep",
                "keep_dtype": "BF16",
                "rules": (),
            }
            plan_entries = build_plan(header.tensors, profile)
            payloads = list(stream_output_payloads(source_path, plan_entries))

        self.assertEqual(len(payloads), 1)
        payload_name, payload_bytes = payloads[0]

        self.assertEqual(payload_name, "test_tensor")
        self.assertEqual(
            payload_bytes,
            tensor_to_raw_bytes(tensor.to(torch.bfloat16)),
        )

    def test_round_trips_bfloat16_storage(self) -> None:
        tensor = torch.tensor(
            [
                [1.0, 0.25, -1.0],
                [0.01, -0.02, 0.02],
            ],
            dtype=torch.bfloat16,
        )

        payload = tensor_to_raw_bytes(tensor)

        self.assertEqual(
            len(payload),
            tensor.numel() * tensor.element_size(),
        )

        restored = torch.frombuffer(
            bytearray(payload),
            dtype=torch.bfloat16,
        ).reshape(tensor.shape)

        self.assertTrue(torch.equal(restored, tensor))

    def test_stream_source_payload_keep(self) -> None:
        tensor = torch.tensor(
            [
                [1.0, 0.25, -1.0],
                [0.01, -0.02, 0.02],
            ],
            dtype=torch.bfloat16,
        )

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "test_tensor.safetensors"

            save_file({"test_tensor": tensor}, str(source_path))

            profile: QuantizationProfile = {
                "default": "keep",
                "rules": ()
            }

            header = read_source_model_header(source_path)

            plan_entries = build_plan(header.tensors, profile)

            payloads = list(stream_output_payloads(source_path, plan_entries))

            self.assertEqual(len(payloads), 1)

            payload_name, payload_bytes = payloads[0]

            self.assertEqual(payload_name, "test_tensor")
            self.assertEqual(payload_bytes, tensor_to_raw_bytes(tensor))

    def test_stream_source_payload_int8(self) -> None:
        tensor = torch.tensor(
            [
                [1.0, 0.25, -1.0],
                [0.01, -0.02, 0.02],
            ],
            dtype=torch.bfloat16,
        )

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "test_tensor.safetensors"

            save_file(
                {
                    "model.diffusion_model.blocks.1.attn.wq.weight": tensor,
                },
                str(source_path),
            )

            header = read_source_model_header(source_path)

            profile: QuantizationProfile = {
                "default": "keep",
                "rules": (
                    {
                        "action": "int8",
                        "prefix": "",
                        "suffixes": (".attn.wq.weight",),
                    },
                ),
            }
            plan_entries = build_plan(header.tensors, profile)

            payloads = list(stream_output_payloads(source_path, plan_entries))

            self.assertEqual(len(payloads), 3)

            weight_name, weight_bytes = payloads[0]
            scale_name, scale_bytes = payloads[1]
            marker_name, marker_bytes = payloads[2]

            self.assertEqual(
                weight_name,
                "model.diffusion_model.blocks.1.attn.wq.weight",
            )
            self.assertEqual(
                scale_name,
                "model.diffusion_model.blocks.1.attn.wq.weight_scale",
            )
            self.assertEqual(
                marker_name,
                "model.diffusion_model.blocks.1.attn.wq.comfy_quant",
            )

            result = quantize_int8_tensorwise(tensor)

            self.assertEqual(weight_bytes, tensor_to_raw_bytes(result.codes))
            self.assertEqual(scale_bytes, tensor_to_raw_bytes(result.scales))
            self.assertEqual(marker_bytes, b'{"format": "int8_tensorwise"}')

    def test_stream_source_payload_int8_convrot(self) -> None:
        tensor = torch.zeros(
            (1, 256),
            dtype=torch.bfloat16,
        )
        tensor[0, 0] = 70.0

        profile: QuantizationProfile = {
            "default": "keep",
            "rules": (
                {
                    "action": "int8_convrot",
                    "prefix": "blocks.",
                    "suffixes": (".attn.wq.weight",),
                },
            ),
        }

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "test_tensor.safetensors"

            save_file(
                {"blocks.0.attn.wq.weight": tensor},
                str(source_path),
            )

            header = read_source_model_header(source_path)
            plan_entries = build_plan(header.tensors, profile)
            payloads = list(
                stream_output_payloads(source_path, plan_entries)
            )

        self.assertEqual(len(payloads), 3)

        weight_name, weight_bytes = payloads[0]
        scale_name, scale_bytes = payloads[1]
        marker_name, marker_bytes = payloads[2]

        self.assertEqual(weight_name, "blocks.0.attn.wq.weight")
        self.assertEqual(
            scale_name,
            "blocks.0.attn.wq.weight_scale",
        )
        self.assertEqual(
            marker_name,
            "blocks.0.attn.wq.comfy_quant",
        )

        result = quantize_int8_convrot(tensor)

        self.assertEqual(
            weight_bytes,
            tensor_to_raw_bytes(result.codes),
        )
        self.assertEqual(
            scale_bytes,
            tensor_to_raw_bytes(result.scales),
        )
        self.assertEqual(
            marker_bytes,
            INT8_CONVROT_MARKER_PAYLOAD,
        )

    def test_stream_source_payload_int6_rowwise_packs_codes(self) -> None:
        tensor = torch.tensor(
            [
                [1.0, 0.25, -1.0, 0.0],
                [0.01, -0.02, 0.02, 0.0],
            ],
            dtype=torch.bfloat16,
        )
        profile: QuantizationProfile = {
            "default": "keep",
            "rules": (
                {
                    "action": "int6_rowwise",
                    "prefix": "blocks.",
                    "suffixes": (".attn.wq.weight",),
                },
            ),
        }

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "test_tensor.safetensors"
            save_file({"blocks.0.attn.wq.weight": tensor}, str(source_path))

            header = read_source_model_header(source_path)
            plan_entries = build_plan(header.tensors, profile)
            payloads = list(stream_output_payloads(source_path, plan_entries))

        result = quantize_int6_rowwise(tensor)

        self.assertEqual(
            [name for name, _ in payloads],
            [
                "blocks.0.attn.wq.weight",
                "blocks.0.attn.wq.weight_scale",
                "blocks.0.attn.wq.comfy_quant",
            ],
        )
        self.assertEqual(payloads[1][1], tensor_to_raw_bytes(result.scales))
        packed = pack_int6_row_major(result.codes)
        self.assertEqual(payloads[0][1], tensor_to_raw_bytes(packed.packed_codes))
        self.assertEqual(payloads[2][1], INT6_ROWWISE_MARKER_PAYLOAD)

    def test_stream_source_payload_int6_convrot_packs_rotated_codes(self) -> None:
        tensor = torch.zeros((1, 256), dtype=torch.bfloat16)
        tensor[0, 0] = 70.0
        profile: QuantizationProfile = {
            "default": "keep",
            "rules": (
                {
                    "action": "int6_convrot",
                    "prefix": "blocks.",
                    "suffixes": (".attn.wq.weight",),
                },
            ),
        }

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "test_tensor.safetensors"
            save_file({"blocks.0.attn.wq.weight": tensor}, str(source_path))
            header = read_source_model_header(source_path)
            payloads = list(
                stream_output_payloads(source_path, build_plan(header.tensors, profile))
            )

        result = quantize_int6_convrot(tensor)
        packed = pack_int6_row_major(result.codes)
        self.assertEqual(payloads[0][1], tensor_to_raw_bytes(packed.packed_codes))
        self.assertEqual(payloads[1][1], tensor_to_raw_bytes(result.scales))
        self.assertEqual(payloads[2][1], INT6_CONVROT_MARKER_PAYLOAD)

    def test_stream_source_payload_convrot_w4a4(self) -> None:
        tensor = torch.zeros(
            (1, 256),
            dtype=torch.bfloat16,
        )
        tensor[0, 0] = 70.0

        profile: QuantizationProfile = {
            "default": "keep",
            "rules": (
                {
                    "action": "convrot_w4a4",
                    "prefix": "blocks.",
                    "suffixes": (".attn.wq.weight",),
                },
            ),
        }

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "test_tensor.safetensors"

            save_file(
                {"blocks.0.attn.wq.weight": tensor},
                str(source_path),
            )

            header = read_source_model_header(source_path)
            plan_entries = build_plan(header.tensors, profile)
            payloads = list(
                stream_output_payloads(source_path, plan_entries)
            )

        self.assertEqual(len(payloads), 3)

        weight_name, weight_bytes = payloads[0]
        scale_name, scale_bytes = payloads[1]
        marker_name, marker_bytes = payloads[2]

        self.assertEqual(weight_name, "blocks.0.attn.wq.weight")
        self.assertEqual(
            scale_name,
            "blocks.0.attn.wq.weight_scale",
        )
        self.assertEqual(
            marker_name,
            "blocks.0.attn.wq.comfy_quant",
        )

        result = quantize_convrot_w4a4(tensor)

        self.assertEqual(
            weight_bytes,
            tensor_to_raw_bytes(result.packed_codes),
        )
        self.assertEqual(
            scale_bytes,
            tensor_to_raw_bytes(result.scales),
        )
        self.assertEqual(
            marker_bytes,
            CONVROT_W4A4_MARKER_PAYLOAD,
        )
