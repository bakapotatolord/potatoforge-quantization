from pathlib import Path
import unittest
import json
from contextlib import redirect_stdout
from io import StringIO
from safetensors.torch import save_file, load_file
from tempfile import TemporaryDirectory
import torch
from unittest.mock import patch

from potatoforge.planning import (
    INT6_ROWWISE_MARKER,
    INT8_MARKER,
    CONVROT_W4A4_MARKER,
    INT8_CONVROT_MARKER,
)
from potatoforge.converter import (
    ResolvedIOMode,
    convert_model,
    convert_model_from_profile,
    resolve_io_mode,
)
from potatoforge.headers.header_reader import read_header_from_safetensors
from potatoforge.lora.lora_merge import (
    AdapterMergeInput,
    merge_bf16_adapters,
)
from potatoforge.quantization import quantize_int8_tensorwise, quantize_int8_convrot
from potatoforge.quantization.int6_rowwise import quantize_int6_rowwise
from potatoforge.quantization.int6_packing import pack_int6_row_major
from potatoforge.profiles import QuantizationProfile
from potatoforge.source_payloads import tensor_to_raw_bytes
from potatoforge.quantization.convrot_w4a4 import (
    quantize_convrot_w4a4,
)


class TestConverter(unittest.TestCase):
    def _write_int8_profile(self, directory: str) -> Path:
        profile_path = Path(directory) / "test-int8-profile.json"
        profile_path.write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "profile_id": "test-int8",
                    "default": "keep",
                    "rules": [
                        {
                            "action": "int8",
                            "prefix": "",
                            "suffixes": [".attn.wq.weight"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return profile_path

    def test_converter_prints_selected_io_mode(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.safetensors"
            output_path = root / "output.safetensors"
            save_file(
                {"tensor": torch.tensor([1.0], dtype=torch.bfloat16)},
                str(source_path),
            )
            profile: QuantizationProfile = {
                "default": "keep",
                "rules": (),
            }
            output = StringIO()

            with (
                patch(
                    "potatoforge.converter.resolve_io_mode",
                    return_value=ResolvedIOMode("serial", "test selection"),
                ),
                redirect_stdout(output),
            ):
                convert_model(
                    source_path,
                    output_path,
                    profile,
                    on_entry_started=lambda *_args: None,
                    io_mode="batched",
                )

        self.assertIn("I/O mode: serial (test selection)", output.getvalue())

    def test_batched_falls_back_to_serial_without_an_input_buffer(self) -> None:
        result = resolve_io_mode("batched", None)

        self.assertEqual(
            result,
            ResolvedIOMode(
                "serial",
                "batched mode has no input buffer; falling back to serial",
            ),
        )

    def test_buffered_conversions_match_serial_for_non_physical_header_order(
        self,
    ) -> None:
        late = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        middle = torch.tensor([4.0], dtype=torch.bfloat16)
        early = torch.tensor([5.0, 6.0], dtype=torch.bfloat16)
        tensors = {
            "late": late,
            "middle": middle,
            "early": early,
        }
        physical_order = ("early", "late", "middle")
        header_order = ("late", "middle", "early")

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.safetensors"
            serial_path = root / "serial.safetensors"
            batched_path = root / "batched.safetensors"

            offsets: dict[str, tuple[int, int]] = {}
            raw_payload = bytearray()
            for name in physical_order:
                payload = tensor_to_raw_bytes(tensors[name])
                start = len(raw_payload)
                raw_payload.extend(payload)
                offsets[name] = (start, len(raw_payload))

            header = {
                name: {
                    "dtype": "BF16",
                    "shape": list(tensors[name].shape),
                    "data_offsets": list(offsets[name]),
                }
                for name in header_order
            }
            header_bytes = json.dumps(
                header,
                separators=(",", ":"),
            ).encode("utf-8")
            source_path.write_bytes(
                len(header_bytes).to_bytes(8, "little")
                + header_bytes
                + raw_payload
            )

            profile: QuantizationProfile = {
                "default": "keep",
                "rules": (),
            }
            convert_model(
                source_path,
                serial_path,
                profile,
                on_entry_started=None,
            )
            convert_model(
                source_path,
                batched_path,
                profile,
                on_entry_started=None,
                io_mode="batched",
                input_buffer_bytes=5,
            )
            self.assertEqual(serial_path.read_bytes(), batched_path.read_bytes())

    def test_batched_propagates_source_read_errors(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.safetensors"
            output_path = root / "output.safetensors"
            header = {
                "tensor": {
                    "dtype": "BF16",
                    "shape": [2],
                    "data_offsets": [0, 4],
                }
            }
            header_bytes = json.dumps(
                header,
                separators=(",", ":"),
            ).encode("utf-8")
            source_path.write_bytes(
                len(header_bytes).to_bytes(8, "little")
                + header_bytes
                + b"\x00\x00"
            )

            profile: QuantizationProfile = {
                "default": "keep",
                "rules": (),
            }

            with self.assertRaisesRegex(ValueError, "truncated"):
                convert_model(
                    source_path,
                    output_path,
                    profile,
                    on_entry_started=None,
                    io_mode="batched",
                    input_buffer_bytes=4,
                )

            self.assertFalse(output_path.exists())

    def test_converter_preserves_top_level_metadata(self) -> None:
        weights = torch.tensor(
            [[1.0, 0.25, -1.0]],
            dtype=torch.bfloat16,
        )
        metadata = {
            "creator": "potatoforge-test",
            "source": "synthetic",
        }

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.safetensors"
            target_path = Path(directory) / "target.safetensors"
            profile_path = self._write_int8_profile(directory)

            save_file(
                {
                    "model.diffusion_model.blocks.1.attn.wq.weight": weights,
                },
                str(source_path),
                metadata=metadata,
            )

            convert_model_from_profile(
                source_path,
                target_path,
                profile_path,
            )

            output_header = read_header_from_safetensors(target_path)

        self.assertEqual(output_header["__metadata__"], metadata)

    def test_converter(self) -> None:
        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "converter_test.safetensors"
            target_path = Path(directory) / "converted_test.safetensors"
            profile_path = self._write_int8_profile(directory)

            weights = torch.tensor(
                [
                    [1.0, 0.25, -1.0],
                    [0.01, -0.02, 0.02],
                ],
                dtype=torch.bfloat16,
            )

            bias = torch.tensor(
                [1.0, -1.0],
                dtype=torch.bfloat16,
            )

            source_tensors = {
                "model.diffusion_model.blocks.1.attn.wq.weight": weights,
                "model.diffusion_model.blocks.1.attn.wq.bias": bias,
            }

            save_file(source_tensors, str(source_path))

            convert_model_from_profile(source_path, target_path, profile_path)

            self.assertTrue(target_path.exists())
            self.assertFalse(target_path.with_name(f"{target_path.name}.partial").exists())

            output_tensors = load_file(target_path)

            result = quantize_int8_tensorwise(weights)

            marker_text = bytes(
                output_tensors[
                    "model.diffusion_model.blocks.1.attn.wq.comfy_quant"
                ].tolist()
            ).decode("utf-8")

            self.assertTrue(
                torch.equal(
                    output_tensors[
                        "model.diffusion_model.blocks.1.attn.wq.weight"
                    ],
                    result.codes,
                )
            )
            self.assertTrue(
                torch.equal(
                    output_tensors[
                        "model.diffusion_model.blocks.1.attn.wq.weight_scale"
                    ],
                    result.scales,
                )
            )
            self.assertEqual(json.loads(marker_text), INT8_MARKER)
            self.assertTrue(
                torch.equal(
                    output_tensors[
                        "model.diffusion_model.blocks.1.attn.wq.bias"
                    ],
                    bias,
                )
            )

            with self.assertRaises(FileExistsError):
                convert_model_from_profile(source_path, target_path, profile_path)

    def test_fused_adapter_quantization_matches_separate_pipeline(self) -> None:
        source_tensors = {
            "layer.weight": torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                dtype=torch.bfloat16,
            ),
            "kept": torch.tensor([0.5, -0.5], dtype=torch.bfloat16),
        }
        adapter_tensors = {
            "layer.lora_A": torch.tensor(
                [[1.0, 0.0, -1.0]],
                dtype=torch.bfloat16,
            ),
            "layer.lora_B": torch.tensor(
                [[1.0], [0.5]],
                dtype=torch.bfloat16,
            ),
        }
        profile: QuantizationProfile = {
            "default": "keep",
            "rules": (
                {
                    "action": "int8",
                    "prefix": "layer.",
                    "suffixes": (".weight",),
                },
            ),
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.safetensors"
            adapter_path = root / "adapter.safetensors"
            merged_path = root / "merged.safetensors"
            separate_path = root / "separate.safetensors"
            fused_path = root / "fused.safetensors"
            adapter_input = AdapterMergeInput(adapter_path, 0.5)

            save_file(source_tensors, str(source_path), metadata={"test": "fused"})
            save_file(adapter_tensors, str(adapter_path))

            merge_bf16_adapters(
                source_path,
                merged_path,
                (adapter_input,),
            )
            convert_model(
                merged_path,
                separate_path,
                profile,
                on_entry_started=None,
            )
            convert_model(
                source_path,
                fused_path,
                profile,
                on_entry_started=None,
                adapters=(adapter_input,),
                io_mode="batched",
                input_buffer_bytes=8,
            )

            self.assertEqual(
                separate_path.read_bytes(),
                fused_path.read_bytes(),
            )

    def test_converter_writes_a_mixed_quantization_artifact(
        self,
    ) -> None:
        w4_weights = torch.zeros(
            (1, 256),
            dtype=torch.bfloat16,
        )
        w4_weights[0, 0] = 70.0

        int8_weights = torch.tensor(
            [[1.0, 0.25, -1.0]],
            dtype=torch.bfloat16,
        )
        kept_bias = torch.tensor(
            [1.0],
            dtype=torch.bfloat16,
        )

        profile: QuantizationProfile = {
            "default": "keep",
            "rules": (
                {
                    "action": "convrot_w4a4",
                    "prefix": "blocks.0.",
                    "suffixes": (".attn.wq.weight",),
                },
                {
                    "action": "int8",
                    "prefix": "blocks.1.",
                    "suffixes": (".attn.wq.weight",),
                },
            ),
        }

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.safetensors"
            target_path = Path(directory) / "target.safetensors"

            save_file(
                {
                    "blocks.0.attn.wq.weight": w4_weights,
                    "blocks.1.attn.wq.weight": int8_weights,
                    "blocks.1.attn.wq.bias": kept_bias,
                },
                str(source_path),
            )

            convert_model(
                source_path,
                target_path,
                profile,
                io_mode="batched",
                input_buffer_bytes=512,
            )

            output_tensors = load_file(target_path)

        expected_w4 = quantize_convrot_w4a4(w4_weights)
        expected_int8 = quantize_int8_tensorwise(int8_weights)

        self.assertTrue(
            torch.equal(
                output_tensors["blocks.0.attn.wq.weight"],
                expected_w4.packed_codes,
            )
        )
        self.assertTrue(
            torch.equal(
                output_tensors["blocks.0.attn.wq.weight_scale"],
                expected_w4.scales,
            )
        )

        w4_marker = bytes(
            output_tensors[
                "blocks.0.attn.wq.comfy_quant"
            ].tolist()
        ).decode("utf-8")
        self.assertEqual(json.loads(w4_marker), CONVROT_W4A4_MARKER)

        self.assertTrue(
            torch.equal(
                output_tensors["blocks.1.attn.wq.weight"],
                expected_int8.codes,
            )
        )
        self.assertTrue(
            torch.equal(
                output_tensors["blocks.1.attn.wq.weight_scale"],
                expected_int8.scales,
            )
        )

        int8_marker = bytes(
            output_tensors[
                "blocks.1.attn.wq.comfy_quant"
            ].tolist()
        ).decode("utf-8")
        self.assertEqual(json.loads(int8_marker), INT8_MARKER)

        self.assertTrue(
            torch.equal(
                output_tensors["blocks.1.attn.wq.bias"],
                kept_bias,
            )
        )

    def test_converter_quantizes_float16_with_all_methods(self) -> None:
        plain_weights = torch.tensor(
            [[1.0, 0.25, -1.0] + [0.0] * 253],
            dtype=torch.float16,
        )
        convrot_weights = torch.zeros(
            (1, 256),
            dtype=torch.float16,
        )
        convrot_weights[0, 0] = 70.0
        w4_weights = torch.zeros(
            (1, 256),
            dtype=torch.float16,
        )
        w4_weights[0, 0] = 70.0
        kept_bias = torch.tensor([1.0], dtype=torch.float16)

        profile: QuantizationProfile = {
            "default": "keep",
            "rules": (
                {
                    "action": "int8",
                    "prefix": "blocks.0.",
                    "suffixes": (".attn.wq.weight",),
                },
                {
                    "action": "int8_convrot",
                    "prefix": "blocks.1.",
                    "suffixes": (".attn.wq.weight",),
                },
                {
                    "action": "convrot_w4a4",
                    "prefix": "blocks.2.",
                    "suffixes": (".attn.wq.weight",),
                },
            ),
        }

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source-float16.safetensors"
            target_path = Path(directory) / "target-float16.safetensors"

            save_file(
                {
                    "blocks.0.attn.wq.weight": plain_weights,
                    "blocks.1.attn.wq.weight": convrot_weights,
                    "blocks.2.attn.wq.weight": w4_weights,
                    "blocks.2.attn.wq.bias": kept_bias,
                },
                str(source_path),
            )

            convert_model(source_path, target_path, profile)
            output_tensors = load_file(target_path)

        expected_plain = quantize_int8_tensorwise(plain_weights)
        expected_convrot = quantize_int8_convrot(convrot_weights)
        expected_w4 = quantize_convrot_w4a4(w4_weights)

        self.assertTrue(
            torch.equal(
                output_tensors["blocks.0.attn.wq.weight"],
                expected_plain.codes,
            )
        )
        self.assertTrue(
            torch.equal(
                output_tensors["blocks.1.attn.wq.weight"],
                expected_convrot.codes,
            )
        )
        self.assertTrue(
            torch.equal(
                output_tensors["blocks.2.attn.wq.weight"],
                expected_w4.packed_codes,
            )
        )
        self.assertTrue(
            torch.equal(
                output_tensors["blocks.2.attn.wq.bias"],
                kept_bias,
            )
        )

    def test_converter_loads_profile_from_json_path(self) -> None:
        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.safetensors"
            target_path = Path(directory) / "target.safetensors"
            profile_path = Path(directory) / "profile.json"

            weights = torch.tensor(
                [
                    [1.0, 0.25, -1.0],
                    [0.01, -0.02, 0.02],
                ],
                dtype=torch.bfloat16,
            )

            bias = torch.tensor(
                [1.0, -1.0],
                dtype=torch.bfloat16,
            )

            save_file(
                {
                    "blocks.0.attn.wq.weight": weights,
                    "blocks.0.attn.wq.bias": bias,
                },
                str(source_path),
            )

            profile_path.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "profile_id": "test-int8",
                        "default": "keep",
                        "rules": [
                            {
                                "action": "int8",
                                "prefix": "blocks.",
                                "suffixes": [
                                    ".attn.wq.weight",
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            convert_model_from_profile(
                source_path,
                target_path,
                profile_path,
            )

            output_tensors = load_file(target_path)

        expected = quantize_int8_tensorwise(weights)

        self.assertTrue(
            torch.equal(
                output_tensors["blocks.0.attn.wq.weight"],
                expected.codes,
            )
        )
        self.assertTrue(
            torch.equal(
                output_tensors["blocks.0.attn.wq.weight_scale"],
                expected.scales,
            )
        )
        self.assertTrue(
            torch.equal(
                output_tensors["blocks.0.attn.wq.bias"],
                bias,
            )
        )

    def test_converter_writes_an_int8_convrot_artifact(
        self,
    ) -> None:
        weights = torch.zeros(
            (1, 256),
            dtype=torch.bfloat16,
        )
        weights[0, 0] = 70.0
        bias = torch.tensor(
            [1.0],
            dtype=torch.bfloat16,
        )

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
            source_path = Path(directory) / "source.safetensors"
            target_path = Path(directory) / "target.safetensors"

            save_file(
                {
                    "blocks.0.attn.wq.weight": weights,
                    "blocks.0.attn.wq.bias": bias,
                },
                str(source_path),
            )

            convert_model(source_path, target_path, profile)

            output_tensors = load_file(target_path)

        expected = quantize_int8_convrot(weights)

        self.assertTrue(
            torch.equal(
                output_tensors["blocks.0.attn.wq.weight"],
                expected.codes,
            )
        )
        self.assertTrue(
            torch.equal(
                output_tensors["blocks.0.attn.wq.weight_scale"],
                expected.scales,
            )
        )

        marker = bytes(
            output_tensors[
                "blocks.0.attn.wq.comfy_quant"
            ].tolist()
        ).decode("utf-8")
        self.assertEqual(json.loads(marker), INT8_CONVROT_MARKER)

        self.assertTrue(
            torch.equal(
                output_tensors["blocks.0.attn.wq.bias"],
                bias,
            )
        )

    def test_converter_writes_packed_int6_rowwise(self) -> None:
        weights = torch.tensor(
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
            source_path = Path(directory) / "source.safetensors"
            target_path = Path(directory) / "target.safetensors"
            save_file({"blocks.0.attn.wq.weight": weights}, str(source_path))

            convert_model(source_path, target_path, profile)
            output_tensors = load_file(target_path)

        expected = quantize_int6_rowwise(weights)
        expected_packed = pack_int6_row_major(expected.codes)
        self.assertTrue(
            torch.equal(
                output_tensors["blocks.0.attn.wq.weight"],
                expected_packed.packed_codes,
            )
        )
        self.assertTrue(
            torch.equal(
                output_tensors["blocks.0.attn.wq.weight_scale"],
                expected.scales,
            )
        )
        marker = bytes(
            output_tensors["blocks.0.attn.wq.comfy_quant"].tolist()
        ).decode("utf-8")
        self.assertEqual(json.loads(marker), INT6_ROWWISE_MARKER)
