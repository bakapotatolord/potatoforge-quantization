import unittest

from potatoforge.planning import (
    BFLOAT16_BYTES_PER_ELEMENT,
    CONVROT_W4A4_MARKER_BYTE_COUNT,
    INT6_CONVROT_MARKER_BYTE_COUNT,
    INT6_ROWWISE_MARKER_BYTE_COUNT,
    INT8_CONVROT_MARKER_BYTE_COUNT,
    TensorDescriptor,
    build_output_layout,
    build_plan,
    layout_to_header,
    plan_input_batches,
)

from potatoforge.profiles import QuantizationProfile


class TestInt8TensorwisePlan(unittest.TestCase):
    def test_plans_kept_float32_as_bfloat16(self) -> None:
        header: dict[str, TensorDescriptor] = {
            "tproj.1.weight": {
                "dtype": "F32",
                "shape": [2, 3],
                "data_offsets": [0, 24],
            },
        }
        profile: QuantizationProfile = {
            "default": "keep",
            "keep_dtype": "BF16",
            "rules": (),
        }

        entries = build_plan(header, profile)
        layout = build_output_layout(entries)

        self.assertEqual(entries[0]["action"], "keep")
        self.assertEqual(
            entries[0]["output_tensors"][0],
            (
                "tproj.1.weight",
                "BF16",
                (2, 3),
                2 * 3 * BFLOAT16_BYTES_PER_ELEMENT,
            ),
        )
        self.assertEqual(layout.raw_data_bytes, 12)

    def test_output_layout_uses_contiguous_offsets(self) -> None:
        header: dict[str, TensorDescriptor] = {
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
        entries = build_plan(header, profile)
        layout = build_output_layout(entries)
        output_header = layout_to_header(layout)

        self.assertEqual(layout.raw_data_bytes, 47)

        self.assertEqual(
            [
                tensor.data_offsets
                for tensor in layout.tensors
            ],
            [
                (0, 6),
                (6, 14),
                (14, 43),
                (43, 47),
            ],
        )

        self.assertEqual(
            output_header[
                "model.diffusion_model.blocks.1.attn.wq.comfy_quant"
            ],
            {
                "dtype": "U8",
                "shape": [29],
                "data_offsets": [14, 43],
            },
        )

        self.assertEqual(
            output_header[
                "model.diffusion_model.blocks.1.attn.wq.bias"
            ],
            {
                "dtype": "BF16",
                "shape": [2],
                "data_offsets": [43, 47],
            },
        )

    def test_plans_a_convrot_w4a4_weight(self) -> None:
        header: dict[str, TensorDescriptor] = {
            "blocks.0.attn.wq.weight": {
                "dtype": "BF16",
                "shape": [2, 256],
                "data_offsets": [0, 1024],
            },
        }
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

        entries = build_plan(header, profile)
        layout = build_output_layout(entries)

        self.assertEqual(entries[0]["action"], "convrot_w4a4")
        self.assertEqual(
            [tensor.spec.shape for tensor in layout.tensors],
            [(2, 128), (2,), (CONVROT_W4A4_MARKER_BYTE_COUNT,)],
        )
        self.assertEqual(
            [tensor.data_offsets for tensor in layout.tensors],
            [
                (0, 256),
                (256, 264),
                (
                    264,
                    264 + CONVROT_W4A4_MARKER_BYTE_COUNT,
                ),
            ],
        )

    def test_plans_an_int8_convrot_weight(self) -> None:
        header: dict[str, TensorDescriptor] = {
            "blocks.0.attn.wq.weight": {
                "dtype": "BF16",
                "shape": [2, 256],
                "data_offsets": [0, 1024],
            },
        }
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

        entries = build_plan(header, profile)
        layout = build_output_layout(entries)

        self.assertEqual(entries[0]["action"], "int8_convrot")
        self.assertEqual(
            [tensor.spec.shape for tensor in layout.tensors],
            [(2, 256), (2, 1), (INT8_CONVROT_MARKER_BYTE_COUNT,)],
        )
        self.assertEqual(
            [tensor.data_offsets for tensor in layout.tensors],
            [
                (0, 512),
                (512, 520),
                (
                    520,
                    520 + INT8_CONVROT_MARKER_BYTE_COUNT,
                ),
            ],
        )

    def test_plans_all_quantization_methods_for_float16_weights(self) -> None:
        actions = (
            ("int8", "int8"),
            ("int6_rowwise", "int6_rowwise"),
            ("int8_convrot", "int8_convrot"),
            ("convrot_w4a4", "convrot_w4a4"),
        )

        for action, expected_action in actions:
            with self.subTest(action=action):
                header: dict[str, TensorDescriptor] = {
                    "blocks.0.attn.wq.weight": {
                        "dtype": "F16",
                        "shape": [2, 256],
                        "data_offsets": [0, 1024],
                    },
                }
                profile: QuantizationProfile = {
                    "default": "keep",
                    "rules": (
                        {
                            "action": action,
                            "prefix": "blocks.",
                            "suffixes": (".attn.wq.weight",),
                        },
                    ),
                }

                entries = build_plan(header, profile)

                self.assertEqual(entries[0]["action"], expected_action)

    def test_plans_int6_rowwise_float32_weights(self) -> None:
        header: dict[str, TensorDescriptor] = {
            "blocks.0.attn.wq.weight": {
                "dtype": "F32",
                "shape": [2, 4],
                "data_offsets": [0, 32],
            },
        }
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

        entries = build_plan(header, profile)
        layout = build_output_layout(entries)

        self.assertEqual(entries[0]["action"], "int6_rowwise")
        self.assertEqual(
            [tensor.spec.shape for tensor in layout.tensors],
            [(2, 3), (2, 1), (INT6_ROWWISE_MARKER_BYTE_COUNT,)],
        )

    def test_plans_int6_convrot_weights(self) -> None:
        header: dict[str, TensorDescriptor] = {
            "blocks.0.attn.wq.weight": {
                "dtype": "F32",
                "shape": [2, 256],
                "data_offsets": [0, 2048],
            },
        }
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

        entries = build_plan(header, profile)
        layout = build_output_layout(entries)

        self.assertEqual(entries[0]["action"], "int6_convrot")
        self.assertEqual(
            [tensor.spec.shape for tensor in layout.tensors],
            [(2, 192), (2, 1), (INT6_CONVROT_MARKER_BYTE_COUNT,)],
        )

    def test_keeps_int6_rowwise_weights_with_ineligible_width(self) -> None:
        header: dict[str, TensorDescriptor] = {
            "blocks.0.attn.wq.weight": {
                "dtype": "F32",
                "shape": [2, 3],
                "data_offsets": [0, 24],
            },
        }
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

        entries = build_plan(header, profile)

        self.assertEqual(entries[0]["action"], "keep")
        self.assertIn("divisible by 4", entries[0]["reason"])

    def test_keeps_int6_convrot_weights_with_ineligible_width(self) -> None:
        header: dict[str, TensorDescriptor] = {
            "blocks.0.attn.wq.weight": {
                "dtype": "F32",
                "shape": [2, 4],
                "data_offsets": [0, 32],
            },
        }
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

        entries = build_plan(header, profile)

        self.assertEqual(entries[0]["action"], "keep")
        self.assertIn("divisible by 256", entries[0]["reason"])

class TestInputBatchPlanning(unittest.TestCase):
    def test_keeps_output_batches_and_reads_each_batch_by_source_offset(self) -> None:
        entries = [
            {
                "tensor_name": "late",
                "source_dtype": "BF16",
                "shape": (3,),
                "input_bytes": 6,
                "action": "keep",
                "estimated_bytes": 6,
                "output_tensors": (),
                "source_data_offsets": (4, 10),
            },
            {
                "tensor_name": "middle",
                "source_dtype": "BF16",
                "shape": (1,),
                "input_bytes": 2,
                "action": "keep",
                "estimated_bytes": 2,
                "output_tensors": (),
                "source_data_offsets": (10, 12),
            },
            {
                "tensor_name": "early",
                "source_dtype": "BF16",
                "shape": (2,),
                "input_bytes": 4,
                "action": "keep",
                "estimated_bytes": 4,
                "output_tensors": (),
                "source_data_offsets": (0, 4),
            },
        ]

        batches = plan_input_batches(entries, 7)

        self.assertEqual(
            [[tensor.name for tensor in batch.tensors] for batch in batches],
            [["late"], ["early", "middle"]],
        )
        self.assertEqual(
            [batch.total_input_bytes for batch in batches],
            [6, 6],
        )
        self.assertFalse(batches[0].oversized)

    def test_plans_oversized_tensor_alone(self) -> None:
        entry = {
            "tensor_name": "large",
            "source_dtype": "BF16",
            "shape": (6,),
            "input_bytes": 12,
            "action": "keep",
            "estimated_bytes": 12,
            "output_tensors": (),
            "source_data_offsets": (0, 12),
        }

        batches = plan_input_batches([entry], 8)

        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].tensors[0].nbytes, 12)
        self.assertTrue(batches[0].oversized)

    def test_rejects_non_positive_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            plan_input_batches([], 0)
