import json
import unittest
from pathlib import Path

from potatoforge.patch_planning import build_patch_metadata, build_patch_plan
from potatoforge.planning import TensorDescriptor, build_quantized_tensor_plan


class TestPatchPlanning(unittest.TestCase):
    def setUp(self) -> None:
        self.source_header: dict[str, TensorDescriptor] = {
            "A.weight": {
                "dtype": "BF16",
                "shape": [1, 4],
                "data_offsets": [16, 24],
            },
            "B.weight": {
                "dtype": "BF16",
                "shape": [1, 4],
                "data_offsets": [0, 8],
            },
            "C.weight": {
                "dtype": "BF16",
                "shape": [1, 4],
                "data_offsets": [8, 16],
            },
        }

    def test_plans_one_selected_family(self) -> None:
        plan = build_patch_plan(
            self.source_header,
            "B.weight",
            "int8",
            "test-patch",
        )

        self.assertEqual(
            [entry.source_tensor_name for entry in plan.entries],
            ["B.weight"],
        )
        self.assertEqual(
            [tensor.spec.name for tensor in plan.layout.tensors],
            [
                "B.weight",
                "B.weight_scale",
                "B.comfy_quant",
            ],
        )
        self.assertEqual(plan.selected_tensor_count, 1)
        self.assertEqual(plan.generated_tensor_count, 3)
        self.assertEqual(plan.source_bytes_to_read, 8)

    def test_plans_every_weight_with_a_prefix(self) -> None:
        source_header = {
            **self.source_header,
            "blocks.0.weight": {
                "dtype": "BF16",
                "shape": [1, 4],
                "data_offsets": [24, 32],
            },
            "blocks.1.weight": {
                "dtype": "BF16",
                "shape": [1, 4],
                "data_offsets": [32, 40],
            },
        }

        plan = build_patch_plan(
            source_header,
            "blocks.*",
            "int8",
            "blocks-patch",
        )

        self.assertEqual(
            [entry.source_tensor_name for entry in plan.entries],
            ["blocks.0.weight", "blocks.1.weight"],
        )
        self.assertEqual(plan.selected_tensor_count, 2)

    def test_plans_a_fused_qkv_family(self) -> None:
        source_header = {
            **self.source_header,
            "C.attn.in_proj_weight": {
                "dtype": "BF16",
                "shape": [3, 256],
                "data_offsets": [28, 28 + 3 * 256 * 2],
            },
        }

        plan = build_patch_plan(
            source_header,
            "C.attn.in_proj_weight",
            "int8_convrot",
            "fused-qkv",
        )

        self.assertEqual(plan.entries[0].logical_layer_name, "C.attn.in_proj")
        self.assertEqual(
            [tensor.spec.name for tensor in plan.layout.tensors],
            [
                "C.attn.in_proj_weight",
                "C.attn.in_proj.weight_scale",
                "C.attn.in_proj.comfy_quant",
            ],
        )

    def test_reuses_normal_format_planner(self) -> None:
        plan = build_patch_plan(
            self.source_header,
            "B.weight",
            "int8",
            "test-patch",
        )

        self.assertEqual(
            plan.entries[0].output_tensors,
            build_quantized_tensor_plan(
                "int8",
                "B.weight",
                self.source_header["B.weight"],
            ).output_tensors,
        )

    def test_builds_deterministic_patch_ownership_metadata(self) -> None:
        plan = build_patch_plan(
            self.source_header,
            "B.weight",
            "int8",
            "test-patch",
        )

        self.assertEqual(
            build_patch_metadata(plan, Path("source.safetensors")),
            {
                "potatoforge_file_type": "quant_patch",
                "potatoforge_patch_format": "1",
                "potatoforge_patch_id": "test-patch",
                "potatoforge_patch_replaces": json.dumps(
                    ["B"], separators=(",", ":")
                ),
                "potatoforge_patch_source": "source.safetensors",
            },
        )

    def test_rejects_missing_and_noncanonical_layers(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not exist"):
            build_patch_plan(
                self.source_header,
                "missing.weight",
                "int8",
                "test-patch",
            )
        with self.assertRaisesRegex(ValueError, "canonical"):
            build_patch_plan(
                {
                    **self.source_header,
                    "B.weight_scale": {
                        "dtype": "F32",
                        "shape": [1, 1],
                        "data_offsets": [24, 28],
                    },
                },
                "B.weight_scale",
                "int8",
                "test-patch",
            )

    def test_reports_target_format_validation_with_tensor_name(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Cannot patch B.weight as int8_convrot.*divisible by 256",
        ):
            build_patch_plan(
                self.source_header,
                "B.weight",
                "int8_convrot",
                "test-patch",
            )


if __name__ == "__main__":
    unittest.main()
