import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file
from typer.testing import CliRunner

from potatoforge.cli import app
from potatoforge.headers.source_header import read_source_model_header
from potatoforge.patch_planning import PatchPlan
from potatoforge.patch_sweep import (
    generate_patch_sweep,
    generate_patch_sweep_from_profile,
    plan_patch_sweep,
)
from potatoforge.planning import CONVROT_W4A4_MARKER, TensorDescriptor
from potatoforge.source_payloads import read_source_tensor_bytes
from potatoforge.sweep_profiles import SweepProfile


def sweep_profile(*groups: dict[str, object]) -> SweepProfile:
    return {
        "profile_id": "sweep-v1",
        "groups": tuple(groups),  # type: ignore[typeddict-item]
    }


class TestPatchSweep(unittest.TestCase):
    def test_plans_group_filename_and_physical_layer_order(self) -> None:
        source_header: dict[str, TensorDescriptor] = {
            "layer/a.weight": {
                "dtype": "BF16",
                "shape": [1, 4],
                "data_offsets": [8, 16],
            },
            "layer_b.weight": {
                "dtype": "BF16",
                "shape": [1, 4],
                "data_offsets": [0, 8],
            },
        }
        profile = sweep_profile(
            {
                "id": "layers",
                "action": "int8",
                "layers": ["layer/a.weight", "layer_b.weight"],
            }
        )

        plan = plan_patch_sweep(source_header, profile)

        self.assertEqual(len(plan.entries), 1)
        entry = plan.entries[0]
        self.assertEqual(entry.group_id, "layers")
        self.assertEqual(entry.filename, "layers-int8.safetensors")
        self.assertEqual(entry.patch_id, "layers-int8")
        self.assertEqual(
            entry.layers,
            ("layer/a.weight", "layer_b.weight"),
        )
        self.assertEqual(
            [item.source_tensor_name for item in entry.plan.entries],
            ["layer_b.weight", "layer/a.weight"],
        )

    def test_rejects_case_insensitive_group_filename_collision(self) -> None:
        source_header: dict[str, TensorDescriptor] = {
            "A.weight": {
                "dtype": "BF16",
                "shape": [1, 4],
                "data_offsets": [0, 8],
            },
            "B.weight": {
                "dtype": "BF16",
                "shape": [1, 4],
                "data_offsets": [8, 16],
            },
        }
        with self.assertRaisesRegex(ValueError, "duplicate output filename"):
            plan_patch_sweep(
                source_header,
                sweep_profile(
                    {
                        "id": "shared",
                        "action": "int8",
                        "layers": ["A.weight"],
                    },
                    {
                        "id": "SHARED",
                        "action": "int8",
                        "layers": ["B.weight"],
                    }
                ),
            )

    def test_rejects_duplicate_layer_within_one_group(self) -> None:
        source_header: dict[str, TensorDescriptor] = {
            "A.weight": {
                "dtype": "BF16",
                "shape": [1, 4],
                "data_offsets": [0, 8],
            },
        }

        with self.assertRaisesRegex(ValueError, "more than once"):
            plan_patch_sweep(
                source_header,
                sweep_profile(
                    {
                        "id": "duplicate",
                        "action": "int8",
                        "layers": ["A.weight", "A.weight"],
                    }
                ),
            )

    def test_rejects_empty_group_at_planning_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one source layer"):
            plan_patch_sweep(
                {},
                sweep_profile(
                    {
                        "id": "empty",
                        "action": "int8",
                        "layers": [],
                    }
                ),
            )

    def test_generates_one_patch_per_group_without_a_sweep_manifest(self) -> None:
        tensors = {
            "A.weight": torch.tensor([[1.0, -2.0, 0.5, 0.25]], dtype=torch.bfloat16),
            "B.weight": torch.arange(256, dtype=torch.float32).reshape(1, 256).to(torch.bfloat16),
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.safetensors"
            profile_path = root / "sweep.json"
            output_dir = root / "patches"
            save_file(tensors, str(source))
            profile_path.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "profile_id": "precision-v1",
                        "groups": [
                            {"action": "int8", "layers": ["A.weight"]},
                            {"action": "int6_convrot", "layers": ["B.weight"]},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            cli_result = CliRunner().invoke(
                app,
                [
                    "patch-sweep",
                    "--source",
                    str(source),
                    "--profile",
                    str(profile_path),
                    "--output-dir",
                    str(output_dir),
                ],
            )
            self.assertEqual(cli_result.exit_code, 0, cli_result.output)
            self.assertIn("generated_patch_count: 2", cli_result.output)
            self.assertFalse((output_dir / "sweep_manifest.json").exists())
            for filename, family in (
                ("group-1-int8.safetensors", "A"),
                ("group-2-int6_convrot.safetensors", "B"),
            ):
                patch_path = output_dir / filename
                header = read_source_model_header(patch_path)
                tensors = load_file(str(patch_path))
                self.assertEqual(
                    list(header.tensors),
                    [
                        family + ".weight",
                        family + ".weight_scale",
                        family + ".comfy_quant",
                    ],
                )
                self.assertEqual(set(tensors), set(header.tensors))
                self.assertEqual(
                    header.metadata["potatoforge_file_type"],
                    "quant_patch",
                )
                self.assertEqual(
                    header.metadata["potatoforge_patch_format"],
                    "1",
                )
                self.assertEqual(
                    json.loads(header.metadata["potatoforge_patch_replaces"]),
                    [family],
                )

    def test_generates_a_standard_w4a4_patch_for_mse_scale_action(self) -> None:
        weight = torch.zeros((1, 256), dtype=torch.bfloat16)
        weight[0, 0] = 3.0
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.safetensors"
            profile = root / "sweep.json"
            output_dir = root / "patches"
            save_file({"B.weight": weight}, str(source))
            profile.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "profile_id": "w4a4-mse",
                        "groups": [
                            {
                                "action": "convrot_w4a4_mse",
                                "layers": ["B.weight"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = generate_patch_sweep_from_profile(
                source,
                profile,
                output_dir,
            )
            patch_tensors = load_file(
                str(output_dir / "group-1-convrot_w4a4_mse.safetensors")
            )

        self.assertEqual(result.generated_patch_count, 1)
        marker = bytes(patch_tensors["B.comfy_quant"].tolist()).decode("utf-8")
        self.assertEqual(json.loads(marker), CONVROT_W4A4_MARKER)

    def test_generates_one_explicitly_named_multi_layer_patch(self) -> None:
        tensors = {
            "A.weight": torch.tensor([[1.0, -2.0, 0.5, 0.25]], dtype=torch.bfloat16),
            "B.weight": torch.tensor([[2.0, -1.0, 0.25, 0.5]], dtype=torch.bfloat16),
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.safetensors"
            profile = root / "sweep.json"
            output_dir = root / "patches"
            save_file(tensors, str(source))
            profile.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "profile_id": "grouped-v1",
                        "groups": [
                            {
                                "id": "shared",
                                "action": "int8",
                                "layers": ["A.weight", "B.weight"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = generate_patch_sweep_from_profile(source, profile, output_dir)

            patch_path = output_dir / "shared-int8.safetensors"
            self.assertTrue(patch_path.exists())
            self.assertFalse((output_dir / "sweep_manifest.json").exists())
            self.assertEqual(result.generated_patch_count, 1)
            header = read_source_model_header(patch_path)
            self.assertEqual(
                set(header.tensors),
                {
                    "A.weight",
                    "A.weight_scale",
                    "A.comfy_quant",
                    "B.weight",
                    "B.weight_scale",
                    "B.comfy_quant",
                },
            )
            self.assertEqual(
                header.metadata["potatoforge_patch_id"],
                "shared-int8",
            )
            self.assertEqual(
                json.loads(header.metadata["potatoforge_patch_replaces"]),
                ["A", "B"],
            )

    def test_reads_only_selected_source_layers_and_preflights_invalid_profile(self) -> None:
        tensors = {
            "A.weight": torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16),
            "B.weight": torch.tensor([[4.0, 3.0, 2.0, 1.0]], dtype=torch.bfloat16),
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.safetensors"
            valid_profile = root / "valid.json"
            invalid_profile = root / "invalid.json"
            output_dir = root / "patches"
            save_file(tensors, str(source))
            source_header = read_source_model_header(source)
            valid_profile.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "profile_id": "valid",
                        "groups": [{"action": "int8", "layers": ["B.weight"]}],
                    }
                ),
                encoding="utf-8",
            )
            invalid_profile.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "profile_id": "invalid",
                        "groups": [
                            {"action": "int8", "layers": ["B.weight", "missing.weight"]}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "potatoforge.patching.read_source_tensor_bytes",
                wraps=read_source_tensor_bytes,
            ) as source_reader:
                generate_patch_sweep_from_profile(source, valid_profile, output_dir)
            self.assertEqual(source_reader.call_count, 1)
            self.assertEqual(
                source_reader.call_args.args[2],
                tuple(source_header.tensors["B.weight"]["data_offsets"]),
            )

            with self.assertRaisesRegex(ValueError, "does not exist"):
                generate_patch_sweep_from_profile(
                    source,
                    invalid_profile,
                    root / "invalid-patches",
                )
            self.assertFalse((root / "invalid-patches").exists())

    def test_preserves_completed_groups_when_a_later_group_fails(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.safetensors"
            output_dir = root / "patches"
            save_file(
                {
                    "A.weight": torch.tensor(
                        [[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16
                    ),
                    "B.weight": torch.tensor(
                        [[4.0, 3.0, 2.0, 1.0]], dtype=torch.bfloat16
                    ),
                },
                str(source),
            )
            profile = sweep_profile(
                {
                    "id": "first",
                    "action": "int8",
                    "layers": ["A.weight"],
                },
                {
                    "id": "second",
                    "action": "int8",
                    "layers": ["B.weight"],
                }
            )

            def fail_on_b(
                _source: Path,
                output_path: Path,
                plan: PatchPlan,
                **_kwargs: object,
            ) -> None:
                layer = plan.entries[0].source_tensor_name
                if layer == "B.weight":
                    raise RuntimeError("quantizer failed")
                output_path.write_bytes(b"completed")

            with patch(
                "potatoforge.patch_sweep.execute_patch_plan",
                side_effect=fail_on_b,
            ):
                with self.assertRaisesRegex(ValueError, "B.weight"):
                    generate_patch_sweep(source, output_dir, profile)

            self.assertTrue((output_dir / "first-int8.safetensors").exists())
            self.assertFalse((output_dir / "second-int8.safetensors").exists())
            self.assertFalse((output_dir / "sweep_manifest.json").exists())

    def test_allows_overlapping_layers_in_distinct_groups(self) -> None:
        source_header: dict[str, TensorDescriptor] = {
            "A.weight": {
                "dtype": "BF16",
                "shape": [1, 4],
                "data_offsets": [0, 8],
            },
            "B.weight": {
                "dtype": "BF16",
                "shape": [1, 4],
                "data_offsets": [8, 16],
            },
        }

        plan = plan_patch_sweep(
            source_header,
            sweep_profile(
                {
                    "id": "wide",
                    "action": "int8",
                    "layers": ["A.weight", "B.weight"],
                },
                {
                    "id": "wide",
                    "action": "int6_rowwise",
                    "layers": ["B.weight"],
                },
            ),
        )

        self.assertEqual(
            [(entry.filename, entry.layers) for entry in plan.entries],
            [
                ("wide-int8.safetensors", ("A.weight", "B.weight")),
                ("wide-int6_rowwise.safetensors", ("B.weight",)),
            ],
        )


if __name__ == "__main__":
    unittest.main()
