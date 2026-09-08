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
    def test_plans_deterministic_ids_filenames_and_physical_order(self) -> None:
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
                "action": "int8",
                "layers": ["layer/a.weight", "layer_b.weight"],
            }
        )

        plan = plan_patch_sweep(source_header, profile)

        self.assertEqual(
            [(entry.layer, entry.filename) for entry in plan.entries],
            [
                ("layer/a.weight", "layer_a__int8.safetensors"),
                ("layer_b.weight", "layer_b__int8.safetensors"),
            ],
        )
        self.assertEqual(
            [entry.patch_id for entry in plan.entries],
            [
                "sweep-v1__layer/a__int8",
                "sweep-v1__layer_b__int8",
            ],
        )

    def test_rejects_sanitized_filename_collision(self) -> None:
        source_header: dict[str, TensorDescriptor] = {
            "layer/a.weight": {
                "dtype": "BF16",
                "shape": [1, 4],
                "data_offsets": [0, 8],
            },
            "layer_a.weight": {
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
                        "action": "int8",
                        "layers": ["layer/a.weight", "layer_a.weight"],
                    }
                ),
            )

    def test_generates_one_standalone_patch_per_layer_and_manifest(self) -> None:
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
            manifest = json.loads(
                (output_dir / "sweep_manifest.json").read_text(encoding="utf-8")
            )
            self.assertIn("generated_patch_count: 2", cli_result.output)
            self.assertEqual(
                [patch["layer"] for patch in manifest["patches"]],
                ["A.weight", "B.weight"],
            )
            self.assertEqual(manifest["profile_id"], "precision-v1")
            self.assertEqual(manifest["source"], "source.safetensors")
            for entry in manifest["patches"]:
                patch_path = output_dir / entry["file"]
                header = read_source_model_header(patch_path)
                tensors = load_file(str(patch_path))
                self.assertEqual(
                    list(header.tensors),
                    [
                        entry["family"] + ".weight",
                        entry["family"] + ".weight_scale",
                        entry["family"] + ".comfy_quant",
                    ],
                )
                self.assertEqual(set(tensors), set(header.tensors))
                self.assertEqual(
                    header.metadata["potatoforge_file_type"],
                    "quant_patch",
                )
                self.assertEqual(
                    json.loads(header.metadata["potatoforge_patch_replaces"]),
                    [entry["family"]],
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
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            patch_tensors = load_file(
                str(output_dir / "B__convrot_w4a4_mse.safetensors")
            )

        self.assertEqual(manifest["patches"][0]["action"], "convrot_w4a4_mse")
        marker = bytes(patch_tensors["B.comfy_quant"].tolist()).decode("utf-8")
        self.assertEqual(json.loads(marker), CONVROT_W4A4_MARKER)

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

    def test_preserves_completed_patches_when_a_later_layer_fails(self) -> None:
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
                    "action": "int8",
                    "layers": ["A.weight", "B.weight"],
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

            self.assertTrue((output_dir / "A__int8.safetensors").exists())
            self.assertFalse((output_dir / "B__int8.safetensors").exists())
            self.assertFalse((output_dir / "sweep_manifest.json").exists())

    def test_processes_source_ranges_by_offset_but_keeps_manifest_profile_order(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.safetensors"
            output_dir = root / "patches"
            save_file(
                {
                    "Z.weight": torch.tensor(
                        [[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16
                    ),
                    "A.weight": torch.tensor(
                        [[4.0, 3.0, 2.0, 1.0]], dtype=torch.bfloat16
                    ),
                },
                str(source),
            )
            source_header = read_source_model_header(source)
            physical_order = [
                name
                for name, _ in sorted(
                    source_header.tensors.items(),
                    key=lambda item: item[1]["data_offsets"][0],
                )
            ]
            profile = sweep_profile(
                {"action": "int8", "layers": list(reversed(physical_order))}
            )
            calls: list[str] = []

            def fake_generate(
                _source: Path,
                output_path: Path,
                plan: PatchPlan,
                **_kwargs: object,
            ) -> None:
                layer = plan.entries[0].source_tensor_name
                calls.append(layer)
                output_path.write_bytes(b"patch")

            with patch(
                "potatoforge.patch_sweep.execute_patch_plan",
                side_effect=fake_generate,
            ):
                result = generate_patch_sweep(source, output_dir, profile)

            manifest = json.loads(
                result.manifest_path.read_text(encoding="utf-8")
            )

        self.assertEqual(calls, physical_order)
        self.assertEqual(
            [entry["layer"] for entry in manifest["patches"]],
            list(reversed(physical_order)),
        )


if __name__ == "__main__":
    unittest.main()
