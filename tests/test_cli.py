import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import call, patch

from typer.testing import CliRunner

from potatoforge.audits.profile_optimizer import OptimizedProfile
from potatoforge.cli import app
from potatoforge.converter import ResolvedIOMode
from potatoforge.lora.lora_merge import AdapterMergeInput
from potatoforge.config import ConfigPaths, OptimizeConfig, QuantizeConfig


class TestCli(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_help_lists_direct_commands(self) -> None:
        result = self.runner.invoke(app, ["--help"])

        self.assertEqual(result.exit_code, 0, result.stdout)
        for command in (
            "inspect-header",
            "inspect-lora",
            "merge-lora",
            "audit",
            "analyze",
            "optimize",
            "quantize",
            "patch-sweep",
            "extract",
            "test",
        ):
            self.assertIn(command, result.stdout)

    def test_version_is_available(self) -> None:
        result = self.runner.invoke(app, ["--version"])

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout.strip(), "0.1.0")

    def test_int6_runtime_toggle_is_not_a_cli_option(self) -> None:
        for command in ("analyze", "optimize"):
            with self.subTest(command=command):
                result = self.runner.invoke(app, [command, "--help"])

                self.assertEqual(result.exit_code, 0, result.stdout)
                self.assertNotIn("enable-potatoforge-int6-runtime", result.stdout)

    def test_inspect_header_displays_quantization_metadata(self) -> None:
        header = {
            "__metadata__": {
                "potatoforge.quantization": "mixed",
                "potatoforge.quantization_layers": json.dumps(
                    {
                        "blocks.1.mlp.weight": "int6_rowwise",
                        "blocks.0.attn.wq.weight": "int8",
                    }
                ),
            },
            "blocks.0.attn.wq.weight": {
                "dtype": "I8",
                "shape": [1, 4],
                "data_offsets": [0, 4],
            },
        }

        with patch(
            "potatoforge.cli.read_header_from_safetensors",
            return_value=header,
        ):
            result = self.runner.invoke(
                app,
                [
                    "inspect-header",
                    "model.safetensors",
                    "--quantization",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.stdout)
        self.assertIn("quantization: mixed", result.stdout)
        self.assertIn("int8: 1 layer", result.stdout)
        self.assertIn("int6_rowwise: 1 layer", result.stdout)
        self.assertIn(
            "blocks.0.attn.wq.weight: int8",
            result.stdout,
        )
        self.assertIn(
            "blocks.1.mlp.weight: int6_rowwise",
            result.stdout,
        )

    def test_inspect_header_reports_fused_qkv_as_one_quantized_layer(self) -> None:
        header = {
            "__metadata__": {
                "potatoforge.quantization": "int8_convrot",
                "potatoforge.quantization_layers": json.dumps(
                    {"foo.attn.in_proj_weight": "int8_convrot"}
                ),
            },
            "foo.attn.in_proj_weight": {
                "dtype": "I8",
                "shape": [3, 256],
                "data_offsets": [0, 768],
            },
            "foo.attn.in_proj.weight_scale": {
                "dtype": "F32",
                "shape": [3, 1],
                "data_offsets": [768, 780],
            },
            "foo.attn.in_proj.comfy_quant": {
                "dtype": "U8",
                "shape": [64],
                "data_offsets": [780, 844],
            },
        }

        with patch(
            "potatoforge.cli.read_header_from_safetensors",
            return_value=header,
        ):
            result = self.runner.invoke(
                app,
                ["inspect-header", "model.safetensors", "--quantization"],
            )

        self.assertEqual(result.exit_code, 0, result.stdout)
        self.assertIn("quantized_layer_count: 1", result.stdout)
        self.assertIn("foo.attn.in_proj_weight: int8_convrot", result.stdout)
        self.assertNotIn("q_proj", result.stdout)
        self.assertNotIn("k_proj", result.stdout)
        self.assertNotIn("v_proj", result.stdout)

    def test_optimize_config_applies_a_typed_target_override(self) -> None:
        config = OptimizeConfig(
            paths=ConfigPaths(
                audit_report=Path("audit.json"),
                profile=Path("profile.json"),
            ),
            profile_id="zit",
            target_size_gib=4.3,
            methods=("int8",),
            max_relative_l2_error=None,
            exclude_prefixes=(),
            exclude_suffixes=(),
            overwrite=False,
        )
        optimized = OptimizedProfile(
            profile={"profile_id": "zit", "default": "keep", "rules": ()},
            output_bytes=123,
            target_bytes=4 * 1024**3,
            reconstruction_sse=0.0,
        )

        with (
            patch("potatoforge.cli.load_optimize_config", return_value=config),
            patch("potatoforge.cli.load_weight_audit", return_value={}),
            patch("potatoforge.cli.audited_global_relative_l2", return_value=0.0),
            patch("potatoforge.cli.optimize_target_size", return_value=optimized) as optimize_mock,
        ):
            result = self.runner.invoke(
                app,
                [
                    "optimize",
                    "--config",
                    "config.toml",
                    "--target-size-gib",
                    "4.0",
                    "--dry-run",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.stdout)
        self.assertEqual(optimize_mock.call_args.args[2], 4 * 1024**3)

    def test_analyze_single_tensor_does_not_write_a_workbook(self) -> None:
        with (
            patch("potatoforge.cli.load_weight_audit", return_value={}),
            patch(
                "potatoforge.cli.validate_weight_audit_against_source",
                return_value=object(),
            ),
            patch("potatoforge.cli.print_tensor_analysis") as print_mock,
            patch("potatoforge.cli.write_analysis_workbook") as write_mock,
        ):
            result = self.runner.invoke(
                app,
                [
                    "analyze",
                    "--audit",
                    "audit.json",
                    "--tensor",
                    "blocks.0.attn.wq.weight",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        print_mock.assert_called_once()
        write_mock.assert_not_called()

    def test_analyze_prints_comma_separated_tensors(self) -> None:
        with (
            patch("potatoforge.cli.load_weight_audit", return_value={}),
            patch(
                "potatoforge.cli.validate_weight_audit_against_source",
                return_value=object(),
            ),
            patch("potatoforge.cli.print_tensor_analysis") as print_mock,
            patch("potatoforge.cli.write_analysis_workbook") as write_mock,
        ):
            result = self.runner.invoke(
                app,
                [
                    "analyze",
                    "--audit",
                    "audit.json",
                    "--tensors",
                    "blocks.0.attn.wq.weight, blocks.1.attn.wq.weight",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            [call.args[2] for call in print_mock.call_args_list],
            [
                "blocks.0.attn.wq.weight",
                "blocks.1.attn.wq.weight",
            ],
        )
        write_mock.assert_not_called()

    def test_analyze_can_audit_one_tensor_from_source(self) -> None:
        audit_document = object()
        source_header = object()

        with (
            patch(
                "potatoforge.cli.audit_bf16_source",
                return_value=audit_document,
            ) as audit_mock,
            patch(
                "potatoforge.cli.read_source_model_header",
                return_value=source_header,
            ) as header_mock,
            patch("potatoforge.cli.print_tensor_analysis") as print_mock,
        ):
            result = self.runner.invoke(
                app,
                [
                    "analyze",
                    "--source",
                    "model.safetensors",
                    "--tensor",
                    "blocks.0.attn.wq.weight",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            audit_mock.call_args.args[0],
            Path("model.safetensors"),
        )
        self.assertEqual(
            audit_mock.call_args.kwargs["tensor_name"],
            "blocks.0.attn.wq.weight",
        )
        header_mock.assert_called_once_with(Path("model.safetensors"))
        print_mock.assert_called_once_with(
            audit_document,
            source_header,
            "blocks.0.attn.wq.weight",
        )

    def test_analyze_can_audit_comma_separated_tensors_from_source(self) -> None:
        audit_document = object()
        source_header = object()
        tensor_names = (
            "blocks.0.attn.wq.weight",
            "blocks.1.attn.wq.weight",
        )

        with (
            patch(
                "potatoforge.cli.audit_bf16_source",
                return_value=audit_document,
            ) as audit_mock,
            patch(
                "potatoforge.cli.read_source_model_header",
                return_value=source_header,
            ),
            patch("potatoforge.cli.print_tensor_analysis") as print_mock,
        ):
            result = self.runner.invoke(
                app,
                [
                    "analyze",
                    "--source",
                    "model.safetensors",
                    "--tensors",
                    "blocks.0.attn.wq.weight, blocks.1.attn.wq.weight",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            audit_mock.call_args.kwargs["tensor_names"],
            tensor_names,
        )
        self.assertEqual(
            [call.args[2] for call in print_mock.call_args_list],
            list(tensor_names),
        )

    def test_analyze_source_requires_a_tensor(self) -> None:
        result = self.runner.invoke(
            app,
            ["analyze", "--source", "model.safetensors"],
        )

        self.assertEqual(result.exit_code, 3, result.output)
        self.assertIn("--source requires --tensor", result.output)

    def test_analyze_rejects_audit_and_source_together(self) -> None:
        result = self.runner.invoke(
            app,
            [
                "analyze",
                "--audit",
                "audit.json",
                "--source",
                "model.safetensors",
                "--tensor",
                "blocks.0.attn.wq.weight",
            ],
        )

        self.assertEqual(result.exit_code, 3, result.output)
        self.assertIn("--audit and --source cannot be combined", result.output)

    def test_analyze_reuses_gib_target_conversion_for_profile_sweep(self) -> None:
        optimized = OptimizedProfile(
            profile={"profile_id": "analysis", "default": "keep", "rules": ()},
            output_bytes=123,
            target_bytes=4 * 1024**3,
            reconstruction_sse=0.0,
        )

        with (
            patch("potatoforge.cli.load_weight_audit", return_value={}),
            patch(
                "potatoforge.cli.validate_weight_audit_against_source",
                return_value=object(),
            ),
            patch(
                "potatoforge.cli.generate_profile_sweep",
                return_value=(optimized,),
            ) as sweep_mock,
            patch("potatoforge.cli.write_analysis_workbook") as write_mock,
        ):
            result = self.runner.invoke(
                app,
                [
                    "analyze",
                    "audit.json",
                    "--target-size-gib",
                    "4.0",
                    "--method",
                    "int8, int6",
                    "--exclude-prefix",
                    "blocks.0., blocks.1.",
                    "--output",
                    "analysis.xlsx",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(sweep_mock.call_args.args[2], 0.5 * 1024**3)
        self.assertEqual(
            sweep_mock.call_args.kwargs["selected_target_bytes"],
            4 * 1024**3,
        )
        self.assertEqual(
            sweep_mock.call_args.kwargs["excluded_prefixes"],
            ("blocks.0.", "blocks.1."),
        )
        self.assertEqual(
            sweep_mock.call_args.args[3],
            frozenset(("int8", "int6")),
        )
        write_mock.assert_called_once()

    def test_optimize_splits_comma_separated_exclusions(self) -> None:
        config = OptimizeConfig(
            paths=ConfigPaths(
                audit_report=Path("audit.json"),
                profile=Path("profile.json"),
            ),
            profile_id="zit",
            target_size_gib=4.3,
            methods=("int8",),
            max_relative_l2_error=None,
            exclude_prefixes=(),
            exclude_suffixes=(),
            overwrite=False,
        )
        optimized = OptimizedProfile(
            profile={"profile_id": "zit", "default": "keep", "rules": ()},
            output_bytes=123,
            target_bytes=4 * 1024**3,
            reconstruction_sse=0.0,
        )

        with (
            patch("potatoforge.cli.load_optimize_config", return_value=config),
            patch("potatoforge.cli.load_weight_audit", return_value={}),
            patch("potatoforge.cli.audited_global_relative_l2", return_value=0.0),
            patch(
                "potatoforge.cli.optimize_target_size",
                return_value=optimized,
            ) as optimize_mock,
        ):
            result = self.runner.invoke(
                app,
                [
                    "optimize",
                    "--config",
                    "config.toml",
                    "--method",
                    "int8, int6",
                    "--exclude-prefix",
                    "first., second.",
                    "--exclude-suffix",
                    ".bias, .scale",
                    "--dry-run",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(optimize_mock.call_args.args[5], ("first.", "second."))
        self.assertEqual(optimize_mock.call_args.args[6], (".bias", ".scale"))
        self.assertEqual(
            optimize_mock.call_args.args[3],
            frozenset(("int8", "int6")),
        )

    def test_analyze_applies_exclusions_to_the_default_profile_sweep(self) -> None:
        optimized = OptimizedProfile(
            profile={"profile_id": "analysis", "default": "keep", "rules": ()},
            output_bytes=123,
            target_bytes=123,
            reconstruction_sse=0.0,
        )
        with (
            patch("potatoforge.cli.load_weight_audit", return_value={}),
            patch(
                "potatoforge.cli.validate_weight_audit_against_source",
                return_value=object(),
            ),
            patch(
                "potatoforge.cli.generate_profile_sweep",
                return_value=(optimized,),
            ) as sweep_mock,
            patch("potatoforge.cli.write_analysis_workbook"),
        ):
            result = self.runner.invoke(
                app,
                [
                    "analyze",
                    "audit.json",
                    "--output",
                    "analysis.xlsx",
                    "--exclude-prefix",
                    "blocks.0.",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            sweep_mock.call_args.kwargs["excluded_prefixes"],
            ("blocks.0.",),
        )
        self.assertIsNone(sweep_mock.call_args.kwargs["selected_target_bytes"])

    def test_quantize_dry_run_reports_size_without_writing_output(self) -> None:
        with (
            patch(
                "potatoforge.cli.estimate_output_bytes",
                return_value=123,
            ) as estimate_mock,
            patch("potatoforge.cli.convert_model_from_profile") as convert_mock,
        ):
            result = self.runner.invoke(
                app,
                [
                    "quantize",
                    "source.safetensors",
                    "--profile",
                    "profile.json",
                    "--dry-run",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        estimate_mock.assert_called_once_with(
            Path("source.safetensors"),
            Path("profile.json"),
        )
        convert_mock.assert_not_called()
        self.assertIn("estimated_output_bytes: 123", result.output)

    def test_quantize_passes_batched_io_options(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output_path = root / "output.safetensors"
            output_path.write_bytes(b"")

            with patch(
                "potatoforge.cli.convert_model_from_profile"
            ) as convert_mock:
                convert_mock.return_value = ResolvedIOMode(
                    "batched",
                    "requested mode",
                )
                result = self.runner.invoke(
                    app,
                    [
                        "quantize",
                        str(root / "source.safetensors"),
                        str(output_path),
                        "--profile",
                        str(root / "profile.json"),
                        "--io-mode",
                        "batched",
                        "--input-buffer-gib",
                        "2",
                        "--adapter-path",
                        str(root / "first.safetensors"),
                        "--adapter-strength",
                        "0.75",
                        "--adapter-path",
                        str(root / "second.safetensors"),
                        "--adapter-strength",
                        "1.25",
                    ],
                )

        self.assertEqual(result.exit_code, 0, result.stdout)
        self.assertEqual(
            convert_mock.call_args.kwargs["io_mode"],
            "batched",
        )
        self.assertEqual(
            convert_mock.call_args.kwargs["input_buffer_bytes"],
            2 * 1024**3,
        )
        self.assertEqual(
            convert_mock.call_args.kwargs["adapters"],
            (
                AdapterMergeInput(root / "first.safetensors", 0.75),
                AdapterMergeInput(root / "second.safetensors", 1.25),
            ),
        )

    def test_quantize_loads_config_and_allows_io_override(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output_path = root / "output.safetensors"
            output_path.write_bytes(b"")
            config = QuantizeConfig(
                paths=ConfigPaths(
                    source=root / "source.safetensors",
                    profile=root / "profile.json",
                    quantized_output=output_path,
                ),
                io_mode="batched",
                input_buffer_gib=4.0,
            )

            with (
                patch(
                    "potatoforge.cli.load_quantize_config",
                    return_value=config,
                ),
                patch(
                    "potatoforge.cli.convert_model_from_profile"
                ) as convert_mock,
            ):
                convert_mock.return_value = ResolvedIOMode(
                    "batched",
                    "requested mode",
                )
                result = self.runner.invoke(
                    app,
                    [
                        "quantize",
                        "--config",
                    str(root / "config.toml"),
                        "--io-mode",
                        "batched",
                        "--adapter-path",
                        str(root / "adapter.safetensors"),
                        "--adapter-strength",
                        "0.5",
                    ],
                )

        self.assertEqual(result.exit_code, 0, result.stdout)
        self.assertEqual(
            convert_mock.call_args.args[:3],
            (
                root / "source.safetensors",
                output_path,
                root / "profile.json",
            ),
        )
        self.assertEqual(
            convert_mock.call_args.kwargs["io_mode"],
            "batched",
        )
        self.assertEqual(
            convert_mock.call_args.kwargs["input_buffer_bytes"],
            4 * 1024**3,
        )
        self.assertEqual(
            convert_mock.call_args.kwargs["adapters"],
            (AdapterMergeInput(root / "adapter.safetensors", 0.5),),
        )

    def test_quantize_uses_config_adapters_without_cli_overrides(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output_path = root / "output.safetensors"
            output_path.write_bytes(b"")
            config = QuantizeConfig(
                paths=ConfigPaths(
                    source=root / "source.safetensors",
                    profile=root / "profile.json",
                    quantized_output=output_path,
                ),
                io_mode="serial",
                input_buffer_gib=None,
                adapters=(AdapterMergeInput(root / "config-adapter.safetensors", 0.65),),
            )

            with (
                patch(
                    "potatoforge.cli.load_quantize_config",
                    return_value=config,
                ),
                patch(
                    "potatoforge.cli.convert_model_from_profile"
                ) as convert_mock,
            ):
                convert_mock.return_value = ResolvedIOMode(
                    "serial",
                    "requested mode",
                )
                result = self.runner.invoke(
                    app,
                    [
                        "quantize",
                        "--config",
                        str(root / "config.toml"),
                    ],
                )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            convert_mock.call_args.kwargs["adapters"],
            (AdapterMergeInput(root / "config-adapter.safetensors", 0.65),),
        )

    def test_quantize_rejects_unpaired_adapter_options(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.runner.invoke(
                app,
                [
                    "quantize",
                    str(root / "source.safetensors"),
                    str(root / "output.safetensors"),
                    "--profile",
                    str(root / "profile.json"),
                    "--adapter-path",
                    str(root / "adapter.safetensors"),
                ],
            )

        self.assertEqual(result.exit_code, 3, result.stdout)
        self.assertIn(
            "--adapter-path and --adapter-strength must be repeated",
            result.output,
        )

if __name__ == "__main__":
    unittest.main()
