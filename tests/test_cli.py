import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

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
            "optimize",
            "quantize",
            "extract",
            "test",
        ):
            self.assertIn(command, result.stdout)

    def test_version_is_available(self) -> None:
        result = self.runner.invoke(app, ["--version"])

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout.strip(), "0.1.0")

    def test_optimize_config_applies_a_typed_target_override(self) -> None:
        config = OptimizeConfig(
            paths=ConfigPaths(
                audit_report=Path("audit.json"),
                profile=Path("profile.json"),
            ),
            profile_id="zit",
            target_size_gib=4.3,
            methods=("int8",),
            enable_potatoforge_int6_runtime=False,
            max_relative_l2_error=None,
            exclude_prefixes=(),
            exclude_suffixes=(),
            overwrite=False,
        )
        optimized = OptimizedProfile(
            profile={"profile_id": "zit", "default": "keep", "rules": ()},
            output_bytes=123,
            target_bytes=4 * 1024**3,
            proxy_distortion=0.0,
        )

        with (
            patch("potatoforge.cli.load_optimize_config", return_value=config),
            patch("potatoforge.cli.load_weight_audit", return_value={}),
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
