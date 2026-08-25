import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from potatoforge.lora.lora_merge import AdapterMergeInput
from potatoforge.config import load_optimize_config, load_quantize_config


class TestConfigs(unittest.TestCase):
    def _write(self, directory: str, text: str) -> Path:
        path = Path(directory) / "config.toml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_loads_and_resolves_an_optimize_config(self) -> None:
        with TemporaryDirectory() as directory:
            config_path = self._write(
                directory,
                """
format_version = 1

[paths]
audit_report = "reports/zit/audit.json"
profile = "profiles/zit/generated.json"

[optimize]
profile_id = "zit"
target_size_gib = 4.3
methods = ["int6", "int8"]
enable_potatoforge_int6_runtime = true
exclude_prefixes = ["final_layer."]
exclude_suffixes = [".bias"]
""",
            )

            config = load_optimize_config(config_path)

        project_root = Path.cwd().resolve()
        self.assertEqual(config.target_size_gib, 4.3)
        self.assertEqual(config.methods, ("int6", "int8"))
        self.assertEqual(config.paths.audit_report, project_root / "reports/zit/audit.json")
        self.assertEqual(config.paths.profile, project_root / "profiles/zit/generated.json")
        self.assertEqual(config.exclude_prefixes, ("final_layer.",))

    def test_loads_and_resolves_a_quantize_config(self) -> None:
        with TemporaryDirectory() as directory:
            config_path = self._write(
                directory,
                """
format_version = 1

[paths]
source = "models/source.safetensors"
profile = "profiles/kroma/kroma-v0.1-balanced.json"
quantized_output = "outputs/quantized.safetensors"

[quantize]
io_mode = "batched"
input_buffer_gib = 4.0

[[quantize.adapters]]
path = "models/style-a.safetensors"
strength = 0.65

[[quantize.adapters]]
path = "models/style-b.safetensors"
strength = 0.40
""",
            )

            config = load_quantize_config(config_path)

        project_root = Path.cwd().resolve()
        self.assertEqual(config.io_mode, "batched")
        self.assertEqual(config.input_buffer_gib, 4.0)
        self.assertEqual(config.paths.source, project_root / "models/source.safetensors")
        self.assertEqual(
            config.paths.quantized_output,
            project_root / "outputs/quantized.safetensors",
        )
        self.assertEqual(
            config.adapters,
            (
                AdapterMergeInput(project_root / "models/style-a.safetensors", 0.65),
                AdapterMergeInput(project_root / "models/style-b.safetensors", 0.40),
            ),
        )

    def test_allows_batched_quantize_without_a_buffer_for_fallback(self) -> None:
        with TemporaryDirectory() as directory:
            config_path = self._write(
                directory,
                """
format_version = 1

[paths]
source = "source.safetensors"
profile = "profile.json"
quantized_output = "output.safetensors"

[quantize]
io_mode = "batched"
""",
            )

            config = load_quantize_config(config_path)

        self.assertEqual(config.io_mode, "batched")
        self.assertIsNone(config.input_buffer_gib)

    def test_rejects_inert_legacy_fields(self) -> None:
        with TemporaryDirectory() as directory:
            config_path = self._write(
                directory,
                """
format_version = 1
name = "legacy"

[paths]
audit_report = "audit.json"
profile = "profile.json"

[optimize]
profile_id = "invalid"
target_size_gib = 1.0
methods = ["int8"]
""",
            )

            with self.assertRaisesRegex(ValueError, "Unknown config field"):
                load_optimize_config(config_path)

    def test_int6_requires_runtime_opt_in(self) -> None:
        with TemporaryDirectory() as directory:
            config_path = self._write(
                directory,
                """
format_version = 1

[paths]
audit_report = "audit.json"
profile = "profile.json"

[optimize]
profile_id = "invalid"
target_size_gib = 1.0
methods = ["int6"]
""",
            )

            with self.assertRaisesRegex(ValueError, "INT6 methods require"):
                load_optimize_config(config_path)


if __name__ == "__main__":
    unittest.main()
