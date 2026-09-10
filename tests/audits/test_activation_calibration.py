import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import save_file

from potatoforge.calibration import ActivationCalibration


class TestActivationCalibration(unittest.TestCase):
    def test_loads_pair_and_supports_exact_lookup(self) -> None:
        with TemporaryDirectory() as directory:
            metadata_path, _ = self._write_pair(Path(directory))

            calibration = ActivationCalibration.load(metadata_path)

        stats = calibration.get("blocks.0.attn.wq.weight")
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats.tensor_name, "blocks.0.attn.wq.weight")
        self.assertEqual(stats.input_features, 3)
        self.assertEqual(stats.sample_count, 10)
        self.assertEqual(stats.invocation_count, 2)
        self.assertEqual(stats.sum_x2.dtype, torch.float32)
        self.assertEqual(stats.sum_x2.device.type, "cpu")
        torch.testing.assert_close(stats.sum_x2, torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(calibration.baseline_label, "int8_convrot")
        self.assertEqual(calibration.tensor_names(), ("blocks.0.attn.wq.weight",))
        self.assertTrue(calibration.has("blocks.0.attn.wq.weight"))
        self.assertFalse(calibration.has("missing.weight"))
        self.assertIsNone(calibration.get("missing.weight"))

    def test_rejects_unsupported_metadata_headers(self) -> None:
        for field, value in (
            ("format", "other"),
            ("version", 2),
            ("activation_basis", "packed"),
            ("activation_axis", "first_dimension"),
        ):
            with self.subTest(field=field), TemporaryDirectory() as directory:
                metadata_path, _ = self._write_pair(
                    Path(directory),
                    metadata_updates={field: value},
                )

                with self.assertRaises(ValueError):
                    ActivationCalibration.load(metadata_path)

    def test_rejects_missing_statistics_key(self) -> None:
        with TemporaryDirectory() as directory:
            metadata_path, _ = self._write_pair(
                Path(directory),
                layer_updates={"stats_key": "missing.sum_x2"},
            )

            with self.assertRaisesRegex(ValueError, "missing"):
                ActivationCalibration.load(metadata_path)

    def test_rejects_malformed_statistics(self) -> None:
        cases = (
            (torch.ones((1, 3)), "rank 1"),
            (torch.ones(2), "length"),
            (torch.tensor([1.0, -1.0, 2.0]), "non-negative"),
            (torch.tensor([1.0, float("nan"), 2.0]), "finite"),
        )
        for tensor, message in cases:
            with self.subTest(message=message), TemporaryDirectory() as directory:
                metadata_path, tensors_path = self._write_pair(Path(directory))
                save_file(
                    {"blocks.0.attn.wq.weight.sum_x2": tensor},
                    str(tensors_path),
                )

                with self.assertRaisesRegex(ValueError, message):
                    ActivationCalibration.load(metadata_path)

    def test_rejects_invalid_layer_metadata(self) -> None:
        cases = (
            ("input_features", 0),
            ("sample_count", -1),
            ("invocation_count", -1),
            ("stats_key", ""),
        )
        for field, value in cases:
            with self.subTest(field=field), TemporaryDirectory() as directory:
                metadata_path, _ = self._write_pair(
                    Path(directory),
                    layer_updates={field: value},
                )

                with self.assertRaises(ValueError):
                    ActivationCalibration.load(metadata_path)

    def _write_pair(
        self,
        directory: Path,
        *,
        metadata_updates: dict[str, object] | None = None,
        layer_updates: dict[str, object] | None = None,
    ) -> tuple[Path, Path]:
        layer = {
            "input_features": 3,
            "sample_count": 10,
            "invocation_count": 2,
            "stats_key": "blocks.0.attn.wq.weight.sum_x2",
        }
        layer.update(layer_updates or {})
        metadata: dict[str, object] = {
            "format": "potatoforge_activation_calibration",
            "version": 1,
            "session_id": "session-1",
            "session_name": "test",
            "baseline_label": "int8_convrot",
            "activation_basis": "logical_linear_input",
            "activation_axis": "last_dimension",
            "layer_count": 1,
            "layers": {"blocks.0.attn.wq.weight": layer},
        }
        metadata.update(metadata_updates or {})
        metadata_path = directory / "calibration.json"
        tensors_path = directory / "calibration.safetensors"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        save_file(
            {"blocks.0.attn.wq.weight.sum_x2": torch.tensor([1.0, 2.0, 3.0])},
            str(tensors_path),
        )
        return metadata_path, tensors_path
