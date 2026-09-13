import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import save_file

from potatoforge.calibration import (
    ActivationCalibration,
    LayerCalibration,
    load_activation_calibration,
    validate_activation_calibration_against_source,
)


class TestActivationCalibration(unittest.TestCase):
    def test_rejects_unsupported_calibration_version(self) -> None:
        with TemporaryDirectory() as directory:
            metadata_path = Path(directory) / "calibration.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "format": "potatoforge_activation_calibration",
                        "version": 2,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "version"):
                ActivationCalibration.load(metadata_path)

    def test_loads_pair_and_flattens_padded_samples(self) -> None:
        with TemporaryDirectory() as directory:
            metadata_path, _ = self._write_pair(Path(directory))

            calibration = load_activation_calibration(metadata_path)

        self.assertEqual(calibration.version, 1)
        self.assertEqual(calibration.session_id, "session-1")
        self.assertEqual(len(calibration.evaluations), 2)
        self.assertEqual(calibration.evaluations[0].timestep, 0.9)
        self.assertEqual(calibration.evaluations[1].sigma, 0.1)
        self.assertEqual(calibration.evaluations[0].root_input_sum_x2, 10.0)
        self.assertIsNone(calibration.evaluations[1].root_output_sum_y2)

        layer = calibration.get("blocks.0.attn.wq.weight")
        self.assertIsInstance(layer, LayerCalibration)
        assert isinstance(layer, LayerCalibration)
        self.assertEqual(layer.output_features, 2)
        torch.testing.assert_close(
            layer.aggregate_sum_x2,
            torch.tensor([17.0, 29.0, 45.0]),
        )
        torch.testing.assert_close(
            layer.eval_sum_y2,
            torch.tensor([[1.0, 4.0], [9.0, 16.0]]),
        )
        torch.testing.assert_close(
            layer.eval_sample_counts,
            torch.tensor([2, 1]),
        )
        torch.testing.assert_close(
            layer.sample_x,
            torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
            ),
        )
        torch.testing.assert_close(
            layer.sample_evaluation_indices,
            torch.tensor([0, 0, 1]),
        )

    def test_validation_rejects_bad_tensor_contracts(self) -> None:
        cases = (
            ("missing", {"eval_sum_y2_key": "missing"}, None, "missing"),
            (
                "shape",
                None,
                {"blocks.0.attn.wq.weight.eval_sum_y2": torch.ones((1, 2))},
                "shape",
            ),
            (
                "negative",
                None,
                {
                    "blocks.0.attn.wq.weight.eval_sum_y2": torch.tensor(
                        [[1.0, -1.0], [2.0, 3.0]]
                    )
                },
                "non-negative",
            ),
            (
                "sample mapping",
                None,
                {
                    "blocks.0.attn.wq.weight.sample_x_valid": torch.tensor([3, 0])
                },
                "cannot exceed",
            ),
        )
        for case, layer_updates, tensor_updates, message in cases:
            with self.subTest(case=case), TemporaryDirectory() as directory:
                metadata_path, tensors_path = self._write_pair(
                    Path(directory),
                    layer_updates=layer_updates,
                    tensor_updates=tensor_updates,
                )

                with self.assertRaisesRegex(ValueError, message):
                    load_activation_calibration(metadata_path, tensors_path)

    def test_source_validation_requires_exact_linear_shape(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path, _ = self._write_pair(root)
            source_path = root / "source.safetensors"
            save_file(
                {"blocks.0.attn.wq.weight": torch.ones((2, 3))},
                str(source_path),
            )
            calibration = load_activation_calibration(metadata_path)

            source_header = validate_activation_calibration_against_source(
                calibration,
                source_path,
            )

        self.assertEqual(
            source_header.tensors["blocks.0.attn.wq.weight"]["shape"],
            [2, 3],
        )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path, _ = self._write_pair(root)
            source_path = root / "source.safetensors"
            save_file(
                {"blocks.0.attn.wq.weight": torch.ones((3, 2))},
                str(source_path),
            )

            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_activation_calibration_against_source(
                    metadata_path,
                    source_path,
                )

    def _write_pair(
        self,
        directory: Path,
        *,
        layer_updates: dict[str, object] | None = None,
        tensor_updates: dict[str, torch.Tensor] | None = None,
    ) -> tuple[Path, Path]:
        tensor_name = "blocks.0.attn.wq.weight"
        layer: dict[str, object] = {
            "input_features": 3,
            "output_features": 2,
            "sample_count": 3,
            "invocation_count": 2,
            "stats_key": f"{tensor_name}.sum_x2",
            "eval_sum_x_key": f"{tensor_name}.eval_sum_x",
            "eval_sum_x2_key": f"{tensor_name}.eval_sum_x2",
            "eval_max_abs_x_key": f"{tensor_name}.eval_max_abs_x",
            "eval_sum_y_key": f"{tensor_name}.eval_sum_y",
            "eval_sum_y2_key": f"{tensor_name}.eval_sum_y2",
            "eval_sample_count_key": f"{tensor_name}.eval_sample_count",
            "eval_invocation_count_key": f"{tensor_name}.eval_invocation_count",
            "sample_x_key": f"{tensor_name}.sample_x",
            "sample_x_valid_key": f"{tensor_name}.sample_x_valid",
        }
        layer.update(layer_updates or {})
        metadata: dict[str, object] = {
            "format": "potatoforge_activation_calibration",
            "version": 1,
            "session_id": "session-1",
            "session_name": "test-1",
            "baseline_label": "bf16",
            "activation_basis": "logical_linear_input",
            "activation_axis": "last_dimension",
            "layer_count": 1,
            "evaluation_count": 2,
            "evaluations": [
                {
                    "evaluation_index": 0,
                    "time_parameter_name": "timestep",
                    "time_value": 0.9,
                    "time_value_truncated": False,
                },
                {
                    "evaluation_index": 1,
                    "time_parameter_name": "sigma",
                    "time_value": 0.1,
                    "time_value_truncated": False,
                },
            ],
            "layers": {tensor_name: layer},
        }
        metadata_path = directory / "calibration.json"
        tensors_path = directory / "calibration.safetensors"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        tensors: dict[str, torch.Tensor] = {
            "__pf__.root_input_sum_x2": torch.tensor([10.0, 20.0]),
            "__pf__.root_input_valid": torch.tensor([True, True]),
            "__pf__.root_output_sum_y2": torch.tensor([30.0, 0.0]),
            "__pf__.root_output_valid": torch.tensor([True, False]),
            f"{tensor_name}.sum_x2": torch.tensor([17.0, 29.0, 45.0]),
            f"{tensor_name}.eval_sum_x": torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            ),
            f"{tensor_name}.eval_sum_x2": torch.tensor(
                [[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]]
            ),
            f"{tensor_name}.eval_max_abs_x": torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            ),
            f"{tensor_name}.eval_sum_y": torch.tensor(
                [[1.0, 2.0], [3.0, 4.0]]
            ),
            f"{tensor_name}.eval_sum_y2": torch.tensor(
                [[1.0, 4.0], [9.0, 16.0]]
            ),
            f"{tensor_name}.eval_sample_count": torch.tensor([2, 1]),
            f"{tensor_name}.eval_invocation_count": torch.tensor([1, 1]),
            f"{tensor_name}.sample_x": torch.tensor(
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[7.0, 8.0, 9.0], [0.0, 0.0, 0.0]],
                ]
            ),
            f"{tensor_name}.sample_x_valid": torch.tensor([2, 1]),
        }
        tensors.update(tensor_updates or {})
        save_file(tensors, str(tensors_path))
        return metadata_path, tensors_path
