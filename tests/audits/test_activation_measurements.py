import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import save_file

from potatoforge.audits.activation_measurements import (
    MEASUREMENT_METHODS,
    measure_activation_candidate,
    measure_activation_candidates,
)
from potatoforge.calibration import load_activation_calibration
from potatoforge.planning import (
    TensorDescriptor,
    build_quantized_tensor_plan,
)
from potatoforge.quantization.int8_tensorwise import (
    dequantize_int8_tensorwise,
    quantize_int8_tensorwise,
)


class TestActivationMeasurements(unittest.TestCase):
    def test_computes_eval_output_error_from_reconstructed_candidate(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            calibration, layer = self._write_calibration(root, 4)
            descriptor = self._descriptor(2, 4)
            weights = torch.tensor(
                [[-1.0, 0.5, 1.0, 2.0], [3.0, -2.0, 0.25, 1.5]],
                dtype=torch.bfloat16,
            )

            result = measure_activation_candidate(
                "blocks.0.attn.wq.weight",
                descriptor,
                weights,
                layer,
                "int8",
            )
            reconstructed = dequantize_int8_tensorwise(
                quantize_int8_tensorwise(weights)
            )
            expected = layer.eval_sum_x2 @ (
                reconstructed.float() - weights.float()
            ).square().transpose(0, 1)

        self.assertTrue(result.available)
        self.assertEqual(result.action, "int8")
        self.assertEqual(
            result.storage_bytes,
            build_quantized_tensor_plan(
                "int8",
                "blocks.0.attn.wq.weight",
                descriptor,
            ).estimated_bytes,
        )
        torch.testing.assert_close(result.error_by_eval_output, expected)
        self.assertEqual(calibration.version, 2)

    def test_keep_is_zero_and_ineligible_methods_are_explicit(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, layer = self._write_calibration(root, 4)
            descriptor = self._descriptor(2, 4)
            weights = torch.ones((2, 4), dtype=torch.bfloat16)

            keep = measure_activation_candidate(
                "blocks.0.attn.wq.weight",
                descriptor,
                weights,
                layer,
                "bf16",
            )
            convrot = measure_activation_candidate(
                "blocks.0.attn.wq.weight",
                descriptor,
                weights,
                layer,
                "int8_convrot",
            )

        self.assertTrue(keep.available)
        self.assertEqual(keep.action, "keep")
        self.assertEqual(keep.storage_bytes, 16)
        assert keep.error_by_eval_output is not None
        self.assertEqual(float(keep.error_by_eval_output.sum()), 0.0)
        self.assertFalse(convrot.available)
        self.assertIsNone(convrot.error_by_eval_output)
        self.assertIn("divisible", convrot.unavailable_reason or "")

    def test_measures_all_current_methods_when_width_is_eligible(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, layer = self._write_calibration(root, 256)
            descriptor = self._descriptor(2, 256)
            weights = torch.linspace(
                -1.0,
                1.0,
                steps=512,
                dtype=torch.bfloat16,
            ).reshape(2, 256)

            results = measure_activation_candidates(
                "blocks.0.attn.wq.weight",
                descriptor,
                weights,
                layer,
            )

        self.assertEqual(
            tuple(result.method for result in results),
            MEASUREMENT_METHODS,
        )
        self.assertTrue(all(result.available for result in results))
        self.assertTrue(
            all(
                result.error_by_eval_output is not None
                and tuple(result.error_by_eval_output.shape) == (2, 2)
                for result in results
            )
        )

    def test_measures_sampled_exact_and_diagonal_diagnostics(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, layer = self._write_calibration(root, 2, include_samples=True)
            weights = torch.eye(2, dtype=torch.bfloat16)

            result = measure_activation_candidate(
                "blocks.0.attn.wq.weight",
                self._descriptor(2, 2),
                weights,
                layer,
                "int8",
            )

        self.assertTrue(result.available)
        self.assertIsNotNone(result.sample_error_sse)
        self.assertIsNotNone(result.sample_reference_energy)
        self.assertIsNotNone(result.sample_direction_error)
        self.assertEqual(result.sample_exact_sse, 0.0)
        self.assertEqual(result.sample_diag_sse, 0.0)
        self.assertEqual(result.sample_cross_term_ratio, 0.0)
        torch.testing.assert_close(
            result.sample_error_sse,
            torch.zeros(2),
        )

    def _write_calibration(
        self,
        root: Path,
        input_features: int,
        include_samples: bool = False,
    ) -> tuple[object, object]:
        tensor_name = "blocks.0.attn.wq.weight"
        eval_sum_x2 = torch.arange(
            1,
            input_features + 1,
            dtype=torch.float32,
        ).repeat(2, 1)
        metadata = {
            "format": "potatoforge_activation_calibration",
            "version": 2,
            "session_id": "measurement-session",
            "baseline_label": "bf16",
            "activation_basis": "logical_linear_input",
            "activation_axis": "last_dimension",
            "layer_count": 1,
            "evaluation_count": 2,
            "evaluations": [
                {"evaluation_index": 0},
                {"evaluation_index": 1},
            ],
            "layers": {
                tensor_name: {
                    "input_features": input_features,
                    "output_features": 2,
                    "sample_count": 2,
                    "invocation_count": 2,
                    "stats_key": f"{tensor_name}.sum_x2",
                    "eval_sum_x_key": f"{tensor_name}.eval_sum_x",
                    "eval_sum_x2_key": f"{tensor_name}.eval_sum_x2",
                    "eval_max_abs_x_key": f"{tensor_name}.eval_max_abs_x",
                    "eval_sum_y_key": f"{tensor_name}.eval_sum_y",
                    "eval_sum_y2_key": f"{tensor_name}.eval_sum_y2",
                    "eval_sample_count_key": (
                        f"{tensor_name}.eval_sample_count"
                    ),
                    "eval_invocation_count_key": (
                        f"{tensor_name}.eval_invocation_count"
                    ),
                    "sample_x_key": (
                        f"{tensor_name}.sample_x" if include_samples else None
                    ),
                    "sample_x_valid_key": (
                        f"{tensor_name}.sample_x_valid"
                        if include_samples
                        else None
                    ),
                }
            },
        }
        metadata_path = root / "calibration.json"
        tensors_path = root / "calibration.safetensors"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        save_file(
            {
                f"{tensor_name}.sum_x2": eval_sum_x2.sum(dim=0),
                f"{tensor_name}.eval_sum_x": torch.zeros(
                    (2, input_features)
                ),
                f"{tensor_name}.eval_sum_x2": eval_sum_x2,
                f"{tensor_name}.eval_max_abs_x": torch.zeros(
                    (2, input_features)
                ),
                f"{tensor_name}.eval_sum_y": torch.zeros((2, 2)),
                f"{tensor_name}.eval_sum_y2": torch.ones((2, 2)),
                f"{tensor_name}.eval_sample_count": torch.ones(
                    2,
                    dtype=torch.int64,
                ),
                f"{tensor_name}.eval_invocation_count": torch.ones(
                    2,
                    dtype=torch.int64,
                ),
                **(
                    {
                        f"{tensor_name}.sample_x": torch.tensor(
                            [[[1.0, 0.0]], [[0.0, 1.0]]],
                            dtype=torch.float32,
                        ),
                        f"{tensor_name}.sample_x_valid": torch.ones(
                            2,
                            dtype=torch.int64,
                        ),
                    }
                    if include_samples
                    else {}
                ),
            },
            str(tensors_path),
        )
        calibration = load_activation_calibration(metadata_path)
        return calibration, calibration.get(tensor_name)

    @staticmethod
    def _descriptor(out_features: int, in_features: int) -> TensorDescriptor:
        return {
            "dtype": "BF16",
            "shape": [out_features, in_features],
            "data_offsets": [0, out_features * in_features * 2],
        }


if __name__ == "__main__":
    unittest.main()
