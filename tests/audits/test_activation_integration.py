import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from potatoforge.audits.activation_error import activation_weighted_error
from potatoforge.audits.weight_audit import audit_bf16_source
from potatoforge.calibration import ActivationCalibration
from potatoforge.quantization.convrot_w4a4 import (
    dequantize_convrot_w4a4,
    quantize_convrot_w4a4,
)
from potatoforge.quantization.int8_tensorwise import (
    dequantize_int8_convrot,
    quantize_int8_convrot,
)


class TestActivationAuditIntegration(unittest.TestCase):
    def test_probe_requires_int8_convrot_calibration(self) -> None:
        weights = torch.ones((2, 256), dtype=torch.bfloat16)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.safetensors"
            save_file({"blocks.0.attn.wq.weight": weights}, str(source_path))

            with self.assertRaisesRegex(
                ValueError,
                "activation_probe_output requires.*int8_convrot",
            ):
                audit_bf16_source(
                    source_path,
                    activation_probe_output=root / "probe",
                )

    def test_reuses_w4a4_reconstruction_without_changing_existing_metrics(self) -> None:
        weights = torch.linspace(
            -0.5,
            0.5,
            steps=512,
            dtype=torch.bfloat16,
        ).reshape(2, 256)
        sum_x2 = torch.linspace(1.0, 2.0, steps=256)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.safetensors"
            save_file({"blocks.0.attn.wq.weight": weights}, str(source_path))
            metadata_path = self._write_calibration(root, sum_x2)
            calibration = ActivationCalibration.load(metadata_path)

            without_calibration = audit_bf16_source(source_path)
            with_calibration = audit_bf16_source(
                source_path,
                activation_calibration=calibration,
            )

        without_record = without_calibration["results"][0]
        with_record = with_calibration["results"][0]
        self.assertNotIn("activation", without_record)
        self.assertEqual(with_record["methods"], without_record["methods"])
        self.assertEqual(
            with_record["error_deltas"],
            without_record["error_deltas"],
        )
        activation = with_record["activation"]
        self.assertEqual(activation["status"], "ok")
        self.assertEqual(
            activation["metric_variant"],
            "diagonal_activation_energy_v1",
        )

        candidate = dequantize_convrot_w4a4(quantize_convrot_w4a4(weights))
        expected = activation_weighted_error(
            weights.float(),
            candidate,
            sum_x2,
        )
        self.assertAlmostEqual(
            activation["error"],
            expected.activation_error,
            places=5,
        )
        self.assertEqual(activation["sample_count"], 1)
        self.assertEqual(activation["invocation_count"], 1)
        self.assertEqual(activation["input_features"], 256)

    def test_missing_tensor_is_explicit_but_does_not_break_audit(self) -> None:
        weights = torch.ones((2, 256), dtype=torch.bfloat16)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.safetensors"
            save_file({"blocks.0.attn.wq.weight": weights}, str(source_path))
            metadata_path = self._write_calibration(
                root,
                torch.ones(1),
                tensor_name="other.weight",
                stats_key="other.weight.sum_x2",
                input_features=1,
            )
            calibration = ActivationCalibration.load(metadata_path)

            document = audit_bf16_source(
                source_path,
                activation_calibration=calibration,
            )

        record = document["results"][0]
        self.assertEqual(record["activation"]["status"], "missing_calibration")
        self.assertIsNone(record["activation"]["error"])
        self.assertIsNotNone(record["methods"]["convrot_w4a4"]["relative_l2_error"])

    def test_int8_convrot_calibration_uses_effective_reference(self) -> None:
        weights = torch.linspace(
            -0.5,
            0.5,
            steps=512,
            dtype=torch.bfloat16,
        ).reshape(2, 256)
        sum_x2 = torch.linspace(1.0, 2.0, steps=256)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.safetensors"
            save_file({"blocks.0.attn.wq.weight": weights}, str(source_path))
            metadata_path = self._write_calibration(
                root,
                sum_x2,
                baseline_label="int8_convrot",
            )
            calibration = ActivationCalibration.load(metadata_path)
            document = audit_bf16_source(
                source_path,
                activation_calibration=calibration,
            )

        activation = document["results"][0]["activation"]
        candidate = dequantize_convrot_w4a4(quantize_convrot_w4a4(weights))
        int8_reference = dequantize_int8_convrot(
            quantize_int8_convrot(weights)
        )
        expected = activation_weighted_error(
            int8_reference,
            candidate,
            sum_x2,
        )
        bf16_score = activation_weighted_error(
            weights.float(),
            candidate,
            sum_x2,
        )

        self.assertEqual(activation["status"], "ok")
        self.assertEqual(activation["calibration_baseline"], "int8_convrot")
        self.assertEqual(activation["activation_reference"], "int8_convrot")
        self.assertEqual(activation["candidate_format"], "convrot_w4a4")
        self.assertAlmostEqual(activation["error"], expected.activation_error, places=5)
        self.assertNotAlmostEqual(
            activation["error"],
            bf16_score.activation_error,
            places=5,
        )

    def test_unsupported_calibration_baseline_is_explicit(self) -> None:
        weights = torch.ones((2, 256), dtype=torch.bfloat16)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.safetensors"
            save_file({"blocks.0.attn.wq.weight": weights}, str(source_path))
            metadata_path = self._write_calibration(
                root,
                torch.ones(256),
                baseline_label="unknown_reference",
            )
            calibration = ActivationCalibration.load(metadata_path)
            document = audit_bf16_source(
                source_path,
                activation_calibration=calibration,
            )

        activation = document["results"][0]["activation"]
        self.assertEqual(activation["status"], "unsupported_reference")
        self.assertIsNone(activation["activation_reference"])
        self.assertIsNone(activation["error"])
        self.assertIsNotNone(
            document["results"][0]["methods"]["convrot_w4a4"][
                "relative_l2_error"
            ]
        )

    def test_calibration_reuses_the_single_w4a4_candidate_reconstruction(self) -> None:
        weights = torch.linspace(
            -0.5,
            0.5,
            steps=512,
            dtype=torch.bfloat16,
        ).reshape(2, 256)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.safetensors"
            save_file({"blocks.0.attn.wq.weight": weights}, str(source_path))
            metadata_path = self._write_calibration(
                root,
                torch.ones(256),
            )
            calibration = ActivationCalibration.load(metadata_path)

            with patch(
                "potatoforge.audits.all_comparison.quantize_convrot_w4a4",
                wraps=quantize_convrot_w4a4,
            ) as quantize_mock:
                audit_bf16_source(
                    source_path,
                    activation_calibration=calibration,
                )

        self.assertEqual(quantize_mock.call_count, 1)

    def test_w4a4_only_audit_skips_other_candidate_quantizers(self) -> None:
        weights = torch.linspace(
            -0.5,
            0.5,
            steps=512,
            dtype=torch.bfloat16,
        ).reshape(2, 256)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.safetensors"
            save_file({"blocks.0.attn.wq.weight": weights}, str(source_path))
            metadata_path = self._write_calibration(
                root,
                torch.ones(256),
                baseline_label="int8_convrot",
            )
            calibration = ActivationCalibration.load(metadata_path)

            with (
                patch(
                    "potatoforge.audits.all_comparison.quantize_int8_tensorwise",
                    side_effect=AssertionError("plain INT8 should be skipped"),
                ),
                patch(
                    "potatoforge.audits.all_comparison.quantize_int6_rowwise",
                    side_effect=AssertionError("INT6 should be skipped"),
                ),
                patch(
                    "potatoforge.audits.all_comparison.quantize_int6_convrot",
                    side_effect=AssertionError("INT6 ConvRot should be skipped"),
                ),
                patch(
                    "potatoforge.audits.all_comparison.quantize_convrot_w4a4_mse",
                    side_effect=AssertionError("W4A4 MSE should be skipped"),
                ),
                patch(
                    "potatoforge.audits.all_comparison.quantize_int8_convrot",
                    wraps=quantize_int8_convrot,
                ) as int8_reference_mock,
            ):
                document = audit_bf16_source(
                    source_path,
                    activation_calibration=calibration,
                    audit_method="convrot_w4a4",
                )

        record = document["results"][0]
        self.assertEqual(int8_reference_mock.call_count, 1)
        self.assertIsNotNone(record["activation"]["error"])
        self.assertIsNotNone(record["methods"]["convrot_w4a4"]["relative_l2_error"])
        for method in (
            "int8",
            "int6",
            "int8_convrot",
            "int6_convrot",
            "convrot_w4a4_mse",
        ):
            self.assertIsNone(record["methods"][method]["relative_l2_error"])

    def _write_calibration(
        self,
        root: Path,
        sum_x2: torch.Tensor,
        *,
        tensor_name: str = "blocks.0.attn.wq.weight",
        stats_key: str = "blocks.0.attn.wq.weight.sum_x2",
        input_features: int = 256,
        baseline_label: str = "bf16",
    ) -> Path:
        metadata_path = root / "calibration.json"
        tensors_path = root / "calibration.safetensors"
        metadata_path.write_text(
            json.dumps(
                {
                    "format": "potatoforge_activation_calibration",
                    "version": 1,
                    "baseline_label": baseline_label,
                    "activation_basis": "logical_linear_input",
                    "activation_axis": "last_dimension",
                    "layer_count": 1,
                    "layers": {
                        tensor_name: {
                            "input_features": input_features,
                            "sample_count": 1,
                            "invocation_count": 1,
                            "stats_key": stats_key,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        save_file({stats_key: sum_x2}, str(tensors_path))
        return metadata_path
