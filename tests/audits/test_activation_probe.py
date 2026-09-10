import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import save_file

from potatoforge.audits.weight_audit import audit_bf16_source
from potatoforge.calibration import ActivationCalibration, ActivationStats
from potatoforge.calibration.activation_probe import (
    ActivationProbeCache,
    ActivationProbeRecord,
    merge_activation_calibrations,
    score_activation_probe,
)
from potatoforge.quantization.int8_tensorwise import (
    dequantize_int8_convrot,
    quantize_int8_convrot,
)


class TestActivationProbe(unittest.TestCase):
    def test_probe_cache_reproduces_full_audit_activation_error(self) -> None:
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
            calibration_path = root / "calibration.json"
            calibration_path.write_text(
                json.dumps(
                    {
                        "format": "potatoforge_activation_calibration",
                        "version": 1,
                        "baseline_label": "int8_convrot",
                        "activation_basis": "logical_linear_input",
                        "activation_axis": "last_dimension",
                        "layer_count": 1,
                        "layers": {
                            "blocks.0.attn.wq.weight": {
                                "input_features": 256,
                                "sample_count": 3,
                                "invocation_count": 4,
                                "stats_key": "blocks.0.attn.wq.weight.sum_x2",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            save_file(
                {
                    "blocks.0.attn.wq.weight.sum_x2": torch.linspace(
                        1.0,
                        2.0,
                        steps=256,
                    )
                },
                str(root / "calibration.safetensors"),
            )
            calibration = ActivationCalibration.load(calibration_path)
            audit = audit_bf16_source(
                source_path,
                activation_calibration=calibration,
                activation_probe_output=root / "probe",
            )

            probe = ActivationProbeCache.load(root / "probe.json")
            scored = score_activation_probe(probe, calibration)
        self.assertEqual(probe.reference_format, "int8_convrot")
        self.assertEqual(probe.candidate_format, "convrot_w4a4")
        probe_record = probe.get("blocks.0.attn.wq.weight")
        self.assertIsNotNone(probe_record)
        assert probe_record is not None
        self.assertIsNotNone(probe_record.reference_power_per_input)
        expected_reference = dequantize_int8_convrot(
            quantize_int8_convrot(weights)
        )
        torch.testing.assert_close(
            probe_record.reference_power_per_input,
            expected_reference.float().square().sum(dim=0),
        )
        self.assertEqual(scored["results"][0]["activation_status"], "ok")
        self.assertAlmostEqual(
            scored["results"][0]["activation_error"],
            audit["results"][0]["activation"]["error"],
            places=5,
        )
        self.assertAlmostEqual(
            scored["results"][0]["activation_error_mean"],
            scored["results"][0]["activation_error"] / 3,
            places=7,
        )
        self.assertEqual(
            scored["metrics"],
            [
                "diagonal_activation_energy_v1",
                "relative_diagonal_output_error_v1",
            ],
        )
        self.assertEqual(
            scored["primary_metric"],
            "relative_diagonal_output_error_v1",
        )

    def test_score_reports_raw_and_mean_ranks(self) -> None:
        tensor_a = "blocks.0.attn.wo.weight"
        tensor_b = "blocks.0.mlp.down.weight"
        tensor_zero = "blocks.0.attn.wq.weight"
        probe = ActivationProbeCache(
            source_model_path=None,
            records={
                tensor_a: ActivationProbeRecord(
                    tensor_a,
                    2,
                    "ok",
                    torch.tensor([1.0, 0.0]),
                    torch.tensor([1.0, 0.0]),
                ),
                tensor_b: ActivationProbeRecord(
                    tensor_b,
                    2,
                    "ok",
                    torch.tensor([1.0, 0.0]),
                    torch.tensor([1.0, 0.0]),
                ),
                tensor_zero: ActivationProbeRecord(
                    tensor_zero,
                    2,
                    "ok",
                    torch.tensor([0.0, 0.0]),
                    torch.tensor([0.0, 0.0]),
                ),
            },
        )
        calibration = ActivationCalibration(
            baseline_label="int8_convrot",
            session_id=None,
            session_name=None,
            activation_basis="logical_linear_input",
            activation_axis="last_dimension",
            stats={
                tensor_a: ActivationStats(
                    tensor_a, 2, 100, 1, torch.tensor([10.0, 0.0])
                ),
                tensor_b: ActivationStats(
                    tensor_b, 2, 1, 1, torch.tensor([9.0, 0.0])
                ),
                tensor_zero: ActivationStats(
                    tensor_zero, 2, 0, 0, torch.tensor([1.0, 0.0])
                ),
            },
        )

        report = score_activation_probe(probe, calibration)
        results = {
            result["tensor_name"]: result for result in report["results"]
        }

        self.assertEqual(results[tensor_a]["activation_rank_raw"], 1)
        self.assertEqual(results[tensor_b]["activation_rank_raw"], 2)
        self.assertEqual(results[tensor_b]["activation_rank_mean"], 1)
        self.assertEqual(results[tensor_a]["activation_rank_mean"], 2)
        self.assertEqual(results[tensor_a]["weight_error_energy"], 1.0)
        self.assertAlmostEqual(
            results[tensor_a]["input_activation_energy_mean"],
            0.05,
        )
        self.assertAlmostEqual(
            results[tensor_a]["activation_alignment_ratio"],
            2.0,
        )
        self.assertAlmostEqual(
            results[tensor_a]["reference_output_energy"],
            10.0,
        )
        self.assertAlmostEqual(
            results[tensor_a]["relative_output_error_sq"],
            1.0,
        )
        self.assertAlmostEqual(
            results[tensor_a]["relative_output_error"],
            1.0,
        )
        self.assertEqual(results[tensor_a]["relative_output_error_rank"], 1)
        self.assertEqual(results[tensor_b]["relative_output_error_rank"], 2)
        self.assertIsNone(results[tensor_zero]["input_activation_energy_mean"])
        self.assertIsNone(results[tensor_zero]["activation_alignment_ratio"])
        self.assertEqual(
            results[tensor_zero]["activation_status"],
            "zero_reference_output_energy",
        )
        self.assertIsNone(results[tensor_zero]["relative_output_error_sq"])
        self.assertIsNone(results[tensor_zero]["relative_output_error"])
        self.assertIsNone(results[tensor_zero]["relative_output_error_rank"])
        self.assertIsNone(results[tensor_zero]["activation_error_mean"])
        self.assertIsNone(results[tensor_zero]["activation_rank_mean"])

    def test_merge_adds_raw_statistics_and_counts(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._write_calibration(
                root / "first",
                torch.tensor([1.0, 2.0, 3.0]),
                sample_count=3,
                invocation_count=4,
            )
            second = self._write_calibration(
                root / "second",
                torch.tensor([4.0, 5.0, 6.0]),
                sample_count=7,
                invocation_count=8,
            )

            metadata_path, tensors_path = merge_activation_calibrations(
                [first, second],
                root / "merged",
            )
            merged = ActivationCalibration.load(metadata_path)
            self.assertTrue(tensors_path.exists())

        stats = merged.get("blocks.0.attn.wq.weight")
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual((stats.sample_count, stats.invocation_count), (10, 12))
        torch.testing.assert_close(stats.sum_x2, torch.tensor([5.0, 7.0, 9.0]))

    def test_merge_rejects_incompatible_baseline(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._write_calibration(
                root / "first",
                torch.ones(3),
            )
            second = self._write_calibration(
                root / "second",
                torch.ones(3),
                baseline_label="bf16",
            )

            with self.assertRaisesRegex(ValueError, "baseline_label"):
                merge_activation_calibrations([first, second], root / "merged")

    def test_probe_rejects_q_key_alias(self) -> None:
        tensor_name = "blocks.0.attn.wq.weight"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path = root / "probe.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "format": "potatoforge_activation_probe",
                        "version": 2,
                        "source_model_path": "source.safetensors",
                        "reference_format": "int8_convrot",
                        "candidate_format": "convrot_w4a4",
                        "metric_basis": "logical_linear_input",
                        "metric": "diagonal_activation_energy_v1",
                        "tensor_count": 1,
                        "tensors": {
                            tensor_name: {
                                "tensor_name": tensor_name,
                                "input_features": 3,
                                "q_key": "other.q_per_input",
                                "reference_power_key": (
                                    f"{tensor_name}.reference_power_per_input"
                                ),
                                "status": "ok",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            save_file(
                {
                    "other.q_per_input": torch.ones(3),
                    f"{tensor_name}.reference_power_per_input": torch.ones(3),
                },
                str(root / "probe.safetensors"),
            )

            with self.assertRaisesRegex(ValueError, "q_key"):
                ActivationProbeCache.load(metadata_path)

    def test_probe_rejects_old_schema_version(self) -> None:
        with TemporaryDirectory() as directory:
            metadata_path = Path(directory) / "probe.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "format": "potatoforge_activation_probe",
                        "version": 1,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "version"):
                ActivationProbeCache.load(metadata_path)

    def _write_calibration(
        self,
        root: Path,
        sum_x2: torch.Tensor,
        *,
        sample_count: int = 1,
        invocation_count: int = 1,
        baseline_label: str = "int8_convrot",
    ) -> Path:
        root.mkdir()
        tensor_name = "blocks.0.attn.wq.weight"
        metadata_path = root / "calibration.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "format": "potatoforge_activation_calibration",
                    "version": 1,
                    "baseline_label": baseline_label,
                    "activation_basis": "logical_linear_input",
                    "activation_axis": "last_dimension",
                    "diffusion_model_class": "Kroma",
                    "layer_count": 1,
                    "layers": {
                        tensor_name: {
                            "input_features": 3,
                            "sample_count": sample_count,
                            "invocation_count": invocation_count,
                            "stats_key": f"{tensor_name}.sum_x2",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        save_file(
            {f"{tensor_name}.sum_x2": sum_x2},
            str(root / "calibration.safetensors"),
        )
        return metadata_path


if __name__ == "__main__":
    unittest.main()
