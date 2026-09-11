import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from potatoforge.audits.activation_audit import (
    ACTIVATION_AUDIT_FORMAT,
    ACTIVATION_AUDIT_VERSION,
    ActivationAuditCache,
    activation_audit_pair_paths,
    inspect_activation_audit,
    run_activation_audit,
    score_activation_audit,
    write_activation_audit_cache,
)
from potatoforge.audits.activation_measurements import (
    CandidateMeasurement,
    measure_activation_candidates,
)
from potatoforge.calibration import load_activation_calibration
from potatoforge.planning import TensorDescriptor


class TestActivationAudit(unittest.TestCase):
    def test_round_trip_preserves_sampled_diagnostics(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            calibration, layer, calibration_metadata, _ = self._write_calibration(
                root,
                2,
                include_samples=True,
            )
            candidates = measure_activation_candidates(
                "blocks.0.attn.wq.weight",
                self._descriptor(2, 2),
                torch.eye(2, dtype=torch.bfloat16),
                layer,
                ("int8",),
            )
            metadata_path, tensors_path = write_activation_audit_cache(
                root / "sampled",
                root / "source.safetensors",
                calibration,
                {"blocks.0.attn.wq.weight": candidates},
                calibration_metadata_path=calibration_metadata,
                requested_methods=("int8",),
            )
            cache = ActivationAuditCache.load(
                metadata_path,
                calibration=calibration,
            )
            stored_tensors = load_file(str(tensors_path), device="cpu")
            score_report = score_activation_audit(
                metadata_path,
                calibration,
                metric="sampled_exact_relative_sse",
            )
            inspection = inspect_activation_audit(
                metadata_path,
                calibration,
                tensor_name="blocks.0.attn.wq.weight",
            )

        loaded = cache.get("blocks.0.attn.wq.weight", "int8")
        assert loaded is not None
        self.assertEqual(
            set(stored_tensors),
            {"m000000", "s000000", "r000000", "d000000"},
        )
        self.assertEqual(loaded.sample_exact_sse, 0.0)
        self.assertEqual(loaded.sample_diag_sse, 0.0)
        self.assertEqual(loaded.sample_cross_term_ratio, 0.0)
        self.assertEqual(
            score_report["results"][0]["objective_cost"],
            0.0,
        )
        self.assertTrue(
            inspection["layers"][0]["methods"][0]["sampled"]["available"]
        )

    def test_sampled_cache_uses_sampled_rows_not_total_sample_count(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            calibration, layer, calibration_metadata, _ = self._write_calibration(
                root,
                2,
                include_samples=True,
                evaluation_sample_counts=(50, 50),
            )
            candidates = measure_activation_candidates(
                "blocks.0.attn.wq.weight",
                self._descriptor(2, 2),
                torch.eye(2, dtype=torch.bfloat16),
                layer,
                ("convrot_w4a4",),
            )
            metadata_path, _ = write_activation_audit_cache(
                root / "sampled",
                root / "source.safetensors",
                calibration,
                {"blocks.0.attn.wq.weight": candidates},
                calibration_metadata_path=calibration_metadata,
                requested_methods=("convrot_w4a4",),
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["layers"]["blocks.0.attn.wq.weight"].pop(
                "sampled_sample_count"
            )
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            cache = ActivationAuditCache.load(
                metadata_path,
                calibration=calibration,
            )

        loaded = cache.get("blocks.0.attn.wq.weight", "convrot_w4a4")
        assert loaded is not None
        assert loaded.sample_error_sse is not None
        self.assertEqual(loaded.sample_error_sse.shape, (2,))
        torch.testing.assert_close(
            loaded.sample_reference_energy,
            torch.ones(2),
        )

    def test_inspection_reports_metrics_and_worst_evaluations(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            calibration, _, calibration_metadata, _ = self._write_calibration(
                root,
                4,
            )
            tensor_name = "blocks.0.attn.wq.weight"
            candidate = CandidateMeasurement(
                method="int8",
                action="int8",
                available=True,
                storage_bytes=16,
                error_by_eval_output=torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0]],
                    dtype=torch.float32,
                ),
            )
            metadata_path, _ = write_activation_audit_cache(
                root / "model",
                root / "source.safetensors",
                calibration,
                {tensor_name: (candidate,)},
                calibration_metadata_path=calibration_metadata,
                requested_methods=("int8",),
            )

            report = inspect_activation_audit(
                metadata_path,
                calibration,
                tensor_name=tensor_name,
                top_n=1,
            )

        layer_report = report["layers"][0]
        method_report = layer_report["methods"][0]
        self.assertEqual(layer_report["shape"], [2, 4])
        self.assertEqual(layer_report["evaluation_count"], 2)
        self.assertEqual(
            method_report["metrics"]["aggregate_observed_relative_sse"],
            2.5,
        )
        self.assertEqual(method_report["worst_evaluations"][0]["index"], 1)
        self.assertEqual(
            method_report["worst_evaluations"][0]["metadata"]["index"],
            1,
        )
        self.assertEqual(
            report["rankings"]["most_temporally_fragile"][0]["tensor_name"],
            tensor_name,
        )

    def test_round_trip_preserves_measurements_and_shared_denominators(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            calibration, layer, calibration_metadata, calibration_tensors = (
                self._write_calibration(root, 4)
            )
            descriptor = self._descriptor(2, 4)
            weights = torch.tensor(
                [[-1.0, 0.5, 1.0, 2.0], [3.0, -2.0, 0.25, 1.5]],
                dtype=torch.bfloat16,
            )
            methods = ("bf16", "int8", "int8_convrot")
            candidates = measure_activation_candidates(
                "blocks.0.attn.wq.weight",
                descriptor,
                weights,
                layer,
                methods,
            )
            metadata_path, tensors_path = write_activation_audit_cache(
                root / "model",
                root / "source.safetensors",
                calibration,
                {"blocks.0.attn.wq.weight": candidates},
                calibration_metadata_path=calibration_metadata,
                calibration_stats_path=calibration_tensors,
                requested_methods=methods,
            )
            cache = ActivationAuditCache.load(
                metadata_path,
                calibration=calibration,
            )
            document = json.loads(metadata_path.read_text(encoding="utf-8"))
            stored_tensors = load_file(str(tensors_path), device="cpu")

        self.assertEqual(
            (metadata_path.name, tensors_path.name),
            ("model.activation-audit.json", "model.activation-audit.safetensors"),
        )
        self.assertEqual(document["format"], ACTIVATION_AUDIT_FORMAT)
        self.assertEqual(document["format_version"], ACTIVATION_AUDIT_VERSION)
        self.assertEqual(tuple(document["requested_methods"]), methods)
        self.assertEqual(set(stored_tensors), {"m000000", "m000001"})
        self.assertNotIn("eval_sum_y2", stored_tensors)
        self.assertEqual(cache.calibration_session_id, "audit-session")
        self.assertFalse(cache.get("blocks.0.attn.wq.weight", "int8_convrot").available)
        self.assertEqual(
            cache.get("blocks.0.attn.wq.weight", "int8").error_by_eval_output.dtype,
            torch.float32,
        )
        torch.testing.assert_close(
            cache.get("blocks.0.attn.wq.weight", "int8").error_by_eval_output,
            candidates[1].error_by_eval_output,
        )

    def test_run_audit_streams_matching_source_weights(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            calibration, _, calibration_metadata, calibration_tensors = (
                self._write_calibration(root, 4)
            )
            weights = torch.tensor(
                [[-1.0, 0.5, 1.0, 2.0], [3.0, -2.0, 0.25, 1.5]],
                dtype=torch.bfloat16,
            )
            source_path = root / "source.safetensors"
            save_file(
                {"blocks.0.attn.wq.weight": weights},
                str(source_path),
            )
            progress: list[tuple[int, int, str]] = []
            metadata_path, _ = run_activation_audit(
                source_path,
                calibration,
                root / "model",
                calibration_metadata_path=calibration_metadata,
                calibration_stats_path=calibration_tensors,
                requested_methods=("bf16", "int8"),
                on_tensor_started=lambda index, total, name: progress.append(
                    (index, total, name)
                ),
            )
            cache = ActivationAuditCache.load(
                metadata_path,
                calibration=calibration,
            )

        self.assertEqual(progress, [(1, 1, "blocks.0.attn.wq.weight")])
        self.assertIsNotNone(cache.get("blocks.0.attn.wq.weight", "int8"))

    def test_run_audit_records_unambiguous_bias_free_energy(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            calibration, _, calibration_metadata, calibration_tensors = (
                self._write_calibration(root, 4)
            )
            tensor_name = "blocks.0.attn.wq.weight"
            source_path = root / "source.safetensors"
            save_file(
                {
                    tensor_name: torch.ones((2, 4), dtype=torch.bfloat16),
                    "blocks.0.attn.wq.bias": torch.tensor(
                        [0.5, -0.5],
                        dtype=torch.bfloat16,
                    ),
                },
                str(source_path),
            )

            metadata_path, _ = run_activation_audit(
                source_path,
                calibration,
                root / "model",
                calibration_metadata_path=calibration_metadata,
                calibration_stats_path=calibration_tensors,
                requested_methods=("bf16",),
            )
            cache = ActivationAuditCache.load(
                metadata_path,
                calibration=calibration,
            )
            score = score_activation_audit(
                metadata_path,
                calibration,
                metric="aggregate_bias_free_observed_relative_sse",
            )
            inspection = inspect_activation_audit(metadata_path, calibration)

        self.assertEqual(cache.bias_names[tensor_name], "blocks.0.attn.wq.bias")
        torch.testing.assert_close(
            cache.bias_free_output_energy[tensor_name],
            torch.full((2, 2), 1.25),
        )
        self.assertEqual(score["results"][0]["objective_cost"], 0.0)
        self.assertTrue(inspection["layers"][0]["bias"]["available"])
        self.assertEqual(
            inspection["layers"][0]["methods"][0]["bias_free_metrics"][
                "aggregate_bias_free_observed_relative_sse"
            ],
            0.0,
        )

    def test_scoring_reuses_cache_without_measurement_calls(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            calibration, layer, calibration_metadata, calibration_tensors = (
                self._write_calibration(root, 4)
            )
            candidates = measure_activation_candidates(
                "blocks.0.attn.wq.weight",
                self._descriptor(2, 4),
                torch.ones((2, 4), dtype=torch.bfloat16),
                layer,
                ("bf16", "int8"),
            )
            metadata_path, _ = write_activation_audit_cache(
                root / "model",
                root / "source.safetensors",
                calibration,
                {"blocks.0.attn.wq.weight": candidates},
                calibration_metadata_path=calibration_metadata,
                calibration_stats_path=calibration_tensors,
                requested_methods=("bf16", "int8"),
            )
            with patch(
                "potatoforge.audits.activation_audit.measure_activation_candidates",
                side_effect=AssertionError("scoring must not measure candidates"),
            ):
                report = score_activation_audit(
                    metadata_path,
                    calibration,
                    metric="eval_cvar20_observed_relative_sse",
                )

        self.assertEqual(
            report["objective"]["metric"],
            "eval_cvar20_observed_relative_sse",
        )
        self.assertEqual(report["summary"]["available_candidate_count"], 2)
        self.assertEqual(
            [result["objective_cost"] for result in report["results"]],
            [0.0, 0.0],
        )

    def test_cache_can_reload_when_every_requested_method_is_unavailable(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            calibration, layer, calibration_metadata, _ = self._write_calibration(
                root,
                4,
            )
            method = "int8_convrot"
            candidate = measure_activation_candidates(
                "blocks.0.attn.wq.weight",
                self._descriptor(2, 4),
                torch.ones((2, 4), dtype=torch.bfloat16),
                layer,
                (method,),
            )
            metadata_path, tensors_path = write_activation_audit_cache(
                root / "unsupported.activation-audit.json",
                root / "source.safetensors",
                calibration,
                {"blocks.0.attn.wq.weight": candidate},
                calibration_metadata_path=calibration_metadata,
                requested_methods=(method,),
            )
            cache = ActivationAuditCache.load(
                metadata_path,
                calibration=calibration,
            )
            stored_tensors = load_file(str(tensors_path), device="cpu")

        self.assertEqual(stored_tensors, {})
        loaded = cache.get("blocks.0.attn.wq.weight", method)
        assert loaded is not None
        self.assertFalse(loaded.available)
        self.assertIn("divisible", loaded.unavailable_reason or "")

    def test_rejects_wrong_session_and_leaves_no_pair_after_validation_failure(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            calibration, layer, calibration_metadata, _ = self._write_calibration(
                root,
                4,
            )
            candidates = measure_activation_candidates(
                "blocks.0.attn.wq.weight",
                self._descriptor(2, 4),
                torch.ones((2, 4), dtype=torch.bfloat16),
                layer,
                ("bf16",),
            )
            metadata_path, _ = write_activation_audit_cache(
                root / "model.activation-audit.json",
                root / "source.safetensors",
                calibration,
                {"blocks.0.attn.wq.weight": candidates},
                calibration_metadata_path=calibration_metadata,
                requested_methods=("bf16",),
            )
            other_root = root / "other"
            other_root.mkdir()
            other_calibration, _, _, _ = self._write_calibration(
                other_root,
                4,
                session_id="different-session",
            )

            with self.assertRaisesRegex(ValueError, "session_id"):
                ActivationAuditCache.load(
                    metadata_path,
                    calibration=other_calibration,
                )

            bad_candidate = CandidateMeasurement(
                method="bf16",
                action="keep",
                available=True,
                storage_bytes=8,
                error_by_eval_output=torch.zeros((1, 1)),
            )
            bad_metadata, bad_tensors = activation_audit_pair_paths(root / "bad")
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                write_activation_audit_cache(
                    root / "bad",
                    root / "source.safetensors",
                    calibration,
                    {"blocks.0.attn.wq.weight": (bad_candidate,)},
                    calibration_metadata_path=calibration_metadata,
                    requested_methods=("bf16",),
                )

        self.assertFalse(bad_metadata.exists())
        self.assertFalse(bad_tensors.exists())

    def _write_calibration(
        self,
        root: Path,
        input_features: int,
        session_id: str = "audit-session",
        include_samples: bool = False,
        evaluation_sample_counts: tuple[int, int] = (1, 1),
    ) -> tuple[object, object, Path, Path]:
        tensor_name = "blocks.0.attn.wq.weight"
        eval_sum_x2 = torch.arange(
            1,
            input_features + 1,
            dtype=torch.float32,
        ).repeat(2, 1)
        metadata = {
            "format": "potatoforge_activation_calibration",
            "version": 2,
            "session_id": session_id,
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
                    "sample_count": sum(evaluation_sample_counts),
                    "invocation_count": 2,
                    "stats_key": f"{tensor_name}.sum_x2",
                    "eval_sum_x_key": f"{tensor_name}.eval_sum_x",
                    "eval_sum_x2_key": f"{tensor_name}.eval_sum_x2",
                    "eval_max_abs_x_key": f"{tensor_name}.eval_max_abs_x",
                    "eval_sum_y_key": f"{tensor_name}.eval_sum_y",
                    "eval_sum_y2_key": f"{tensor_name}.eval_sum_y2",
                    "eval_sample_count_key": f"{tensor_name}.eval_sample_count",
                    "eval_invocation_count_key": f"{tensor_name}.eval_invocation_count",
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
                f"{tensor_name}.eval_sum_x": torch.zeros((2, input_features)),
                f"{tensor_name}.eval_sum_x2": eval_sum_x2,
                f"{tensor_name}.eval_max_abs_x": torch.zeros(
                    (2, input_features)
                ),
                f"{tensor_name}.eval_sum_y": torch.zeros((2, 2)),
                f"{tensor_name}.eval_sum_y2": torch.ones((2, 2)),
                f"{tensor_name}.eval_sample_count": torch.tensor(
                    evaluation_sample_counts,
                    dtype=torch.int64,
                ),
                f"{tensor_name}.eval_invocation_count": torch.ones(
                    2,
                    dtype=torch.int64,
                ),
                **(
                    {
                        f"{tensor_name}.sample_x": torch.eye(
                            input_features,
                            dtype=torch.float32,
                        ).reshape(
                            2,
                            1,
                            input_features,
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
        layer = calibration.get(tensor_name)
        return calibration, layer, metadata_path, tensors_path

    @staticmethod
    def _descriptor(out_features: int, in_features: int) -> TensorDescriptor:
        return {
            "dtype": "BF16",
            "shape": [out_features, in_features],
            "data_offsets": [0, out_features * in_features * 2],
        }


if __name__ == "__main__":
    unittest.main()
