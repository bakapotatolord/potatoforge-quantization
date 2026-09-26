import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from potatoforge.audits.activation_audit import (
    ActivationAuditCache,
    ActivationMetricCapabilityError,
    inspect_activation_audit,
    run_activation_audit,
    score_activation_audit,
)
from potatoforge.calibration import (
    ActivationCalibration,
    LayerCalibration,
)
from potatoforge.calibration.activation import merge_activation_calibrations


_TENSOR_NAME = "blocks.0.attn.wq.weight"
_WEIGHTS = torch.tensor(
    [[0.13, -0.22, 0.31], [-0.4, 0.21, 0.07]],
    dtype=torch.float32,
)
_BIAS = torch.tensor([0.1, -0.2], dtype=torch.float32)
_INPUTS = (
    torch.arange(45, dtype=torch.float32).reshape(5, 3, 3) / 13.0 - 1.0
)
_OUTPUTS = torch.matmul(_INPUTS, _WEIGHTS.transpose(0, 1)) + _BIAS
_EVAL_SUM_X = _INPUTS.sum(dim=1)
_EVAL_SUM_X2 = _INPUTS.square().sum(dim=1)
_EVAL_MAX_ABS_X = _INPUTS.abs().amax(dim=1)
_EVAL_SUM_Y = _OUTPUTS.sum(dim=1)
_EVAL_SUM_Y2 = _OUTPUTS.square().sum(dim=1)
_EVAL_OUTPUT_ENERGY = _EVAL_SUM_Y2.sum(dim=1)


def _write_pair(
    directory: Path,
    version: int,
    collection_mode: str | None = None,
    *,
    name: str | None = None,
    tensor_updates: dict[str, torch.Tensor] | None = None,
) -> tuple[Path, Path]:
    stem = name or f"v{version}-{collection_mode or 'legacy'}"
    metadata_path = directory / f"{stem}.json"
    tensors_path = directory / f"{stem}.safetensors"
    evaluations = [
        {
            "evaluation_index": index,
            "time_parameter_name": "timestep",
            "time_value": 1.0 - index * 0.1,
            "time_value_truncated": False,
            "root_input_sum_x2": float(index + 1),
            "root_output_sum_y2": float(index + 11),
        }
        for index in range(_INPUTS.shape[0])
    ]
    prefix = _TENSOR_NAME
    layer: dict[str, object] = {
        "input_features": 3,
        "output_features": 2,
        "sample_count": int(_INPUTS.shape[0] * _INPUTS.shape[1]),
        "invocation_count": int(_INPUTS.shape[0]),
    }
    tensors: dict[str, torch.Tensor] = {}
    if version == 1:
        layer.update(
            {
                "stats_key": f"{prefix}.sum_x2",
                "eval_sum_x_key": f"{prefix}.eval_sum_x",
                "eval_sum_x2_key": f"{prefix}.eval_sum_x2",
                "eval_max_abs_x_key": f"{prefix}.eval_max_abs_x",
                "eval_sum_y_key": f"{prefix}.eval_sum_y",
                "eval_sum_y2_key": f"{prefix}.eval_sum_y2",
                "eval_sample_count_key": f"{prefix}.eval_sample_count",
                "eval_invocation_count_key": f"{prefix}.eval_invocation_count",
                "sample_x_key": f"{prefix}.sample_x",
                "sample_x_valid_key": f"{prefix}.sample_x_valid",
            }
        )
        tensors.update(
            {
                f"{prefix}.sum_x2": _EVAL_SUM_X2.sum(dim=0),
                f"{prefix}.eval_sum_x": _EVAL_SUM_X,
                f"{prefix}.eval_sum_x2": _EVAL_SUM_X2,
                f"{prefix}.eval_max_abs_x": _EVAL_MAX_ABS_X,
                f"{prefix}.eval_sum_y": _EVAL_SUM_Y,
                f"{prefix}.eval_sum_y2": _EVAL_SUM_Y2,
                f"{prefix}.eval_sample_count": torch.full((5,), 3, dtype=torch.int64),
                f"{prefix}.eval_invocation_count": torch.ones(5, dtype=torch.int64),
                f"{prefix}.sample_x": _INPUTS,
                f"{prefix}.sample_x_valid": torch.full((5,), 3, dtype=torch.int64),
                "__pf__.root_input_sum_x2": torch.arange(1, 6, dtype=torch.float32),
                "__pf__.root_input_valid": torch.ones(5, dtype=torch.bool),
                "__pf__.root_output_sum_y2": torch.arange(11, 16, dtype=torch.float32),
                "__pf__.root_output_valid": torch.ones(5, dtype=torch.bool),
            }
        )
    elif collection_mode == "cvar":
        layer.update(
            {
                "eval_sum_x2_key": f"{prefix}.eval_sum_x2",
                "eval_output_energy_key": f"{prefix}.eval_output_energy",
            }
        )
        tensors.update(
            {
                f"{prefix}.eval_sum_x2": _EVAL_SUM_X2,
                f"{prefix}.eval_output_energy": _EVAL_OUTPUT_ENERGY,
            }
        )
    else:
        layer.update(
            {
                "eval_sum_x_key": f"{prefix}.eval_sum_x",
                "eval_sum_x2_key": f"{prefix}.eval_sum_x2",
                "eval_max_abs_x_key": f"{prefix}.eval_max_abs_x",
                "eval_sum_y_key": f"{prefix}.eval_sum_y",
                "eval_sum_y2_key": f"{prefix}.eval_sum_y2",
                "eval_sample_count_key": f"{prefix}.eval_sample_count",
                "eval_invocation_count_key": f"{prefix}.eval_invocation_count",
                "sample_x_key": f"{prefix}.sample_x",
                "sample_x_valid_key": f"{prefix}.sample_x_valid",
            }
        )
        tensors.update(
            {
                f"{prefix}.eval_sum_x": _EVAL_SUM_X,
                f"{prefix}.eval_sum_x2": _EVAL_SUM_X2,
                f"{prefix}.eval_max_abs_x": _EVAL_MAX_ABS_X,
                f"{prefix}.eval_sum_y": _EVAL_SUM_Y,
                f"{prefix}.eval_sum_y2": _EVAL_SUM_Y2,
                f"{prefix}.eval_sample_count": torch.full((5,), 3, dtype=torch.int64),
                f"{prefix}.eval_invocation_count": torch.ones(5, dtype=torch.int64),
                f"{prefix}.sample_x": _INPUTS,
                f"{prefix}.sample_x_valid": torch.full((5,), 3, dtype=torch.int64),
            }
        )

    metadata: dict[str, object] = {
        "format": "potatoforge_activation_calibration",
        "version": version,
        "session_id": "shared-session",
        "session_name": "test",
        "baseline_label": "bf16",
        "activation_basis": "logical_linear_input",
        "activation_axis": "last_dimension",
        "layer_count": 1,
        "evaluation_count": 5,
        "evaluations": evaluations,
        "layers": {prefix: layer},
    }
    if collection_mode is not None:
        metadata["collection_mode"] = collection_mode
    if tensor_updates:
        tensors.update(tensor_updates)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    save_file(tensors, str(tensors_path))
    return metadata_path, tensors_path


class TestActivationCalibrationV2(unittest.TestCase):
    def test_cvar_reader_keeps_compact_energy_and_derives_input_aggregate(self) -> None:
        with TemporaryDirectory() as directory:
            metadata_path, tensors_path = _write_pair(Path(directory), 2, "cvar")
            calibration = ActivationCalibration.load(metadata_path)
            layer = calibration.get(_TENSOR_NAME)
            self.assertIsInstance(layer, LayerCalibration)
            assert isinstance(layer, LayerCalibration)
            with safe_open(str(tensors_path), framework="pt", device="cpu") as file:
                keys = set(file.keys())

        self.assertEqual(calibration.version, 2)
        self.assertEqual(calibration.collection_mode, "cvar")
        torch.testing.assert_close(layer.eval_output_energy, _EVAL_OUTPUT_ENERGY)
        torch.testing.assert_close(layer.aggregate_sum_x2, _EVAL_SUM_X2.sum(dim=0))
        self.assertIsNone(layer.eval_sum_y2)
        self.assertIsNone(layer.eval_sum_y)
        self.assertIsNone(layer.eval_sum_x)
        self.assertIsNone(layer.sample_x)
        self.assertIsNone(layer.eval_sample_counts)
        self.assertFalse(calibration.supports_sampled_metrics)
        self.assertFalse(calibration.supports_bias_free_metrics)
        self.assertNotIn(f"{_TENSOR_NAME}.sum_x2", keys)
        self.assertNotIn("__pf__.root_input_sum_x2", keys)
        self.assertEqual(calibration.evaluations[0].root_input_sum_x2, 1.0)
        self.assertEqual(calibration.evaluations[0].timestep, 1.0)

    def test_full_reader_derives_energy_and_keeps_optional_statistics(self) -> None:
        with TemporaryDirectory() as directory:
            metadata_path, tensors_path = _write_pair(Path(directory), 2, "full")
            calibration = ActivationCalibration.load(metadata_path)
            layer = calibration.get(_TENSOR_NAME)
            self.assertIsInstance(layer, LayerCalibration)
            assert isinstance(layer, LayerCalibration)
            with safe_open(str(tensors_path), framework="pt", device="cpu") as file:
                keys = set(file.keys())

        torch.testing.assert_close(layer.eval_output_energy, _EVAL_OUTPUT_ENERGY)
        torch.testing.assert_close(layer.aggregate_sum_x2, _EVAL_SUM_X2.sum(dim=0))
        torch.testing.assert_close(layer.eval_sum_x, _EVAL_SUM_X)
        torch.testing.assert_close(layer.eval_max_abs_x, _EVAL_MAX_ABS_X)
        torch.testing.assert_close(layer.sample_x, _INPUTS.reshape(-1, 3))
        self.assertTrue(layer.supports_bias_free_metrics)
        self.assertTrue(layer.supports_sampled_metrics)
        self.assertNotIn(f"{_TENSOR_NAME}.sum_x2", keys)
        self.assertNotIn("__pf__.root_output_sum_y2", keys)

    def test_v2_requires_a_valid_mode_and_checks_cvar_energy(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path, _ = _write_pair(root, 2, "cvar")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.pop("collection_mode")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "collection_mode"):
                ActivationCalibration.load(metadata_path)

            metadata["collection_mode"] = "compact"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "collection_mode"):
                ActivationCalibration.load(metadata_path)

        with TemporaryDirectory() as directory:
            metadata_path, _ = _write_pair(
                Path(directory),
                2,
                "cvar",
                tensor_updates={
                    f"{_TENSOR_NAME}.eval_output_energy": torch.tensor(
                        [-1e-6, 2.0, 3.0, 4.0, 5.0]
                    )
                },
            )
            layer = ActivationCalibration.load(metadata_path).get(_TENSOR_NAME)
            self.assertIsInstance(layer, LayerCalibration)
            assert isinstance(layer, LayerCalibration)
            self.assertEqual(float(layer.eval_output_energy[0].item()), 0.0)

        with TemporaryDirectory() as directory:
            metadata_path, _ = _write_pair(
                Path(directory),
                2,
                "cvar",
                tensor_updates={
                    f"{_TENSOR_NAME}.eval_output_energy": torch.full((5,), -1.0)
                },
            )
            with self.assertRaisesRegex(ValueError, "non-negative"):
                ActivationCalibration.load(metadata_path)

    def test_v2_observed_metrics_match_v1_and_full_only_metrics_are_gated(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pair_paths = {
                "v1": _write_pair(root, 1),
                "full": _write_pair(root, 2, "full"),
                "cvar": _write_pair(root, 2, "cvar"),
            }
            source_path = root / "source.safetensors"
            save_file(
                {
                    _TENSOR_NAME: _WEIGHTS,
                    "blocks.0.attn.wq.bias": _BIAS,
                },
                str(source_path),
            )
            calibrations = {
                name: ActivationCalibration.load(paths[0])
                for name, paths in pair_paths.items()
            }
            caches = {}
            for name, (metadata_path, _) in pair_paths.items():
                cache_path = root / f"{name}.audit.json"
                run_activation_audit(
                    source_path,
                    calibrations[name],
                    cache_path,
                    calibration_metadata_path=metadata_path,
                    requested_methods=("int8",),
                )
                caches[name] = ActivationAuditCache.load(
                    cache_path,
                    calibration=calibrations[name],
                )

            observed_metrics = (
                "aggregate_observed_relative_sse",
                "eval_mean_observed_relative_sse",
                "eval_p95_observed_relative_sse",
                "eval_cvar20_observed_relative_sse",
                "channel_p95_global_contribution",
                "channel_cvar20_global_contribution",
            )
            for metric in observed_metrics:
                with self.subTest(metric=metric):
                    scores = [
                        score_activation_audit(
                            caches[name], calibrations[name], metric=metric
                        )["results"][0]["objective_cost"]
                        for name in ("v1", "full", "cvar")
                    ]
                    self.assertAlmostEqual(scores[0], scores[1], places=6)
                    self.assertAlmostEqual(scores[0], scores[2], places=6)

            full_only_metrics = (
                "eval_cvar20_bias_free_observed_relative_sse",
                "sampled_exact_relative_sse",
                "sampled_cvar20_relative_sse",
            )
            for metric in full_only_metrics:
                with self.subTest(metric=metric):
                    legacy_score = score_activation_audit(
                        caches["v1"], calibrations["v1"], metric=metric
                    )["results"][0]["objective_cost"]
                    full_score = score_activation_audit(
                        caches["full"], calibrations["full"], metric=metric
                    )["results"][0]["objective_cost"]
                    self.assertAlmostEqual(legacy_score, full_score, places=6)
                    with self.assertRaises(ActivationMetricCapabilityError) as error:
                        score_activation_audit(
                            caches["cvar"], calibrations["cvar"], metric=metric
                        )
                    self.assertIn("in 'cvar' mode", str(error.exception))

            with self.assertRaisesRegex(ValueError, "collection_mode"):
                ActivationAuditCache.load(
                    root / "cvar.audit.json",
                    calibration=calibrations["full"],
                )
            with self.assertRaisesRegex(ValueError, "versions do not match"):
                ActivationAuditCache.load(
                    root / "v1.audit.json",
                    calibration=calibrations["full"],
                )
            self.assertEqual(caches["cvar"].calibration_collection_mode, "cvar")
            self.assertEqual(
                caches["cvar"].error_representation,
                "eval_output_marginals",
            )
            self.assertEqual(
                caches["full"].error_representation,
                "eval_output_matrix",
            )
            cvar_candidate = caches["cvar"].get(_TENSOR_NAME, "int8")
            full_candidate = caches["full"].get(_TENSOR_NAME, "int8")
            assert cvar_candidate is not None
            assert full_candidate is not None
            assert full_candidate.error_by_eval_output is not None
            assert cvar_candidate.error_by_eval is not None
            assert cvar_candidate.error_by_output is not None
            self.assertIsNone(cvar_candidate.error_by_eval_output)
            torch.testing.assert_close(
                cvar_candidate.error_by_eval,
                full_candidate.error_by_eval_output.sum(dim=1),
            )
            torch.testing.assert_close(
                cvar_candidate.error_by_output,
                full_candidate.error_by_eval_output.sum(dim=0),
            )
            cvar_audit_document = json.loads(
                (root / "cvar.audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                cvar_audit_document["error_representation"],
                "eval_output_marginals",
            )
            with safe_open(
                str((root / "cvar.audit.safetensors")),
                framework="pt",
                device="cpu",
            ) as file:
                self.assertEqual(set(file.keys()), {"e000000", "o000000"})
            self.assertFalse(
                cvar_audit_document["optional_metric_families"]["sampled_metrics"][
                    "available"
                ]
            )
            inspection = inspect_activation_audit(
                caches["cvar"],
                calibrations["cvar"],
            )

        self.assertEqual(inspection["calibration_collection_mode"], "cvar")
        self.assertFalse(
            inspection["optional_metric_families"]["sampled_metrics"]["available"]
        )
        self.assertEqual(
            inspection["layers"][0]["output_activation_energy"],
            float(_EVAL_OUTPUT_ENERGY.sum().item()),
        )

    def test_legacy_v2_cvar_matrix_cache_remains_loadable(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path, _ = _write_pair(root, 2, "cvar")
            calibration = ActivationCalibration.load(metadata_path)
            source_path = root / "source.safetensors"
            save_file({_TENSOR_NAME: _WEIGHTS}, str(source_path))
            cache_path = root / "legacy.audit.json"
            cache_metadata, cache_tensors = run_activation_audit(
                source_path,
                calibration,
                cache_path,
                calibration_metadata_path=metadata_path,
                requested_methods=("int8",),
            )
            document = json.loads(cache_metadata.read_text(encoding="utf-8"))
            document.pop("error_representation")
            method = document["layers"][_TENSOR_NAME]["methods"]["int8"]
            method.pop("error_by_eval_key")
            method.pop("error_by_output_key")
            method["error_tensor_key"] = "m000000"
            matrix = torch.arange(10, dtype=torch.float32).reshape(5, 2)
            save_file({"m000000": matrix}, str(cache_tensors))
            cache_metadata.write_text(json.dumps(document), encoding="utf-8")

            cache = ActivationAuditCache.load(
                cache_metadata,
                calibration=calibration,
            )
            candidate = cache.get(_TENSOR_NAME, "int8")

        self.assertEqual(cache.error_representation, "eval_output_matrix")
        assert candidate is not None
        self.assertIsNone(candidate.error_by_eval)
        self.assertIsNone(candidate.error_by_output)
        torch.testing.assert_close(candidate.error_by_eval_output, matrix)

    def test_merge_rules_and_v2_payloads(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cvar_pair = _write_pair(root, 2, "cvar", name="cvar")
            full_pair = _write_pair(root, 2, "full", name="full")
            v1_pair = _write_pair(root, 1, name="v1")

            with self.assertRaisesRegex(ValueError, "modes 'cvar' and 'full'"):
                merge_activation_calibrations(
                    [cvar_pair[0], full_pair[0]],
                    root / "mixed-mode.json",
                )
            with self.assertRaisesRegex(ValueError, "versions 1 and 2"):
                merge_activation_calibrations(
                    [v1_pair[0], cvar_pair[0]],
                    root / "mixed-version.json",
                )

            for mode, pair in (("cvar", cvar_pair), ("full", full_pair)):
                output = root / f"merged-{mode}.json"
                merge_activation_calibrations([pair[0], pair[0]], output)
                merged = ActivationCalibration.load(output)
                layer = merged.get(_TENSOR_NAME)
                self.assertIsInstance(layer, LayerCalibration)
                assert isinstance(layer, LayerCalibration)
                self.assertEqual(merged.evaluation_count, 10)
                self.assertEqual(merged.collection_mode, mode)
                with safe_open(
                    str(output.with_suffix(".safetensors")),
                    framework="pt",
                    device="cpu",
                ) as file:
                    keys = set(file.keys())
                self.assertNotIn(f"{_TENSOR_NAME}.sum_x2", keys)
                self.assertFalse(any(key.startswith("__pf__.root_") for key in keys))
                if mode == "cvar":
                    torch.testing.assert_close(
                        layer.eval_output_energy,
                        torch.cat((_EVAL_OUTPUT_ENERGY, _EVAL_OUTPUT_ENERGY)),
                    )
                else:
                    torch.testing.assert_close(
                        layer.eval_sum_x2,
                        torch.cat((_EVAL_SUM_X2, _EVAL_SUM_X2)),
                    )
                    self.assertEqual(layer.sample_x.shape[0], 30)


if __name__ == "__main__":
    unittest.main()
