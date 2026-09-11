import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, call, patch

from potatoforge.audits.activation_comparison import (
    generate_activation_comparison_workbook,
)
from potatoforge.audits.activation_profiles import ActivationProfileResult
from potatoforge.audits.profile_optimizer import OptimizedProfile


class TestActivationComparison(unittest.TestCase):
    def test_loads_shared_inputs_once_and_reuses_each_score(self) -> None:
        cache = Mock(
            source_model_path="model.safetensors",
            calibration_session_id="session",
            measurements={"blocks.0.weight": {}},
        )
        calibration = Mock()
        source_header = object()
        score_report = {"results": [], "objective": {"metric_version": 1}}
        generated = ActivationProfileResult(
            OptimizedProfile(
                {
                    "profile_id": "comparison",
                    "default": "keep",
                    "rules": (),
                },
                100,
                200,
                0.0,
            ),
            {
                "target_size_bytes": 200,
                "estimated_final_size_bytes": 100,
                "minimum_output_bytes": 100,
                "selected_action_counts": {"keep": 1},
                "profile_rule_count": 0,
                "layer_count": 1,
            },
        )
        with TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.xlsx"
            with (
                patch(
                    "potatoforge.audits.activation_comparison.ActivationCalibration.load",
                    return_value=calibration,
                ) as load_calibration,
                patch(
                    "potatoforge.audits.activation_comparison.ActivationAuditCache.load",
                    return_value=cache,
                ) as load_cache,
                patch(
                    "potatoforge.audits.activation_comparison.read_source_model_header",
                    return_value=source_header,
                ) as read_header,
                patch(
                    "potatoforge.audits.activation_comparison.score_activation_audit",
                    return_value=score_report,
                ) as score,
                patch(
                    "potatoforge.audits.activation_comparison.generate_activation_cache_profile",
                    return_value=generated,
                ) as generate_profile,
                patch(
                    "potatoforge.audits.activation_comparison._write_workbook"
                ) as write_workbook,
            ):
                result = generate_activation_comparison_workbook(
                    Path("cache.json"),
                    Path("calibration.json"),
                    output,
                    target_bytes=200,
                    metrics=("metric_a", "metric_b"),
                )

        load_calibration.assert_called_once_with(Path("calibration.json"))
        load_cache.assert_called_once_with(
            Path("cache.json"),
            calibration=calibration,
        )
        read_header.assert_called_once_with("model.safetensors")
        self.assertEqual(
            score.call_args_list,
            [
                call(cache, calibration, metric="metric_a"),
                call(cache, calibration, metric="metric_b"),
            ],
        )
        self.assertEqual(generate_profile.call_count, 2)
        for profile_call, metric in zip(
            generate_profile.call_args_list,
            ("metric_a", "metric_b"),
        ):
            self.assertIs(profile_call.kwargs["source_header"], source_header)
            self.assertIs(profile_call.kwargs["score_report"], score_report)
            self.assertEqual(profile_call.kwargs["metric"], metric)
        write_workbook.assert_called_once()
        self.assertEqual(result["metric_count"], 2)
        self.assertEqual(result["layer_count"], 1)


if __name__ == "__main__":
    unittest.main()
