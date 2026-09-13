import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from potatoforge.audits.activation_comparison import ACTIVATION_METRICS
from potatoforge.cli import app


class TestActivationCli(unittest.TestCase):
    def test_activation_audit_passes_cache_arguments(self) -> None:
        calibration = Mock(
            session_id="test-session",
            tensor_names=Mock(return_value=("blocks.0.attn.wq.weight",)),
        )
        with (
            patch(
                "potatoforge.cli.ActivationCalibration.load",
                return_value=calibration,
            ),
            patch(
                "potatoforge.cli.run_activation_audit",
                return_value=(Path("cache.json"), Path("cache.safetensors")),
            ) as audit_mock,
        ):
            result = CliRunner().invoke(
                app,
                [
                    "activation-audit",
                    "model.safetensors",
                    "--device",
                    "cuda",
                    "--activation-calibration",
                    "calibration.json",
                    "--output",
                    "cache",
                    "--method",
                    "int8, bf16",
                    "--tensor",
                    "blocks.0.attn.wq.weight",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        audit_mock.assert_called_once()
        self.assertEqual(
            audit_mock.call_args.args[:3],
            (
                Path("model.safetensors"),
                calibration,
                Path("cache"),
            ),
        )
        self.assertEqual(
            audit_mock.call_args.kwargs["requested_methods"],
            ("int8", "bf16"),
        )
        self.assertEqual(
            audit_mock.call_args.kwargs["tensor_names"],
            ("blocks.0.attn.wq.weight",),
        )
        self.assertEqual(audit_mock.call_args.kwargs["device"], "cuda")
        self.assertIsNone(audit_mock.call_args.kwargs["method_timings"])
        self.assertNotIn("Activation audit timing", result.output)

    def test_activation_audit_timing_option_reports(self) -> None:
        calibration = Mock(
            session_id="test-session",
            tensor_names=Mock(return_value=("blocks.0.attn.wq.weight",)),
        )
        with (
            patch(
                "potatoforge.cli.ActivationCalibration.load",
                return_value=calibration,
            ),
            patch(
                "potatoforge.cli.run_activation_audit",
                return_value=(Path("cache.json"), Path("cache.safetensors")),
            ) as audit_mock,
            patch("potatoforge.cli.perf_counter", side_effect=(10.0, 12.5)),
        ):
            result = CliRunner().invoke(
                app,
                [
                    "activation-audit",
                    "model.safetensors",
                    "--activation-calibration",
                    "calibration.json",
                    "--output",
                    "cache",
                    "--timing",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Activation audit timing", result.output)
        self.assertIn("total", result.output)
        self.assertEqual(audit_mock.call_args.kwargs["method_timings"], {})

    def test_activation_score_can_score_audit_cache(self) -> None:
        report = {"summary": {"available_candidate_count": 2}}
        with (
            patch(
                "potatoforge.cli.score_activation_audit",
                return_value=report,
            ) as score_mock,
            patch("potatoforge.cli._write_json"),
        ):
            result = CliRunner().invoke(
                app,
                [
                    "activation-score",
                    "--audit-cache",
                    "cache.json",
                    "--activation-calibration",
                    "calibration.json",
                    "--output",
                    "scores.json",
                    "--metric",
                    "eval_p95_observed_relative_sse",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        score_mock.assert_called_once_with(
            Path("cache.json"),
            Path("calibration.json"),
            metric="eval_p95_observed_relative_sse",
        )

    def test_activation_inspect_writes_report(self) -> None:
        report = {"summary": {"layer_count": 1}}
        with (
            patch(
                "potatoforge.cli.inspect_activation_audit",
                return_value=report,
            ) as inspect_mock,
            patch("potatoforge.cli._write_json") as write_mock,
        ):
            result = CliRunner().invoke(
                app,
                [
                    "activation-inspect",
                    "--audit-cache",
                    "cache.json",
                    "--activation-calibration",
                    "calibration.json",
                    "--tensor",
                    "blocks.0.attn.wq.weight",
                    "--top-n",
                    "3",
                    "--output",
                    "inspection.json",
                    "--overwrite",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        inspect_mock.assert_called_once_with(
            Path("cache.json"),
            Path("calibration.json"),
            tensor_name="blocks.0.attn.wq.weight",
            top_n=3,
        )
        write_mock.assert_called_once_with(
            Path("inspection.json"),
            report,
            True,
        )

    def test_activation_optimize_canonical_dispatches_to_optimizer(self) -> None:
        generated = SimpleNamespace(
            optimized=SimpleNamespace(
                profile={"profile_id": "activation-audit"},
                target_bytes=1024,
                output_bytes=768,
            ),
            summary={"candidate_count": 1},
        )
        with (
            patch(
                "potatoforge.cli.generate_activation_cache_profile",
                return_value=generated,
            ) as optimize_mock,
            patch("potatoforge.cli.write_profile"),
            patch("potatoforge.cli._write_json"),
        ):
            result = CliRunner().invoke(
                app,
                [
                    "activation-optimize",
                    "--audit-cache",
                    "cache.json",
                    "--activation-calibration",
                    "calibration.json",
                    "--target-size-gib",
                    "1",
                    "--output",
                    "profile.json",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        optimize_mock.assert_called_once_with(
            Path("cache.json"),
            Path("calibration.json"),
            source_path=None,
            target_bytes=1024**3,
            promotion_budget_bytes=None,
            profile_id="activation-audit",
            metric="aggregate_observed_relative_sse",
            allowed_methods=None,
            baseline_method="bf16",
            excluded_prefixes=(),
            excluded_suffixes=(),
        )

    def test_calibration_merge_passes_all_input_paths(self) -> None:
        with (
            patch(
                "potatoforge.cli.merge_activation_calibrations",
                return_value=(Path("merged.json"), Path("merged.safetensors")),
            ) as merge_mock,
        ):
            result = CliRunner().invoke(
                app,
                [
                    "calibration-merge",
                    "first.json",
                    "second.json",
                    "--output",
                    "merged",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        merge_mock.assert_called_once_with(
            [Path("first.json"), Path("second.json")],
            Path("merged"),
            overwrite=False,
        )

    def test_activation_compare_passes_all_profile_options(self) -> None:
        summary = {"workbook_path": "comparison.xlsx", "metric_count": 14}
        with patch(
            "potatoforge.cli.generate_activation_comparison_workbook",
            return_value=summary,
        ) as compare_mock:
            result = CliRunner().invoke(
                app,
                [
                    "activation-compare",
                    "--audit-cache",
                    "cache.json",
                    "--activation-calibration",
                    "calibration.json",
                    "--output",
                    "comparison.xlsx",
                    "--target-size-gib",
                    "8.5",
                    "--method",
                    "convrot_w4a4,int8_convrot",
                    "--baseline-method",
                    "convrot_w4a4",
                    "--exclude-prefix",
                    "first.,last.",
                    "--top-n",
                    "10",
                    "--overwrite",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        compare_mock.assert_called_once_with(
            Path("cache.json"),
            Path("calibration.json"),
            Path("comparison.xlsx"),
            source_path=None,
            target_bytes=int(8.5 * 1024**3),
            promotion_budget_bytes=None,
            allowed_methods=frozenset({"convrot_w4a4", "int8_convrot"}),
            baseline_method="convrot_w4a4",
            excluded_prefixes=("first.", "last."),
            excluded_suffixes=(),
            metrics=ACTIVATION_METRICS,
            top_n=10,
            overwrite=True,
        )

if __name__ == "__main__":
    unittest.main()
