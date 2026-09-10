import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from potatoforge.cli import app
from potatoforge.audits.activation_profiles import ActivationProfileResult
from potatoforge.audits.profile_optimizer import OptimizedProfile


class TestActivationCli(unittest.TestCase):
    def test_audit_loads_activation_calibration(self) -> None:
        document = {
            "results": [],
            "selection": {"dtype": "BF16"},
            "summary": {
                "audited_layer_count": 0,
                "skipped_tensor_count": 0,
            },
        }
        calibration = object()
        with (
            patch(
                "potatoforge.cli.ActivationCalibration.load",
                return_value=calibration,
            ) as load_mock,
            patch(
                "potatoforge.cli.audit_bf16_source",
                return_value=document,
            ) as audit_mock,
            patch("potatoforge.cli._write_json"),
            patch("potatoforge.cli.print_weight_audit_table"),
        ):
            result = CliRunner().invoke(
                app,
                [
                    "audit",
                    "model.safetensors",
                    "--output",
                    "audit.json",
                    "--activation-calibration",
                    "calibration.json",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        load_mock.assert_called_once_with(Path("calibration.json"))
        self.assertIs(
            audit_mock.call_args.kwargs["activation_calibration"],
            calibration,
        )

    def test_audit_passes_activation_probe_output(self) -> None:
        document = {
            "results": [],
            "selection": {"dtype": "BF16"},
            "summary": {
                "audited_layer_count": 0,
                "skipped_tensor_count": 0,
            },
        }
        with (
            patch(
                "potatoforge.cli.audit_bf16_source",
                return_value=document,
            ) as audit_mock,
            patch("potatoforge.cli._write_json"),
            patch("potatoforge.cli.print_weight_audit_table"),
        ):
            result = CliRunner().invoke(
                app,
                [
                    "audit",
                    "model.safetensors",
                    "--output",
                    "audit.json",
                    "--method",
                    "convrot_w4a4",
                    "--activation-probe-output",
                    "probe",
                    "--overwrite",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            audit_mock.call_args.kwargs["audit_method"],
            "convrot_w4a4",
        )
        self.assertEqual(
            audit_mock.call_args.kwargs["activation_probe_output"],
            Path("probe"),
        )
        self.assertTrue(
            audit_mock.call_args.kwargs["activation_probe_overwrite"]
        )

    def test_activation_score_writes_lightweight_report(self) -> None:
        report = {"summary": {"scored_tensor_count": 2}}
        with (
            patch(
                "potatoforge.cli.score_activation_probe",
                return_value=report,
            ) as score_mock,
            patch("potatoforge.cli._write_json") as write_mock,
        ):
            result = CliRunner().invoke(
                app,
                [
                    "activation-score",
                    "--probe-cache",
                    "probe.json",
                    "--activation-calibration",
                    "calibration.json",
                    "--output",
                    "scores.json",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        score_mock.assert_called_once_with(
            Path("probe.json"),
            Path("calibration.json"),
        )
        write_mock.assert_called_once_with(
            Path("scores.json"),
            report,
            False,
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

    def test_profile_from_activation_writes_profile_and_summary(self) -> None:
        optimized = OptimizedProfile(
            profile={
                "profile_id": "activation-relative",
                "default": "keep",
                "rules": (),
            },
            output_bytes=100,
            target_bytes=200,
            reconstruction_sse=0.0,
        )
        generated = ActivationProfileResult(
            optimized,
            {
                "metric": "relative_output_sse",
                "promoted_int8cr_tensor_count": 1,
            },
        )
        with (
            patch(
                "potatoforge.cli.generate_activation_profile",
                return_value=generated,
            ) as generate_mock,
            patch("potatoforge.cli.write_profile") as write_profile_mock,
            patch("potatoforge.cli._write_json") as write_json_mock,
        ):
            result = CliRunner().invoke(
                app,
                [
                    "profile-from-activation",
                    "--activation-score",
                    "score.json",
                    "--weight-audit",
                    "audit.json",
                    "--target-size-mib",
                    "7",
                    "--output",
                    "relative.json",
                    "--summary-output",
                    "relative-summary.json",
                    "--overwrite",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        generate_mock.assert_called_once_with(
            Path("score.json"),
            Path("audit.json"),
            target_bytes=7 * 1024**2,
            promotion_budget_bytes=None,
            profile_id="activation-relative",
            metric="relative_output_sse",
            include_regex=r"^blocks\.",
        )
        write_profile_mock.assert_called_once_with(
            Path("relative.json"),
            generated.optimized.profile,
            overwrite=True,
        )
        write_json_mock.assert_called_once_with(
            Path("relative-summary.json"),
            generated.summary,
            True,
        )

    def test_source_analyze_passes_activation_calibration(self) -> None:
        calibration = object()
        with (
            patch(
                "potatoforge.cli.ActivationCalibration.load",
                return_value=calibration,
            ) as load_mock,
            patch(
                "potatoforge.cli.audit_bf16_source",
                return_value={},
            ) as audit_mock,
            patch(
                "potatoforge.cli.read_source_model_header",
                return_value=object(),
            ),
            patch("potatoforge.cli.print_tensor_analysis"),
        ):
            result = CliRunner().invoke(
                app,
                [
                    "analyze",
                    "--source",
                    "model.safetensors",
                    "--tensor",
                    "blocks.0.attn.wq.weight",
                    "--activation-calibration",
                    "calibration.json",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        load_mock.assert_called_once_with(Path("calibration.json"))
        self.assertIs(
            audit_mock.call_args.kwargs["activation_calibration"],
            calibration,
        )


if __name__ == "__main__":
    unittest.main()
