import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from potatoforge.audits.activation_audit import ActivationAuditCache
from potatoforge.audits.activation_measurements import CandidateMeasurement
from potatoforge.audits.activation_profiles import (
    _validate_profile_semantics,
    generate_activation_cache_profile,
)
from potatoforge.audits.profile_compaction import compact_runtime_rules
from potatoforge.profiles import resolve_profile


class TestActivationProfiles(unittest.TestCase):
    def test_cache_profile_reuses_existing_optimizer_choices(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.safetensors"
            tensor_name = "blocks.0.attn.wq.weight"
            save_file(
                {tensor_name: torch.zeros((2, 256), dtype=torch.bfloat16)},
                str(source_path),
            )
            cache = ActivationAuditCache(
                source_model_path=str(source_path.resolve()),
                calibration_metadata_path=str((root / "calibration.json").resolve()),
                calibration_stats_path=str(
                    (root / "calibration.safetensors").resolve()
                ),
                calibration_session_id="session",
                calibration_version=1,
                baseline_label="bf16",
                requested_methods=("bf16", "int8"),
                measurements={
                    tensor_name: {
                        "bf16": CandidateMeasurement(
                            "bf16",
                            "keep",
                            True,
                            1024,
                            torch.zeros((1, 2)),
                        ),
                        "int8": CandidateMeasurement(
                            "int8",
                            "int8",
                            True,
                            600,
                            torch.ones((1, 2)),
                        ),
                    }
                },
                layer_shapes={tensor_name: (256, 2, 1)},
            )
            score_report = {
                "objective": {"metric_version": 1},
                "results": [
                    {
                        "tensor_name": tensor_name,
                        "method": "bf16",
                        "objective_cost": 0.0,
                    },
                    {
                        "tensor_name": tensor_name,
                        "method": "int8",
                        "objective_cost": 1.0,
                    },
                ],
            }
            with (
                patch(
                    "potatoforge.audits.activation_profiles.ActivationCalibration.load",
                    return_value=object(),
                ),
                patch(
                    "potatoforge.audits.activation_profiles.score_activation_audit",
                    return_value=score_report,
                ),
                patch(
                    "potatoforge.audits.activation_profiles.compact_runtime_rules",
                    wraps=compact_runtime_rules,
                ) as compact_rules,
            ):
                generated = generate_activation_cache_profile(
                    cache,
                    root / "calibration.json",
                    promotion_budget_bytes=0,
                )

        self.assertEqual(
            resolve_profile(generated.optimized.profile, tensor_name),
            "int8",
        )
        self.assertTrue(compact_rules.called)
        self.assertEqual(compact_rules.call_args.args[1], "keep")
        self.assertEqual(
            generated.summary["metric"],
            "aggregate_observed_relative_sse",
        )

    def test_compactor_uses_the_supplied_default_action(self) -> None:
        desired_actions = {
            "module.0.weight": "convrot_w4a4_mse",
            "module.1.weight": "convrot_w4a4_mse",
        }
        self.assertEqual(
            compact_runtime_rules(desired_actions, "convrot_w4a4_mse"),
            (),
        )

    def test_semantic_validation_reports_the_mismatched_action(self) -> None:
        profile = {
            "default": "convrot_w4a4",
            "rules": (
                {
                    "action": "int8_convrot",
                    "prefix": "solo.weight",
                    "suffixes": ("",),
                },
            ),
        }
        with self.assertRaisesRegex(
            ValueError,
            "solo\\.weight.*expected keep.*actual int8_convrot",
        ):
            _validate_profile_semantics(
                profile,
                {"solo.weight": "keep"},
            )


if __name__ == "__main__":
    unittest.main()
