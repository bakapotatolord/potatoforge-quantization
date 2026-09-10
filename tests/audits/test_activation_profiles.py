import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from potatoforge.audits.activation_profiles import (
    _runtime_profile,
    generate_activation_profile,
    load_activation_score,
)
from potatoforge.audits.profile_compaction import compact_runtime_rules
from potatoforge.audits.profile_optimizer import (
    estimate_profile_bytes,
    write_profile,
)
from potatoforge.audits.weight_audit import audit_bf16_source
from potatoforge.headers.source_header import read_source_model_header
from potatoforge.profiles import load_profile
from potatoforge.profiles import resolve_profile


class TestActivationProfiles(unittest.TestCase):
    def test_rejects_score_without_relative_metric_schema(self) -> None:
        with TemporaryDirectory() as directory:
            score_path = Path(directory) / "old-score.json"
            score_path.write_text(
                json.dumps(
                    {
                        "format": "potatoforge_activation_score",
                        "version": 1,
                        "results": [],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "relative_output_error is unavailable",
            ):
                load_activation_score(score_path)

    def test_relative_output_sse_promotes_by_squared_activation_error(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.safetensors"
            names = (
                "blocks.0.attn.wq.weight",
                "blocks.1.attn.wq.weight",
            )
            save_file(
                {
                    name: torch.ones((2, 256), dtype=torch.bfloat16)
                    for name in names
                },
                str(source_path),
            )
            audit = audit_bf16_source(source_path)
            header = read_source_model_header(source_path)
            baseline = {
                "profile_id": "baseline",
                "default": "keep",
                "rules": tuple(
                    {
                        "action": "convrot_w4a4",
                        "prefix": name,
                        "suffixes": ("",),
                    }
                    for name in names
                ),
            }
            promoted = {
                **baseline,
                "rules": (
                    {
                        "action": "int8_convrot",
                        "prefix": names[0],
                        "suffixes": ("",),
                    },
                    baseline["rules"][1],
                ),
            }
            score_path = root / "score.json"
            score_path.write_text(
                json.dumps(
                    {
                        "format": "potatoforge_activation_score",
                        "version": 2,
                        "reference_format": "int8_convrot",
                        "candidate_format": "convrot_w4a4",
                        "metric_basis": "logical_linear_input",
                        "results": [
                            {
                                "tensor_name": names[0],
                                "activation_status": "ok",
                                "relative_output_error": 0.1,
                                "relative_output_error_sq": 4.0,
                            },
                            {
                                "tensor_name": names[1],
                                "activation_status": "ok",
                                "relative_output_error": 0.9,
                                "relative_output_error_sq": 0.01,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            generated = generate_activation_profile(
                score_path,
                audit,
                target_bytes=estimate_profile_bytes(header, promoted),
                profile_id="relative-sse",
            )

        self.assertEqual(
            [
                rule["prefix"]
                for rule in generated.optimized.profile["rules"]
                if rule["action"] == "int8_convrot"
            ],
            [names[0]],
        )
        self.assertEqual(generated.summary["metric"], "relative_output_sse")
        self.assertEqual(
            generated.summary["total_relative_output_sse_before_promotion"],
            4.01,
        )

    def test_generated_profile_groups_common_prefixes_and_suffixes(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.safetensors"
            encoder_names = (
                "encoder.layers.0.attention.key.weight",
                "encoder.layers.0.attention.output.weight",
                "encoder.layers.1.attention.key.weight",
                "encoder.layers.1.attention.output.weight",
            )
            keep_names = (
                "decoder.layers.0.weight",
                "decoder.layers.1.weight",
            )
            names = encoder_names + keep_names
            save_file(
                {
                    name: torch.ones((2, 256), dtype=torch.bfloat16)
                    for name in names
                },
                str(source_path),
            )
            audit = audit_bf16_source(source_path)
            header = read_source_model_header(source_path)
            promoted = {
                "profile_id": "target",
                "default": "keep",
                "rules": tuple(
                    {
                        "action": "int8_convrot",
                        "prefix": name,
                        "suffixes": ("",),
                    }
                    for name in encoder_names
                ),
            }
            score_path = root / "score.json"
            score_path.write_text(
                json.dumps(
                    {
                        "format": "potatoforge_activation_score",
                        "version": 2,
                        "reference_format": "int8_convrot",
                        "candidate_format": "convrot_w4a4",
                        "metric_basis": "logical_linear_input",
                        "results": [
                            {
                                "tensor_name": name,
                                "activation_status": "ok",
                                "relative_output_error": 1.0,
                                "relative_output_error_sq": 1.0,
                            }
                            for name in encoder_names
                        ],
                    }
                ),
                encoding="utf-8",
            )

            generated = generate_activation_profile(
                score_path,
                audit,
                target_bytes=estimate_profile_bytes(header, promoted),
                profile_id="grouped",
                include_regex=r"^encoder\.",
            )

        rules = generated.optimized.profile["rules"]
        self.assertLess(len(rules), len(names))
        self.assertTrue(
            any(
                rule["prefix"] == "encoder.layers.0.attention."
                and set(rule["suffixes"]) == {"key.weight", "output.weight"}
                for rule in rules
            )
        )
        self.assertTrue(
            any(
                rule["prefix"] == "decoder.layers."
                and set(rule["suffixes"]) == {"0.weight", "1.weight"}
                for rule in rules
            )
        )
        for name in encoder_names:
            self.assertEqual(
                resolve_profile(generated.optimized.profile, name),
                "int8_convrot",
            )
        for name in keep_names:
            self.assertEqual(
                resolve_profile(generated.optimized.profile, name),
                "keep",
            )

    def test_grouped_profile_puts_specific_exceptions_before_broad_rules(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.safetensors"
            promoted_names = (
                "encoder.layers.0.attention.key.weight",
                "encoder.layers.0.attention.output.weight",
                "encoder.layers.0.attention.value.weight",
            )
            exception_name = "encoder.layers.0.attention.extra.key.weight"
            names = promoted_names + (exception_name,)
            save_file(
                {
                    name: torch.ones((2, 256), dtype=torch.bfloat16)
                    for name in names
                },
                str(source_path),
            )
            audit = audit_bf16_source(source_path)
            header = read_source_model_header(source_path)
            target = {
                "profile_id": "target",
                "default": "keep",
                "rules": (
                    *(
                        {
                            "action": "int8_convrot",
                            "prefix": name,
                            "suffixes": ("",),
                        }
                        for name in promoted_names
                    ),
                    {
                        "action": "convrot_w4a4",
                        "prefix": exception_name,
                        "suffixes": ("",),
                    },
                ),
            }
            score_path = root / "score.json"
            score_path.write_text(
                json.dumps(
                    {
                        "format": "potatoforge_activation_score",
                        "version": 2,
                        "reference_format": "int8_convrot",
                        "candidate_format": "convrot_w4a4",
                        "metric_basis": "logical_linear_input",
                        "results": [
                            {
                                "tensor_name": name,
                                "activation_status": "ok",
                                "relative_output_error": 1.0,
                                "relative_output_error_sq": 1.0,
                            }
                            for name in promoted_names
                        ],
                    }
                ),
                encoding="utf-8",
            )

            generated = generate_activation_profile(
                score_path,
                audit,
                target_bytes=estimate_profile_bytes(header, target),
                profile_id="precedence",
                include_regex=r"^encoder\.",
            )

        rules = generated.optimized.profile["rules"]
        self.assertEqual(rules[0]["action"], "convrot_w4a4")
        self.assertEqual(rules[0]["prefix"], exception_name)
        self.assertEqual(rules[1]["action"], "int8_convrot")
        self.assertEqual(rules[1]["prefix"], "encoder.layers.0.attention.")
        self.assertEqual(
            resolve_profile(generated.optimized.profile, exception_name),
            "convrot_w4a4",
        )
        for name in promoted_names:
            self.assertEqual(
                resolve_profile(generated.optimized.profile, name),
                "int8_convrot",
            )

    def test_compacted_profile_matches_verbose_profile_for_complete_namespace(self) -> None:
        default_action = "convrot_w4a4"
        desired_actions = {
            "blocks.1.attn.wq.weight": "int8_convrot",
            "blocks.10.attn.wq.weight": "keep",
            "blocks.11.attn.wq.weight": "keep",
            "blocks.12.attn.wq.weight": default_action,
            "encoder.1.attn.wq.weight": "keep",
            "encoder.10.attn.wq.weight": "keep",
            "decoder.1.attn.wq.weight": default_action,
            "single.weight": default_action,
        }
        compact_profile = {
            "default": default_action,
            "rules": compact_runtime_rules(desired_actions, default_action),
        }
        verbose_profile = {
            "default": default_action,
            "rules": tuple(
                {
                    "action": action,
                    "prefix": tensor_name,
                    "suffixes": ("",),
                }
                for tensor_name, action in desired_actions.items()
                if action != default_action
            ),
        }

        for tensor_name, expected_action in desired_actions.items():
            self.assertEqual(
                resolve_profile(compact_profile, tensor_name),
                resolve_profile(verbose_profile, tensor_name),
            )
            self.assertEqual(
                resolve_profile(compact_profile, tensor_name),
                expected_action,
            )

    def test_compactor_uses_the_supplied_default_action(self) -> None:
        for default_action in ("keep", "int8_convrot", "convrot_w4a4_mse"):
            desired_actions = {
                "module.0.weight": default_action,
                "module.1.weight": default_action,
            }
            self.assertEqual(
                compact_runtime_rules(desired_actions, default_action),
                (),
            )
            profile = _runtime_profile(
                "test",
                "test",
                set(),
                set(),
                set(),
                set(desired_actions),
                default_action=default_action,
            )
            self.assertEqual(profile["default"], default_action)
            self.assertEqual(profile["rules"], ())

    def test_no_compaction_candidate_retains_exact_rule(self) -> None:
        self.assertEqual(
            compact_runtime_rules(
                {"solo.weight": "keep"},
                "convrot_w4a4",
            ),
            (
                {
                    "action": "keep",
                    "prefix": "solo.weight",
                    "suffixes": ("",),
                },
            ),
        )

    def test_runtime_profile_requires_complete_tensor_namespace(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Runtime tensor namespace is missing",
        ):
            _runtime_profile(
                "test",
                "test",
                {"missing.weight"},
                set(),
                set(),
                {"present.weight"},
                default_action="convrot_w4a4",
            )

    def test_semantic_verification_reports_the_matching_compact_rule(self) -> None:
        incorrect_rule = {
            "action": "int8_convrot",
            "prefix": "solo.weight",
            "suffixes": ("",),
        }
        with patch(
            "potatoforge.audits.activation_profiles.compact_runtime_rules",
            return_value=(incorrect_rule,),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "solo\\.weight.*expected keep.*actual int8_convrot.*"
                "matching compact rule",
            ):
                _runtime_profile(
                    "test",
                    "test",
                    set(),
                    {"solo.weight"},
                    set(),
                    {"solo.weight"},
                    default_action="convrot_w4a4",
                )

    def test_selection_uses_relative_output_error_and_main_block_filter(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.safetensors"
            tensor_names = (
                "blocks.0.attn.wq.weight",
                "blocks.1.attn.wq.weight",
                "txtfusion.layer.weight",
            )
            save_file(
                {
                    name: torch.randn((2, 256), dtype=torch.bfloat16)
                    for name in tensor_names
                },
                str(source_path),
            )
            audit = audit_bf16_source(source_path)
            for result in audit["results"]:
                if result["tensor_name"].startswith("blocks."):
                    result["methods"]["int8_convrot"] = {
                        "storage_bytes": None,
                        "relative_l2_error": None,
                    }
            source_header = read_source_model_header(source_path)
            baseline_profile = {
                "profile_id": "baseline",
                "default": "keep",
                "rules": tuple(
                    {
                        "action": "convrot_w4a4",
                        "prefix": name,
                        "suffixes": ("",),
                    }
                    for name in tensor_names[:2]
                ),
            }
            promoted_profile = {
                **baseline_profile,
                "rules": (
                    {
                        "action": "int8_convrot",
                        "prefix": tensor_names[1],
                        "suffixes": ("",),
                    },
                    *baseline_profile["rules"][:1],
                    baseline_profile["rules"][1],
                ),
            }
            target_bytes = estimate_profile_bytes(
                source_header,
                promoted_profile,
            )
            score_path = root / "score.json"
            score_path.write_text(
                json.dumps(
                    {
                        "format": "potatoforge_activation_score",
                        "version": 2,
                        "reference_format": "int8_convrot",
                        "candidate_format": "convrot_w4a4",
                        "metric_basis": "logical_linear_input",
                        "results": [
                            {
                                "tensor_name": tensor_names[0],
                                "activation_status": "ok",
                                "activation_error": 1000.0,
                                "relative_output_error": 1.0,
                            },
                            {
                                "tensor_name": tensor_names[1],
                                "activation_status": "ok",
                                "activation_error": 1.0,
                                "relative_output_error": 2.0,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            generated = generate_activation_profile(
                score_path,
                audit,
                target_bytes=target_bytes,
                profile_id="relative",
                metric="relative_output_error",
            )
            changed_score = json.loads(score_path.read_text(encoding="utf-8"))
            changed_score["results"][0]["activation_error"] = 0.001
            score_path.write_text(json.dumps(changed_score), encoding="utf-8")
            generated_again = generate_activation_profile(
                score_path,
                audit,
                target_bytes=target_bytes,
                profile_id="relative",
                metric="relative_output_error",
            )
            tighter = generate_activation_profile(
                score_path,
                audit,
                target_bytes=estimate_profile_bytes(
                    source_header,
                    baseline_profile,
                ),
                profile_id="relative-tight",
                metric="relative_output_error",
            )
            write_profile(
                root / "profile.json",
                generated.optimized.profile,
            )
            loaded_profile = load_profile(root / "profile.json")

        self.assertEqual(
            [
                rule["prefix"]
                for rule in generated.optimized.profile["rules"]
                if rule["action"] == "int8_convrot"
            ],
            [tensor_names[1]],
        )
        self.assertEqual(
            next(
                rule["action"]
                for rule in generated.optimized.profile["rules"]
                if rule["prefix"] == "txtfusion.layer.weight"
            ),
            "keep",
        )
        self.assertEqual(generated.summary["metric"], "relative_output_error")
        self.assertEqual(generated.optimized.profile, generated_again.optimized.profile)
        self.assertEqual(generated.summary, generated_again.summary)
        self.assertEqual(generated.optimized.profile, loaded_profile)
        promoted_names = [
            rule["prefix"]
            for rule in generated.optimized.profile["rules"]
            if rule["action"] == "int8_convrot"
        ]
        self.assertTrue(all(name.startswith("blocks.") for name in promoted_names))
        self.assertLessEqual(
            generated.optimized.output_bytes,
            generated.optimized.target_bytes,
        )
        self.assertLessEqual(
            tighter.optimized.output_bytes,
            generated.optimized.output_bytes,
        )

    def test_unsupported_relative_rows_are_excluded(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.safetensors"
            names = (
                "blocks.0.attn.wq.weight",
                "blocks.1.attn.wq.weight",
            )
            save_file(
                {
                    name: torch.ones((2, 256), dtype=torch.bfloat16)
                    for name in names
                },
                str(source_path),
            )
            audit = audit_bf16_source(source_path)
            header = read_source_model_header(source_path)
            baseline = {
                "profile_id": "baseline",
                "default": "keep",
                "rules": tuple(
                    {
                        "action": "convrot_w4a4",
                        "prefix": name,
                        "suffixes": ("",),
                    }
                    for name in names[:1]
                ),
            }
            score_path = root / "score.json"
            score_path.write_text(
                json.dumps(
                    {
                        "format": "potatoforge_activation_score",
                        "version": 2,
                        "reference_format": "int8_convrot",
                        "candidate_format": "convrot_w4a4",
                        "metric_basis": "logical_linear_input",
                        "results": [
                            {
                                "tensor_name": names[0],
                                "activation_status": "ok",
                                "relative_output_error": 1.0,
                            },
                            {
                                "tensor_name": names[1],
                                "activation_status": "zero_reference_output_energy",
                                "relative_output_error": None,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            target_bytes = estimate_profile_bytes(header, baseline)

            generated = generate_activation_profile(
                score_path,
                audit,
                target_bytes=target_bytes,
            )

        self.assertEqual(generated.summary["baseline_w4a4_tensor_count"], 2)
        self.assertNotIn(
            names[1],
            [rule["prefix"] for rule in generated.optimized.profile["rules"]],
        )

    def test_legacy_profile_can_use_the_same_target_size(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.safetensors"
            names = (
                "blocks.0.attn.wq.weight",
                "blocks.1.attn.wq.weight",
            )
            save_file(
                {
                    name: torch.randn((2, 256), dtype=torch.bfloat16)
                    for name in names
                },
                str(source_path),
            )
            audit = audit_bf16_source(source_path)
            header = read_source_model_header(source_path)
            baseline_rules = tuple(
                {
                    "action": "convrot_w4a4",
                    "prefix": name,
                    "suffixes": ("",),
                }
                for name in names
            )
            one_promotion = {
                "profile_id": "target",
                "default": "keep",
                "rules": (
                    {
                        "action": "int8_convrot",
                        "prefix": names[0],
                        "suffixes": ("",),
                    },
                    baseline_rules[1],
                ),
            }
            baseline_profile = {
                "profile_id": "baseline",
                "default": "keep",
                "rules": baseline_rules,
            }
            target_bytes = estimate_profile_bytes(header, one_promotion)
            baseline_bytes = estimate_profile_bytes(header, baseline_profile)
            score_path = root / "score.json"
            score_path.write_text(
                json.dumps(
                    {
                        "format": "potatoforge_activation_score",
                        "version": 2,
                        "reference_format": "int8_convrot",
                        "candidate_format": "convrot_w4a4",
                        "metric_basis": "logical_linear_input",
                        "results": [
                            {
                                "tensor_name": names[0],
                                "activation_status": "ok",
                                "relative_output_error": 1.0,
                            },
                            {
                                "tensor_name": names[1],
                                "activation_status": "ok",
                                "relative_output_error": 0.5,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            relative = generate_activation_profile(
                score_path,
                audit,
                target_bytes=target_bytes,
                metric="relative_output_error",
                profile_id="relative",
            )
            legacy = generate_activation_profile(
                score_path,
                audit,
                target_bytes=target_bytes,
                metric="relative_l2_error",
                profile_id="legacy",
            )
            budgeted = generate_activation_profile(
                score_path,
                audit,
                promotion_budget_bytes=target_bytes - baseline_bytes,
                profile_id="budgeted",
                metric="relative_output_error",
            )

        self.assertEqual(relative.optimized.target_bytes, legacy.optimized.target_bytes)
        self.assertEqual(budgeted.optimized.target_bytes, target_bytes)
        self.assertLessEqual(relative.optimized.output_bytes, target_bytes)
        self.assertLessEqual(legacy.optimized.output_bytes, target_bytes)
        self.assertEqual(legacy.summary["metric"], "relative_l2_error")


if __name__ == "__main__":
    unittest.main()
