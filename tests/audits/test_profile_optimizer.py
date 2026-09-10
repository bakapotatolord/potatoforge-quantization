import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from potatoforge.audits.all_comparison import relative_l2_error
from potatoforge.audits.profile_optimizer import (
    MethodChoice,
    _KNAPSACK_BUDGET_UNIT_BYTES,
    _build_knapsack_frontier,
    _indices_from_frontier,
    _method_choices,
    _optimize_profile_target_worker,
    _profile_for_assignments,
    estimate_profile_bytes,
    generate_profile_sweep,
    load_weight_audit,
    next_method_choice,
    optimize_target_size,
    write_profile,
)
from potatoforge.audits.weight_audit import audit_bf16_source, write_weight_audit_report
from potatoforge.converter import convert_model, convert_model_from_profile
from potatoforge.headers.source_header import read_source_model_header
from potatoforge.planning import build_plan
from potatoforge.profiles import load_profile
from potatoforge.quantization.convrot_w4a4 import (
    ConvRotW4A4Result,
    dequantize_convrot_w4a4,
)


class TestProfileOptimizer(unittest.TestCase):
    def _write_source(self, directory: str) -> Path:
        source_path = Path(directory) / "source.safetensors"
        torch.manual_seed(0)
        save_file(
            {
                "blocks.0.attn.wq.weight": torch.randn(
                    (2, 256),
                    dtype=torch.bfloat16,
                ),
                "embedding": torch.ones((2, 256), dtype=torch.bfloat16),
            },
            str(source_path),
        )
        return source_path

    def _write_source_with_excluded_weight(self, directory: str) -> Path:
        source_path = Path(directory) / "source-with-exclusion.safetensors"
        torch.manual_seed(0)
        save_file(
            {
                "blocks.0.attn.wq.weight": torch.randn(
                    (2, 256),
                    dtype=torch.bfloat16,
                ),
                "txtfusion.layer.weight": torch.ones(
                    (2, 256),
                    dtype=torch.bfloat16,
                ),
            },
            str(source_path),
        )
        return source_path

    def _write_repeated_block_source(self, directory: str) -> Path:
        source_path = Path(directory) / "repeated-blocks.safetensors"
        torch.manual_seed(0)
        save_file(
            {
                "blocks.0.attn.wq.weight": torch.randn(
                    (2, 256),
                    dtype=torch.bfloat16,
                ),
                "blocks.1.attn.wq.weight": torch.randn(
                    (2, 256),
                    dtype=torch.bfloat16,
                ),
            },
            str(source_path),
        )
        return source_path

    def test_target_size_uses_the_complete_output_file_size(self) -> None:
        with TemporaryDirectory() as directory:
            source_path = self._write_source(directory)
            audit = audit_bf16_source(source_path)
            source_header = read_source_model_header(source_path)
            smallest_bytes = estimate_profile_bytes(
                source_header,
                {
                    "profile_id": "smallest",
                    "default": "keep",
                    "rules": (
                        {
                            "action": "convrot_w4a4",
                            "prefix": "blocks.0.attn.wq.weight",
                            "suffixes": ("",),
                        },
                    ),
                },
            )
            smallest = optimize_target_size(
                audit,
                "smallest",
                smallest_bytes,
                frozenset(("convrot_w4a4",)),
            )
            full_precision_bytes = estimate_profile_bytes(
                source_header,
                {
                    "profile_id": "full-precision",
                    "default": "keep",
                    "rules": (),
                },
            )
            optimized = optimize_target_size(
                audit,
                "high-quality",
                full_precision_bytes,
            )

        self.assertLess(smallest.output_bytes, full_precision_bytes)
        self.assertEqual(optimized.output_bytes, full_precision_bytes)
        self.assertLessEqual(optimized.output_bytes, optimized.target_bytes)
        self.assertEqual(optimized.profile["rules"], ())

    def test_generates_a_sweep_from_minimum_to_full_precision(self) -> None:
        with TemporaryDirectory() as directory:
            source_path = self._write_source(directory)
            audit = audit_bf16_source(source_path)
            progress: list[tuple[int, int, int]] = []

            sweep = generate_profile_sweep(
                audit,
                "sweep",
                10**9,
                frozenset(("int8",)),
                on_profile_started=lambda index, total, target: progress.append(
                    (index, total, target)
                ),
            )

        self.assertGreaterEqual(len(sweep), 2)
        self.assertEqual(sweep[0].output_bytes, sweep[0].target_bytes)
        self.assertEqual(sweep[-1].profile["rules"], ())
        self.assertEqual(len(progress), len(sweep))
        self.assertEqual(progress[-1][0], progress[-1][1])
        self.assertTrue(
            all(
                left.reconstruction_sse >= right.reconstruction_sse
                for left, right in zip(sweep, sweep[1:])
            )
        )

    def test_profile_sweep_parallelizes_multiple_targets(self) -> None:
        with TemporaryDirectory() as directory:
            audit = audit_bf16_source(self._write_source(directory))
            expected = (object(), object(), object())
            with patch(
                "potatoforge.audits.profile_optimizer.ProcessPoolExecutor"
            ) as pool_type:
                pool = pool_type.return_value.__enter__.return_value
                pool.map.return_value = iter(expected)
                sweep = generate_profile_sweep(
                    audit,
                    "parallel-sweep",
                    1,
                    frozenset(("int8",)),
                )

        tasks = pool.map.call_args.args[1]
        self.assertGreaterEqual(len(tasks), 3)
        self.assertEqual(
            pool_type.call_args.kwargs["max_workers"],
            min(4, len(tasks)),
        )
        self.assertEqual(
            pool.map.call_args.args[0],
            _optimize_profile_target_worker,
        )
        self.assertEqual(
            [task[2] for task in tasks],
            sorted(task[2] for task in tasks),
        )
        self.assertEqual(sweep, expected)

    def test_profile_sweep_runs_with_spawned_workers(self) -> None:
        with TemporaryDirectory() as directory:
            source_path = self._write_source(directory)
            audit = audit_bf16_source(source_path)
            source_header = read_source_model_header(source_path)
            minimum_output_bytes = estimate_profile_bytes(
                source_header,
                {
                    "profile_id": "parallel-sweep",
                    "default": "keep",
                    "rules": tuple(
                        {
                            "action": "int8",
                            "prefix": result["tensor_name"],
                            "suffixes": ("",),
                        }
                        for result in audit["results"]
                    ),
                },
            )
            maximum_output_bytes = estimate_profile_bytes(
                source_header,
                {
                    "profile_id": "parallel-sweep",
                    "default": "keep",
                    "rules": (),
                },
            )
            step_bytes = max(
                1,
                (maximum_output_bytes - minimum_output_bytes) // 2,
            )

            sweep = generate_profile_sweep(
                audit,
                "parallel-sweep",
                step_bytes,
                frozenset(("int8",)),
            )

        self.assertGreaterEqual(len(sweep), 3)
        self.assertTrue(
            all(
                left.target_bytes <= right.target_bytes
                for left, right in zip(sweep, sweep[1:])
            )
        )

    def test_optimization_is_deterministic(self) -> None:
        with TemporaryDirectory() as directory:
            source_path = self._write_source(directory)
            audit = audit_bf16_source(source_path)
            source_header = read_source_model_header(source_path)
            minimum_bytes = estimate_profile_bytes(
                source_header,
                {
                    "profile_id": "minimum",
                    "default": "keep",
                    "rules": tuple(
                        {
                            "action": "int8",
                            "prefix": result["tensor_name"],
                            "suffixes": ("",),
                        }
                        for result in audit["results"]
                    ),
                },
            )
            maximum_bytes = estimate_profile_bytes(
                source_header,
                {
                    "profile_id": "maximum",
                    "default": "keep",
                    "rules": (),
                },
            )
            target_bytes = (minimum_bytes + maximum_bytes) // 2

            first = optimize_target_size(
                audit,
                "deterministic",
                target_bytes,
                frozenset(("int8",)),
            )
            second = optimize_target_size(
                audit,
                "deterministic",
                target_bytes,
                frozenset(("int8",)),
            )

        self.assertEqual(first.profile, second.profile)
        self.assertEqual(first.output_bytes, second.output_bytes)
        self.assertEqual(first.target_bytes, second.target_bytes)
        self.assertEqual(first.reconstruction_sse, second.reconstruction_sse)

    def test_next_method_choice_matches_optimizer_order(self) -> None:
        with TemporaryDirectory() as directory:
            result = audit_bf16_source(self._write_source(directory))["results"][0]
            choices = _method_choices(
                result,
                frozenset(("bf16", "int8", "int8_convrot")),
                None,
            )

        self.assertGreaterEqual(len(choices), 2)
        self.assertEqual(
            next_method_choice(
                result,
                choices[0].method,
                frozenset(("int8", "int8_convrot")),
            ),
            choices[1],
        )

    def test_preserves_w4a4_mse_as_a_generated_profile_action(self) -> None:
        with TemporaryDirectory() as directory:
            source_path = self._write_source(directory)
            audit = audit_bf16_source(source_path)
            source_header = read_source_model_header(source_path)
            target_bytes = estimate_profile_bytes(
                source_header,
                {
                    "profile_id": "w4a4-mse",
                    "default": "keep",
                    "rules": (
                        {
                            "action": "convrot_w4a4_mse",
                            "prefix": "blocks.0.attn.wq.weight",
                            "suffixes": ("",),
                        },
                    ),
                },
            )

            optimized = optimize_target_size(
                audit,
                "w4a4-mse",
                target_bytes,
                frozenset(("convrot_w4a4_mse",)),
            )

        self.assertEqual(
            optimized.profile["rules"][0]["action"],
            "convrot_w4a4_mse",
        )

    def test_generated_w4a4_mse_profile_reproduces_the_audited_conversion(
        self,
    ) -> None:
        weights = torch.linspace(
            -0.1,
            0.1,
            steps=512,
            dtype=torch.bfloat16,
        ).reshape(2, 256)
        weights[0, 0] = 3.0

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.safetensors"
            profile_path = Path(directory) / "profile.json"
            output_path = Path(directory) / "output.safetensors"
            save_file({"blocks.0.attn.wq.weight": weights}, str(source_path))
            audit = audit_bf16_source(source_path)
            source_header = read_source_model_header(source_path)
            target_bytes = estimate_profile_bytes(
                source_header,
                {
                    "profile_id": "w4a4-mse",
                    "default": "keep",
                    "rules": (
                        {
                            "action": "convrot_w4a4_mse",
                            "prefix": "blocks.0.attn.wq.weight",
                            "suffixes": ("",),
                        },
                    ),
                },
            )
            optimized = optimize_target_size(
                audit,
                "w4a4-mse",
                target_bytes,
                frozenset(("convrot_w4a4_mse",)),
            )
            write_profile(profile_path, optimized.profile)
            convert_model_from_profile(
                source_path,
                output_path,
                profile_path,
                on_entry_started=None,
            )
            converted = load_file(str(output_path))

        self.assertEqual(
            optimized.profile["rules"][0]["action"],
            "convrot_w4a4_mse",
        )
        reconstructed = dequantize_convrot_w4a4(
            ConvRotW4A4Result(
                converted["blocks.0.attn.wq.weight"],
                converted["blocks.0.attn.wq.weight_scale"],
            )
        )
        self.assertAlmostEqual(
            relative_l2_error(weights, reconstructed),
            audit["results"][0]["methods"]["convrot_w4a4_mse"][
                "relative_l2_error"
            ],
            places=6,
        )

    def test_same_size_lower_w4a4_mse_dominates_plain_w4a4(self) -> None:
        with TemporaryDirectory() as directory:
            result = audit_bf16_source(self._write_source(directory))["results"][0]
            w4a4_error = result["methods"]["convrot_w4a4"]["relative_l2_error"]
            result["methods"]["convrot_w4a4_mse"][
                "relative_l2_error"
            ] = w4a4_error / 2

        choices = _method_choices(
            result,
            frozenset(("convrot_w4a4", "convrot_w4a4_mse")),
            None,
        )

        self.assertEqual(
            [choice.method for choice in choices],
            ["convrot_w4a4_mse"],
        )

    def test_equal_size_and_error_prefers_plain_w4a4(self) -> None:
        with TemporaryDirectory() as directory:
            result = audit_bf16_source(self._write_source(directory))["results"][0]
            result["methods"]["convrot_w4a4_mse"]["relative_l2_error"] = result[
                "methods"
            ]["convrot_w4a4"]["relative_l2_error"]

        choices = _method_choices(
            result,
            frozenset(("convrot_w4a4", "convrot_w4a4_mse")),
            None,
        )

        self.assertEqual(
            [choice.method for choice in choices],
            ["convrot_w4a4"],
        )

    def test_estimate_matches_output_with_quantization_metadata(self) -> None:
        with TemporaryDirectory() as directory:
            source_path = self._write_source(directory)
            output_path = Path(directory) / "quantized.safetensors"
            profile: QuantizationProfile = {
                "default": "keep",
                "rules": (
                    {
                        "action": "int8",
                        "prefix": "blocks.0.attn.wq.weight",
                        "suffixes": ("",),
                    },
                ),
            }
            source_header = read_source_model_header(source_path)
            estimated_bytes = estimate_profile_bytes(source_header, profile)

            convert_model(
                source_path,
                output_path,
                profile,
                on_entry_started=None,
            )

            output_bytes = output_path.stat().st_size

        self.assertEqual(estimated_bytes, output_bytes)

    def test_prefers_convrot_when_its_actual_size_fits(self) -> None:
        with TemporaryDirectory() as directory:
            source_path = self._write_source(directory)
            audit = audit_bf16_source(source_path)
            convrot_bytes = estimate_profile_bytes(
                read_source_model_header(source_path),
                {
                    "profile_id": "same-tier",
                    "default": "keep",
                    "rules": (
                        {
                            "action": "int8_convrot",
                            "prefix": "blocks.0.attn.wq.weight",
                            "suffixes": ("",),
                        },
                    ),
                },
            )
            optimized = optimize_target_size(
                audit,
                "same-tier",
                convrot_bytes,
                frozenset(("int8", "int8_convrot")),
            )

        self.assertEqual(optimized.profile["rules"][0]["action"], "int8_convrot")

    def test_excluded_prefixes_force_bf16(self) -> None:
        with TemporaryDirectory() as directory:
            source_path = self._write_source_with_excluded_weight(directory)
            audit = audit_bf16_source(source_path)
            source_header = read_source_model_header(source_path)
            target_bytes = estimate_profile_bytes(
                source_header,
                {
                    "profile_id": "exclude-txtfusion",
                    "default": "keep",
                    "rules": (
                        {
                            "action": "int8",
                            "prefix": "blocks.0.attn.wq.weight",
                            "suffixes": ("",),
                        },
                    ),
                },
            )
            optimized = optimize_target_size(
                audit,
                "exclude-txtfusion",
                target_bytes,
                frozenset(("int8",)),
                excluded_prefixes=("txtfusion.",),
            )

        self.assertEqual(
            optimized.profile["rules"],
            (
                {
                    "action": "int8",
                    "prefix": "blocks.0.attn.wq.weight",
                    "suffixes": ("",),
                },
            ),
        )

    def test_excluded_suffixes_force_bf16(self) -> None:
        with TemporaryDirectory() as directory:
            source_path = self._write_source_with_excluded_weight(directory)
            audit = audit_bf16_source(source_path)
            source_header = read_source_model_header(source_path)
            target_bytes = estimate_profile_bytes(
                source_header,
                {
                    "profile_id": "exclude-suffix",
                    "default": "keep",
                    "rules": (
                        {
                            "action": "int8",
                            "prefix": "blocks.0.attn.wq.weight",
                            "suffixes": ("",),
                        },
                    ),
                },
            )
            optimized = optimize_target_size(
                audit,
                "exclude-suffix",
                target_bytes,
                frozenset(("int8",)),
                excluded_suffixes=(".layer.weight",),
            )

        self.assertEqual(len(optimized.profile["rules"]), 1)
        self.assertEqual(optimized.profile["rules"][0]["action"], "int8")

    def test_emits_one_rule_per_non_keep_assignment(self) -> None:
        with TemporaryDirectory() as directory:
            source_path = self._write_repeated_block_source(directory)
            audit = audit_bf16_source(source_path)
            source_header = read_source_model_header(source_path)
            target_bytes = estimate_profile_bytes(
                source_header,
                {
                    "profile_id": "grouped",
                    "default": "keep",
                    "rules": (
                        {
                            "action": "int8",
                            "prefix": "blocks.0.attn.wq.weight",
                            "suffixes": ("",),
                        },
                        {
                            "action": "int8",
                            "prefix": "blocks.1.attn.wq.weight",
                            "suffixes": ("",),
                        },
                    ),
                },
            )
            optimized = optimize_target_size(
                audit,
                "grouped",
                target_bytes,
                frozenset(("int8",)),
            )

        self.assertEqual(
            optimized.profile["rules"],
            (
                {
                    "action": "int8",
                    "prefix": "blocks.0.attn.wq.weight",
                    "suffixes": ("",),
                },
                {
                    "action": "int8",
                    "prefix": "blocks.1.attn.wq.weight",
                    "suffixes": ("",),
                },
            ),
        )

    def test_omits_keep_assignments_from_generated_rules(self) -> None:
        profile = _profile_for_assignments(
            {
                "blocks.0.attn.wq.weight": MethodChoice(
                    "int8",
                    "int8",
                    1,
                    0.1,
                    0.01,
                ),
                "blocks.1.attn.wq.weight": MethodChoice(
                    "bf16",
                    "keep",
                    2,
                    0.0,
                    0.0,
                ),
                "blocks.2.attn.wq.weight": MethodChoice(
                    "int8",
                    "int8",
                    1,
                    0.1,
                    0.01,
                ),
            },
            "mixed",
        )

        self.assertEqual(
            profile["rules"],
            (
                {
                    "action": "int8",
                    "prefix": "blocks.0.attn.wq.weight",
                    "suffixes": ("",),
                },
                {
                    "action": "int8",
                    "prefix": "blocks.2.attn.wq.weight",
                    "suffixes": ("",),
                },
            ),
        )

    def test_writes_a_loadable_profile_from_a_version_five_audit(self) -> None:
        with TemporaryDirectory() as directory:
            source_path = self._write_source(directory)
            audit_path = Path(directory) / "audit.json"
            profile_path = Path(directory) / "generated.json"
            write_weight_audit_report(audit_path, audit_bf16_source(source_path))
            source_header = read_source_model_header(source_path)
            target_bytes = estimate_profile_bytes(
                source_header,
                {
                    "profile_id": "generated",
                    "default": "keep",
                    "rules": (
                        {
                            "action": "convrot_w4a4",
                            "prefix": "blocks.0.attn.wq.weight",
                            "suffixes": ("",),
                        },
                    ),
                },
            )

            optimized = optimize_target_size(
                load_weight_audit(audit_path),
                "generated",
                target_bytes,
            )
            write_profile(profile_path, optimized.profile)
            profile = load_profile(profile_path)
            plan = build_plan(read_source_model_header(source_path).tensors, profile)

        self.assertEqual(profile["profile_id"], "generated")
        self.assertTrue(any(entry["action"] != "keep" for entry in plan))

    def test_rejects_a_version_three_audit_with_regeneration_message(self) -> None:
        with TemporaryDirectory() as directory:
            source_path = self._write_source(directory)
            audit = audit_bf16_source(source_path)
            audit["format_version"] = 3
            audit_path = Path(directory) / "old-audit.json"
            write_weight_audit_report(audit_path, audit)

            with self.assertRaisesRegex(
                ValueError,
                "format_version 5.*Regenerate the weight audit",
            ):
                load_weight_audit(audit_path)

    def test_rejects_missing_or_invalid_tensor_energy(self) -> None:
        invalid_values = [None, float("nan"), float("inf"), -1.0, True, "25"]
        for invalid_value in invalid_values:
            with self.subTest(invalid_value=invalid_value):
                with TemporaryDirectory() as directory:
                    source_path = self._write_source(directory)
                    audit = audit_bf16_source(source_path)
                    if invalid_value is None:
                        del audit["results"][0]["weight_l2_sq"]
                    else:
                        audit["results"][0]["weight_l2_sq"] = invalid_value
                    audit_path = Path(directory) / "invalid-audit.json"
                    write_weight_audit_report(audit_path, audit)

                    with self.assertRaisesRegex(ValueError, "weight_l2_sq"):
                        load_weight_audit(audit_path)

    def test_derives_reconstruction_sse_from_relative_error_and_energy(self) -> None:
        with TemporaryDirectory() as directory:
            result = audit_bf16_source(self._write_source(directory))["results"][0]
        result["weight_l2_sq"] = 100.0
        result["methods"]["int8"]["relative_l2_error"] = 0.2

        choices = _method_choices(result, frozenset(("int8",)), None)

        self.assertAlmostEqual(choices[0].reconstruction_sse, 4.0)
        bf16 = _method_choices(result, frozenset(("bf16",)), None)[0]
        self.assertEqual(bf16.reconstruction_sse, 0.0)

    def test_tensor_energy_changes_knapsack_allocation(self) -> None:
        unit = _KNAPSACK_BUDGET_UNIT_BYTES
        choices = {
            "tensor_a.weight": (
                MethodChoice("int8", "int8", 0, 0.2, 4.0),
                MethodChoice("bf16", "keep", unit, 0.1, 1.0),
            ),
            "tensor_b.weight": (
                MethodChoice("int8", "int8", 0, 0.2, 40.0),
                MethodChoice("bf16", "keep", unit, 0.1, 10.0),
            ),
        }

        frontier = _build_knapsack_frontier(choices, 1)
        indices = _indices_from_frontier(choices, frontier, 1)

        self.assertEqual(indices, {"tensor_a.weight": 0, "tensor_b.weight": 1})

    def test_multiple_choice_knapsack_beats_greedy_upgrade_ratio(self) -> None:
        unit = _KNAPSACK_BUDGET_UNIT_BYTES
        choices = {
            "tensor_a.weight": (
                MethodChoice("int8", "int8", 0, 1.0, 6.0),
                MethodChoice("bf16", "keep", 6 * unit, 0.0, 0.0),
            ),
            "tensor_b.weight": (
                MethodChoice("int8", "int8", 0, 1.0, 4.9),
                MethodChoice("bf16", "keep", 5 * unit, 0.0, 0.0),
            ),
            "tensor_c.weight": (
                MethodChoice("int8", "int8", 0, 1.0, 4.9),
                MethodChoice("bf16", "keep", 5 * unit, 0.0, 0.0),
            ),
        }

        frontier = _build_knapsack_frontier(choices, 10)
        indices = _indices_from_frontier(choices, frontier, 10)

        self.assertEqual(
            indices,
            {
                "tensor_a.weight": 0,
                "tensor_b.weight": 1,
                "tensor_c.weight": 1,
            },
        )

    def test_rejects_an_audit_missing_current_methods(self) -> None:
        with TemporaryDirectory() as directory:
            source_path = self._write_source(directory)
            audit = audit_bf16_source(source_path)
            for result in audit["results"]:
                del result["methods"]["convrot_w4a4_mse"]

            with self.assertRaisesRegex(ValueError, "unsupported or incomplete"):
                optimize_target_size(
                    audit,
                    "incomplete",
                    10**9,
                    frozenset(("int8",)),
                )

    def test_rejects_incomplete_or_malformed_audits(self) -> None:
        with TemporaryDirectory() as directory:
            source_path = self._write_source(directory)
            incomplete = audit_bf16_source(source_path)
            incomplete["results"] = []
            with self.assertRaisesRegex(ValueError, "does not cover"):
                optimize_target_size(incomplete, "incomplete", 10**9)

            malformed = audit_bf16_source(source_path)
            malformed["results"][0]["methods"]["int8"] = {
                "storage_bytes": 1,
                "relative_l2_error": None,
            }
            audit_path = Path(directory) / "malformed.json"
            write_weight_audit_report(audit_path, malformed)
            with self.assertRaisesRegex(ValueError, "invalid method record"):
                load_weight_audit(audit_path)

    def test_write_profile_can_explicitly_overwrite(self) -> None:
        with TemporaryDirectory() as directory:
            profile_path = Path(directory) / "generated.json"
            profile = {
                "profile_id": "generated",
                "default": "keep",
                "rules": (),
            }
            write_profile(profile_path, profile)
            write_profile(profile_path, profile, overwrite=True)

            self.assertEqual(load_profile(profile_path)["profile_id"], "generated")

    def test_write_profile_preserves_optional_rule_fallback(self) -> None:
        with TemporaryDirectory() as directory:
            profile_path = Path(directory) / "fallback.json"
            write_profile(
                profile_path,
                {
                    "profile_id": "fallback",
                    "default": "keep",
                    "rules": (
                        {
                            "action": "int8_convrot",
                            "fallback": "int8",
                            "prefix": "model.diffusion_model.",
                            "suffixes": ("",),
                        },
                        {
                            "action": "int8",
                            "prefix": "model.other.",
                            "suffixes": ("",),
                        },
                    ),
                },
            )
            document = json.loads(profile_path.read_text(encoding="utf-8"))

        self.assertEqual(
            document["rules"],
            [
                {
                    "action": "int8_convrot",
                    "prefix": "model.diffusion_model.",
                    "suffixes": [""],
                    "fallback": "int8",
                },
                {
                    "action": "int8",
                    "prefix": "model.other.",
                    "suffixes": [""],
                },
            ],
        )

    def test_rejects_audit_for_a_different_tensor_shape(self) -> None:
        with TemporaryDirectory() as directory:
            audit = audit_bf16_source(self._write_source(directory))
            audit["results"][0]["shape"] = [1, 512]

            with self.assertRaisesRegex(ValueError, "checkpoint shapes"):
                optimize_target_size(audit, "wrong-shape", 10**9)

    def test_method_list_is_the_int6_opt_in(self) -> None:
        with TemporaryDirectory() as directory:
            source_path = self._write_source(directory)
            audit = audit_bf16_source(source_path)
            source_header = read_source_model_header(source_path)
            int6_profile = {
                "profile_id": "int6",
                "default": "keep",
                "rules": (
                    {
                        "action": "int6_rowwise",
                        "prefix": "blocks.0.attn.wq.weight",
                        "suffixes": ("",),
                    },
                ),
            }

            optimized = optimize_target_size(
                audit,
                "int6",
                estimate_profile_bytes(source_header, int6_profile),
                frozenset(("int6",)),
            )

        self.assertEqual(
            optimized.profile["rules"][0]["action"],
            "int6_rowwise",
        )


if __name__ == "__main__":
    unittest.main()
