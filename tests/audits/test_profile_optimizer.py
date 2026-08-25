import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import save_file

from potatoforge.audits.profile_optimizer import (
    MethodChoice,
    _profile_for_assignments,
    estimate_profile_bytes,
    load_weight_audit,
    optimize_target_size,
    write_profile,
)
from potatoforge.audits.weight_audit import audit_bf16_source, write_weight_audit_report
from potatoforge.headers.source_header import read_source_model_header
from potatoforge.planning import build_plan
from potatoforge.profiles import load_profile


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
                ),
                "blocks.1.attn.wq.weight": MethodChoice(
                    "bf16",
                    "keep",
                    2,
                    0.0,
                ),
                "blocks.2.attn.wq.weight": MethodChoice(
                    "int8",
                    "int8",
                    1,
                    0.1,
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

    def test_writes_a_loadable_profile_from_a_version_three_audit(self) -> None:
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

    def test_rejects_audit_for_a_different_tensor_shape(self) -> None:
        with TemporaryDirectory() as directory:
            audit = audit_bf16_source(self._write_source(directory))
            audit["results"][0]["shape"] = [1, 512]

            with self.assertRaisesRegex(ValueError, "checkpoint shapes"):
                optimize_target_size(audit, "wrong-shape", 10**9)

    def test_requires_an_explicit_int6_runtime_opt_in(self) -> None:
        with TemporaryDirectory() as directory:
            audit = audit_bf16_source(self._write_source(directory))

            with self.assertRaisesRegex(ValueError, "INT6 methods require"):
                optimize_target_size(
                    audit,
                    "int6",
                    10**9,
                    frozenset(("int6",)),
                )


if __name__ == "__main__":
    unittest.main()
