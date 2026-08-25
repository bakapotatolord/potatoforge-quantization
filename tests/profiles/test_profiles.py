import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from potatoforge.profiles import load_profile, resolve_profile

from tests.profile_paths import (
    ILLUSTRIOUS_COMBINED_PROFILE_PATH,
    KROMA_BALANCED_PROFILE_PATH,
)

class TestProfiles(unittest.TestCase):
    def _base_document(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "profile_id": "test-profile",
            "default": "keep",
            "rules": [],
        }

    def _write_document(
        self,
        directory: str,
        document: object,
    ) -> Path:
        path = Path(directory) / "profile.json"
        path.write_text(
            json.dumps(document),
            encoding="utf-8",
        )
        return path

    def _assert_invalid_document(
        self,
        document: object,
        message: str,
    ) -> None:
        with TemporaryDirectory() as directory:
            path = self._write_document(directory, document)

            with self.assertRaisesRegex(ValueError, message):
                load_profile(path)

    def test_loads_kroma_balanced_profile(self) -> None:
        profile = load_profile(KROMA_BALANCED_PROFILE_PATH)

        self.assertEqual(profile["default"], "keep")
        self.assertEqual(
            profile["profile_id"],
            "kroma-v0.1-balanced",
        )
        self.assertEqual(
            profile["rules"][0],
            {
                "action": "int8_convrot",
                "prefix": "blocks.",
                "suffixes": (
                    ".attn.wk.weight",
                    ".attn.wv.weight",
                    ".attn.wo.weight",
                ),
            },
        )

        self.assertEqual(
            resolve_profile(profile, "blocks.0.attn.wk.weight"),
            "int8_convrot",
        )
        self.assertEqual(
            resolve_profile(profile, "blocks.0.attn.wq.weight"),
            "convrot_w4a4",
        )
        self.assertEqual(
            resolve_profile(profile, "blocks.0.attn.gate.weight"),
            "convrot_w4a4",
        )
        self.assertEqual(
            resolve_profile(profile, "blocks.0.attn.gate.bias"),
            "keep",
        )

    def test_loads_illustrious_combined_profile(self) -> None:
        profile = load_profile(ILLUSTRIOUS_COMBINED_PROFILE_PATH)

        self.assertEqual(profile["default"], "keep")
        self.assertEqual(
            resolve_profile(
                profile,
                "conditioner.embedders.0.transformer.text_model.encoder."
                "layers.0.self_attn.q_proj.weight",
            ),
            "int8_convrot",
        )
        self.assertEqual(
            resolve_profile(
                profile,
                "conditioner.embedders.1.model.transformer.resblocks.0."
                "attn.out_proj.weight",
            ),
            "int8",
        )
        self.assertEqual(
            resolve_profile(
                profile,
                "model.diffusion_model.input_blocks.4.1."
                "transformer_blocks.0.attn1.to_q.weight",
            ),
            "int8",
        )
        self.assertEqual(
            resolve_profile(
                profile,
                "model.diffusion_model.input_blocks.0.1."
                "transformer_blocks.0.attn1.to_q.weight",
            ),
            "int8_convrot",
        )
        self.assertEqual(
            resolve_profile(
                profile,
                "model.diffusion_model.time_embed.0.weight",
            ),
            "keep",
        )

    def test_loads_int6_rowwise_action(self) -> None:
        document = self._base_document()
        document["rules"] = [
            {
                "action": "int6_rowwise",
                "prefix": "blocks.",
                "suffixes": [".weight"],
            }
        ]

        with TemporaryDirectory() as directory:
            path = self._write_document(directory, document)
            profile = load_profile(path)

        self.assertEqual(
            resolve_profile(profile, "blocks.0.attn.wq.weight"),
            "int6_rowwise",
        )

    def test_loads_int6_convrot_action(self) -> None:
        document = self._base_document()
        document["rules"] = [
            {
                "action": "int6_convrot",
                "prefix": "blocks.",
                "suffixes": [".weight"],
            }
        ]

        with TemporaryDirectory() as directory:
            profile = load_profile(self._write_document(directory, document))

        self.assertEqual(
            resolve_profile(profile, "blocks.0.attn.wq.weight"),
            "int6_convrot",
        )

    def test_rejects_invalid_json(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text("{", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Invalid JSON profile"):
                load_profile(path)

    def test_rejects_invalid_utf8(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_bytes(b"\xff")

            with self.assertRaisesRegex(ValueError, "Invalid JSON profile"):
                load_profile(path)

    def test_rejects_non_object_document(self) -> None:
        self._assert_invalid_document(
            [],
            "top-level JSON object",
        )

    def test_rejects_missing_required_field(self) -> None:
        document = self._base_document()
        del document["profile_id"]

        self._assert_invalid_document(
            document,
            "missing required fields",
        )

    def test_rejects_unknown_top_level_field(self) -> None:
        document = self._base_document()
        document["extra"] = True

        self._assert_invalid_document(
            document,
            "unknown fields",
        )

    def test_rejects_unsupported_format_version(self) -> None:
        document = self._base_document()
        document["format_version"] = 2

        self._assert_invalid_document(
            document,
            "format_version must be 1",
        )

    def test_rejects_boolean_format_version(self) -> None:
        document = self._base_document()
        document["format_version"] = True

        self._assert_invalid_document(
            document,
            "format_version must be an integer",
        )

    def test_rejects_unknown_action(self) -> None:
        document = self._base_document()
        document["rules"] = [
            {
                "action": "in8",
                "prefix": "blocks.",
                "suffixes": [".weight"],
            }
        ]

        self._assert_invalid_document(
            document,
            "action must be one of",
        )

    def test_rejects_missing_or_non_string_prefix(self) -> None:
        missing_prefix = self._base_document()
        missing_prefix["rules"] = [
            {
                "action": "int8",
                "suffixes": [".weight"],
            }
        ]

        non_string_prefix = self._base_document()
        non_string_prefix["rules"] = [
            {
                "action": "int8",
                "prefix": 123,
                "suffixes": [".weight"],
            }
        ]

        for document in (missing_prefix, non_string_prefix):
            with self.subTest(document=document):
                self._assert_invalid_document(document, "prefix")

    def test_rejects_invalid_suffixes(self) -> None:
        invalid_suffixes = (
            [],
            ".weight",
            [".weight", 123],
        )

        for suffixes in invalid_suffixes:
            document = self._base_document()
            document["rules"] = [
                {
                    "action": "int8",
                    "prefix": "blocks.",
                    "suffixes": suffixes,
                }
            ]

            with self.subTest(suffixes=suffixes):
                self._assert_invalid_document(document, "suffixes")

    def test_rejects_non_string_description(self) -> None:
        document = self._base_document()
        document["description"] = None

        self._assert_invalid_document(
            document,
            "description must be a string",
        )

    def test_rejects_unknown_rule_field(self) -> None:
        document = self._base_document()
        document["rules"] = [
            {
                "action": "int8",
                "prefix": "blocks.",
                "suffixes": [".weight"],
                "extra": True,
            }
        ]

        self._assert_invalid_document(
            document,
            "contains unknown fields",
        )
