import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from potatoforge.sweep_profiles import load_sweep_profile


class TestSweepProfiles(unittest.TestCase):
    def _load(self, document: object):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sweep.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            return load_sweep_profile(path)

    def test_loads_multiple_action_groups(self) -> None:
        profile = self._load(
            {
                "format_version": 1,
                "profile_id": "kroma-sensitivity-v1",
                "groups": [
                    {
                        "action": "int8_convrot",
                        "layers": ["blocks.0.attn.wq.weight"],
                    },
                    {
                        "action": "int6_convrot",
                        "layers": ["blocks.1.mlp.down.weight"],
                    },
                ],
            }
        )

        self.assertEqual(profile["profile_id"], "kroma-sensitivity-v1")
        self.assertEqual(
            [(group["action"], group["layers"]) for group in profile["groups"]],
            [
                ("int8_convrot", ("blocks.0.attn.wq.weight",)),
                ("int6_convrot", ("blocks.1.mlp.down.weight",)),
            ],
        )

    def test_rejects_missing_unknown_empty_and_keep_fields(self) -> None:
        base = {
            "format_version": 1,
            "profile_id": "test",
            "groups": [
                {"action": "int8", "layers": ["layer.weight"]},
            ],
        }
        cases = (
            ({key: value for key, value in base.items() if key != "groups"}, "missing required"),
            ({**base, "extra": True}, "unknown fields"),
            ({**base, "groups": []}, "non-empty list"),
            (
                {
                    **base,
                    "groups": [{"action": "keep", "layers": ["layer.weight"]}],
                },
                "action must be one of",
            ),
        )
        for document, error in cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    self._load(document)

    def test_rejects_invalid_group_and_layer_values(self) -> None:
        base = {
            "format_version": 1,
            "profile_id": "test",
            "groups": [{"action": "int8", "layers": ["layer.weight"]}],
        }
        cases = (
            ({**base, "groups": [{"action": "int8"}]}, "missing fields"),
            (
                {
                    **base,
                    "groups": [{"action": "int8", "layers": [""]}],
                },
                "non-empty strings",
            ),
            (
                {
                    **base,
                    "groups": [
                        {"action": "int8", "layers": ["layer.weight"], "extra": True}
                    ],
                },
                "unknown fields",
            ),
        )
        for document, error in cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    self._load(document)


if __name__ == "__main__":
    unittest.main()
