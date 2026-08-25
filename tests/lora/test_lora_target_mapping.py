import unittest

from potatoforge.lora.lora_discovery import (
    discover_additive_deltas,
    discover_linear_pairs,
)
from potatoforge.lora.lora_merge import (
    resolve_additive_delta_source_key,
    resolve_linear_pair_source_key,
)
from potatoforge.headers.source_header import SourceModelHeader


def make_header(*keys: str) -> SourceModelHeader:
    return SourceModelHeader(
        tensors={
            key: {
                "dtype": "BF16",
                "shape": [2, 2],
                "data_offsets": [0, 8],
            }
            for key in keys
        },
        metadata={},
    )


class TestResolveSourceTensorKey(unittest.TestCase):
    def test_resolves_discovered_linear_pair_exactly(self) -> None:
        adapter_header = make_header(
            "layer.lora_A",
            "layer.lora_B",
        )
        pair = discover_linear_pairs(adapter_header)["pairs"][0]

        result = resolve_linear_pair_source_key(
            make_header("layer.weight"),
            pair,
        )

        self.assertEqual(result, "layer.weight")

    def test_resolves_discovered_additive_delta_exactly(self) -> None:
        adapter_header = make_header("layer.scale.diff")
        delta = discover_additive_deltas(adapter_header)[0]

        result = resolve_additive_delta_source_key(
            make_header("layer.scale"),
            delta,
        )

        self.assertEqual(result, "layer.scale")

    def test_resolves_discovered_pair_with_unique_suffix_match(self) -> None:
        adapter_header = make_header(
            "diffusion_model.blocks.0.attn.wq.lora_A",
            "diffusion_model.blocks.0.attn.wq.lora_B",
        )
        pair = discover_linear_pairs(adapter_header)["pairs"][0]

        for source_name in (
            "blocks.0.attn.wq.weight",
            "model.diffusion_model.blocks.0.attn.wq.weight",
        ):
            with self.subTest(source_name=source_name):
                result = resolve_linear_pair_source_key(
                    make_header(source_name),
                    pair,
                )

                self.assertEqual(result, source_name)

    def test_rejects_unmatched_target(self) -> None:
        adapter_header = make_header(
            "layer.lora_A",
            "layer.lora_B",
        )
        pair = discover_linear_pairs(adapter_header)["pairs"][0]

        with self.assertRaisesRegex(
            ValueError,
            "No source tensor matched adapter target",
        ):
            resolve_linear_pair_source_key(
                make_header("other.weight"),
                pair,
            )

    def test_rejects_ambiguous_target(self) -> None:
        adapter_header = make_header(
            "diffusion_model.layer.lora_A",
            "diffusion_model.layer.lora_B",
        )
        pair = discover_linear_pairs(adapter_header)["pairs"][0]

        with self.assertRaisesRegex(
            ValueError,
            "Ambiguous source tensor match",
        ):
            resolve_linear_pair_source_key(
                make_header(
                    "model.diffusion_model.layer.weight",
                    "other.diffusion_model.layer.weight",
                ),
                pair,
            )


if __name__ == "__main__":
    unittest.main()
