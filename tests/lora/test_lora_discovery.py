import unittest

from potatoforge.lora.lora_discovery import (
    PAIR_CONVENTIONS,
    classify_adapter_tensor_key,
    derive_pair_candidate,
    discover_additive_deltas,
    discover_linear_pairs,
    inspect_adapter_header,
    inspect_linear_pair,
)
from potatoforge.headers.source_header import SourceModelHeader


def make_header(a_shape: list[int], b_shape: list[int]) -> SourceModelHeader:
    return SourceModelHeader(
        tensors={
            "layer.lora_A": {
                "dtype": "BF16",
                "shape": a_shape,
                "data_offsets": [0, 48],
            },
            "layer.lora_B": {
                "dtype": "BF16",
                "shape": b_shape,
                "data_offsets": [48, 112],
            },
        },
        metadata={},
    )


class TestLinearPairInspection(unittest.TestCase):
    def test_arbitrary_rank_is_read_from_factor_shapes(self) -> None:
        header = make_header([4, 6], [8, 4])

        pair = inspect_linear_pair(
            header,
            down_key="layer.lora_A",
            up_key="layer.lora_B",
        )

        self.assertEqual(pair["rank"], 4)
        self.assertEqual(pair["input_features"], 6)
        self.assertEqual(pair["output_features"], 8)

    def test_mismatched_factor_rank_is_rejected(self) -> None:
        header = make_header([4, 6], [8, 5])

        with self.assertRaisesRegex(ValueError, "Ranks do not match"):
            inspect_linear_pair(
                header,
                down_key="layer.lora_A",
                up_key="layer.lora_B",
            )

    def test_non_matrix_factor_is_rejected(self) -> None:
        header = make_header([4, 6, 1], [8, 4])

        with self.assertRaisesRegex(
            ValueError,
            "LoRA factors must be rank-2 tensors",
        ):
            inspect_linear_pair(
                header,
                down_key="layer.lora_A",
                up_key="layer.lora_B",
            )

    def test_non_positive_feature_dimension_is_rejected(self) -> None:
        header = make_header([4, 0], [8, 4])

        with self.assertRaisesRegex(
            ValueError,
            "LoRA feature dimensions must be positive",
        ):
            inspect_linear_pair(
                header,
                down_key="layer.lora_A",
                up_key="layer.lora_B",
            )


class TestPairCandidateDerivation(unittest.TestCase):
    def test_kroma_convention_derives_up_key(self) -> None:
        convention = next(
            item
            for item in PAIR_CONVENTIONS
            if item["name"] == "bare_ab"
        )

        candidate = derive_pair_candidate(
            "layer.lora_A",
            convention,
        )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate["target"], "layer")
        self.assertEqual(candidate["down_key"], "layer.lora_A")
        self.assertEqual(candidate["up_key"], "layer.lora_B")

    def test_comfy_convention_derives_up_key(self) -> None:
        convention = next(
            item
            for item in PAIR_CONVENTIONS
            if item["name"] == "comfy_up_down"
        )

        candidate = derive_pair_candidate(
            "layer.lora_down.weight",
            convention,
        )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate["target"], "layer")
        self.assertEqual(candidate["up_key"], "layer.lora_up.weight")

    def test_additional_pair_conventions_derive_source_target(self) -> None:
        cases = (
            (
                "diffusers_legacy_ab",
                "layer_lora.down.weight",
                "layer_lora.up.weight",
                "layer",
            ),
            (
                "diffusers_dotted_ab",
                "layer.lora.down.weight",
                "layer.lora.up.weight",
                "layer",
            ),
            (
                "transformers_lora_linear_layer",
                "layer.lora_linear_layer.down.weight",
                "layer.lora_linear_layer.up.weight",
                "layer",
            ),
            (
                "qwen_default_ab",
                "layer.lora_A.default.weight",
                "layer.lora_B.default.weight",
                "layer",
            ),
        )

        for name, down_key, up_key, target in cases:
            with self.subTest(name=name):
                convention = next(
                    item
                    for item in PAIR_CONVENTIONS
                    if item["name"] == name
                )

                candidate = derive_pair_candidate(
                    down_key,
                    convention,
                )

                self.assertIsNotNone(candidate)
                assert candidate is not None
                self.assertEqual(candidate["target"], target)
                self.assertEqual(candidate["up_key"], up_key)


class TestLinearPairDiscovery(unittest.TestCase):
    def test_discovers_kroma_pair(self) -> None:
        result = discover_linear_pairs(make_header([4, 6], [8, 4]))

        self.assertEqual(len(result["pairs"]), 1)
        self.assertEqual(result["pairs"][0]["rank"], 4)
        self.assertEqual(result["pairs"][0]["input_features"], 6)
        self.assertEqual(result["pairs"][0]["output_features"], 8)
        self.assertEqual(result["pairs"][0]["convention_name"], "bare_ab")
        self.assertEqual(result["pairs"][0]["target"], "layer")
        self.assertEqual(result["unpaired_down_keys"], [])
        self.assertEqual(result["unpaired_up_keys"], [])

    def test_reports_unpaired_down_key(self) -> None:
        header = SourceModelHeader(
            tensors={
                "layer.lora_A": {
                    "dtype": "BF16",
                    "shape": [4, 6],
                    "data_offsets": [0, 48],
                },
            },
            metadata={},
        )

        result = discover_linear_pairs(header)

        self.assertEqual(result["pairs"], [])
        self.assertEqual(result["unpaired_down_keys"], ["layer.lora_A"])
        self.assertEqual(result["unpaired_up_keys"], [])

    def test_reports_unpaired_up_key(self) -> None:
        header = SourceModelHeader(
            tensors={
                "layer.lora_B": {
                    "dtype": "BF16",
                    "shape": [8, 4],
                    "data_offsets": [0, 48],
                },
            },
            metadata={},
        )

        result = discover_linear_pairs(header)

        self.assertEqual(result["pairs"], [])
        self.assertEqual(result["unpaired_down_keys"], [])
        self.assertEqual(result["unpaired_up_keys"], ["layer.lora_B"])


class TestAdapterTensorClassification(unittest.TestCase):
    def test_classifies_diff_key(self) -> None:
        result = classify_adapter_tensor_key("layer.diff")

        self.assertEqual(result["kind"], "additive_tensor_delta")
        self.assertEqual(result["target"], "layer")

    def test_classifies_alpha_key(self) -> None:
        result = classify_adapter_tensor_key("layer.alpha")

        self.assertEqual(result["kind"], "alpha")
        self.assertEqual(result["target"], "layer")

    def test_unknown_remaining_key_is_unsupported(self) -> None:
        result = classify_adapter_tensor_key("layer.some_unknown_format")

        self.assertEqual(result["kind"], "unsupported")
        self.assertIsNone(result["target"])
        self.assertEqual(result["contract"], "unsupported")

    def test_classifies_explicit_unsupported_contract_markers(self) -> None:
        cases = (
            ("layer.hada_w1_a", "loha"),
            ("layer.lokr_w1", "lokr"),
            ("layer.oft_blocks", "oft_or_boft"),
            ("layer.dora_scale", "dora"),
            ("layer.w_norm", "weight_norm"),
            ("layer.b_norm", "bias_norm"),
            ("layer.diff_b", "additive_bias_delta"),
            ("layer.reshape_weight", "reshape"),
            ("layer.set_weight", "set_weight"),
        )

        for key, contract in cases:
            with self.subTest(key=key):
                result = classify_adapter_tensor_key(key)

                self.assertEqual(result["kind"], "unsupported")
                self.assertEqual(result["target"], "layer")
                self.assertEqual(result["contract"], contract)


class TestAdditiveDeltaDiscovery(unittest.TestCase):
    def test_discovers_diff_tensor(self) -> None:
        header = SourceModelHeader(
            tensors={
                "norm.weight.diff": {
                    "dtype": "BF16",
                    "shape": [8],
                    "data_offsets": [0, 16],
                },
            },
            metadata={},
        )

        result = discover_additive_deltas(header)

        self.assertEqual(
            result,
            [
                {
                    "key": "norm.weight.diff",
                    "target": "norm.weight",
                    "shape": [8],
                    "dtype": "BF16",
                }
            ],
        )

    def test_ignores_non_diff_keys(self) -> None:
        header = SourceModelHeader(
            tensors={
                "layer.diff": {
                    "dtype": "BF16",
                    "shape": [8],
                    "data_offsets": [0, 16],
                },
                "layer.alpha": {
                    "dtype": "BF16",
                    "shape": [1],
                    "data_offsets": [16, 18],
                },
                "layer.unknown": {
                    "dtype": "BF16",
                    "shape": [4],
                    "data_offsets": [18, 26],
                },
            },
            metadata={},
        )

        result = discover_additive_deltas(header)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["key"], "layer.diff")

    def test_rejects_empty_diff_target(self) -> None:
        header = SourceModelHeader(
            tensors={
                ".diff": {
                    "dtype": "BF16",
                    "shape": [8],
                    "data_offsets": [0, 16],
                },
            },
            metadata={},
        )

        with self.assertRaisesRegex(
            ValueError,
            "Invalid additive delta key",
        ):
            discover_additive_deltas(header)


class TestAdapterInspection(unittest.TestCase):
    def test_inventory_records_every_adapter_header_key(self) -> None:
        header = SourceModelHeader(
            tensors={
                "layer.lora_A": {
                    "dtype": "BF16",
                    "shape": [4, 6],
                    "data_offsets": [0, 48],
                },
                "layer.lora_B": {
                    "dtype": "BF16",
                    "shape": [8, 4],
                    "data_offsets": [48, 112],
                },
                "layer.diff": {
                    "dtype": "BF16",
                    "shape": [8, 6],
                    "data_offsets": [112, 208],
                },
                "layer.alpha": {
                    "dtype": "BF16",
                    "shape": [1],
                    "data_offsets": [208, 210],
                },
                "layer.unknown": {
                    "dtype": "BF16",
                    "shape": [2],
                    "data_offsets": [210, 214],
                },
                "orphan.lora_A": {
                    "dtype": "BF16",
                    "shape": [2, 3],
                    "data_offsets": [214, 226],
                },
            },
            metadata={},
        )

        result = inspect_adapter_header(header)
        records = {
            record["key"]: record
            for record in result["tensors"]
        }

        self.assertEqual(len(result["pairs"]), 1)
        self.assertEqual(result["pairs"][0]["rank"], 4)
        self.assertEqual(result["additive_deltas"][0]["shape"], [8, 6])
        self.assertEqual(list(records), sorted(header.tensors))
        self.assertEqual(set(records), set(header.tensors))
        self.assertEqual(records["layer.lora_A"]["kind"], "linear_down")
        self.assertEqual(records["layer.lora_B"]["kind"], "linear_up")
        self.assertEqual(
            records["layer.diff"]["kind"],
            "additive_tensor_delta",
        )
        self.assertEqual(records["layer.alpha"]["kind"], "alpha")
        self.assertEqual(records["layer.unknown"]["kind"], "unsupported")
        self.assertEqual(
            records["orphan.lora_A"]["kind"],
            "unpaired_down",
        )

    def test_rejects_empty_alpha_target(self) -> None:
        header = SourceModelHeader(
            tensors={
                ".alpha": {
                    "dtype": "BF16",
                    "shape": [1],
                    "data_offsets": [0, 2],
                },
            },
            metadata={},
        )

        with self.assertRaisesRegex(
            ValueError,
            "Invalid alpha key",
        ):
            inspect_adapter_header(header)

    def test_inventory_distinguishes_oft_and_boft_blocks(self) -> None:
        header = SourceModelHeader(
            tensors={
                "oft.oft_blocks": {
                    "dtype": "BF16",
                    "shape": [2, 4, 4],
                    "data_offsets": [0, 128],
                },
                "boft.oft_blocks": {
                    "dtype": "BF16",
                    "shape": [2, 2, 4, 4],
                    "data_offsets": [128, 256],
                },
            },
            metadata={},
        )

        records = {
            record["key"]: record
            for record in inspect_adapter_header(header)["tensors"]
        }

        self.assertEqual(records["oft.oft_blocks"]["contract"], "oft")
        self.assertEqual(records["boft.oft_blocks"]["contract"], "boft")


if __name__ == "__main__":
    unittest.main()
