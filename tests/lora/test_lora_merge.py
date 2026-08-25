import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import load_file, save_file

from potatoforge.lora.lora_merge import (
    AdapterMergeInput,
    build_merge_plan,
    merge_bf16_adapters,
)
from potatoforge.headers.source_header import (
    SourceModelHeader,
    read_source_model_header,
)


def make_header(
    descriptors: dict[str, tuple[str, list[int]]],
) -> SourceModelHeader:
    return SourceModelHeader(
        tensors={
            key: {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [0, 8],
            }
            for key, (dtype, shape) in descriptors.items()
        },
        metadata={},
    )


def read_raw_payload(
    file_path: Path,
    tensor_name: str,
) -> bytes:
    header = read_source_model_header(file_path)
    start, end = header.tensors[tensor_name]["data_offsets"]

    with file_path.open("rb") as file:
        header_length = int.from_bytes(file.read(8), "little")
        file.seek(8 + header_length + start)
        payload = file.read(end - start)

    return payload


class TestBuildMergePlan(unittest.TestCase):
    def test_resolves_and_validates_linear_and_additive_patches(self) -> None:
        adapter_header = make_header(
            {
                "diffusion_model.linear.lora_A": ("BF16", [2, 3]),
                "diffusion_model.linear.lora_B": ("BF16", [4, 2]),
                "diffusion_model.linear.alpha": ("F32", [1]),
                "diffusion_model.norm.scale.diff": ("F32", [4]),
            }
        )
        source_header = make_header(
            {
                "linear.weight": ("BF16", [4, 3]),
                "norm.scale": ("BF16", [4]),
            }
        )

        plan = build_merge_plan(
            source_header,
            adapter_header,
        )

        self.assertEqual(
            plan["linear_pairs"][0]["source_key"],
            "linear.weight",
        )
        self.assertEqual(
            plan["linear_pairs"][0]["alpha_key"],
            "diffusion_model.linear.alpha",
        )
        self.assertEqual(
            plan["additive_deltas"][0]["source_key"],
            "norm.scale",
        )

    def test_rejects_linear_source_shape_mismatch(self) -> None:
        adapter_header = make_header(
            {
                "diffusion_model.linear.lora_A": ("BF16", [2, 3]),
                "diffusion_model.linear.lora_B": ("BF16", [4, 2]),
            }
        )
        source_header = make_header(
            {
                "linear.weight": ("BF16", [5, 3]),
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "Source Linear tensor shape does not match",
        ):
            build_merge_plan(
                source_header,
                adapter_header,
            )

    def test_rejects_non_float_linear_source(self) -> None:
        adapter_header = make_header(
            {
                "diffusion_model.linear.lora_A": ("BF16", [2, 3]),
                "diffusion_model.linear.lora_B": ("BF16", [4, 2]),
            }
        )
        source_header = make_header(
            {
                "linear.weight": ("I8", [4, 3]),
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "Source tensor linear.weight must use BF16 or F32",
        ):
            build_merge_plan(
                source_header,
                adapter_header,
            )

    def test_accepts_f32_linear_and_additive_sources(self) -> None:
        adapter_header = make_header(
            {
                "diffusion_model.linear.lora_A": ("BF16", [2, 3]),
                "diffusion_model.linear.lora_B": ("BF16", [4, 2]),
                "diffusion_model.norm.scale.diff": ("F32", [4]),
            }
        )
        source_header = make_header(
            {
                "linear.weight": ("F32", [4, 3]),
                "norm.scale": ("F32", [4]),
            }
        )

        plan = build_merge_plan(
            source_header,
            adapter_header,
        )

        self.assertEqual(
            plan["linear_pairs"][0]["source_key"],
            "linear.weight",
        )
        self.assertEqual(
            plan["additive_deltas"][0]["source_key"],
            "norm.scale",
        )

    def test_rejects_unsupported_adapter_tensor(self) -> None:
        adapter_header = make_header(
            {
                "layer.some_unsupported_format": ("BF16", [2, 2]),
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "unsupported or unpaired tensors",
        ):
            build_merge_plan(
                make_header({}),
                adapter_header,
            )

    def test_reports_known_unsupported_adapter_contract(self) -> None:
        adapter_header = make_header(
            {
                "layer.hada_w1_a": ("BF16", [2, 2]),
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "Unsupported contracts: \['loha'\]",
        ):
            build_merge_plan(
                make_header({}),
                adapter_header,
            )

    def test_rejects_additive_source_shape_mismatch(self) -> None:
        adapter_header = make_header(
            {
                "diffusion_model.norm.scale.diff": ("F32", [4]),
            }
        )
        source_header = make_header(
            {
                "norm.scale": ("BF16", [5]),
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "Additive delta shape does not match source tensor",
        ):
            build_merge_plan(
                source_header,
                adapter_header,
            )


class TestStreamingBf16Merge(unittest.TestCase):
    def test_merges_hybrid_adapter_and_preserves_untouched_payloads(self) -> None:
        source_tensors = {
            "layer.weight": torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ],
                dtype=torch.bfloat16,
            ),
            "layer.bias": torch.tensor(
                [0.25, -0.5],
                dtype=torch.bfloat16,
            ),
            "norm.scale": torch.tensor(
                [1.0, 2.0],
                dtype=torch.bfloat16,
            ),
        }
        adapter_tensors = {
            "adapter.layer.lora_A": torch.tensor(
                [
                    [1.0, 0.0, -1.0],
                    [0.5, 1.0, 0.0],
                ],
                dtype=torch.bfloat16,
            ),
            "adapter.layer.lora_B": torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 2.0],
                ],
                dtype=torch.bfloat16,
            ),
            "adapter.layer.alpha": torch.tensor(
                [2.0],
                dtype=torch.float32,
            ),
            "adapter.norm.scale.diff": torch.tensor(
                [0.5, -0.25],
                dtype=torch.float32,
            ),
        }

        with TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_path = directory_path / "source.safetensors"
            adapter_path = directory_path / "adapter.safetensors"
            output_path = directory_path / "merged.safetensors"
            progress: list[tuple[int, int, str]] = []

            save_file(
                source_tensors,
                str(source_path),
                metadata={"fixture": "hybrid"},
            )
            save_file(adapter_tensors, str(adapter_path))

            untouched_payload = read_raw_payload(
                source_path,
                "layer.bias",
            )

            merge_bf16_adapters(
                source_path,
                output_path,
                (
                    AdapterMergeInput(
                        path=adapter_path,
                        strength=0.5,
                    ),
                ),
                on_tensor_started=(
                    lambda index, count, name: progress.append(
                        (index, count, name)
                    )
                ),
            )

            merged_tensors = load_file(str(output_path))
            merged_header = read_source_model_header(output_path)
            merged_untouched_payload = read_raw_payload(
                output_path,
                "layer.bias",
            )
            source_order = list(
                read_source_model_header(source_path).tensors
            )

        expected_weight = torch.tensor(
            [
                [1.5, 2.0, 2.5],
                [4.5, 6.0, 6.0],
            ],
            dtype=torch.bfloat16,
        )
        expected_scale = torch.tensor(
            [1.25, 1.875],
            dtype=torch.bfloat16,
        )

        self.assertTrue(
            torch.equal(
                merged_tensors["layer.weight"],
                expected_weight,
            )
        )
        self.assertTrue(
            torch.equal(
                merged_tensors["norm.scale"],
                expected_scale,
            )
        )
        self.assertEqual(
            merged_tensors["layer.bias"].dtype,
            torch.bfloat16,
        )
        self.assertEqual(
            list(merged_header.tensors),
            source_order,
        )
        self.assertEqual(
            merged_header.metadata,
            {"fixture": "hybrid"},
        )
        self.assertEqual(
            merged_untouched_payload,
            untouched_payload,
        )
        self.assertEqual(
            [item[0] for item in progress],
            [1, 2, 3],
        )
        self.assertTrue(
            all(item[1] == 3 for item in progress)
        )

    def test_sums_multiple_adapters_on_the_same_weight(self) -> None:
        source_tensors = {
            "layer.weight": torch.tensor(
                [[2.0, 3.0]],
                dtype=torch.bfloat16,
            ),
        }
        first_adapter = {
            "layer.lora_A": torch.tensor(
                [[1.0, 0.0]],
                dtype=torch.bfloat16,
            ),
            "layer.lora_B": torch.tensor(
                [[1.0]],
                dtype=torch.bfloat16,
            ),
        }
        second_adapter = {
            "layer.lora_A": torch.tensor(
                [[0.0, 1.0]],
                dtype=torch.bfloat16,
            ),
            "layer.lora_B": torch.tensor(
                [[1.0]],
                dtype=torch.bfloat16,
            ),
        }

        with TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_path = directory_path / "source.safetensors"
            first_path = directory_path / "first.safetensors"
            second_path = directory_path / "second.safetensors"
            output_path = directory_path / "merged.safetensors"

            save_file(source_tensors, str(source_path))
            save_file(first_adapter, str(first_path))
            save_file(second_adapter, str(second_path))

            merge_bf16_adapters(
                source_path,
                output_path,
                (
                    AdapterMergeInput(first_path, 1.0),
                    AdapterMergeInput(second_path, 1.0),
                ),
            )

            merged_tensors = load_file(str(output_path))

        self.assertTrue(
            torch.equal(
                merged_tensors["layer.weight"],
                torch.tensor(
                    [[3.0, 4.0]],
                    dtype=torch.bfloat16,
                ),
            )
        )

    def test_rejects_unsupported_adapter_before_creating_output(self) -> None:
        source_tensors = {
            "layer.weight": torch.ones(
                (1, 1),
                dtype=torch.bfloat16,
            ),
        }
        adapter_tensors = {
            "layer.some_unsupported_format": torch.ones(
                (1, 1),
                dtype=torch.bfloat16,
            ),
        }

        with TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_path = directory_path / "source.safetensors"
            adapter_path = directory_path / "adapter.safetensors"
            output_path = directory_path / "merged.safetensors"

            save_file(source_tensors, str(source_path))
            save_file(adapter_tensors, str(adapter_path))

            with self.assertRaisesRegex(
                ValueError,
                "unsupported or unpaired tensors",
            ):
                merge_bf16_adapters(
                    source_path,
                    output_path,
                    (AdapterMergeInput(adapter_path, 1.0),),
                )

            self.assertFalse(output_path.exists())
            self.assertFalse(
                output_path.with_name(
                    f"{output_path.name}.partial"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
