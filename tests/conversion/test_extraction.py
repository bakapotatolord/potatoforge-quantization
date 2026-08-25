import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import load_file, save_file

from potatoforge.extraction import extract_tensors


class TestExtraction(unittest.TestCase):
    def test_extracts_and_renames_a_prefix_without_loading_the_full_model(self) -> None:
        source_tensors = {
            "conditioner.embedders.0.transformer.text_model.weight": torch.tensor(
                [[1.0, 2.0]],
                dtype=torch.float16,
            ),
            "conditioner.embedders.0.other.weight": torch.tensor(
                [3.0],
                dtype=torch.float16,
            ),
        }

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.safetensors"
            output_path = Path(directory) / "clip_l.safetensors"
            save_file(source_tensors, str(source_path))

            selected_count, selected_bytes = extract_tensors(
                source_path,
                output_path,
                source_prefix="conditioner.embedders.0.transformer.",
            )

            output_tensors = load_file(str(output_path))

        self.assertEqual(selected_count, 1)
        self.assertEqual(selected_bytes, 4)
        self.assertEqual(
            tuple(output_tensors),
            ("text_model.weight",),
        )
        self.assertTrue(
            torch.equal(
                output_tensors["text_model.weight"],
                source_tensors[
                    "conditioner.embedders.0.transformer.text_model.weight"
                ],
            )
        )
