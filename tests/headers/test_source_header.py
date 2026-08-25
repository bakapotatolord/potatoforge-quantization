from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from safetensors.torch import save_file

from potatoforge.headers.source_header import read_source_model_header


class TestSourceModelHeader(unittest.TestCase):
    def test_reads_tensor_descriptors_and_metadata(self) -> None:
        weights = torch.tensor(
            [
                [1.0, 0.25, -1.0],
                [0.01, -0.02, 0.02],
            ],
            dtype=torch.bfloat16,
        )

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.safetensors"

            save_file(
                {"demo.weight": weights},
                str(source_path),
                metadata={"creator": "test"},
            )

            header = read_source_model_header(source_path)

        self.assertEqual(header.metadata, {"creator": "test"})
        self.assertNotIn("__metadata__", header.tensors)
        self.assertEqual(
            header.tensors["demo.weight"]["dtype"],
            "BF16",
        )
        self.assertEqual(
            header.tensors["demo.weight"]["shape"],
            [2, 3],
        )
        self.assertEqual(
            header.tensors["demo.weight"]["data_offsets"],
            [0, 12],
        )
