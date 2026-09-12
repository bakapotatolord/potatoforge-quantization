import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from potatoforge.headers.source_header import read_source_model_header
from potatoforge.patch_planning import build_patch_plan
from potatoforge.patching import execute_patch_plan


class TestPatchWriter(unittest.TestCase):
    def test_writes_one_patch_for_a_tensor_prefix(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.safetensors"
            output = root / "patch.safetensors"
            save_file(
                {
                    "blocks.0.weight": torch.tensor(
                        [[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16
                    ),
                    "blocks.1.weight": torch.tensor(
                        [[4.0, 3.0, 2.0, 1.0]], dtype=torch.bfloat16
                    ),
                },
                str(source),
            )
            plan = build_patch_plan(
                read_source_model_header(source).tensors,
                "blocks.*",
                "int8",
                "blocks",
            )

            execute_patch_plan(source, output, plan)

            header = read_source_model_header(output)
            self.assertEqual(
                list(header.tensors),
                [
                    "blocks.0.weight",
                    "blocks.0.weight_scale",
                    "blocks.0.comfy_quant",
                    "blocks.1.weight",
                    "blocks.1.weight_scale",
                    "blocks.1.comfy_quant",
                ],
            )

    def test_refuses_collisions_and_cleans_partial_output(self) -> None:
        weight = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.safetensors"
            output = root / "patch.safetensors"
            save_file({"B.weight": weight}, str(source))
            plan = build_patch_plan(
                read_source_model_header(source).tensors,
                "B.weight",
                "int8",
                "test-patch",
            )

            with self.assertRaisesRegex(ValueError, "same"):
                execute_patch_plan(source, source, plan)

            output.write_bytes(b"exists")
            with self.assertRaises(FileExistsError):
                execute_patch_plan(source, output, plan)
            output.unlink()

            def fail_writer(path: Path, *_: object, **__: object) -> None:
                path.write_bytes(b"partial")
                raise ValueError("write failed")

            with patch("potatoforge.patching.write_safetensors_file", fail_writer):
                with self.assertRaisesRegex(ValueError, "write failed"):
                    execute_patch_plan(source, output, plan)

            self.assertFalse(output.exists())
            self.assertFalse(output.with_name("patch.safetensors.partial").exists())


if __name__ == "__main__":
    unittest.main()
