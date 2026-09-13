import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import save_file

from potatoforge.audits.weight_audit import (
    audit_bf16_source,
    print_weight_audit_table,
    write_weight_audit_report,
)


class TestWeightAudit(unittest.TestCase):
    def test_audits_w4a4_mse_alongside_the_baseline(self) -> None:
        weights = torch.zeros((2, 256), dtype=torch.bfloat16)
        weights[0] = torch.linspace(
            -0.1,
            0.1,
            steps=256,
            dtype=torch.bfloat16,
        )
        weights[0, 0] = 3.0

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.safetensors"
            save_file(
                {"blocks.0.attn.wq.weight": weights},
                str(source_path),
            )
            document = audit_bf16_source(source_path)

        methods = document["results"][0]["methods"]
        self.assertEqual(
            set(methods),
            {
                "bf16",
                "int8",
                "int6",
                "int8_convrot",
                "int6_convrot",
                "convrot_w4a4",
                "convrot_w4a4_mse",
            },
        )
        self.assertEqual(
            methods["convrot_w4a4_mse"]["storage_bytes"],
            methods["convrot_w4a4"]["storage_bytes"],
        )
        self.assertLessEqual(
            methods["convrot_w4a4_mse"]["relative_l2_error"],
            methods["convrot_w4a4"]["relative_l2_error"] + 1e-7,
        )

    def test_can_skip_plain_int8_and_int6_methods(self) -> None:
        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.safetensors"
            save_file(
                {
                    "blocks.0.attn.wq.weight": torch.ones(
                        (2, 256),
                        dtype=torch.bfloat16,
                    ),
                },
                str(source_path),
            )

            methods = audit_bf16_source(
                source_path,
                include_plain_methods=False,
            )["results"][0]["methods"]

        self.assertIsNone(methods["int8"]["relative_l2_error"])
        self.assertIsNone(methods["int6"]["relative_l2_error"])
        self.assertIsNotNone(
            methods["int8_convrot"]["relative_l2_error"]
        )
        self.assertIsNotNone(
            methods["int6_convrot"]["relative_l2_error"]
        )
        self.assertIsNotNone(
            methods["convrot_w4a4"]["relative_l2_error"]
        )
        self.assertIsNotNone(
            methods["convrot_w4a4_mse"]["relative_l2_error"]
        )

    def test_audits_eligible_weights_without_a_profile(self) -> None:
        weights = torch.linspace(
            -1.0,
            1.0,
            steps=512,
            dtype=torch.bfloat16,
        ).reshape(2, 256)
        narrow_weights = torch.ones(
            (2, 128),
            dtype=torch.bfloat16,
        )

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.safetensors"
            save_file(
                {
                    "blocks.0.attn.wq.weight": weights,
                    "blocks.0.mlp.down.weight": narrow_weights,
                    "blocks.0.attn.bias": torch.ones(
                        (256,),
                        dtype=torch.bfloat16,
                    ),
                    "embedding": torch.ones(
                        (2, 256),
                        dtype=torch.float32,
                    ),
                },
                str(source_path),
            )

            document = audit_bf16_source(source_path)

        results = document["results"]
        self.assertEqual(len(results), 2)
        self.assertEqual(document["summary"]["audited_layer_count"], 2)
        self.assertEqual(document["summary"]["skipped_tensor_count"], 2)
        self.assertEqual(document["format_version"], 5)

        wide_result = results[0]
        self.assertEqual(
            wide_result["tensor_name"],
            "blocks.0.attn.wq.weight",
        )
        self.assertEqual(
            wide_result["methods"]["bf16"]["relative_l2_error"],
            0.0,
        )
        self.assertLess(
            wide_result["methods"]["convrot_w4a4"]["storage_bytes"],
            wide_result["methods"]["int8"]["storage_bytes"],
        )
        self.assertLess(
            wide_result["methods"]["int6"]["storage_bytes"],
            wide_result["methods"]["int8"]["storage_bytes"],
        )
        self.assertIsNotNone(
            wide_result["error_deltas"]["int8_convrot_vs_int8"]
        )
        self.assertIsNotNone(
            wide_result["methods"]["int6_convrot"]["relative_l2_error"]
        )
        narrow_result = results[1]
        self.assertIsNotNone(
            narrow_result["methods"]["int6"]["relative_l2_error"]
        )
        self.assertIsNone(
            narrow_result["methods"]["int8_convrot"]["relative_l2_error"]
        )
        self.assertIsNone(
            narrow_result["methods"]["int6_convrot"]["relative_l2_error"]
        )
        self.assertIsNone(
            narrow_result["methods"]["convrot_w4a4"]["storage_bytes"]
        )

    def test_audits_only_the_requested_tensor(self) -> None:
        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.safetensors"
            save_file(
                {
                    "blocks.0.attn.wq.weight": torch.ones(
                        (2, 256),
                        dtype=torch.bfloat16,
                    ),
                    "blocks.1.attn.wq.weight": torch.ones(
                        (2, 256),
                        dtype=torch.bfloat16,
                    ),
                },
                str(source_path),
            )

            document = audit_bf16_source(
                source_path,
                tensor_name="blocks.1.attn.wq.weight",
            )

        self.assertEqual(
            [result["tensor_name"] for result in document["results"]],
            ["blocks.1.attn.wq.weight"],
        )
        self.assertEqual(document["summary"]["audited_layer_count"], 1)
        self.assertEqual(document["summary"]["skipped_tensor_count"], 1)

    def test_audits_multiple_requested_tensors(self) -> None:
        tensor_names = (
            "blocks.0.attn.wq.weight",
            "blocks.1.attn.wq.weight",
        )
        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.safetensors"
            save_file(
                {
                    name: torch.ones((2, 256), dtype=torch.bfloat16)
                    for name in tensor_names
                },
                str(source_path),
            )

            document = audit_bf16_source(
                source_path,
                tensor_names=tensor_names,
            )

        self.assertEqual(
            [result["tensor_name"] for result in document["results"]],
            list(tensor_names),
        )

    def test_contains_measurements_without_a_recommendation(self) -> None:
        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.safetensors"
            save_file(
                {
                    "blocks.0.attn.wq.weight": torch.zeros(
                        (2, 256),
                        dtype=torch.bfloat16,
                    ),
                },
                str(source_path),
            )

            document = audit_bf16_source(source_path)

        self.assertNotIn("recommendation", document["results"][0])
        self.assertNotIn("policy", document)

    def test_audits_float16_weights_with_all_methods(self) -> None:
        weights = torch.linspace(
            -1.0,
            1.0,
            steps=256,
            dtype=torch.float16,
        ).reshape(1, 256)

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source-float16.safetensors"
            save_file(
                {"blocks.0.attn.wq.weight": weights},
                str(source_path),
            )

            document = audit_bf16_source(source_path)

        result = document["results"][0]
        self.assertEqual(document["selection"]["dtype"], "F16")
        self.assertIsNotNone(result["methods"]["int8"]["relative_l2_error"])
        self.assertIsNotNone(
            result["methods"]["int8_convrot"]["relative_l2_error"]
        )
        self.assertIsNotNone(
            result["methods"]["convrot_w4a4"]["relative_l2_error"]
        )

    def test_audits_float32_weights_with_all_methods(self) -> None:
        weights = torch.linspace(
            -1.0,
            1.0,
            steps=256,
            dtype=torch.float32,
        ).reshape(1, 256)

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source-float32.safetensors"
            save_file(
                {"blocks.0.attn.wq.weight": weights},
                str(source_path),
            )

            document = audit_bf16_source(source_path)

        result = document["results"][0]
        self.assertEqual(document["selection"]["dtype"], "F32")
        self.assertIsNotNone(result["methods"]["int8"]["relative_l2_error"])
        self.assertIsNotNone(
            result["methods"]["int8_convrot"]["relative_l2_error"]
        )
        self.assertIsNotNone(
            result["methods"]["convrot_w4a4"]["relative_l2_error"]
        )

    def test_prints_the_actual_source_dtype(self) -> None:
        weights = torch.zeros(
            (1, 256),
            dtype=torch.float16,
        )

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source-float16.safetensors"
            save_file(
                {"blocks.0.attn.wq.weight": weights},
                str(source_path),
            )
            document = audit_bf16_source(source_path)

        output = StringIO()
        with redirect_stdout(output):
            print_weight_audit_table(
                document["results"],
                source_dtype=str(document["selection"]["dtype"]),
            )

        self.assertIn("F16 KiB", output.getvalue())
        self.assertIn("W4A4 MSE KiB", output.getvalue())
        self.assertIn("against F16", output.getvalue())

    def test_writes_a_report_without_overwriting(self) -> None:
        document = {
            "format_version": 5,
            "source_path": "source.safetensors",
            "selection": {
                "dtype": "BF16",
                "rank": 2,
                "name_suffix": ".weight",
            },
            "summary": {
                "source_tensor_count": 0,
                "audited_layer_count": 0,
                "skipped_tensor_count": 0,
                "storage_bytes": {
                    "bf16": 0,
                    "int8": 0,
                    "int6": 0,
                    "int8_convrot": 0,
                    "int6_convrot": 0,
                    "convrot_w4a4": 0,
                    "convrot_w4a4_mse": 0,
                },
            },
            "results": [],
        }

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "weight-audit.json"

            write_weight_audit_report(output_path, document)

            with output_path.open(encoding="utf-8") as report_file:
                saved_document = json.load(report_file)

            with self.assertRaises(FileExistsError):
                write_weight_audit_report(output_path, document)

        self.assertEqual(saved_document["format_version"], 5)
        self.assertEqual(saved_document["results"], [])

    def test_records_known_tensor_energy(self) -> None:
        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "tiny.safetensors"
            save_file(
                {"tiny.weight": torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16)},
                str(source_path),
            )

            document = audit_bf16_source(source_path)

        self.assertAlmostEqual(document["results"][0]["weight_l2_sq"], 25.0)
