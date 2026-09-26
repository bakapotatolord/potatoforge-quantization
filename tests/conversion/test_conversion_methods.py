import unittest

import torch

from potatoforge.conversion_methods import (
    convrot_conversion_methods,
    get_conversion_method,
    supported_quantizing_actions,
)


class TestConversionMethods(unittest.TestCase):
    def test_catalog_actions_and_convrot_order(self) -> None:
        self.assertEqual(
            set(supported_quantizing_actions()),
            {
                "int8",
                "int6_rowwise",
                "int7_rowwise",
                "int5_convrot",
                "int5_convrot_mse",
                "int6_convrot",
                "int6_convrot_mse",
                "int7_convrot",
                "int7_convrot_mse",
                "int8_convrot",
                "convrot_w4a4",
                "convrot_w4a4_mse",
                "convrot_w4a4_mse_gptq",
            },
        )
        self.assertEqual(
            tuple(method.action for method in convrot_conversion_methods()),
            (
                "int8_convrot",
                "convrot_w4a4",
                "convrot_w4a4_mse",
                "convrot_w4a4_mse_gptq",
                "int5_convrot",
                "int5_convrot_mse",
                "int6_convrot",
                "int6_convrot_mse",
                "int7_convrot",
                "int7_convrot_mse",
            ),
        )

    def test_lookup_rejects_keep_and_unknown_actions(self) -> None:
        for action in ("keep", "unknown"):
            with self.subTest(action=action):
                with self.assertRaisesRegex(
                    ValueError,
                    "Unsupported quantization action",
                ):
                    get_conversion_method(action)

    def test_mse_methods_reuse_their_ordinary_plan_and_marker(self) -> None:
        for ordinary, mse, label in (
            ("int5_convrot", "int5_convrot_mse", "INT5 ConvRot MSE"),
            ("int6_convrot", "int6_convrot_mse", "INT6 ConvRot MSE"),
            ("int7_convrot", "int7_convrot_mse", "INT7 ConvRot MSE"),
        ):
            with self.subTest(mse=mse):
                ordinary_method = get_conversion_method(ordinary)
                mse_method = get_conversion_method(mse)

                self.assertEqual(mse_method.label, label)
                self.assertTrue(mse_method.is_convrot)
                self.assertIs(mse_method.plan, ordinary_method.plan)
                self.assertEqual(
                    mse_method.marker_payload,
                    ordinary_method.marker_payload,
                )
                self.assertEqual(
                    mse_method.internal_stages,
                    (
                        "prepare",
                        "rotation",
                        "scale_init",
                        "zero_row_check",
                        "coarse_search",
                        "fine_search",
                        "final_quantize",
                        "pack",
                        "finalize",
                    ),
                )

    def test_encoders_match_their_planned_cpu_payloads(self) -> None:
        weights = torch.zeros((2, 256), dtype=torch.bfloat16)
        descriptor = {
            "dtype": "BF16",
            "shape": [2, 256],
            "data_offsets": [0, 1024],
        }
        expected_dtypes = {
            "I8": torch.int8,
            "U8": torch.uint8,
            "F32": torch.float32,
        }

        for action in supported_quantizing_actions():
            with self.subTest(action=action):
                method = get_conversion_method(action)
                plan = method.plan("blocks.0.weight", descriptor)
                if action == "convrot_w4a4_mse_gptq":
                    with self.assertRaisesRegex(ValueError, "requires a calibration"):
                        method.encode(weights, "cpu", None)
                    self.assertEqual(method.marker_payload, get_conversion_method("convrot_w4a4_mse").marker_payload)
                    continue
                encoded = method.encode(weights, "cpu", None)
                code_spec, scale_spec, _ = plan.output_tensors

                self.assertEqual(encoded.codes.device.type, "cpu")
                self.assertEqual(encoded.scales.device.type, "cpu")
                self.assertEqual(
                    encoded.codes.dtype,
                    expected_dtypes[code_spec.dtype],
                )
                self.assertEqual(
                    encoded.scales.dtype,
                    expected_dtypes[scale_spec.dtype],
                )
                self.assertEqual(tuple(encoded.codes.shape), code_spec.shape)
                self.assertEqual(tuple(encoded.scales.shape), scale_spec.shape)


if __name__ == "__main__":
    unittest.main()
