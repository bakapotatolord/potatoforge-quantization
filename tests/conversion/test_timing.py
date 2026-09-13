import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import save_file

from potatoforge.converter import convert_model
from potatoforge.profiles import QuantizationProfile
from potatoforge.timing import (
    SIZE_BUCKETS,
    TIMING_STAGES,
    TensorTiming,
    TimingCollector,
    timed_stage,
)


def timing_record(
    name: str,
    action: str,
    source_bytes: int,
    total: float,
    stages: dict[str, float],
    internal_stages: dict[str, float] | None = None,
) -> TensorTiming:
    return TensorTiming(
        tensor_name=name,
        action=action,
        shape=(2, 2),
        source_bytes=source_bytes,
        total=total,
        stages=stages,
        internal_stages=internal_stages or {},
    )


class TestTimingCollector(unittest.TestCase):
    def test_timed_stage_records_without_timing_a_disabled_target(self) -> None:
        stages: dict[str, float] = {}

        with timed_stage(stages, "quantize"):
            pass
        with timed_stage(None, "quantize"):
            pass

        self.assertIn("quantize", stages)
        self.assertGreaterEqual(stages["quantize"], 0.0)

    def test_stage_action_and_size_aggregates(self) -> None:
        collector = TimingCollector()
        collector.records = [
            timing_record(
                "keep",
                "keep",
                512 * 1024,
                0.5,
                {"read": 0.2, "write": 0.1},
            ),
            timing_record(
                "int8",
                "int8",
                2 * 1024**2,
                1.5,
                {
                    "read": 0.2,
                    "materialize": 0.3,
                    "quantize": 0.8,
                    "payload_bytes": 0.1,
                    "write": 0.1,
                },
            ),
            timing_record(
                "convrot",
                "int8_convrot",
                17 * 1024**2,
                2.5,
                {"read": 0.3, "quantize": 1.0, "write": 0.2},
            ),
        ]
        collector.wall_seconds = 5.0

        self.assertEqual(collector.stage_totals()["read"], 0.7)
        self.assertEqual(
            {
                action: (count, total)
                for action, count, total in collector.action_totals()
            },
            {
                "keep": (1, 0.5),
                "int8": (1, 1.5),
                "int8_convrot": (1, 2.5),
            },
        )

        size_totals = collector.size_totals()
        self.assertEqual(
            [size_totals[bucket]["tensors"] for bucket in SIZE_BUCKETS],
            [1, 1, 1],
        )
        self.assertAlmostEqual(collector.other_seconds(), 1.7)

    def test_missing_stages_and_zero_total_render_without_division_error(self) -> None:
        collector = TimingCollector()
        collector.records = [
            timing_record("keep", "keep", 1, 0.0, {"read": 0.0})
        ]
        collector.wall_seconds = 0.0

        report = collector.render()

        self.assertIn("Quantization timing", report)
        self.assertIn("0.0%", report)
        for stage in TIMING_STAGES:
            self.assertIn(stage, report)

    def test_other_time_is_clamped_when_stages_exceed_wall_time(self) -> None:
        collector = TimingCollector()
        collector.records = [
            timing_record("tensor", "int8", 1, 0.1, {"quantize": 0.2})
        ]
        collector.wall_seconds = 0.1

        self.assertEqual(collector.other_seconds(), 0.0)
        self.assertIn("other", collector.render())

    def test_int8_convrot_internal_aggregates_and_render(self) -> None:
        collector = TimingCollector()
        collector.records = [
            timing_record(
                "small",
                "int8_convrot",
                512 * 1024,
                1.2,
                {"quantize": 1.0},
                {
                    "rotation": 0.6,
                    "scale": 0.1,
                    "quantize_values": 0.2,
                    "finalize": 0.05,
                },
            ),
            timing_record(
                "large",
                "int8_convrot",
                17 * 1024**2,
                2.2,
                {"quantize": 2.0},
                {
                    "rotation": 1.0,
                    "scale": 0.2,
                    "quantize_values": 0.5,
                    "finalize": 0.1,
                },
            ),
            timing_record("plain", "int8", 2, 0.1, {"quantize": 0.1}),
        ]
        collector.wall_seconds = 4.0

        internal, quantize_total, other = (
            collector.convrot_internal_totals(("int8_convrot",))
        )
        self.assertAlmostEqual(internal["rotation"], 1.6)
        self.assertAlmostEqual(quantize_total, 3.0)
        self.assertAlmostEqual(other, 0.25)

        size_totals = collector.convrot_internal_size_totals(("int8_convrot",))
        self.assertEqual(
            [size_totals[bucket]["tensors"] for bucket in SIZE_BUCKETS],
            [1, 0, 1],
        )
        self.assertAlmostEqual(
            float(size_totals["<1 MiB"]["stages"]["rotation"]),
            0.6,
        )

        clamped = TimingCollector()
        clamped.records = [
            timing_record(
                "noise",
                "int8_convrot",
                1,
                0.1,
                {"quantize": 0.1},
                {"rotation": 0.2},
            )
        ]
        self.assertEqual(
            clamped.convrot_internal_totals(("int8_convrot",))[2],
            0.0,
        )

        report = collector.render()
        self.assertIn("INT8 ConvRot breakdown", report)
        self.assertIn("other_internal", report)
        self.assertIn("INT8 ConvRot slowest tensors", report)

    def test_new_convrot_actions_include_pack_in_nested_timing(self) -> None:
        collector = TimingCollector()
        collector.records = [
            timing_record(
                "w4",
                "convrot_w4a4",
                2 * 1024**2,
                2.0,
                {"quantize": 1.5},
                {
                    "rotation": 0.5,
                    "scale": 0.2,
                    "quantize_values": 0.4,
                    "pack": 0.3,
                    "finalize": 0.1,
                },
            ),
            timing_record(
                "w6",
                "int6_convrot",
                17 * 1024**2,
                3.0,
                {"quantize": 2.5},
                {
                    "prepare": 0.2,
                    "rotation": 0.8,
                    "scale": 0.3,
                    "quantize_values": 0.6,
                    "pack": 0.4,
                    "finalize": 0.2,
                },
            ),
        ]
        collector.wall_seconds = 5.5

        internal, quantize_total, other = collector.convrot_internal_totals(
            ("convrot_w4a4", "int6_convrot")
        )
        self.assertAlmostEqual(internal["pack"], 0.7)
        self.assertAlmostEqual(quantize_total, 4.0)
        self.assertAlmostEqual(other, 0.0)

        report = collector.render()
        self.assertIn("W4A4 ConvRot breakdown", report)
        self.assertIn("INT6 ConvRot breakdown", report)
        self.assertIn("W4A4 ConvRot slowest tensors", report)
        self.assertIn("INT6 ConvRot slowest tensors", report)

    def test_int6_validation_stages_are_counted_as_nested_timing(self) -> None:
        collector = TimingCollector()
        collector.records = [
            timing_record(
                "w6",
                "int6_convrot",
                17 * 1024**2,
                3.0,
                {"quantize": 2.5},
                {
                    "resolve_device": 0.01,
                    "validate_metadata": 0.01,
                    "validate_finite": 1.2,
                    "prepare": 0.2,
                    "rotation": 0.8,
                    "scale": 0.1,
                    "quantize_values": 0.1,
                    "pack": 0.05,
                    "finalize": 0.02,
                },
            )
        ]

        internal, quantize_total, other = collector.convrot_internal_totals(
            ("int6_convrot",)
        )

        self.assertAlmostEqual(internal["validate_finite"], 1.2)
        self.assertAlmostEqual(quantize_total, 2.5)
        self.assertAlmostEqual(other, 0.01)
        self.assertIn("validate_finite", collector.render())

    def test_w4a4_mse_internal_stages_render_separately(self) -> None:
        collector = TimingCollector()
        collector.records = [
            timing_record(
                "mse",
                "convrot_w4a4_mse",
                17 * 1024**2,
                4.0,
                {"quantize": 3.5},
                {
                    "prepare": 0.2,
                    "rotation": 0.3,
                    "scale_init": 0.1,
                    "zero_row_check": 0.01,
                    "coarse_search": 1.5,
                    "fine_search": 1.0,
                    "final_quantize": 0.2,
                    "pack": 0.1,
                    "finalize": 0.09,
                },
            )
        ]

        internal, quantize_total, other = collector.convrot_internal_totals(
            ("convrot_w4a4_mse",)
        )

        self.assertAlmostEqual(internal["coarse_search"], 1.5)
        self.assertAlmostEqual(quantize_total, 3.5)
        self.assertAlmostEqual(other, 0.0)
        self.assertIn("W4A4 ConvRot MSE breakdown", collector.render())

    def test_size_bucket_boundaries_are_source_byte_based(self) -> None:
        self.assertEqual(TimingCollector.size_bucket(1024**2 - 1), "<1 MiB")
        self.assertEqual(TimingCollector.size_bucket(1024**2), "1–16 MiB")
        self.assertEqual(
            TimingCollector.size_bucket(16 * 1024**2),
            "1–16 MiB",
        )
        self.assertEqual(
            TimingCollector.size_bucket(16 * 1024**2 + 1),
            ">16 MiB",
        )

    def test_write_payload_finishes_a_tensor_after_all_payloads(self) -> None:
        collector = TimingCollector()
        entry = {
            "tensor_name": "tensor",
            "action": "int8",
            "shape": (2, 2),
            "input_bytes": 8,
        }
        record = collector.start_tensor(entry)
        collector.register_payloads(record, ("tensor", "tensor_scale"))

        with collector.write_payload("tensor"):
            pass
        self.assertEqual(record.total, 0.0)
        with collector.write_payload("tensor_scale"):
            pass

        self.assertGreaterEqual(record.total, 0.0)
        self.assertGreaterEqual(record.stages["write"], 0.0)
        self.assertFalse(record.pending_payloads)


class TestTimedConversion(unittest.TestCase):
    PROFILE: QuantizationProfile = {
        "default": "keep",
        "rules": (
            {
                "action": "int8",
                "prefix": "",
                "suffixes": (".weight",),
            },
        ),
    }

    def test_timing_is_observational_and_preserves_payload(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.safetensors"
            untimed_path = root / "untimed.safetensors"
            timed_path = root / "timed.safetensors"
            save_file(
                {
                    "blocks.0.weight": torch.tensor(
                        [[1.0, -1.0, 0.5, 0.0], [0.25, 0.0, -0.5, 1.0]],
                        dtype=torch.bfloat16,
                    ),
                    "blocks.0.bias": torch.tensor(
                        [1.0, -1.0],
                        dtype=torch.bfloat16,
                    ),
                },
                str(source_path),
            )

            convert_model(
                source_path,
                untimed_path,
                self.PROFILE,
                on_entry_started=None,
            )
            timing = TimingCollector()
            convert_model(
                source_path,
                timed_path,
                self.PROFILE,
                on_entry_started=None,
                timing=timing,
            )

            self.assertEqual(untimed_path.read_bytes(), timed_path.read_bytes())

        self.assertIsNotNone(timing.wall_seconds)
        records = {record.tensor_name: record for record in timing.records}
        self.assertEqual(set(records), {"blocks.0.weight", "blocks.0.bias"})
        self.assertNotIn("materialize", records["blocks.0.bias"].stages)
        self.assertNotIn("quantize", records["blocks.0.bias"].stages)
        for stage in ("read", "materialize", "quantize", "payload_bytes", "write"):
            self.assertIn(stage, records["blocks.0.weight"].stages)

    def test_timed_convrot_preserves_payload_and_records_internal_stages(
        self,
    ) -> None:
        profile: QuantizationProfile = {
            "default": "keep",
            "rules": (
                {
                    "action": "int8_convrot",
                    "prefix": "",
                    "suffixes": (".weight",),
                },
            ),
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.safetensors"
            untimed_path = root / "untimed.safetensors"
            timed_path = root / "timed.safetensors"
            save_file(
                {
                    "blocks.0.weight": torch.arange(
                        256,
                        dtype=torch.float32,
                    ).reshape(1, 256).to(torch.bfloat16),
                    "blocks.0.bias": torch.tensor(
                        [1.0, -1.0],
                        dtype=torch.bfloat16,
                    ),
                },
                str(source_path),
            )

            convert_model(
                source_path,
                untimed_path,
                profile,
                on_entry_started=None,
            )
            timing = TimingCollector()
            convert_model(
                source_path,
                timed_path,
                profile,
                on_entry_started=None,
                timing=timing,
            )

            self.assertEqual(untimed_path.read_bytes(), timed_path.read_bytes())

        convrot = next(
            record
            for record in timing.records
            if record.action == "int8_convrot"
        )
        self.assertIn("rotation", convrot.internal_stages)
        self.assertIn("scale", convrot.internal_stages)
        self.assertIn("quantize_values", convrot.internal_stages)
        self.assertIn("finalize", convrot.internal_stages)


if __name__ == "__main__":
    unittest.main()
