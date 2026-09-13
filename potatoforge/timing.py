from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Mapping, Sequence

import torch


TIMING_STAGES = (
    "read",
    "materialize",
    "adapter_merge",
    "quantize",
    "pack",
    "payload_bytes",
    "write",
)
CONVROT_ACTIONS = (
    "int8_convrot",
    "convrot_w4a4",
    "convrot_w4a4_mse",
    "int6_convrot",
)
CONVROT_INTERNAL_STAGES = (
    "resolve_device",
    "validate_metadata",
    "validate_finite",
    "prepare",
    "rotation",
    "scale",
    "quantize_values",
    "pack",
    "finalize",
)
W4A4_MSE_INTERNAL_STAGES = (
    "prepare",
    "rotation",
    "scale_init",
    "zero_row_check",
    "coarse_search",
    "fine_search",
    "final_quantize",
    "pack",
    "finalize",
)
SIZE_BUCKETS = ("<1 MiB", "1–16 MiB", ">16 MiB")
_MIB = 1024**2


def _internal_stages_for_actions(
    actions: tuple[str, ...],
) -> tuple[str, ...]:
    stages: list[str] = []
    for action in actions:
        action_stages = (
            W4A4_MSE_INTERNAL_STAGES
            if action == "convrot_w4a4_mse"
            else CONVROT_INTERNAL_STAGES
        )
        for stage in action_stages:
            if stage not in stages:
                stages.append(stage)
    return tuple(stages)


@dataclass
class TensorTiming:
    tensor_name: str
    action: str
    shape: tuple[int, ...]
    source_bytes: int
    stages: dict[str, float] = field(default_factory=dict)
    internal_stages: dict[str, float] = field(default_factory=dict)
    total: float = 0.0
    started_at: float = field(default=0.0, repr=False)
    pending_payloads: set[str] = field(default_factory=set, repr=False)


@contextmanager
def timed_stage(
    target: dict[str, float] | TensorTiming | None,
    stage: str,
    *,
    synchronize: Callable[[object], None] | None = None,
    synchronize_arg: object = None,
) -> Iterator[None]:
    """Record a stage only when a timing target is present."""

    if target is None:
        yield
        return

    if synchronize is not None:
        synchronize(synchronize_arg)
    started_at = perf_counter()
    try:
        yield
    finally:
        if synchronize is not None:
            synchronize(synchronize_arg)
        elapsed = perf_counter() - started_at
        if isinstance(target, TensorTiming):
            target.stages[stage] = target.stages.get(stage, 0.0) + elapsed
        else:
            target[stage] = target.get(stage, 0.0) + elapsed


def timed_internal_stage(
    timings: dict[str, float] | None,
    stage: str,
    device: torch.device | None,
):
    return timed_stage(
        timings,
        stage,
        synchronize=(
            torch.cuda.synchronize
            if timings is not None and device is not None
            else None
        ),
        synchronize_arg=device,
    )


class TimingCollector:
    """Small opt-in collector for conversion-boundary timings."""

    def __init__(self) -> None:
        self.started_at = perf_counter()
        self.wall_seconds: float | None = None
        self.records: list[TensorTiming] = []
        self._payload_records: dict[str, TensorTiming] = {}

    def start_tensor(self, entry: Mapping[str, object]) -> TensorTiming:
        record = TensorTiming(
            tensor_name=str(entry["tensor_name"]),
            action=str(entry["action"]),
            shape=tuple(int(dimension) for dimension in entry["shape"]),
            source_bytes=int(entry["input_bytes"]),
            started_at=perf_counter(),
        )
        self.records.append(record)
        return record

    def register_payloads(
        self,
        record: TensorTiming,
        payload_names: Sequence[str],
    ) -> None:
        record.pending_payloads.update(payload_names)
        for payload_name in payload_names:
            self._payload_records[payload_name] = record

    def _finish_payload_write(self, payload_name: str) -> None:
        record = self._payload_records.pop(payload_name, None)
        if record is None:
            return

        record.pending_payloads.discard(payload_name)
        if not record.pending_payloads:
            record.total = max(0.0, perf_counter() - record.started_at)

    @contextmanager
    def write_payload(self, payload_name: str) -> Iterator[None]:
        """Time a successful writer operation for one payload."""

        if payload_name not in self._payload_records:
            yield
            return

        record = self._payload_records[payload_name]
        with timed_stage(record, "write"):
            yield

        self._finish_payload_write(payload_name)

    def finish(self) -> float:
        if self.wall_seconds is None:
            self.wall_seconds = max(0.0, perf_counter() - self.started_at)
        return self.wall_seconds

    def stage_totals(self) -> dict[str, float]:
        return {
            stage: sum(record.stages.get(stage, 0.0) for record in self.records)
            for stage in TIMING_STAGES
        }

    def other_seconds(self) -> float:
        total = max(0.0, self.wall_seconds or 0.0)
        return max(0.0, total - sum(self.stage_totals().values()))

    def action_totals(self) -> list[tuple[str, int, float]]:
        totals: dict[str, list[float]] = {}
        for record in self.records:
            count_and_time = totals.setdefault(record.action, [0.0, 0.0])
            count_and_time[0] += 1
            count_and_time[1] += record.total
        return sorted(
            [
                (
                    action,
                    int(count_and_time[0]),
                    count_and_time[1],
                )
                for action, count_and_time in totals.items()
            ],
            key=lambda item: (-item[2], item[0]),
        )

    @staticmethod
    def size_bucket(source_bytes: int) -> str:
        if source_bytes < _MIB:
            return SIZE_BUCKETS[0]
        if source_bytes <= 16 * _MIB:
            return SIZE_BUCKETS[1]
        return SIZE_BUCKETS[2]

    def size_totals(self) -> dict[str, dict[str, object]]:
        totals = {
            bucket: {
                "tensors": 0,
                "total": 0.0,
                "stages": {stage: 0.0 for stage in TIMING_STAGES},
            }
            for bucket in SIZE_BUCKETS
        }
        for record in self.records:
            bucket = totals[self.size_bucket(record.source_bytes)]
            bucket["tensors"] = int(bucket["tensors"]) + 1
            bucket["total"] = float(bucket["total"]) + record.total
            stages = bucket["stages"]
            for stage in TIMING_STAGES:
                stages[stage] = float(stages[stage]) + record.stages.get(
                    stage,
                    0.0,
                )
        return totals

    def convrot_internal_totals(
        self,
        actions: tuple[str, ...],
    ) -> tuple[dict[str, float], float, float]:
        internal_stages = _internal_stages_for_actions(actions)
        internal_totals = {
            stage: 0.0 for stage in internal_stages
        }
        quantize_total = 0.0
        for record in self.records:
            if record.action not in actions:
                continue
            quantize_total += record.stages.get("quantize", 0.0)
            for stage in internal_stages:
                internal_totals[stage] += record.internal_stages.get(
                    stage,
                    0.0,
                )
        other_internal = max(
            0.0,
            quantize_total - sum(internal_totals.values()),
        )
        return internal_totals, quantize_total, other_internal

    def convrot_internal_size_totals(
        self,
        actions: tuple[str, ...],
    ) -> dict[str, dict[str, object]]:
        internal_stages = _internal_stages_for_actions(actions)
        totals = {
            bucket: {
                "tensors": 0,
                "quantize": 0.0,
                "stages": {
                    stage: 0.0 for stage in internal_stages
                },
            }
            for bucket in SIZE_BUCKETS
        }
        for record in self.records:
            if record.action not in actions:
                continue
            bucket = totals[self.size_bucket(record.source_bytes)]
            bucket["tensors"] = int(bucket["tensors"]) + 1
            bucket["quantize"] = float(bucket["quantize"]) + record.stages.get(
                "quantize",
                0.0,
            )
            stages = bucket["stages"]
            for stage in internal_stages:
                stages[stage] = float(stages[stage]) + record.internal_stages.get(
                    stage,
                    0.0,
                )
        return totals

    @staticmethod
    def _percent(seconds: float, total: float) -> float:
        return 0.0 if total <= 0.0 else seconds / total * 100.0

    @staticmethod
    def _mib(source_bytes: int) -> str:
        return f"{source_bytes / _MIB:.1f} MiB"

    @staticmethod
    def _shape(shape: tuple[int, ...]) -> str:
        return "x".join(str(dimension) for dimension in shape) or "scalar"

    @staticmethod
    def _shorten(value: str, width: int) -> str:
        if len(value) <= width:
            return value
        return "..." + value[-(width - 3):]

    def render(self, *, top_n: int = 20) -> str:
        total = max(0.0, self.wall_seconds or 0.0)
        stage_totals = self.stage_totals()
        lines = [
            "Quantization timing",
            "=" * 64,
            f"Tensors: {len(self.records)}",
            "",
            f"{'Stage':<18} {'Time':>12} {'% total':>10}",
            "-" * 44,
        ]
        for stage in TIMING_STAGES:
            seconds = stage_totals[stage]
            lines.append(
                f"{stage:<18} {seconds:>8.3f} s {self._percent(seconds, total):>8.1f}%"
            )
        other = self.other_seconds()
        lines.extend(
            (
                f"{'other':<18} {other:>8.3f} s {self._percent(other, total):>8.1f}%",
                "-" * 44,
                f"{'total':<18} {total:>8.3f} s {self._percent(total, total):>8.1f}%",
                "",
                "By action",
                "=" * 64,
                f"{'Action':<28} {'Tensors':>8} {'Time':>12}",
                "-" * 52,
            )
        )
        for action, count, seconds in self.action_totals():
            lines.append(f"{action:<28} {count:>8} {seconds:>8.3f} s")

        lines.extend(("", "By tensor size", "=" * 64))
        size_totals = self.size_totals()
        for bucket in SIZE_BUCKETS:
            summary = size_totals[bucket]
            count = int(summary["tensors"])
            bucket_total = float(summary["total"])
            stages = summary["stages"]
            average = 0.0 if count == 0 else bucket_total / count
            lines.extend(
                (
                    bucket,
                    f"  tensors:       {count}",
                    f"  total:         {bucket_total:.3f} s",
                    f"  avg/tensor:    {average:.3f} s",
                    f"  materialize:   {float(stages['materialize']):.3f} s",
                    f"  quantize:      {float(stages['quantize']):.3f} s",
                    f"  pack:          {float(stages['pack']):.3f} s",
                    f"  write:         {float(stages['write']):.3f} s",
                    "",
                )
            )

        lines.extend(
            (
                "Slowest tensors",
                "=" * 64,
                f"{'Tensor':<38} {'Action':<20} {'Shape':>16} {'Source':>12} {'Total':>10}",
                "-" * 104,
            )
        )
        for record in sorted(
            self.records,
            key=lambda item: item.total,
            reverse=True,
        )[:top_n]:
            lines.append(
                f"{self._shorten(record.tensor_name, 38):<38} "
                f"{self._shorten(record.action, 20):<20} "
                f"{self._shape(record.shape):>16} "
                f"{self._mib(record.source_bytes):>12} "
                f"{record.total:>8.3f} s"
            )

        convrot_labels = {
            "int8_convrot": "INT8 ConvRot",
            "convrot_w4a4": "W4A4 ConvRot",
            "convrot_w4a4_mse": "W4A4 ConvRot MSE",
            "int6_convrot": "INT6 ConvRot",
        }
        for action in CONVROT_ACTIONS:
            convrot_records = [
                record
                for record in self.records
                if record.action == action
            ]
            if not convrot_records:
                continue
            label = convrot_labels[action]
            internal_stages = _internal_stages_for_actions((action,))
            internal_totals, quantize_total, other_internal = (
                self.convrot_internal_totals((action,))
            )
            lines.extend(
                (
                    "",
                    f"{label} breakdown",
                    "=" * 64,
                    f"{'Stage':<18} {'Time':>12} {'% quantize':>12}",
                    "-" * 48,
                )
            )
            for stage in internal_stages:
                seconds = internal_totals[stage]
                lines.append(
                    f"{stage:<18} {seconds:>8.3f} s "
                    f"{self._percent(seconds, quantize_total):>10.1f}%"
                )
            lines.extend(
                (
                    f"{'other_internal':<18} {other_internal:>8.3f} s "
                    f"{self._percent(other_internal, quantize_total):>10.1f}%",
                    "-" * 48,
                    f"{'quantize wall':<18} {quantize_total:>8.3f} s "
                    f"{self._percent(quantize_total, quantize_total):>10.1f}%",
                    "",
                    f"{label} by tensor size",
                    "=" * 64,
                )
            )
            internal_size_totals = self.convrot_internal_size_totals((action,))
            for bucket in SIZE_BUCKETS:
                summary = internal_size_totals[bucket]
                count = int(summary["tensors"])
                bucket_total = float(summary["quantize"])
                average = 0.0 if count == 0 else bucket_total / count
                stages = summary["stages"]
                lines.extend(
                    (
                        bucket,
                        f"  tensors:       {count}",
                        f"  quantize:      {bucket_total:.3f} s",
                        f"  avg/quantize:  {average:.3f} s",
                    )
                )
                for stage in internal_stages:
                    lines.append(
                        f"  {stage + ':':<17} "
                        f"{float(stages[stage]):.3f} s"
                    )
                lines.append("")

            lines.extend(
                (
                    f"{label} slowest tensors",
                    "=" * 64,
                )
            )
            for record in sorted(
                convrot_records,
                key=lambda item: item.total,
                reverse=True,
            )[:top_n]:
                lines.append(
                    f"{self._shorten(record.tensor_name, 38)} "
                    f"({self._shape(record.shape)}, "
                    f"{self._mib(record.source_bytes)}, "
                    f"total {record.total:.3f} s)"
                )
                for stage in internal_stages:
                    seconds = record.internal_stages.get(stage, 0.0)
                    if seconds > 0.0:
                        lines.append(f"  {stage:<16} {seconds:.3f} s")
                record_other = max(
                    0.0,
                    record.stages.get("quantize", 0.0)
                    - sum(
                        record.internal_stages.get(stage, 0.0)
                        for stage in internal_stages
                    ),
                )
                lines.append(f"  {'other_internal':<16} {record_other:.3f} s")

        lines.extend(
            (
                "",
                "Notes:",
                "  write is file.write() wall time; no flush or fsync is included.",
                "  other includes CLI/config, header/plan/layout, and iteration.",
                "  ConvRot details, including CUDA packing, are nested inside quantize and excluded from global stage totals.",
                "  INT6 validate_finite is the full input-tensor torch.isfinite(...).all() scan; CUDA runs it after H2D.",
                "  INT6 CUDA range validation is included in the measured pack stage.",
                "  CPU W4A4 INT4 packing remains inside its existing quantize action.",
            )
        )
        return "\n".join(lines)
