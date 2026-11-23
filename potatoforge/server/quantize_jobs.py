"""Quantize-specific command and progress handling for local jobs."""

from __future__ import annotations

from typing import Any, Callable

from ..converter import format_conversion_progress
from .jobs import CommandJobManager

QuantizeCommand = Callable[..., Any]


def quantize_result_payload(result: Any, *, timing: bool = False) -> dict[str, Any]:
    if result.estimated_bytes is not None:
        return {"estimated_output_bytes": result.estimated_bytes, "dry_run": True}
    output_path = result.output_path
    if output_path is None:
        raise ValueError("Quantize conversion did not produce an output path.")
    payload = {
        "output_path": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "adapter_count": result.adapter_count,
    }
    if timing and result.timing is not None:
        payload["timing_report"] = result.timing.render()
    return payload


class QuantizeJobManager(CommandJobManager):
    """Run one Quantize command with replayable progress and cancellation."""

    label = "Quantization"

    def start(self, request: Any, command: QuantizeCommand) -> dict[str, Any]:
        return super().start(
            lambda job: self.run_command(
                job,
                lambda: command(
                    request.source_path,
                    request.output_path,
                    profile=request.profile,
                    device=request.device,
                    adapter_path=request.adapter_path,
                    adapter_strength=request.adapter_strength,
                    dry_run=request.dry_run,
                    timing=request.timing,
                    on_entry_started=lambda index, total, entry: self.report_tensor(
                        job, index, total, entry["tensor_name"],
                        format_conversion_progress(index, total, entry),
                    ),
                ),
                lambda result: quantize_result_payload(result, timing=request.timing),
            )
        )
