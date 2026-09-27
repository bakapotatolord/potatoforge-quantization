"""Quantize-specific command and progress handling for local jobs."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from ..converter import ConversionCancelled, format_conversion_progress
from .jobs import Job, LocalJobManager

logger = logging.getLogger(__name__)
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


class QuantizeJobManager(LocalJobManager):
    """Run one Quantize command with replayable progress and cancellation."""

    def start(self, request: Any, command: QuantizeCommand) -> dict[str, Any]:
        return super().start(lambda job: self._run(job, request, command))

    async def _run(self, job: Job, request: Any, command: QuantizeCommand) -> None:
        try:
            self.set_running(job)
            result = await asyncio.to_thread(
                command,
                request.source_path,
                request.output_path,
                profile=request.profile,
                device=request.device,
                adapter_path=request.adapter_path,
                adapter_strength=request.adapter_strength,
                dry_run=request.dry_run,
                timing=request.timing,
                on_entry_started=lambda index, total, entry: self._on_entry_started(
                    job, index, total, entry
                ),
            )
            final_result = quantize_result_payload(result, timing=request.timing)
        except ConversionCancelled:
            self.finish(job, "cancelled", error="Quantization stopped.")
        except (FileExistsError, FileNotFoundError, ValueError) as error:
            self.finish(job, "failed", error=str(error) or error.__class__.__name__)
        except Exception:
            logger.exception("Quantize job %s failed", job.id)
            self.finish(job, "failed", error="Quantization failed.")
        else:
            self.finish(job, "succeeded", result=final_result)

    def _on_entry_started(self, job: Job, index: int, total: int, entry: Any) -> None:
        if self.is_cancelled(job):
            raise ConversionCancelled
        self.record_progress(
            job,
            {
                "current": index,
                "total": total,
                "tensor_name": entry["tensor_name"],
            },
            format_conversion_progress(index, total, entry),
        )
