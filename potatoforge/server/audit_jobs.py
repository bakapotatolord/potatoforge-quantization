"""Audit-specific command and progress handling for local jobs."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from ..converter import ConversionCancelled
from .jobs import Job, LocalJobManager

logger = logging.getLogger(__name__)
AuditCommand = Callable[..., Any]


def audit_result_payload(request: Any, result: Any) -> dict[str, Any]:
    summary = result.document["summary"]
    return {
        "source_path": str(request.source_path),
        "audit_path": str(request.output),
        "audited_layer_count": summary["audited_layer_count"],
        "skipped_tensor_count": summary["skipped_tensor_count"],
        "warning": "Weight reconstruction only; not runtime speed or image quality.",
    }


class AuditJobManager(LocalJobManager):
    """Run one Audit command with replayable progress and cancellation."""

    def start(self, request: Any, command: AuditCommand) -> dict[str, Any]:
        return super().start(lambda job: self._run(job, request, command))

    async def _run(self, job: Job, request: Any, command: AuditCommand) -> None:
        try:
            self.set_running(job)
            result = await asyncio.to_thread(
                command,
                request.source_path,
                request.output,
                device=request.device,
                overwrite=request.overwrite,
                methods=getattr(request, "methods", None),
                on_entry_started=lambda index, total, entry: self._on_entry_started(
                    job, index, total, entry
                ),
            )
            final_result = audit_result_payload(request, result)
        except ConversionCancelled:
            self.finish(job, "cancelled", error="Audit stopped.")
        except (FileExistsError, FileNotFoundError, ValueError) as error:
            self.finish(job, "failed", error=str(error) or error.__class__.__name__)
        except Exception:
            logger.exception("Audit job %s failed", job.id)
            self.finish(job, "failed", error="Audit failed.")
        else:
            self.finish(job, "succeeded", result=final_result)

    def _on_entry_started(self, job: Job, index: int, total: int, tensor_name: str) -> None:
        if self.is_cancelled(job):
            raise ConversionCancelled
        self.record_progress(
            job,
            {"current": index, "total": total, "tensor_name": tensor_name},
            f"[{index}/{total}] {tensor_name}",
        )
