"""Activation-audit command and progress handling for local jobs."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from ..application.commands import activation_audit_model
from ..converter import ConversionCancelled
from .jobs import Job, LocalJobManager

logger = logging.getLogger(__name__)
ActivationAuditCommand = Callable[..., Any]


class ActivationAuditJobManager(LocalJobManager):
    """Run one activation audit with replayable progress and cancellation."""

    def start(
        self,
        request: Any,
        command: ActivationAuditCommand = activation_audit_model,
    ) -> dict[str, Any]:
        return super().start(lambda job: self._run(job, request, command))

    async def _run(
        self,
        job: Job,
        request: Any,
        command: ActivationAuditCommand,
    ) -> None:
        try:
            self.set_running(job)
            result = await asyncio.to_thread(
                command,
                request.source_path,
                request.activation_calibration,
                request.output,
                device=request.device,
                overwrite=request.overwrite,
                methods=getattr(request, "methods", None),
                check_cancelled=lambda: self._check_cancelled(job),
                on_tensor_started=lambda index, total, tensor_name: self._on_tensor_started(
                    job,
                    index,
                    total,
                    tensor_name,
                ),
            )
            final_result = self._result_payload(result)
        except ConversionCancelled:
            self.finish(job, "cancelled", error="Activation audit stopped.")
        except (FileExistsError, FileNotFoundError, ValueError) as error:
            self.finish(job, "failed", error=str(error) or error.__class__.__name__)
        except Exception:
            logger.exception("Activation audit job %s failed", job.id)
            self.finish(job, "failed", error="Activation audit failed.")
        else:
            self.finish(job, "succeeded", result=final_result)

    def _on_tensor_started(
        self,
        job: Job,
        index: int,
        total: int,
        tensor_name: str,
    ) -> None:
        self._check_cancelled(job)
        self.record_progress(
            job,
            {"current": index, "total": total, "tensor_name": tensor_name},
            f"[{index}/{total}] {tensor_name}",
        )

    def _check_cancelled(self, job: Job) -> None:
        if self.is_cancelled(job):
            raise ConversionCancelled

    @staticmethod
    def _result_payload(result: Any) -> dict[str, Any]:
        return {
            "source_path": str(result.source_path),
            "activation_calibration": str(result.activation_calibration),
            "cache_metadata_path": str(result.cache_metadata_path),
            "cache_tensors_path": str(result.cache_tensors_path),
            "method_count": result.method_count,
            "layer_count": result.layer_count,
            "warning": (
                "Activation measurements guide profile selection; generated "
                "artifacts and runtime output still need validation."
            ),
        }
