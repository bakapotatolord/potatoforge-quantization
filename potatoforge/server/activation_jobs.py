"""Activation-audit command and progress handling for local jobs."""

from __future__ import annotations

from typing import Any, Callable

from ..application.commands import activation_audit_model
from .jobs import CommandJobManager

ActivationAuditCommand = Callable[..., Any]


class ActivationAuditJobManager(CommandJobManager):
    """Run one activation audit with replayable progress and cancellation."""

    label = "Activation audit"

    def start(
        self,
        request: Any,
        command: ActivationAuditCommand = activation_audit_model,
    ) -> dict[str, Any]:
        return super().start(
            lambda job: self.run_command(
                job,
                lambda: command(
                    request.source_path,
                    request.activation_calibration,
                    request.output,
                    device=request.device,
                    overwrite=request.overwrite,
                    methods=getattr(request, "methods", None),
                    check_cancelled=lambda: self.check_cancelled(job),
                    on_tensor_started=lambda index, total, name: self.report_tensor(
                        job, index, total, name
                    ),
                ),
                _result_payload,
            )
        )


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
