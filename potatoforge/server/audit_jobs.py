"""Audit-specific command and progress handling for local jobs."""

from __future__ import annotations

from typing import Any, Callable

from .jobs import CommandJobManager

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


class AuditJobManager(CommandJobManager):
    """Run one Audit command with replayable progress and cancellation."""

    label = "Audit"

    def start(self, request: Any, command: AuditCommand) -> dict[str, Any]:
        return super().start(
            lambda job: self.run_command(
                job,
                lambda: command(
                    request.source_path,
                    request.output,
                    device=request.device,
                    overwrite=request.overwrite,
                    methods=getattr(request, "methods", None),
                    on_entry_started=lambda index, total, name: self.report_tensor(
                        job, index, total, name
                    ),
                ),
                lambda result: audit_result_payload(request, result),
            )
        )
