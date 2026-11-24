"""LoRA-extraction command and progress handling for local jobs."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable

from ..lora.lora_extract import ExtractionOptions, parse_split_rule, summarize
from .jobs import CommandJobManager

ExtractCommand = Callable[..., Any]


def build_extraction_options(request: Any) -> ExtractionOptions:
    return ExtractionOptions(
        ranks=tuple(request.ranks),
        energy=request.energy,
        split=tuple(parse_split_rule(rule) for rule in request.split_fused),
        include=request.include or None,
        exclude=request.exclude or None,
        include_diffs=request.include_diffs,
        dtype=request.dtype,
        device=request.device,
    )


def _reason_group(reason: str) -> str:
    return re.split(r" [(\[\d]", reason, maxsplit=1)[0]


def extract_result_payload(result: Any) -> dict[str, Any]:
    summary = summarize(result)
    return {
        "outputs": [
            {"label": label, **details} for label, details in summary.items()
        ],
        "tensor_count": len(result.tensors),
        "skipped_tensors": len(result.skipped),
        "skipped_reasons": dict(
            Counter(_reason_group(reason) for reason in result.skipped.values())
        ),
        "unmatched_base": len(result.unmatched_base),
        "unmatched_tuned": len(result.unmatched_tuned),
    }


class ExtractJobManager(CommandJobManager):
    """Run one LoRA extraction with replayable progress and cancellation."""

    label = "LoRA extraction"

    def start(self, request: Any, command: ExtractCommand) -> dict[str, Any]:
        return super().start(
            lambda job: self.run_command(
                job,
                lambda: command(
                    request.base_path,
                    request.tuned_path,
                    request.output_path,
                    build_extraction_options(request),
                    on_tensor_started=lambda index, total, name: self.report_tensor(
                        job, index, total, name
                    ),
                ),
                extract_result_payload,
            )
        )
