import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, NamedTuple, TypedDict

import torch

from ..planning import (
    TensorDescriptor,
    plan_convrot_w4a4,
    plan_int6_convrot,
    plan_int6_rowwise,
    plan_int8_convrot,
    plan_int8_tensorwise,
    source_bytes,
)
from .all_comparison import compare_all_reconstructions
from ..quantization.hadamard import CONVROT_GROUP_SIZE
from ..headers.source_header import read_source_model_header
from ..source_payloads import stream_bf16_source_tensors


QuantizationMethod = Literal[
    "bf16",
    "int8",
    "int6",
    "int8_convrot",
    "int6_convrot",
    "convrot_w4a4",
]

AuditProgressReporter = Callable[[int, int, str], None]


class MethodAudit(NamedTuple):
    relative_l2_error: float | None
    storage_bytes: int | None
    storage_saved_bytes: int | None
    storage_saving_fraction: float | None


class MethodAudits(NamedTuple):
    bf16: MethodAudit
    int8: MethodAudit
    int6: MethodAudit
    int8_convrot: MethodAudit
    int6_convrot: MethodAudit
    convrot_w4a4: MethodAudit


class ErrorDeltas(NamedTuple):
    int8_vs_bf16: float
    int6_vs_int8: float | None
    int8_convrot_vs_int8: float | None
    int6_convrot_vs_int6: float | None
    int6_convrot_vs_int8_convrot: float | None
    convrot_w4a4_vs_int8: float | None
    convrot_w4a4_vs_int8_convrot: float | None


class WeightAuditResult(NamedTuple):
    tensor_name: str
    shape: tuple[int, ...]
    methods: MethodAudits
    error_deltas: ErrorDeltas


class MethodAuditRecord(TypedDict):
    relative_l2_error: float | None
    storage_bytes: int | None
    storage_saved_bytes: int | None
    storage_saving_fraction: float | None


class MethodAuditsRecord(TypedDict):
    bf16: MethodAuditRecord
    int8: MethodAuditRecord
    int6: MethodAuditRecord
    int8_convrot: MethodAuditRecord
    int6_convrot: MethodAuditRecord
    convrot_w4a4: MethodAuditRecord


class ErrorDeltasRecord(TypedDict):
    int8_vs_bf16: float
    int6_vs_int8: float | None
    int8_convrot_vs_int8: float | None
    int6_convrot_vs_int6: float | None
    int6_convrot_vs_int8_convrot: float | None
    convrot_w4a4_vs_int8: float | None
    convrot_w4a4_vs_int8_convrot: float | None


class WeightAuditRecord(TypedDict):
    tensor_name: str
    shape: list[int]
    methods: MethodAuditsRecord
    error_deltas: ErrorDeltasRecord


class WeightAuditSummary(TypedDict):
    source_tensor_count: int
    audited_layer_count: int
    skipped_tensor_count: int
    storage_bytes: dict[str, int]


class WeightAuditDocument(TypedDict):
    format_version: int
    source_path: str
    selection: dict[str, str | int]
    summary: WeightAuditSummary
    results: list[WeightAuditRecord]


def select_auditable_bf16_weights(
    tensors: Mapping[str, TensorDescriptor],
) -> tuple[tuple[str, TensorDescriptor], ...]:
    return tuple(
        (
            tensor_name,
            descriptor,
        )
        for tensor_name, descriptor in tensors.items()
        if descriptor["dtype"] in ("BF16", "F16")
        and len(descriptor["shape"]) == 2
        and tensor_name.endswith(".weight")
    )


def _method_audit(
    relative_l2_error: float | None,
    storage_bytes: int | None,
    bf16_bytes: int,
) -> MethodAudit:
    if relative_l2_error is None or storage_bytes is None:
        return MethodAudit(None, None, None, None)

    storage_saved_bytes = bf16_bytes - storage_bytes
    storage_saving_fraction = (
        None
        if bf16_bytes == 0
        else storage_saved_bytes / bf16_bytes
    )
    return MethodAudit(
        relative_l2_error=relative_l2_error,
        storage_bytes=storage_bytes,
        storage_saved_bytes=storage_saved_bytes,
        storage_saving_fraction=storage_saving_fraction,
    )


def _audit_weight(
    tensor_name: str,
    descriptor: TensorDescriptor,
    weights: torch.Tensor,
) -> WeightAuditResult:
    comparison = compare_all_reconstructions(weights)
    bf16_bytes = source_bytes(descriptor)

    int8_plan = plan_int8_tensorwise(tensor_name, descriptor)
    int6_plan = None
    int8_convrot_plan = None
    int6_convrot_plan = None
    w4a4_plan = None

    if weights.shape[1] % 4 == 0:
        int6_plan = plan_int6_rowwise(tensor_name, descriptor)

    if weights.shape[1] % CONVROT_GROUP_SIZE == 0:
        int8_convrot_plan = plan_int8_convrot(tensor_name, descriptor)
        int6_convrot_plan = plan_int6_convrot(tensor_name, descriptor)
        w4a4_plan = plan_convrot_w4a4(tensor_name, descriptor)

    methods = MethodAudits(
        bf16=_method_audit(0.0, bf16_bytes, bf16_bytes),
        int8=_method_audit(
            comparison.int8_relative_l2_error,
            int8_plan.estimated_bytes,
            bf16_bytes,
        ),
        int6=_method_audit(
            comparison.int6_relative_l2_error,
            None if int6_plan is None else int6_plan.estimated_bytes,
            bf16_bytes,
        ),
        int8_convrot=_method_audit(
            comparison.int8_convrot_relative_l2_error,
            None
            if int8_convrot_plan is None
            else int8_convrot_plan.estimated_bytes,
            bf16_bytes,
        ),
        int6_convrot=_method_audit(
            comparison.int6_convrot_relative_l2_error,
            None
            if int6_convrot_plan is None
            else int6_convrot_plan.estimated_bytes,
            bf16_bytes,
        ),
        convrot_w4a4=_method_audit(
            comparison.w4a4_relative_l2_error,
            None if w4a4_plan is None else w4a4_plan.estimated_bytes,
            bf16_bytes,
        ),
    )
    error_deltas = ErrorDeltas(
        int8_vs_bf16=comparison.int8_relative_l2_error,
        int6_vs_int8=(
            None
            if comparison.int6_relative_l2_error is None
            else comparison.int6_relative_l2_error
            - comparison.int8_relative_l2_error
        ),
        int8_convrot_vs_int8=(
            None
            if comparison.int8_convrot_relative_l2_error is None
            else comparison.int8_convrot_relative_l2_error
            - comparison.int8_relative_l2_error
        ),
        int6_convrot_vs_int6=(
            None
            if (
                comparison.int6_convrot_relative_l2_error is None
                or comparison.int6_relative_l2_error is None
            )
            else comparison.int6_convrot_relative_l2_error
            - comparison.int6_relative_l2_error
        ),
        int6_convrot_vs_int8_convrot=(
            None
            if (
                comparison.int6_convrot_relative_l2_error is None
                or comparison.int8_convrot_relative_l2_error is None
            )
            else comparison.int6_convrot_relative_l2_error
            - comparison.int8_convrot_relative_l2_error
        ),
        convrot_w4a4_vs_int8=(
            None
            if comparison.w4a4_relative_l2_error is None
            else comparison.w4a4_relative_l2_error
            - comparison.int8_relative_l2_error
        ),
        convrot_w4a4_vs_int8_convrot=(
            None
            if (
                comparison.w4a4_relative_l2_error is None
                or comparison.int8_convrot_relative_l2_error is None
            )
            else comparison.w4a4_relative_l2_error
            - comparison.int8_convrot_relative_l2_error
        ),
    )
    return WeightAuditResult(
        tensor_name=tensor_name,
        shape=tuple(weights.shape),
        methods=methods,
        error_deltas=error_deltas,
    )


def audit_bf16_weight_entries(
    source_path: str | Path,
    descriptors: Sequence[tuple[str, TensorDescriptor]],
    on_entry_started: AuditProgressReporter | None = None,
) -> tuple[WeightAuditResult, ...]:
    descriptor_by_name = dict(descriptors)
    results: list[WeightAuditResult] = []

    for tensor_name, weights in stream_bf16_source_tensors(
        source_path,
        descriptors,
        on_entry_started,
    ):
        results.append(
            _audit_weight(
                tensor_name,
                descriptor_by_name[tensor_name],
                weights,
            )
        )

    return tuple(results)


def _method_to_record(method: MethodAudit) -> MethodAuditRecord:
    return {
        "relative_l2_error": method.relative_l2_error,
        "storage_bytes": method.storage_bytes,
        "storage_saved_bytes": method.storage_saved_bytes,
        "storage_saving_fraction": method.storage_saving_fraction,
    }


def audit_result_to_record(result: WeightAuditResult) -> WeightAuditRecord:
    return {
        "tensor_name": result.tensor_name,
        "shape": list(result.shape),
        "methods": {
            "bf16": _method_to_record(result.methods.bf16),
            "int8": _method_to_record(result.methods.int8),
            "int6": _method_to_record(result.methods.int6),
            "int8_convrot": _method_to_record(result.methods.int8_convrot),
            "int6_convrot": _method_to_record(result.methods.int6_convrot),
            "convrot_w4a4": _method_to_record(result.methods.convrot_w4a4),
        },
        "error_deltas": {
            "int8_vs_bf16": result.error_deltas.int8_vs_bf16,
            "int6_vs_int8": result.error_deltas.int6_vs_int8,
            "int8_convrot_vs_int8": (
                result.error_deltas.int8_convrot_vs_int8
            ),
            "int6_convrot_vs_int6": (
                result.error_deltas.int6_convrot_vs_int6
            ),
            "int6_convrot_vs_int8_convrot": (
                result.error_deltas.int6_convrot_vs_int8_convrot
            ),
            "convrot_w4a4_vs_int8": (
                result.error_deltas.convrot_w4a4_vs_int8
            ),
            "convrot_w4a4_vs_int8_convrot": (
                result.error_deltas.convrot_w4a4_vs_int8_convrot
            ),
        },
    }


def _storage_summary(
    results: Sequence[WeightAuditResult],
) -> dict[str, int]:
    return {
        "bf16": sum(
            result.methods.bf16.storage_bytes or 0
            for result in results
        ),
        "int8": sum(
            result.methods.int8.storage_bytes or 0
            for result in results
        ),
        "int6": sum(
            result.methods.int6.storage_bytes or 0
            for result in results
        ),
        "int8_convrot": sum(
            result.methods.int8_convrot.storage_bytes or 0
            for result in results
        ),
        "int6_convrot": sum(
            result.methods.int6_convrot.storage_bytes or 0
            for result in results
        ),
        "convrot_w4a4": sum(
            result.methods.convrot_w4a4.storage_bytes or 0
            for result in results
        ),
    }


def audit_bf16_source(
    source_path: str | Path,
    on_entry_started: AuditProgressReporter | None = None,
) -> WeightAuditDocument:
    source_header = read_source_model_header(source_path)
    descriptors = select_auditable_bf16_weights(source_header.tensors)
    results = audit_bf16_weight_entries(
        source_path,
        descriptors,
        on_entry_started,
    )

    source_dtypes = {
        descriptor["dtype"]
        for _, descriptor in descriptors
    }
    selection_dtype = (
        "BF16"
        if not source_dtypes or source_dtypes == {"BF16"}
        else "F16"
        if source_dtypes == {"F16"}
        else "BF16 or F16"
    )

    return {
        "format_version": 3,
        "source_path": str(Path(source_path).resolve()),
        "selection": {
            "dtype": selection_dtype,
            "rank": 2,
            "name_suffix": ".weight",
        },
        "summary": {
            "source_tensor_count": len(source_header.tensors),
            "audited_layer_count": len(results),
            "skipped_tensor_count": (
                len(source_header.tensors) - len(results)
            ),
            "storage_bytes": _storage_summary(results),
        },
        "results": [audit_result_to_record(result) for result in results],
    }


def write_weight_audit_report(
    output_path: str | Path,
    document: WeightAuditDocument,
) -> None:
    with Path(output_path).open("x", encoding="utf-8") as output_file:
        json.dump(document, output_file, indent=2)
        output_file.write("\n")


def _format_error(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2%}" if signed else f"{value:.2%}"


def _format_kib(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / 1024:.1f}"


def print_weight_audit_table(
    results: Sequence[WeightAuditRecord],
    source_dtype: str = "BF16",
) -> None:
    headers = [
        "layer",
        "shape",
        f"{source_dtype} KiB",
        "INT8 KiB",
        "INT6 KiB",
        "C-INT8 KiB",
        "C-INT6 KiB",
        "W4A4 KiB",
        f"{source_dtype} err",
        "INT8 err",
        "INT6 err",
        "C-INT8 err",
        "C-INT6 err",
        "W4A4 err",
        "d6-I",
        "dC-I",
        "dC6-6",
        "dC6-C",
        "dW-I",
        "dW-C",
    ]
    rows = [headers]

    for result in results:
        methods = result["methods"]
        deltas = result["error_deltas"]
        rows.append(
            [
                result["tensor_name"],
                str(result["shape"]),
                _format_kib(methods["bf16"]["storage_bytes"]),
                _format_kib(methods["int8"]["storage_bytes"]),
                _format_kib(methods["int6"]["storage_bytes"]),
                _format_kib(methods["int8_convrot"]["storage_bytes"]),
                _format_kib(methods["int6_convrot"]["storage_bytes"]),
                _format_kib(methods["convrot_w4a4"]["storage_bytes"]),
                _format_error(methods["bf16"]["relative_l2_error"]),
                _format_error(methods["int8"]["relative_l2_error"]),
                _format_error(methods["int6"]["relative_l2_error"]),
                _format_error(
                    methods["int8_convrot"]["relative_l2_error"]
                ),
                _format_error(
                    methods["int6_convrot"]["relative_l2_error"]
                ),
                _format_error(
                    methods["convrot_w4a4"]["relative_l2_error"]
                ),
                _format_error(
                    deltas["int6_vs_int8"],
                    signed=True,
                ),
                _format_error(
                    deltas["int8_convrot_vs_int8"],
                    signed=True,
                ),
                _format_error(
                    deltas["int6_convrot_vs_int6"],
                    signed=True,
                ),
                _format_error(
                    deltas["int6_convrot_vs_int8_convrot"],
                    signed=True,
                ),
                _format_error(
                    deltas["convrot_w4a4_vs_int8"],
                    signed=True,
                ),
                _format_error(
                    deltas["convrot_w4a4_vs_int8_convrot"],
                    signed=True,
                ),
            ]
        )

    widths = [
        max(len(row[column]) for row in rows)
        for column in range(len(headers))
    ]
    print(
        "Errors are relative L2 reconstruction error against "
        f"{source_dtype}; "
        "d6-I=INT6-INT8, dC-I=ConvRot INT8-INT8, "
        "dC6-6=ConvRot INT6-INT6, dC6-C=ConvRot INT6-ConvRot INT8."
    )
    print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows[1:]:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
