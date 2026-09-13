import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, NamedTuple, TypedDict, cast

import torch

from ..planning import (
    QUANTIZATION_SOURCE_DTYPES,
    TensorDescriptor,
    plan_convrot_w4a4,
    plan_int6_convrot,
    plan_int6_rowwise,
    plan_int8_convrot,
    plan_int8_tensorwise,
    source_bytes,
)
from .all_comparison import (
    compare_all_reconstructions,
)
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
    "convrot_w4a4_mse",
]

AuditProgressReporter = Callable[[int, int, str], None]
def _validate_audit_device(device: str) -> None:
    if device not in ("cpu", "cuda"):
        raise ValueError("Audit device must be cpu or cuda.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable.")


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
    convrot_w4a4_mse: MethodAudit


class ErrorDeltas(NamedTuple):
    int8_vs_bf16: float | None
    int6_vs_int8: float | None
    int8_convrot_vs_int8: float | None
    int6_convrot_vs_int6: float | None
    int6_convrot_vs_int8_convrot: float | None
    convrot_w4a4_vs_int8: float | None
    convrot_w4a4_vs_int8_convrot: float | None


class WeightAuditResult(NamedTuple):
    tensor_name: str
    shape: tuple[int, ...]
    weight_l2_sq: float
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
    convrot_w4a4_mse: MethodAuditRecord


class ErrorDeltasRecord(TypedDict):
    int8_vs_bf16: float | None
    int6_vs_int8: float | None
    int8_convrot_vs_int8: float | None
    int6_convrot_vs_int6: float | None
    int6_convrot_vs_int8_convrot: float | None
    convrot_w4a4_vs_int8: float | None
    convrot_w4a4_vs_int8_convrot: float | None


class WeightAuditRecord(TypedDict):
    tensor_name: str
    shape: list[int]
    weight_l2_sq: float
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
        if descriptor["dtype"] in QUANTIZATION_SOURCE_DTYPES
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
    *,
    device: str = "cpu",
    include_plain_methods: bool = True,
) -> WeightAuditResult:
    original_float = weights.float()
    source_l2 = torch.linalg.vector_norm(original_float)
    comparison = compare_all_reconstructions(
        weights,
        original_float=original_float,
        source_l2=source_l2,
        device=device,
        include_plain_methods=include_plain_methods,
    )
    bf16_bytes = source_bytes(descriptor)

    int8_plan = (
        None
        if not include_plain_methods
        else plan_int8_tensorwise(tensor_name, descriptor)
    )
    int6_plan = None
    int8_convrot_plan = None
    int6_convrot_plan = None
    w4a4_plan = None

    if (
        include_plain_methods and weights.shape[1] % 4 == 0
    ):
        int6_plan = plan_int6_rowwise(tensor_name, descriptor)

    if weights.shape[1] % CONVROT_GROUP_SIZE == 0:
        int8_convrot_plan = plan_int8_convrot(tensor_name, descriptor)
        int6_convrot_plan = plan_int6_convrot(tensor_name, descriptor)
        w4a4_plan = plan_convrot_w4a4(tensor_name, descriptor)

    methods = MethodAudits(
        bf16=_method_audit(0.0, bf16_bytes, bf16_bytes),
        int8=_method_audit(
            comparison.int8_relative_l2_error,
            None if int8_plan is None else int8_plan.estimated_bytes,
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
        convrot_w4a4_mse=_method_audit(
            comparison.w4a4_mse_relative_l2_error,
            None if w4a4_plan is None else w4a4_plan.estimated_bytes,
            bf16_bytes,
        ),
    )
    error_deltas = ErrorDeltas(
        int8_vs_bf16=comparison.int8_relative_l2_error,
        int6_vs_int8=(
            None
            if (
                comparison.int6_relative_l2_error is None
                or comparison.int8_relative_l2_error is None
            )
            else comparison.int6_relative_l2_error
            - comparison.int8_relative_l2_error
        ),
        int8_convrot_vs_int8=(
            None
            if (
                comparison.int8_convrot_relative_l2_error is None
                or comparison.int8_relative_l2_error is None
            )
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
            if (
                comparison.w4a4_relative_l2_error is None
                or comparison.int8_relative_l2_error is None
            )
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
        weight_l2_sq=float((source_l2 * source_l2).item()),
        methods=methods,
        error_deltas=error_deltas,
    )


def audit_bf16_weight_entries(
    source_path: str | Path,
    descriptors: Sequence[tuple[str, TensorDescriptor]],
    on_entry_started: AuditProgressReporter | None = None,
    *,
    device: str = "cpu",
    include_plain_methods: bool = True,
) -> tuple[WeightAuditResult, ...]:
    _validate_audit_device(device)
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
                device=device,
                include_plain_methods=include_plain_methods,
            )
        )

    return tuple(results)


def _method_to_record(method: MethodAudit) -> MethodAuditRecord:
    return cast(MethodAuditRecord, method._asdict())


def audit_result_to_record(result: WeightAuditResult) -> WeightAuditRecord:
    record: WeightAuditRecord = {
        "tensor_name": result.tensor_name,
        "shape": list(result.shape),
        "weight_l2_sq": result.weight_l2_sq,
        "methods": {
            name: _method_to_record(method)
            for name, method in result.methods._asdict().items()
        },
        "error_deltas": cast(ErrorDeltasRecord, result.error_deltas._asdict()),
    }
    return record


def _storage_summary(
    results: Sequence[WeightAuditResult],
) -> dict[str, int]:
    return {
        method_name: sum(
            getattr(result.methods, method_name).storage_bytes or 0
            for result in results
        )
        for method_name in MethodAudits._fields
    }


def audit_bf16_source(
    source_path: str | Path,
    on_entry_started: AuditProgressReporter | None = None,
    *,
    tensor_name: str | None = None,
    tensor_names: Sequence[str] | None = None,
    device: str = "cpu",
    include_plain_methods: bool = True,
) -> WeightAuditDocument:
    _validate_audit_device(device)
    source_header = read_source_model_header(source_path)
    descriptors = select_auditable_bf16_weights(source_header.tensors)
    if tensor_name is not None and tensor_names is not None:
        raise ValueError("tensor_name and tensor_names cannot be combined")
    requested_names = (
        (tensor_name,)
        if tensor_name is not None
        else tensor_names
    )
    if requested_names is not None:
        auditable_descriptors = dict(descriptors)
        selected_descriptors = []
        for requested_name in requested_names:
            descriptor = auditable_descriptors.get(requested_name)
            if descriptor is None:
                if requested_name not in source_header.tensors:
                    raise ValueError(
                        f"Tensor not found in source model: {requested_name}"
                    )
                raise ValueError(
                    "Tensor is not an auditable BF16/F16/F32 matrix weight: "
                    f"{requested_name}"
                )
            selected_descriptors.append((requested_name, descriptor))
        descriptors = tuple(selected_descriptors)
    results = audit_bf16_weight_entries(
        source_path,
        descriptors,
        on_entry_started,
        device=device,
        include_plain_methods=include_plain_methods,
    )

    source_dtypes = {
        descriptor["dtype"]
        for _, descriptor in descriptors
    }
    selection_dtype = " or ".join(sorted(source_dtypes)) or "BF16"

    return {
        "format_version": 5,
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
        "W4A4 MSE KiB",
        f"{source_dtype} err",
        "INT8 err",
        "INT6 err",
        "C-INT8 err",
        "C-INT6 err",
        "W4A4 err",
        "W4A4 MSE err",
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
                _format_kib(methods["convrot_w4a4_mse"]["storage_bytes"]),
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
                    methods["convrot_w4a4_mse"]["relative_l2_error"]
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
