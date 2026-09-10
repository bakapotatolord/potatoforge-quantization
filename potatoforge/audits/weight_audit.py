import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, NamedTuple, NotRequired, TypedDict, cast

import torch

from ..calibration import ActivationCalibration, ActivationStats
from ..calibration.activation_probe import (
    ActivationProbeRecord,
    write_activation_probe_cache,
)
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
    compare_w4a4_reconstruction,
)
from ..quantization.hadamard import CONVROT_GROUP_SIZE
from ..headers.source_header import read_source_model_header
from ..source_payloads import stream_bf16_source_tensors
from .activation_error import ActivationErrorResult


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
ActivationStatus = Literal[
    "ok",
    "missing_calibration",
    "unsupported_reference",
    "unsupported_candidate",
]
_ACTIVATION_METRIC_VARIANT = "diagonal_activation_energy_v1"
_ACTIVATION_CANDIDATE = "convrot_w4a4"
_ACTIVATION_PROBE_REFERENCE = "int8_convrot"
_SUPPORTED_ACTIVATION_REFERENCES = frozenset(("bf16", "int8_convrot"))
_SUPPORTED_AUDIT_METHODS = frozenset(("convrot_w4a4",))


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


class ActivationAuditResult(NamedTuple):
    status: ActivationStatus
    metric_variant: str
    calibration_baseline: str | None
    activation_reference: str | None
    candidate_format: str
    error: float | None
    activation_energy_sum: float | None
    sample_count: int | None
    invocation_count: int | None
    input_features: int | None


class WeightAuditResult(NamedTuple):
    tensor_name: str
    shape: tuple[int, ...]
    weight_l2_sq: float
    methods: MethodAudits
    error_deltas: ErrorDeltas
    activation: ActivationAuditResult | None = None


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


class ActivationRecord(TypedDict):
    status: ActivationStatus
    metric_variant: str
    calibration_baseline: str | None
    activation_reference: str | None
    candidate_format: str
    error: float | None
    activation_energy_sum: float | None
    sample_count: int | None
    invocation_count: int | None
    input_features: int | None


class WeightAuditRecord(TypedDict):
    tensor_name: str
    shape: list[int]
    weight_l2_sq: float
    methods: MethodAuditsRecord
    error_deltas: ErrorDeltasRecord
    activation: NotRequired[ActivationRecord]


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


def _build_activation_audit(
    calibration: ActivationCalibration | None,
    stats: ActivationStats | None,
    metric: ActivationErrorResult | None,
    activation_reference: str | None,
) -> ActivationAuditResult | None:
    if calibration is None:
        return None
    baseline = calibration.baseline_label
    common = {
        "calibration_baseline": baseline,
        "activation_reference": activation_reference,
        "candidate_format": _ACTIVATION_CANDIDATE,
    }
    if stats is None:
        return ActivationAuditResult(
            status="missing_calibration",
            metric_variant=_ACTIVATION_METRIC_VARIANT,
            **common,
            error=None,
            activation_energy_sum=None,
            sample_count=None,
            invocation_count=None,
            input_features=None,
        )
    if activation_reference is None:
        return ActivationAuditResult(
            status="unsupported_reference",
            metric_variant=_ACTIVATION_METRIC_VARIANT,
            **common,
            error=None,
            activation_energy_sum=None,
            sample_count=stats.sample_count,
            invocation_count=stats.invocation_count,
            input_features=stats.input_features,
        )
    if metric is None:
        return ActivationAuditResult(
            status="unsupported_candidate",
            metric_variant=_ACTIVATION_METRIC_VARIANT,
            **common,
            error=None,
            activation_energy_sum=None,
            sample_count=stats.sample_count,
            invocation_count=stats.invocation_count,
            input_features=stats.input_features,
        )
    return ActivationAuditResult(
        status="ok",
        metric_variant=_ACTIVATION_METRIC_VARIANT,
        **common,
        error=metric.activation_error,
        activation_energy_sum=metric.activation_energy_sum,
        sample_count=stats.sample_count,
        invocation_count=stats.invocation_count,
        input_features=metric.input_features,
    )


def _audit_weight(
    tensor_name: str,
    descriptor: TensorDescriptor,
    weights: torch.Tensor,
    activation_calibration: ActivationCalibration | None = None,
    audit_method: str | None = None,
    activation_probe_records: dict[str, ActivationProbeRecord] | None = None,
) -> WeightAuditResult:
    original_float = weights.float()
    source_l2 = torch.linalg.vector_norm(original_float)
    activation_stats = (
        None
        if activation_calibration is None
        else activation_calibration.get(tensor_name)
    )
    activation_reference = (
        None
        if activation_calibration is None
        else (
            activation_calibration.baseline_label
            if activation_calibration.baseline_label
            in _SUPPORTED_ACTIVATION_REFERENCES
            else None
        )
    )
    compare = (
        compare_w4a4_reconstruction
        if audit_method == "convrot_w4a4"
        else compare_all_reconstructions
    )
    comparison = compare(
        weights,
        original_float=original_float,
        source_l2=source_l2,
        activation_sum_x2=(
            None
            if activation_stats is None or activation_reference is None
            else activation_stats.sum_x2
        ),
        activation_reference_label=activation_reference,
        activation_probe=activation_probe_records is not None,
    )
    if activation_probe_records is not None:
        q_per_input = comparison.activation_probe_q_per_input
        reference_power_per_input = (
            comparison.activation_probe_reference_power_per_input
        )
        activation_probe_records[tensor_name] = ActivationProbeRecord(
            tensor_name=tensor_name,
            input_features=weights.shape[1],
            status=(
                "ok"
                if q_per_input is not None
                and reference_power_per_input is not None
                else "unsupported_candidate"
            ),
            q_per_input=(
                None
                if q_per_input is None
                else q_per_input.detach().to(
                    device="cpu",
                    dtype=torch.float32,
                )
            ),
            reference_power_per_input=(
                None
                if reference_power_per_input is None
                else reference_power_per_input.detach().to(
                    device="cpu",
                    dtype=torch.float32,
                )
            ),
        )
    bf16_bytes = source_bytes(descriptor)

    int8_plan = (
        None
        if audit_method == "convrot_w4a4"
        else plan_int8_tensorwise(tensor_name, descriptor)
    )
    int6_plan = None
    int8_convrot_plan = None
    int6_convrot_plan = None
    w4a4_plan = None

    if audit_method != "convrot_w4a4" and weights.shape[1] % 4 == 0:
        int6_plan = plan_int6_rowwise(tensor_name, descriptor)

    if weights.shape[1] % CONVROT_GROUP_SIZE == 0:
        if audit_method != "convrot_w4a4":
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
    activation = _build_activation_audit(
        activation_calibration,
        activation_stats,
        comparison.activation_error,
        activation_reference,
    )
    return WeightAuditResult(
        tensor_name=tensor_name,
        shape=tuple(weights.shape),
        weight_l2_sq=float((source_l2 * source_l2).item()),
        methods=methods,
        error_deltas=error_deltas,
        activation=activation,
    )


def audit_bf16_weight_entries(
    source_path: str | Path,
    descriptors: Sequence[tuple[str, TensorDescriptor]],
    on_entry_started: AuditProgressReporter | None = None,
    *,
    activation_calibration: ActivationCalibration | None = None,
    audit_method: str | None = None,
    activation_probe_records: dict[str, ActivationProbeRecord] | None = None,
) -> tuple[WeightAuditResult, ...]:
    if audit_method is not None and audit_method not in _SUPPORTED_AUDIT_METHODS:
        raise ValueError(
            "Unsupported audit method: "
            f"{audit_method!r}; only convrot_w4a4 is supported."
        )
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
                activation_calibration,
                audit_method,
                activation_probe_records,
            )
        )

    return tuple(results)


def _method_to_record(method: MethodAudit) -> MethodAuditRecord:
    return cast(MethodAuditRecord, method._asdict())


def _activation_to_record(
    activation: ActivationAuditResult,
) -> ActivationRecord:
    return cast(ActivationRecord, activation._asdict())


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
    if result.activation is not None:
        record["activation"] = _activation_to_record(result.activation)
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
    activation_calibration: ActivationCalibration | None = None,
    audit_method: str | None = None,
    activation_probe_output: str | Path | None = None,
    activation_probe_overwrite: bool = False,
) -> WeightAuditDocument:
    if audit_method is not None and audit_method not in _SUPPORTED_AUDIT_METHODS:
        raise ValueError(
            "Unsupported audit method: "
            f"{audit_method!r}; only convrot_w4a4 is supported."
        )
    if activation_probe_output is not None and (
        activation_calibration is None
        or activation_calibration.baseline_label != _ACTIVATION_PROBE_REFERENCE
    ):
        raise ValueError(
            "activation_probe_output requires activation calibration with "
            f"baseline_label={_ACTIVATION_PROBE_REFERENCE!r}."
        )
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
    activation_probe_records = (
        {} if activation_probe_output is not None else None
    )
    results = audit_bf16_weight_entries(
        source_path,
        descriptors,
        on_entry_started,
        activation_calibration=activation_calibration,
        audit_method=audit_method,
        activation_probe_records=activation_probe_records,
    )

    if activation_probe_output is not None:
        write_activation_probe_cache(
            activation_probe_output,
            source_path,
            activation_probe_records or {},
            overwrite=activation_probe_overwrite,
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
