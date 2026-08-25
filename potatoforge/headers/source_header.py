from pathlib import Path
from typing import NamedTuple

from .header_reader import read_header_from_safetensors
from ..planning import TensorDescriptor, METADATA_KEY


class SourceModelHeader(NamedTuple):
    tensors: dict[str, TensorDescriptor]
    metadata: dict[str, str]

def validate_shape(shape: object) -> list[int]:
    if not isinstance(shape, list):
        raise ValueError("Tensor shape must be a list.")

    validated_shape: list[int] = []

    for dimension in shape:
        if type(dimension) is not int:
            raise ValueError(
                "Tensor shape dimensions must be integers."
            )

        validated_shape.append(dimension)

    return validated_shape

def validate_data_offsets(data_offsets: object) -> list[int]:
    if not isinstance(data_offsets, list):
        raise ValueError("Data Offsets must be a list")

    if len(data_offsets) != 2:
        raise ValueError("Data Offsets must have only two elements, [start, end)")


    validated_data_offsets: list[int] = []

    for dimension in data_offsets:
        if type(dimension) is not int:
            raise ValueError(
                "Tensor shape dimensions must be integers."
            )

        validated_data_offsets.append(dimension)

    start, end = validated_data_offsets

    if start < 0:
        raise ValueError("Start of data offset cannot be below zero")

    if end < start:
        raise ValueError("End of data offset cannot be less than start")

    return validated_data_offsets

def validate_dtype(dtype: object) -> str:
    if type(dtype) is not str:
        raise ValueError("Dtype must be a string")

    return dtype

def validate_tensor_descriptor(tensor_name: str, raw_descriptor: object) -> TensorDescriptor:
    if not isinstance(raw_descriptor, dict):
        raise ValueError(f"{tensor_name} must be a dictionary")

    if type(tensor_name) is not str:
        raise ValueError("Tensor name must be a string")

    validated_descriptor: TensorDescriptor = {
        "shape": validate_shape(raw_descriptor.get("shape")),
        "data_offsets": validate_data_offsets(raw_descriptor.get("data_offsets")),
        "dtype": validate_dtype(raw_descriptor.get("dtype"))
    }

    return validated_descriptor

def validate_metadata(metadata: object) -> dict[str, str]:
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a dictionary")

    validated_metadata: dict[str, str] = {}

    for key, value in metadata.items():
        if type(key) is not str:
            raise ValueError(f"{key} must be a string")
        if type(value) is not str:
            raise ValueError(f"{value} must be a string")

        validated_metadata[key] = value

    return validated_metadata

def read_source_model_header(source_path: str | Path) -> SourceModelHeader:
    raw_headers = read_header_from_safetensors(source_path)

    if METADATA_KEY in raw_headers:
        metadata = validate_metadata(raw_headers[METADATA_KEY])
    else:
        metadata = {}

    tensors: dict[str, TensorDescriptor] = {}

    for tensor_name, descriptor in raw_headers.items():
        if tensor_name == METADATA_KEY:
            continue

        descriptor_obj = validate_tensor_descriptor(tensor_name, descriptor)

        tensors[tensor_name] = descriptor_obj

    return SourceModelHeader(tensors=tensors, metadata=metadata)
