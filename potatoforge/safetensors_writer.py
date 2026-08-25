import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import TypeAlias

from .planning import METADATA_KEY, SafetensorsLayout, layout_to_header


TensorPayload: TypeAlias = tuple[str, bytes]
TensorPayloadStream: TypeAlias = Iterable[TensorPayload]


def encode_safetensors_header(
    layout: SafetensorsLayout,
    metadata: Mapping[str, str] | None = None,
) -> bytes:
    header: dict[str, object] = {
        **layout_to_header(layout),
    }

    if metadata is not None:
        header[METADATA_KEY] = dict(metadata)

    return json.dumps(
        header,
        separators=(",", ":"),
    ).encode("utf-8")


def write_safetensors_file(
    output_path: str | Path,
    layout: SafetensorsLayout,
    payloads: TensorPayloadStream,
    metadata: Mapping[str, str] | None = None,
) -> None:
    header_bytes = encode_safetensors_header(layout, metadata)
    payload_iterator: Iterator[TensorPayload] = iter(payloads)

    with Path(output_path).open("xb") as file:
        file.write(
            len(header_bytes).to_bytes(
                8,
                byteorder="little",
            )
        )
        file.write(header_bytes)

        for tensor in layout.tensors:
            try:
                payload_name, payload = next(payload_iterator)
            except StopIteration as error:
                raise ValueError(
                    "Missing payload for "
                    f"{tensor.spec.name}."
                ) from error

            if payload_name != tensor.spec.name:
                raise ValueError(
                    "Payload order does not match the layout. "
                    f"Expected {tensor.spec.name}, got "
                    f"{payload_name}."
                )

            if len(payload) != tensor.spec.byte_count:
                raise ValueError(
                    f"{payload_name} needs "
                    f"{tensor.spec.byte_count} bytes, got "
                    f"{len(payload)}."
                )

            file.write(payload)

        try:
            unexpected_name, _ = next(payload_iterator)
        except StopIteration:
            return

        raise ValueError(
            f"Unexpected payload: {unexpected_name}."
        )
