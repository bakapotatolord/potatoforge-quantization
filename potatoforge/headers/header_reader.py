import json
import os
from pathlib import Path
from typing import BinaryIO


MAX_HEADER_BYTES = 100 * 1024 * 1024


def read_raw_data_start(
    file: BinaryIO,
    *,
    file_label: str = "File",
) -> int:
    length_bytes = file.read(8)

    if len(length_bytes) != 8:
        raise ValueError(
            f"{file_label} is shorter than the safetensors prefix."
        )

    return 8 + int.from_bytes(length_bytes, "little")


def read_header_from_safetensors(file_path: str | Path) -> dict[str, object]:
    try:
        with open(file_path, "rb") as f:
            file_size = os.fstat(f.fileno()).st_size
            if file_size < 8:
                raise ValueError("File is too small to contain a valid header.")

            header_end = read_raw_data_start(f)
            header_length = header_end - 8

            if header_length == 0:
                raise ValueError("Header length is zero, indicating an empty header.")

            if header_length > MAX_HEADER_BYTES:
                raise ValueError(f"Header length {header_length} exceeds maximum allowed size of {MAX_HEADER_BYTES} bytes.")

            if header_end > file_size:
                raise ValueError(
                    f"Header ends at byte {header_end}, "
                    f"but the file contains only {file_size} bytes."
                )

            header_bytes = f.read(header_length)
            if len(header_bytes) != header_length:
                raise ValueError(f"Expected to read {header_length} bytes for the header, but only read {len(header_bytes)} bytes.")

            header_text = header_bytes.decode("utf-8")
            header = json.loads(header_text)

            if not isinstance(header, dict):
                raise ValueError("Header JSON is not a dictionary.")

            return header

    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode header JSON: {e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse header JSON: {e}") from e
