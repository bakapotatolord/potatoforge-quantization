import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from potatoforge.headers.header_reader import read_header_from_safetensors, MAX_HEADER_BYTES

# [8-byte header length][header JSON bytes][tensor data]

class TestReadHeader(unittest.TestCase):
    # Header tests
    def test_valid_empty_header(self):
        header_bytes = b"{}"
        file_bytes = len(header_bytes).to_bytes(8, "little") + header_bytes

        with TemporaryDirectory() as directory:
            path = Path(directory) / "valid.safetensors"
            path.write_bytes(file_bytes)

            result = read_header_from_safetensors(path)

        self.assertEqual(result, {})

    def test_file_shorter_than_eight_bytes(self):
        header_bytes = b"{}"
        file_bytes = len(header_bytes).to_bytes(4, "little") + header_bytes

        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.safetensors"
            path.write_bytes(file_bytes)

            with self.assertRaises(ValueError):
                read_header_from_safetensors(path)

    def test_zero_header_length(self):
        file_bytes = b"\x00" * 8

        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.safetensors"
            path.write_bytes(file_bytes)

            with self.assertRaises(ValueError):
                read_header_from_safetensors(path)

    def test_header_too_large(self):
        header_length = MAX_HEADER_BYTES + 1
        file_bytes = header_length.to_bytes(8, "little")

        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.safetensors"
            path.write_bytes(file_bytes)

            with self.assertRaises(ValueError):
                read_header_from_safetensors(path)

    def test_header_extends_beyond_file(self):
        header_length = 3
        file_bytes = header_length.to_bytes(8, "little") + b"{}"

        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.safetensors"
            path.write_bytes(file_bytes)

            with self.assertRaises(ValueError):
                read_header_from_safetensors(path)

    def test_invalid_utf8(self):
        header_bytes = b"\xff"
        file_bytes = len(header_bytes).to_bytes(8, "little") + header_bytes

        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.safetensors"
            path.write_bytes(file_bytes)

            with self.assertRaises(ValueError):
                read_header_from_safetensors(path)

    def test_invalid_json(self):
        header_bytes = b"{"
        file_bytes = len(header_bytes).to_bytes(8, "little") + header_bytes

        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.safetensors"
            path.write_bytes(file_bytes)

            with self.assertRaises(ValueError):
                read_header_from_safetensors(path)

    def test_json_not_object(self):
        header_bytes = b"[]"
        file_bytes = len(header_bytes).to_bytes(8, "little") + header_bytes

        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.safetensors"
            path.write_bytes(file_bytes)

            with self.assertRaises(ValueError):
                read_header_from_safetensors(path)

    # Tensor Description tests

if __name__ == "__main__":
    unittest.main()
