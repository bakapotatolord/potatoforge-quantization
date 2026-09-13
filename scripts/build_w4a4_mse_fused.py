"""Build the optional CUDA W4A4-MSE candidate evaluator."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "potatoforge" / "quantization" / "w4a4_mse_fused.cu"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output library path (defaults to the repository build directory).",
    )
    args = parser.parse_args()

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise SystemExit("nvcc was not found on PATH.")
    if not SOURCE.is_file():
        raise SystemExit(f"CUDA source was not found: {SOURCE}")

    library_name = (
        "potatoforge_w4a4_mse_fused.dll"
        if os.name == "nt"
        else "libpotatoforge_w4a4_mse_fused.so"
    )
    output = args.output or PROJECT_ROOT / "build" / library_name
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    command = [nvcc, "-O3", "--shared"]
    command.append("-Xcompiler=/MD" if os.name == "nt" else "-Xcompiler=-fPIC")
    command.extend([str(SOURCE), "-o", str(output)])
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        return completed.returncode

    print(f"Built {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
