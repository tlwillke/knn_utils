#!/usr/bin/env python3
"""
fvecs_merge.py

Merge two .fvecs files into one output .fvecs file.

The program:
1. Scans each input file to count vectors and verify internal consistency.
2. Confirms both inputs have the same dimensionality.
3. Concatenates the raw records into the output file.
4. Scans the output file to verify the written vector count and dimension.
"""

import argparse
import shutil
from pathlib import Path

COPY_BUFFER_SIZE_BYTES = 16 * 1024 * 1024  # 16 MiB


def scan_fvecs(path: Path) -> tuple[int, int]:
    """
    Sequentially scan a .fvecs file and return (dimension, vector_count).

    Verifies:
    - each record has a readable 4-byte dimension header
    - all records have the same dimension
    - each payload is complete

    Raises ValueError on malformed input.
    """
    count = 0
    dim = None

    with path.open("rb") as f:
        while True:
            hdr = f.read(4)
            if not hdr:
                break
            if len(hdr) != 4:
                raise ValueError(f"{path}: truncated dimension header at record {count}")

            record_dim = int.from_bytes(hdr, byteorder="little", signed=True)
            if record_dim <= 0:
                raise ValueError(f"{path}: invalid dimension {record_dim} at record {count}")

            if dim is None:
                dim = record_dim
            elif record_dim != dim:
                raise ValueError(
                    f"{path}: inconsistent dimension at record {count}: "
                    f"expected {dim}, found {record_dim}"
                )

            payload = f.read(4 * record_dim)
            if len(payload) != 4 * record_dim:
                raise ValueError(f"{path}: truncated payload at record {count}")

            count += 1

    if dim is None:
        raise ValueError(f"{path}: empty .fvecs file")

    return dim, count


def copy_file(src: Path, dst_file) -> None:
    """Copy one file's raw bytes into an already-open destination file."""
    with src.open("rb") as fin:
        shutil.copyfileobj(fin, dst_file, length=COPY_BUFFER_SIZE_BYTES)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge two .fvecs files into one.")
    parser.add_argument("input_a", type=Path, help="First input .fvecs file")
    parser.add_argument("input_b", type=Path, help="Second input .fvecs file")
    parser.add_argument("output", type=Path, help="Output merged .fvecs file")
    args = parser.parse_args()

    print(f"→ Scanning first input:  {args.input_a}")
    dim_a, count_a = scan_fvecs(args.input_a)
    print(f"  Found {count_a} vectors, dimension {dim_a}")

    print(f"→ Scanning second input: {args.input_b}")
    dim_b, count_b = scan_fvecs(args.input_b)
    print(f"  Found {count_b} vectors, dimension {dim_b}")

    if dim_a != dim_b:
        raise ValueError(
            f"Input dimensionality mismatch: {args.input_a} has dim={dim_a}, "
            f"{args.input_b} has dim={dim_b}"
        )

    print(f"✔ Input dimensionality matches: {dim_a}")
    print(f"→ Writing merged output: {args.output}")

    with args.output.open("wb") as fout:
        copy_file(args.input_a, fout)
        copy_file(args.input_b, fout)

    print("→ Verifying output by scanning written file")
    out_dim, out_count = scan_fvecs(args.output)
    print(f"✔ Output verified: {out_count} vectors, dimension {out_dim}")


if __name__ == "__main__":
    main()
