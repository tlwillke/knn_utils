#!/usr/bin/env python3

import struct
import sys


def print_row(filepath, target_row):
    with open(filepath, "rb") as f:
        row = 0
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                raise IndexError(f"Row {target_row} not found")
            if len(dim_bytes) < 4:
                raise IOError(f"Incomplete dimension header at row {row}")

            dim = struct.unpack("<i", dim_bytes)[0]
            if dim <= 0:
                raise ValueError(f"Non-positive dimension {dim} at row {row}")

            data = f.read(4 * dim)
            if len(data) < 4 * dim:
                raise IOError(f"Truncated data at row {row}")

            if row == target_row:
                values = struct.unpack(f"<{dim}i", data)
                print(f"row={row}, length={dim}")
                print(list(values))
                return

            row += 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <file.ivecs> <row_index>", file=sys.stderr)
        sys.exit(2)

    print_row(sys.argv[1], int(sys.argv[2]))