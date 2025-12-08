import os
import numpy as np
import argparse
import random
import struct

def check_dim(filename, struct_fmt_int="<i", numpy_fmt_float=np.float32, max_samples=100):
    """
    Opens an fvecs file and validates that all sampled records have the same
    dimensionality. Uses a random sample of up to `max_samples` records.

    Returns:
        first_d      : vector dimensionality
        record_size  : bytes per record
        num_records  : total number of vectors

    Raises:
        ValueError if any sampled record has mismatched dimensionality or if the
        file size is not a proper multiple of the record size.
    """

    with open(filename, "rb") as f:
        # --- Read first dimension ---
        first_d = struct.unpack(struct_fmt_int, f.read(4))[0]
        record_size = 4 + 4 * first_d

        # --- Compute number of records from file size ---
        f.seek(0, 2)
        file_size = f.tell()

        if file_size % record_size != 0:
            raise ValueError(
                f"Invalid fvecs file: size {file_size} is not a multiple of "
                f"record size {record_size}."
            )

        num_records = file_size // record_size

        # --- Choose random sample positions ---
        sample_count = min(num_records, max_samples)

        # Always include record 0 (already checked), so sample from 1..num_records-1
        if num_records > 1:
            sample_indices = random.sample(range(1, num_records), sample_count - 1)
            sample_indices.insert(0, 0)  # Ensure index 0 always included
        else:
            sample_indices = [0]

        # --- Mandatory dimensionality check using random sample ---
        for idx in sample_indices:
            offset = idx * record_size
            f.seek(offset)
            d = struct.unpack(struct_fmt_int, f.read(4))[0]
            if d != first_d:
                raise ValueError(
                    f"Dimension mismatch at record {idx}: expected {first_d}, found {d}. "
                    "This file does not conform to fixed-d fvecs format."
                )

        return first_d, record_size, num_records

def read_vector_at(filename, index, first_d, record_size, num_records, numpy_fmt_float=np.float32):
    """O(1) random-access read of the vector at record index."""

    if index < 0 or index >= num_records:
        raise IndexError(
            f"Requested index {index}, but file contains {num_records} records "
            f"(valid range: 0 .. {num_records-1})."
        )

    with open(filename, "rb") as f:
        offset = index * record_size
        f.seek(offset + 4)  # skip the dimension int
        data = f.read(4 * first_d)
        if len(data) != 4 * first_d:
            raise ValueError(
                f"File ended unexpectedly when reading vector {index}. "
                "File may be corrupted or truncated."
            )
        vec = np.frombuffer(data, dtype=numpy_fmt_float).copy()
        return vec

def main():
    parser = argparse.ArgumentParser(
        description="Print the vector at a given line number from an .fvecs file."
    )
    parser.add_argument("filename", help="Path to the .fvecs file.")
    parser.add_argument("line_number", type=int, help="0-based index of the line/vector to read.")
    parser.add_argument(
        "--endian",
        choices=["little", "big"],
        default="little",
        help="Byte order of the .fvecs file (default: little-endian)."
    )
    args = parser.parse_args()

    fname = os.path.expanduser(args.filename)

    endian_map = {
        "little": ("<i", "<f4"),
        "big":    (">i", ">f4"),
    }
    struct_fmt_int, numpy_fmt_float = endian_map[args.endian]

    try:
        dim, record_size, num_records = check_dim(fname, struct_fmt_int, numpy_fmt_float, max_samples=100)
        vec = read_vector_at(fname, args.line_number, dim, record_size, num_records, numpy_fmt_float)
    except Exception as e:
        parser.error(str(e))

    print(f"{dim}-dim vector at line {args.line_number} of {num_records} lines:")
    print(vec)

if __name__ == "__main__":
    main()
