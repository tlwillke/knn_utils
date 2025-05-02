#!/usr/bin/env python3
"""
shuffle_fvecs.py

Shuffle all vectors in an .fvecs file in a memory-efficient way:
– Reads the header to determine vector dimensionality.
– Computes total_vectors from file size and record size.
- Re-writes the dimension for each line.
– Builds and shuffles an index array (4 bytes per vector).
– Streams each vector record in random order to the output file.
"""

# Usage:
# ./shuffle_fvecs.py sorted_index.fvecs shuffled_index.fvecs
#!/usr/bin/env python3

import os
import struct
import argparse
import random
import sys
from array import array

def shuffle_fvecs(input_path: str, output_path: str, buffer_size: int = 1024*1024):
    # --- read expected dimension from the very first record
    with open(input_path, "rb") as f:
        hdr = f.read(4)
        if len(hdr) < 4:
            raise ValueError("Input file is too small or empty.")
        expected_dim = struct.unpack("<i", hdr)[0]

    record_size = 4 + expected_dim * 4  # 4 bytes for int32 + 4 bytes per float

    # --- compute total vectors
    total_bytes = os.path.getsize(input_path)
    if total_bytes % record_size != 0:
        raise ValueError(
            f"File size ({total_bytes}) is not a multiple of record size ({record_size})."
        )
    total_vectors = total_bytes // record_size
    print(f"Shuffling {total_vectors:,} vectors (dim={expected_dim}, record_size={record_size} bytes)")

    # --- build & shuffle index array
    idx = array('I', range(total_vectors))
    random.shuffle(idx)
    print(f"  ▸ Shuffled index array in RAM ({len(idx)*4/1e6:.1f} MB)")

    # --- open files for unbuffered I/O
    with open(input_path,  "rb", buffering=0) as fin, \
         open(output_path, "wb", buffering=0) as fout:

        for count, vec_id in enumerate(idx, 1):
            # Seek to the start of this record
            off = vec_id * record_size
            fin.seek(off)

            # 1) Read & re-write the dimension header
            dim_bytes = fin.read(4)
            if len(dim_bytes) != 4:
                raise EOFError(f"Unexpected EOF reading dimension of vector {vec_id}")
            dim_i = struct.unpack("<i", dim_bytes)[0]
            if dim_i != expected_dim:
                raise ValueError(
                    f"Dimension mismatch at vector {vec_id}: "
                    f"found {dim_i}, expected {expected_dim}"
                )
            fout.write(dim_bytes)

            # 2) Read & write the float data
            floats_to_read = expected_dim * 4
            while floats_to_read:
                chunk = fin.read(min(buffer_size, floats_to_read))
                if not chunk:
                    raise EOFError(f"Unexpected EOF reading floats of vector {vec_id}")
                fout.write(chunk)
                floats_to_read -= len(chunk)

            # Progress reporting
            if count % 1_000_000 == 0 or count == total_vectors:
                print(f"    ▸ written {count:,} / {total_vectors:,}")

    print("Shuffle complete.")

def main():
    parser = argparse.ArgumentParser(
        description="Shuffle an .fvecs file, preserving per-vector headers."
    )
    parser.add_argument("input_file",  help="Path to the input .fvecs")
    parser.add_argument("output_file", help="Path for shuffled .fvecs (must not exist)")
    parser.add_argument(
        "--buffer-size", type=int, default=1024*1024,
        help="I/O buffer size in bytes (default: 1 MiB)"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print(f"ERROR: Input file {args.input_file!r} does not exist.", file=sys.stderr)
        sys.exit(1)
    if os.path.exists(args.output_file):
        print(f"ERROR: Output file {args.output_file!r} already exists.", file=sys.stderr)
        sys.exit(1)

    try:
        shuffle_fvecs(args.input_file, args.output_file, args.buffer_size)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

