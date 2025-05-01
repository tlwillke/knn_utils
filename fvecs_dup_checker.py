#!/usr/bin/env python3
"""
External merge-sort for .fvecs files with duplicate counting.

Reads an input .fvecs file, sorts it using external merge-sort under memory constraints,
and reports vectors duplicated more than N times.
"""
import os
import argparse
import struct
import tempfile
import shutil
import heapq
import numpy as np


def read_record(f, struct_int, struct_floats, d):
    """Read one fvec record from file f. Return tuple of floats or None at EOF."""
    int_bytes = f.read(4)
    if not int_bytes:
        return None
    if len(int_bytes) < 4:
        raise ValueError("Incomplete dimension bytes")
    d_current = struct.unpack(struct_int, int_bytes)[0]
    if d_current != d:
        raise ValueError(f"Inconsistent dimension: expected {d}, got {d_current}")
    vec_bytes = f.read(4 * d)
    if len(vec_bytes) < 4 * d:
        raise ValueError("Incomplete vector bytes")
    return struct.unpack(struct_floats, vec_bytes)


def generate_runs(input_path, chunk_size, endian, temp_dir=None):
    """Split input .fvecs into sorted runs."""
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="fvecs_runs_")
        cleanup = True
    else:
        os.makedirs(temp_dir, exist_ok=True)
        cleanup = False

    prefix = '<' if endian == 'little' else '>'
    struct_int = prefix + 'i'

    # read dimension
    with open(input_path, 'rb') as f:
        d_bytes = f.read(4)
        if len(d_bytes) < 4:
            raise ValueError("Cannot read dimension from empty file")
        d = struct.unpack(struct_int, d_bytes)[0]

    struct_floats = prefix + 'f' * d

    run_paths = []
    run_index = 0
    with open(input_path, 'rb') as fin:
        while True:
            buffer = []
            for _ in range(chunk_size):
                rec = read_record(fin, struct_int, struct_floats, d)
                if rec is None:
                    break
                buffer.append(rec)
            if not buffer:
                break
            buffer.sort()
            run_path = os.path.join(temp_dir, f"run_{run_index}.fvecs")
            with open(run_path, 'wb') as fout:
                for vec in buffer:
                    fout.write(struct.pack(struct_int, d))
                    fout.write(struct.pack(struct_floats, *vec))
            print(f"Wrote run {run_path} with {len(buffer)} records")
            run_paths.append(run_path)
            run_index += 1

    return run_paths, d, struct_int, struct_floats, cleanup, temp_dir


def merge_runs(run_paths, d, struct_int, struct_floats, threshold, cleanup, temp_dir):
    """Merge sorted runs, count duplicates > threshold, and report."""
    files = [open(p, 'rb') for p in run_paths]
    heap = []
    for i, f in enumerate(files):
        rec = read_record(f, struct_int, struct_floats, d)
        if rec is not None:
            heapq.heappush(heap, (rec, i))

    prev = None
    count = 0
    duplicates = []
    while heap:
        rec, i = heapq.heappop(heap)
        if prev is None:
            prev = rec
            count = 1
        elif rec == prev:
            count += 1
        else:
            if count > threshold:
                duplicates.append((prev, count))
            prev = rec
            count = 1

        nxt = read_record(files[i], struct_int, struct_floats, d)
        if nxt is not None:
            heapq.heappush(heap, (nxt, i))

    if prev is not None and count > threshold:
        duplicates.append((prev, count))

    for f in files:
        f.close()

    print(f"Vectors duplicated more than {threshold} times:")
    for vec, c in duplicates:
        print(f"Count={c}\tVector={vec}")

    if cleanup:
        shutil.rmtree(temp_dir)
        print(f"Removed temporary runs at {temp_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="External merge-sort and duplicate count for .fvecs"
    )
    parser.add_argument(
        "filename", help="Input .fvecs file"
    )
    parser.add_argument(
        "-n", "--threshold", type=int, default=1,
        help="Only report vectors duplicated more than N times"
    )
    parser.add_argument(
        "-c", "--chunk_size", type=int, default=500000,
        help="Number of vectors per in-memory chunk"
    )
    parser.add_argument(
        "--endian", choices=['little', 'big'], default='little',
        help="Byte order of .fvecs file"
    )
    parser.add_argument(
        "--temp_dir", help="Directory for temporary runs"
    )
    args = parser.parse_args()

    run_paths, d, struct_int, struct_floats, cleanup, temp_dir = generate_runs(
        args.filename, args.chunk_size, args.endian, args.temp_dir
    )
    merge_runs(
        run_paths, d, struct_int, struct_floats,
        args.threshold, cleanup, temp_dir
    )

if __name__ == '__main__':
    main()