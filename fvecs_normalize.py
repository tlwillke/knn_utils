#!/usr/bin/env python3

import argparse
import os
import shutil
import struct
import numpy as np


def read_fvecs(fname):
    fname = os.path.expanduser(fname)
    data = np.fromfile(fname, dtype=np.float32)

    if data.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    dim = struct.unpack("<I", data[:1].tobytes())[0]
    if dim <= 0:
        raise ValueError(f"Invalid dimension {dim} in {fname}")

    row_width = dim + 1
    if data.size % row_width != 0:
        raise ValueError(
            f"File size is not consistent with fvecs format: "
            f"{fname}, dim={dim}, float_count={data.size}"
        )

    data = data.reshape(-1, row_width)
    dims = data[:, 0].view(np.int32)

    if not np.all(dims == dim):
        raise ValueError(f"Inconsistent vector dimensions in {fname}")

    return np.ascontiguousarray(data[:, 1:], dtype=np.float32)


def write_fvecs(fname, arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")

    n, d = arr.shape
    fname = os.path.expanduser(fname)

    d_repr = struct.unpack("<f", np.uint32(d))[0]
    formatted = np.concatenate(
        (np.full((n, 1), d_repr, dtype=np.float32), arr),
        axis=1
    )

    if n > 0:
        assert struct.unpack("<I", formatted[0, 0].tobytes()) == (d,)

    with open(fname, "wb") as f:
        formatted.tofile(f)


def check_normalization(vecs, tol=1e-3):
    """
    Returns True if all vectors in the array are approximately normalized
    (L2 norm close to 1 within the specified tolerance).
    """
    norms = np.linalg.norm(vecs, axis=1)
    return np.all(np.abs(norms - 1) < tol)


def normalize_vectors(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(arr / norms, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Normalize vectors in an fvecs file.")
    parser.add_argument("--input", required=True, help="Input fvecs file")
    parser.add_argument("--output", required=True, help="Output normalized fvecs file")
    args = parser.parse_args()

    vectors = read_fvecs(args.input)

    normalized_before = check_normalization(vectors)
    print("Vectors normalized before:", "Yes" if normalized_before else "No")
    print(f"Vectors: {vectors.shape[0]}")
    print(f"Dimension: {vectors.shape[1] if vectors.size > 0 else 0}")

    input_path = os.path.expanduser(args.input)
    output_path = os.path.expanduser(args.output)

    if normalized_before:
        print("Input is already normalized. Skipping normalization.")
        if os.path.abspath(input_path) != os.path.abspath(output_path):
            shutil.copyfile(input_path, output_path)
            print(f"Copied input to output without changes: {output_path}")
        else:
            print("Input and output are the same file. No action needed.")
        return

    normalized = normalize_vectors(vectors)

    normalized_after = check_normalization(normalized)
    print("Vectors normalized after:", "Yes" if normalized_after else "No")

    write_fvecs(output_path, normalized)
    print(f"Wrote normalized file to: {output_path}")


if __name__ == "__main__":
    main()
    