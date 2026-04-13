#!/usr/bin/env python3

import os
import sys
import shutil
from pathlib import Path
import numpy as np
import faiss
import argparse
import struct
import yaml

ZERO_NORM_TOLERANCE = 1e-6
NORMALIZATION_TOLERANCE = 1e-5
DEFAULT_YAML_CONFIG_DIR = "yaml-configs"


def resolve_yaml_config_path(path):
    """
    Resolve a YAML config path.

    If path is explicit and exists, use it as-is.
    Otherwise, look for it under the repo's yaml-configs directory.
    """
    expanded = Path(os.path.expanduser(path))

    if expanded.exists():
        return expanded

    candidate = Path.cwd() / DEFAULT_YAML_CONFIG_DIR / path
    if candidate.exists():
        return candidate

    raise ValueError(
        f"YAML config not found: {path}. "
        f"Tried '{expanded}' and '{candidate}'."
    )


def load_yaml_config(path):
    """Load a YAML config file into a dict."""
    resolved = resolve_yaml_config_path(path)
    with open(resolved, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("YAML config must contain a top-level mapping.")

    return data


def read_fvecs(fname):
    """
    Read an fvecs file and return a NumPy array of shape (n, d).
    """
    with open(fname, "rb") as f:
        data = np.fromfile(f, dtype=np.int32)
        dim = data[0].item()
        f.seek(0)
        data = np.fromfile(f, dtype=np.float32)
        num_vectors = len(data) // (dim + 1)
        return data.reshape(num_vectors, dim + 1)[:, 1:]


def read_hdf5(fname, key="data"):
    """
    Read an HDF5 file and return a NumPy array from the dataset with the given key.
    """
    import h5py
    with h5py.File(fname, "r") as hf:
        if key not in hf:
            raise ValueError(f"Key '{key}' not found in HDF5 file: {fname}")
        dset = hf[key]
        return np.array(dset)


def read_hdf5_tensor(fname, key="data"):
    """
    Read an HDF5 tensor dataset and flatten the leading dimensions into rows.
    """
    import h5py
    with h5py.File(fname, "r") as hf:
        if key not in hf:
            raise ValueError(f"Key '{key}' not found in HDF5 file: {fname}")
        tensor_arr = np.array(hf[key])
        return tensor_arr.reshape(-1, tensor_arr.shape[-1])


def read_vectors(fname):
    """
    Determine whether the file is HDF5 or fvecs.

    For HDF5 files, use the format "file.h5:key" or "file.hdf5:key".
    Otherwise the file is read as fvecs.
    """
    fname = os.path.expanduser(fname)
    if ":" in fname:
        file_path, key = fname.split(":", 1)
        if file_path.endswith(".h5") or file_path.endswith(".hdf5"):
            return read_hdf5(file_path, key)
        raise ValueError("For HDF5, use the format 'file.h5:key'")
    return read_fvecs(fname)


def write_fvecs(fname, arr):
    """
    Write a NumPy array of shape (n, d) to an fvecs file.
    """
    n, d = arr.shape
    fname = os.path.expanduser(fname)
    with open(fname, "wb") as f:
        d_repr = struct.unpack("<f", np.uint32(d))[0]
        formatted = np.concatenate(
            (np.full((n, 1), d_repr, dtype=np.float32), arr.astype(np.float32)),
            axis=1,
        )
        assert struct.unpack("<I", formatted[0][0]) == (d,)
        formatted.tofile(f)


def write_ivecs(fname, ivecs):
    """
    Write an array of integer vectors to an ivecs file.
    """
    n, k = ivecs.shape
    fname = os.path.expanduser(fname)
    with open(fname, "wb") as f:
        formatted = np.concatenate(
            (np.full((n, 1), k, dtype=np.int32), ivecs.astype(np.int32)),
            axis=1,
        )
        formatted.tofile(f)


def write_processed_output(input_path, output_path, arr, changed, label):
    """
    Write a processed output file.

    If this side did not change, copy the input file to the output when the
    paths differ. Otherwise write the processed in-memory array as fvecs.
    """
    input_path = os.path.expanduser(input_path)
    output_path = os.path.expanduser(output_path)

    print(f"Writing processed {label} vectors to:", output_path)

    if not changed:
        if os.path.abspath(input_path) != os.path.abspath(output_path):
            shutil.copyfile(input_path, output_path)
            print(f"Copied unchanged {label} input to output.")
        else:
            print(f"{label.capitalize()} input and output are the same file. No action needed.")
        return

    write_fvecs(output_path, arr)


def count_zero_vectors(vecs, tol=ZERO_NORM_TOLERANCE):
    """Count vectors whose L2 norm is less than or equal to tol."""
    norms = np.linalg.norm(vecs, axis=1)
    return int(np.sum(norms <= tol))


def remove_zero_vectors(arr, name, tol=ZERO_NORM_TOLERANCE):
    """Remove vectors whose L2 norm is less than or equal to tol."""
    norms = np.linalg.norm(arr, axis=1)
    keep = norms > tol
    removed = int((~keep).sum())
    if removed:
        print(
            f"Removed {removed} zero-like vectors from {name} "
            f"(kept {int(keep.sum())} / {arr.shape[0]})."
        )
    else:
        print(f"Removed 0 zero-like vectors from {name}.")
    return np.ascontiguousarray(arr[keep], dtype=np.float32)


def check_normalization(vecs, tol=NORMALIZATION_TOLERANCE):
    """Return True if every vector norm is within tol of 1.0."""
    norms = np.linalg.norm(vecs, axis=1)
    return np.all(np.abs(norms - 1.0) < tol)


def normalize_vectors(arr, zero_tol=ZERO_NORM_TOLERANCE):
    """
    Normalize each vector to unit L2 norm.

    Zero-like vectors are left unchanged by replacing very small norms with 1.0
    before division.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms <= zero_tol] = 1.0
    return np.ascontiguousarray(arr / norms, dtype=np.float32)


def build_index(base, d, metric, gpu_ids):
    """
    Build a FAISS index for the given base vectors, dimension, and metric.
    """
    if metric == "l2":
        cpu_index = faiss.IndexFlatL2(d)
    elif metric == "ip":
        cpu_index = faiss.IndexFlatIP(d)
    else:
        raise ValueError(
            f"Unsupported metric: {metric}. Allowed values are: ip, l2."
        )

    if gpu_ids[0] < 0:
        print("Using device: cpu")
        index = cpu_index
    elif len(gpu_ids) == 1:
        print(f"Using device: cuda({gpu_ids[0]})")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, gpu_ids[0], cpu_index)
    else:
        print("Using devices:", ", ".join(f"cuda({g})" for g in gpu_ids))
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.devices = gpu_ids
        index = faiss.index_cpu_to_all_gpus(cpu_index, co=co)

    index.add(base)
    return index


def main():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to a YAML config file, or a file name under yaml-configs/.",
    )
    bootstrap_args, remaining_argv = bootstrap.parse_known_args()

    config = {}
    if bootstrap_args.config:
        config = load_yaml_config(bootstrap_args.config)

    parser = argparse.ArgumentParser(
        description="Compute ground truth for nearest neighbor search using a GPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[bootstrap],
    )

    parser.add_argument(
        "--base",
        type=str,
        required="base" not in config,
        help='Path to the base vectors file (fvecs or HDF5). For HDF5, use the format "file.h5:key".',
    )
    parser.add_argument(
        "--query",
        type=str,
        required="query" not in config,
        help='Path to the query vectors file (fvecs or HDF5). For HDF5, use the format "file.h5:key".',
    )
    parser.add_argument(
        "--ground_truth_out",
        type=str,
        required="ground_truth_out" not in config,
        help="Output ivecs file to write ground truth indices.",
    )
    parser.add_argument(
        "--num_base",
        type=int,
        default=0,
        help="Number of base vectors for truncation. Use 0 to skip truncation.",
    )
    parser.add_argument(
        "--num_query",
        type=int,
        default=0,
        help="Number of query vectors for truncation. Use 0 to skip truncation.",
    )
    parser.add_argument(
        "--remove_zeros",
        action="store_true",
        default=False,
        help="If set, remove zero-like vectors from both base and query.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=False,
        help="If set, shuffle both base and query vectors.",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="Random seed used when shuffling base and query vectors.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="If set, normalize both base and query vectors.",
    )
    parser.add_argument(
        "--zero_tolerance",
        type=float,
        default=ZERO_NORM_TOLERANCE,
        help="Treat vectors with L2 norm <= this value as zero-like for counting, removal, and zero-safe normalization.",
    )
    parser.add_argument(
        "--normalization_tolerance",
        type=float,
        default=NORMALIZATION_TOLERANCE,
        help="Tolerance used when checking whether vectors are already normalized to unit L2 norm.",
    )
    parser.add_argument(
        "--processed_base_out",
        type=str,
        default="",
        help="Output file for processed base vectors when processing is requested.",
    )
    parser.add_argument(
        "--processed_query_out",
        type=str,
        default="",
        help="Output file for processed query vectors when processing is requested.",
    )
    parser.add_argument(
        "--k",
        type=int,
        required="k" not in config,
        help="Number of nearest neighbors to compute ground truth indices for.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="-1",
        help='Comma-separated list of GPU ids to use. Use "-1" for CPU.',
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="l2",
        choices=["l2", "ip"],
        help='Distance metric to use: "l2" or "ip".',
    )

    parser.set_defaults(**config)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(2)

    args = parser.parse_args(remaining_argv)
    args.config = bootstrap_args.config

    if args.config:
        print("Using config:", args.config)

    if args.zero_tolerance < 0:
        raise ValueError("--zero_tolerance must be non-negative")
    if args.normalization_tolerance < 0:
        raise ValueError("--normalization_tolerance must be non-negative")

    gpu_ids = [int(x) for x in args.gpus.split(",")]

    print("Loading base vectors from:", args.base)
    base = read_vectors(args.base)
    print(f"Loaded {base.shape[0]} base vectors of dimension {base.shape[1]}.")

    print("Loading query vectors from:", args.query)
    query = read_vectors(args.query)
    print(f"Loaded {query.shape[0]} query vectors of dimension {query.shape[1]}.")

    base_changed = False
    query_changed = False
    original_base_count = base.shape[0]
    original_query_count = query.shape[0]

    d = base.shape[1]
    if query.shape[1] != d:
        raise ValueError(
            "Dimension mismatch: base vectors have dimension {} but query vectors have dimension {}."
            .format(d, query.shape[1])
        )

    print(f"Zero tolerance: {args.zero_tolerance}")
    base_zero = count_zero_vectors(base, tol=args.zero_tolerance)
    query_zero = count_zero_vectors(query, tol=args.zero_tolerance)
    print(f"Base zero-like vectors: {base_zero} / {base.shape[0]}")
    print(f"Query zero-like vectors: {query_zero} / {query.shape[0]}")

    if args.remove_zeros:
        print("Removing zero-like vectors from both base and query.")
        if base_zero > 0:
            base = remove_zero_vectors(base, "base", tol=args.zero_tolerance)
            base_changed = True
        else:
            print("Removed 0 zero-like vectors from base.")
        if query_zero > 0:
            query = remove_zero_vectors(query, "query", tol=args.zero_tolerance)
            query_changed = True
        else:
            print("Removed 0 zero-like vectors from query.")

        if base.shape[0] == 0:
            raise ValueError("All base vectors were zero after removal.")
        if query.shape[0] == 0:
            raise ValueError("All query vectors were zero after removal.")

    print(f"Normalization tolerance: {args.normalization_tolerance}")
    base_normalized = check_normalization(base, tol=args.normalization_tolerance)
    query_normalized = check_normalization(query, tol=args.normalization_tolerance)
    print("Base vectors normalized:", "Yes" if base_normalized else "No")
    print("Query vectors normalized:", "Yes" if query_normalized else "No")

    if args.shuffle:
        print(f"Shuffling both base and query vectors with seed {args.shuffle_seed}.")
        np.random.seed(args.shuffle_seed)
        np.random.shuffle(base)
        np.random.shuffle(query)
        base_changed = True
        query_changed = True

    if args.num_base > 0:
        if args.num_base > original_base_count:
            raise ValueError("Truncated base size exceeds full dataset size.")
        if args.num_base < original_base_count:
            base = base[:args.num_base]
            base_changed = True
            print(f"Using truncated base: {args.num_base} vectors.")

    if args.num_query > 0:
        if args.num_query > original_query_count:
            raise ValueError("Truncated query size exceeds full dataset size.")
        if args.num_query < original_query_count:
            query = query[:args.num_query]
            query_changed = True
            print(f"Using truncated query: {args.num_query} vectors.")

    if args.normalize:
        normalized_base = False
        normalized_query = False

        if not base_normalized:
            base = normalize_vectors(base, zero_tol=args.zero_tolerance)
            base_changed = True
            normalized_base = True

        if not query_normalized:
            query = normalize_vectors(query, zero_tol=args.zero_tolerance)
            query_changed = True
            normalized_query = True

        if normalized_base and normalized_query:
            print("Normalized both base and query vectors.")
        elif normalized_base:
            print("Normalized base vectors.")
        elif normalized_query:
            print("Normalized query vectors.")
        else:
            print("Normalization requested, but no normalization was needed.")

    requested_processing = (
        args.remove_zeros
        or args.normalize
        or args.shuffle
        or args.num_base > 0
        or args.num_query > 0
    )

    if requested_processing:
        if not args.processed_base_out or not args.processed_query_out:
            raise ValueError(
                "When removing zeros, normalization, shuffling, or truncation is applied, "
                "processed_base_out and processed_query_out must be provided."
            )

        write_processed_output(
            args.base, args.processed_base_out, base, base_changed, "base"
        )
        write_processed_output(
            args.query, args.processed_query_out, query, query_changed, "query"
        )

    print("Adding base vectors to the index...")
    index = build_index(base, d, args.metric, gpu_ids)

    print("Performing nearest neighbor search for k =", args.k)
    distances, indices = index.search(query, args.k)
    print("Search completed.")

    print("Writing results to output file:", args.ground_truth_out)
    write_ivecs(args.ground_truth_out, indices)
    print("Done.")


if __name__ == "__main__":
    main()
