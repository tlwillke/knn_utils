#!/usr/bin/env python3
import argparse
import os

import numpy as np


def _collect_candidate_dataset_names(hf):
    """
    Return all 2D numeric dataset names in the HDF5 file.
    """
    candidates = []

    def visitor(name, obj):
        if hasattr(obj, "shape") and hasattr(obj, "dtype"):
            if len(obj.shape) == 2 and np.issubdtype(obj.dtype, np.number):
                candidates.append(name)

    hf.visititems(visitor)
    return candidates


def resolve_hdf5_keys(fname, keys=None):
    """
    Resolve which dataset keys to analyze.

    Rules:
    - If keys are provided, validate and use them.
    - Otherwise, auto-detect all 2D numeric datasets.
    """
    import h5py

    with h5py.File(fname, "r") as hf:
        candidates = _collect_candidate_dataset_names(hf)

        if keys:
            resolved = []
            for key in keys:
                if key not in hf:
                    raise ValueError(f"Key '{key}' not found in HDF5 file: {fname}")
                dset = hf[key]
                if not (hasattr(dset, "shape") and hasattr(dset, "dtype")):
                    raise ValueError(f"Key '{key}' is not a dataset in HDF5 file: {fname}")
                if len(dset.shape) != 2:
                    raise ValueError(f"Key '{key}' is not 2D in HDF5 file: {fname}; shape={dset.shape}")
                if not np.issubdtype(dset.dtype, np.number):
                    raise ValueError(f"Key '{key}' is not numeric in HDF5 file: {fname}; dtype={dset.dtype}")
                resolved.append(key)
            return resolved

        if not candidates:
            raise ValueError(f"No 2D numeric datasets found in HDF5 file: {fname}")

        return candidates


def read_hdf5(fname, key):
    """
    Reads an HDF5 file and returns a numpy array from the dataset with the given key.
    """
    import h5py

    with h5py.File(fname, "r") as hf:
        if key not in hf:
            raise ValueError(f"Key '{key}' not found in HDF5 file: {fname}")
        dset = hf[key]
        return np.array(dset)


def check_hdf5_dataset(fname, key, tol_norm=1e-3, tol_zero=1e-6, plot=False):
    """
    Check a single HDF5 dataset of embeddings.

    Expected shape:
        [num_vectors, dim]

    Returns a result dict for the dataset.
    """
    fname = os.path.expanduser(fname)
    vectors = read_hdf5(fname, key=key)

    if not isinstance(vectors, np.ndarray):
        vectors = np.array(vectors)

    if vectors.ndim != 2:
        raise ValueError(f"Expected a 2D dataset for key '{key}', got shape {vectors.shape}")

    if vectors.shape[0] == 0:
        raise ValueError(f"No vectors found in dataset '{key}'")

    vectors = np.asarray(vectors, dtype=np.float32)
    total_vectors, dim = vectors.shape
    first_embedding = vectors[0].copy()

    norms = np.linalg.norm(vectors, axis=1)
    normalized = bool(np.all(np.abs(norms - 1.0) <= tol_norm))
    zero_count = int(np.sum(norms < tol_zero))
    norms_list = norms.tolist() if plot else None

    return {
        "key": key,
        "total_vectors": total_vectors,
        "dim": dim,
        "normalized": normalized,
        "zero_count": zero_count,
        "first_embedding": first_embedding,
        "norms_list": norms_list,
    }


def _sanitize_key_for_filename(key):
    return key.replace("/", "__")


def render_histogram_and_table(filename, key, norms_list):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import NullFormatter
    except ImportError:
        print("matplotlib is not installed. Please install it with: pip install matplotlib")
        return

    norms_array = np.array(norms_list, dtype=np.float64)

    exact_zeros_mask = (norms_array == 0.0)
    exact_zeros_count = int(np.sum(exact_zeros_mask))
    non_zero_norms = norms_array[~exact_zeros_mask]

    print(f"Exact zero vectors in '{key}': {exact_zeros_count:,}")

    if len(non_zero_norms) == 0:
        print(f"All vectors in '{key}' are exact zero; skipping non-zero histogram.")
        print("\nHistogram summary:")
        print(f"{'Bin start':>18} {'Bin end':>18} {'Count':>10}")
        print(f"{'0.0 (exact)':>18} {'0.0 (exact)':>18} {exact_zeros_count:10d}")
        return

    plots_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    base_name = os.path.basename(filename)
    key_suffix = _sanitize_key_for_filename(key)
    out_png = os.path.join(
        plots_dir,
        f"{os.path.splitext(base_name)[0]}_{key_suffix}_norm_hist.png",
    )

    min_norm = float(np.min(non_zero_norms))
    max_norm = float(np.max(non_zero_norms))
    if min_norm == max_norm:
        bin_edges = np.array([min_norm, min_norm * 1.01], dtype=np.float64)
    else:
        bin_edges = np.logspace(np.log10(min_norm), np.log10(max_norm), num=101)

    plt.figure()
    counts, edges, _ = plt.hist(non_zero_norms, bins=bin_edges, log=True)
    plt.xscale("log")
    plt.xlabel("L2 Norm Magnitude (Log Scale, non-zero only)")
    plt.ylabel("Frequency (Log Scale)")
    plt.title(f"Distribution of Non-Zero Embedding Norms\n{base_name} [{key}]")

    ax = plt.gca()
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.text(
        0.98,
        0.98,
        f"Perfect zeros: {exact_zeros_count:,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Histogram saved to {out_png}")

    def format_bin_value(value, overall_span, max_abs_edge):
        if overall_span > 0 and 1e-4 <= max_abs_edge <= 1e4 and overall_span < 1.0:
            decimals = int(np.clip(np.ceil(-np.log10(overall_span)) + 2, 0, 12))
            return f"{value:.{decimals}f}"
        return f"{value:.6e}"

    overall_span = float(edges[-1] - edges[0])
    max_abs_edge = float(np.max(np.abs(edges)))

    print("\nHistogram summary:")
    print(f"{'Bin start':>18} {'Bin end':>18} {'Count':>10}")

    if exact_zeros_count > 0:
        print(f"{'0.0 (exact)':>18} {'0.0 (exact)':>18} {exact_zeros_count:10d}")

    for left, right, count in zip(edges[:-1], edges[1:], counts.astype(int)):
        if count > 0:
            left_str = format_bin_value(float(left), overall_span, max_abs_edge)
            right_str = format_bin_value(float(right), overall_span, max_abs_edge)
            print(f"{left_str:>18} {right_str:>18} {count:10d}")


def main():
    parser = argparse.ArgumentParser(
        description="Check and validate HDF5 embedding datasets."
    )
    parser.add_argument("filename", help="Path to the HDF5 file to check")
    parser.add_argument(
        "--key",
        action="append",
        default=None,
        help="Dataset key to analyze. May be specified more than once. "
             "If omitted, all 2D numeric datasets are analyzed.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a histogram of vector norms to plots/ for each analyzed dataset.",
    )
    parser.add_argument(
        "--tol-norm",
        type=float,
        default=1e-3,
        help="Tolerance for considering a vector normalized. Default: 1e-3",
    )
    parser.add_argument(
        "--tol-zero",
        type=float,
        default=1e-6,
        help="Tolerance for considering a vector approximately zero. Default: 1e-6",
    )
    args = parser.parse_args()

    try:
        resolved_keys = resolve_hdf5_keys(args.filename, keys=args.key)

        print(f"Input file: {args.filename}")
        print(f"Datasets to analyze: {', '.join(resolved_keys)}")

        report_fname = f"{os.path.splitext(args.filename)[0]}_hdf5_check_report.txt"
        with open(report_fname, "w", encoding="utf-8") as report_file:
            report_file.write(f"Input file: {args.filename}\n")
            report_file.write(f"Datasets analyzed: {', '.join(resolved_keys)}\n\n")

            for key in resolved_keys:
                print()
                print("=" * 100)
                print(f"Dataset: {key}")
                print("=" * 100)

                result = check_hdf5_dataset(
                    args.filename,
                    key=key,
                    tol_norm=args.tol_norm,
                    tol_zero=args.tol_zero,
                    plot=args.plot,
                )

                total_vectors = result["total_vectors"]
                dim = result["dim"]
                normalized = result["normalized"]
                zero_count = result["zero_count"]
                first_embedding = result["first_embedding"]
                norms_list = result["norms_list"]

                print(f"✅ Successfully processed {total_vectors} embeddings from key '{key}'.")
                print(f"🔹 Each embedding has {dim} dimensions")
                print(f"🔍 First embedding: {first_embedding}")

                if normalized:
                    print("✅ Embeddings are normalized (L2 norm ≈ 1).")
                else:
                    print("❌ Embeddings are not normalized (L2 norm not ≈ 1).")

                if zero_count > 0:
                    print(f"⚠️ Warning: Found {zero_count} ≈zero vectors in the embeddings.")
                else:
                    print("✅ No zero vectors found in the embeddings.")

                report_file.write(f"Dataset key: {key}\n")
                report_file.write(f"Total embeddings: {total_vectors}\n")
                report_file.write(f"Dimension: {dim}\n")
                report_file.write(f"Normalized: {normalized}\n")
                report_file.write(f"Zero vectors: {zero_count}\n\n")

                if args.plot:
                    render_histogram_and_table(args.filename, key, norms_list)

        print(f"\n✅ Report saved to {report_fname}")

    except FileNotFoundError:
        print(f"❌ Error: File not found at {args.filename}")
    except ValueError as e:
        print(f"❌ Error processing file: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()