#!/usr/bin/env python3

import os
import numpy as np
import argparse
import struct

PROGRESS_INTERVAL = 1_000_000
DEFAULT_TOL_NORM = 1e-5
DEFAULT_TOL_ZERO = 1e-6
DEFAULT_FIRST_VECTOR_DIMS = 100

def format_first_embedding(embedding, max_dims):
    """Format the first embedding, truncating to max_dims if needed."""
    if embedding is None:
        return "None"

    shown = embedding[:max_dims]
    formatted = np.array2string(
        shown,
        threshold=max_dims,
        max_line_width=120,
    )

    if len(embedding) <= max_dims:
        return formatted

    return f"{formatted} ... (truncated to first {max_dims} of {len(embedding)} dims)"


def format_bin_value(value, overall_span, max_abs_edge):
    """Format histogram bin edges for readable text output."""
    if overall_span > 0 and 1e-4 <= max_abs_edge <= 1e4 and overall_span < 1.0:
        decimals = int(np.clip(np.ceil(-np.log10(overall_span)) + 2, 0, 12))
        return f"{value:.{decimals}f}"
    return f"{value:.6e}"


def compute_norm_stats(norms_list):
    """Compute summary statistics for vector norms."""
    norms_array = np.asarray(norms_list, dtype=np.float64)
    return {
        "min": float(np.min(norms_array)),
        "max": float(np.max(norms_array)),
        "mean": float(np.mean(norms_array)),
        "max_abs_deviation_from_1": float(np.max(np.abs(norms_array - 1.0))),
    }


def compute_histogram_data(norms_list):
    """Compute histogram inputs and counts for norm reporting."""
    norms_array = np.asarray(norms_list, dtype=np.float64)

    exact_zeros_mask = (norms_array == 0.0)
    exact_zeros_count = int(np.sum(exact_zeros_mask))
    non_zero_norms = norms_array[~exact_zeros_mask]

    if len(non_zero_norms) == 0:
        return exact_zeros_count, non_zero_norms, None, None

    min_norm = float(np.min(non_zero_norms))
    max_norm = float(np.max(non_zero_norms))

    if min_norm == max_norm:
        bin_edges = np.array([min_norm, min_norm * 1.01], dtype=np.float64)
    else:
        bin_edges = np.logspace(np.log10(min_norm), np.log10(max_norm), num=101)

    counts, edges = np.histogram(non_zero_norms, bins=bin_edges)
    return exact_zeros_count, non_zero_norms, counts.astype(int), edges


def format_histogram_summary(exact_zeros_count, counts, edges):
    """Return the text histogram summary for console and report output."""
    lines = []
    lines.append("Histogram summary:")
    lines.append(f"{'Bin start':>18} {'Bin end':>18} {'Count':>10}")

    if exact_zeros_count > 0:
        lines.append(f"{'0.0 (exact)':>18} {'0.0 (exact)':>18} {exact_zeros_count:10d}")

    if counts is None or edges is None:
        lines.append("All vectors are exact zero; skipping non-zero histogram bins.")
        return "\n".join(lines)

    overall_span = float(edges[-1] - edges[0])
    max_abs_edge = float(np.max(np.abs(edges)))

    for left, right, count in zip(edges[:-1], edges[1:], counts):
        if count > 0:
            left_str = format_bin_value(float(left), overall_span, max_abs_edge)
            right_str = format_bin_value(float(right), overall_span, max_abs_edge)
            lines.append(f"{left_str:>18} {right_str:>18} {int(count):10d}")

    return "\n".join(lines)

def check_fvecs(fname, tol_norm=DEFAULT_TOL_NORM, tol_zero=DEFAULT_TOL_ZERO, endian='little'):
    """
    Stream process an .fvecs file to check for normalization and zero vectors.
    Allows specifying the endianness ('little' or 'big').

    Each vector is stored as:
        [d (int32), float, float, ..., float] (using specified endianness)

    Returns:
        total_vectors (int): Number of embeddings processed.
        dim (int): Dimension of the embeddings.
        normalized (bool): True IF AND ONLY IF *all* processed embeddings have L2 norm ≈ 1 within tol_norm.
        zero_count (int): Count of embeddings with L2 norm below tol_zero.
        first_embedding (np.array): The first embedding vector.
        norms_list (list or None): A list of L2 norms (if plot=True) or None.
    """
    fname = os.path.expanduser(fname)
    total_vectors = 0
    normalized = True  # Assume true until proven otherwise
    zero_count = 0
    norms_list = []
    first_embedding = None
    d = -1  # Initialize dimension

    # --- Determine format specifiers based on endian argument ---
    if endian == 'little':
        struct_prefix = '<'
        numpy_prefix = '<'
    elif endian == 'big':
        struct_prefix = '>'
        numpy_prefix = '>'
    else:
        # This case should be prevented by argparse choices
        raise ValueError(f"Invalid endianness specified: {endian}")

    struct_fmt_int = struct_prefix + 'i'   # e.g., '<i' or '>i' for int32
    numpy_fmt_float = numpy_prefix + 'f4' # e.g., '<f4' or '>f4' for float32

    with open(fname, "rb") as f:
        while True:  # Loop structure simplified slightly
            # --- Read dimension for current vector ---
            d_bytes = f.read(4)
            if not d_bytes:
                # Clean break at EOF if no bytes read for dimension
                if total_vectors == 0:
                    print("Warning: File appears empty.")  # Handle empty file case
                break  # End of file

            if len(d_bytes) < 4:
                raise ValueError(
                    f"Incomplete dimension read at vector {total_vectors + 1}. Expected 4 bytes, got {len(d_bytes)}.")

            # --- Use the determined struct format ---
            d_current = struct.unpack(struct_fmt_int, d_bytes)[0]

            if total_vectors == 0:
                # This is the first vector, set the expected dimension
                d = d_current
                if d <= 0:
                    raise ValueError(
                        f"Invalid dimension read for first vector ({d}) using '{endian}' endian. Check file format/endianness.")
            elif d_current != d:
                # Check consistency for subsequent vectors
                raise ValueError(
                    f"Inconsistent dimension: expected {d}, got {d_current} at vector {total_vectors + 1}.")

            # --- Read vector data ---
            vec_byte_count = 4 * d
            vec_bytes = f.read(vec_byte_count)
            if len(vec_bytes) < vec_byte_count:
                raise ValueError(
                    f"Incomplete vector data at vector {total_vectors + 1}. Expected {vec_byte_count} bytes, got {len(vec_bytes)}.")

            # --- Use the determined numpy dtype ---
            # Use .copy() only if you intend to modify 'first_embedding' later, otherwise it's unnecessary overhead
            vector = np.frombuffer(vec_bytes, dtype=numpy_fmt_float)

            total_vectors += 1

            # Process the first vector specifically if needed
            if total_vectors == 1:
                first_embedding = vector.copy()  # Make a copy if you want to return it unmodified

            if total_vectors % PROGRESS_INTERVAL == 0:
                print(f"[Progress] processed {total_vectors} vectors…", flush=True)

            # --- Perform checks (norm, zero) ---
            norm_val = np.linalg.norm(vector)
            norms_list.append(norm_val)

            # Update overall normalized status
            if (abs(norm_val - 1) > tol_norm):
                print(f"Vector # {total_vectors} norm value is {norm_val}")
                normalized = False  # Once false, stays false

            # Count zero vectors
            if norm_val < tol_zero:
                zero_count += 1

    # If no vectors were read at all
    if d == -1 and total_vectors == 0:
        raise ValueError("No valid vector data found in the file.")

    return total_vectors, d, normalized, zero_count, first_embedding, norms_list

def main():
    """Main function to check and validate an .fvecs file using stream processing."""
    parser = argparse.ArgumentParser(
        description="Check and validate an .fvecs file using stream processing."
    )
    parser.add_argument("filename", help="Path to the .fvecs file to check")
    parser.add_argument(
        "--endian",
        choices=['little', 'big'],
        default='little',         # Default to little-endian (fvec standard)
        help="Specify the byte order (endianness) of the file. Default: little"
    )
    parser.add_argument(
        "--tol-norm",
        type=float,
        default=DEFAULT_TOL_NORM,
        help=f"Normalization tolerance. Default: {DEFAULT_TOL_NORM}"
    )
    parser.add_argument(
        "--tol-zero",
        type=float,
        default=DEFAULT_TOL_ZERO,
        help=f"Zero-vector tolerance. Default: {DEFAULT_TOL_ZERO}"
    )
    parser.add_argument(
        "--first-vector-dims",
        type=int,
        default=DEFAULT_FIRST_VECTOR_DIMS,
        help=f"Number of dimensions of the first vector to display/save. Default: {DEFAULT_FIRST_VECTOR_DIMS}"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a histogram plot of vector norms as a PNG"
    )
    args = parser.parse_args()
    if args.first_vector_dims <= 0:
        raise ValueError("--first-vector-dims must be > 0")
    try:
        total_vectors, dim, normalized, zero_count, first_embedding, norms_list = check_fvecs(
            args.filename,
            tol_norm=args.tol_norm,
            tol_zero=args.tol_zero,
            endian=args.endian
        )

        first_embedding_str = format_first_embedding(first_embedding, args.first_vector_dims)

        norm_stats = compute_norm_stats(norms_list)
        unnormalized_count = int(np.sum(np.abs(np.asarray(norms_list, dtype=np.float64) - 1.0) > args.tol_norm))
        exact_zeros_count, non_zero_norms, counts, edges = compute_histogram_data(norms_list)
        histogram_summary = format_histogram_summary(exact_zeros_count, counts, edges)

        summary_lines = [
            f"Input file: {args.filename}",
            f"Endianness: {args.endian}",
            f"Total embeddings: {total_vectors}",
            f"Dimensionality: {dim}",
            f"Zero vectors (< {args.tol_zero}): {zero_count}",
            f"Unnormalized vectors (abs(||v||_2 - 1.0) > {args.tol_norm}): {unnormalized_count}",
            f"Norm max abs deviation from 1.0: {norm_stats['max_abs_deviation_from_1']:.9e}",
            f"Norm mean: {norm_stats['mean']:.9e}",
        ]

        print(f"✅ Successfully processed {total_vectors} embeddings using '{args.endian}' endian format.")
        print()
        print(f"First embedding: {first_embedding_str}")
        print()

        for line in summary_lines:
            print(line)

        print()
        print(histogram_summary)

        report_fname = f"{os.path.splitext(args.filename)[0]}_fvecs_check_report.txt"
        with open(report_fname, "w") as report_file:
            report_file.write(f"First embedding: {first_embedding_str}\n\n")
            report_file.write("\n".join(summary_lines))
            report_file.write("\n\n")
            report_file.write(histogram_summary)
            report_file.write("\n")

        print(f"\n✅ Report saved to {report_fname}")

        if args.plot:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                from matplotlib.ticker import NullFormatter
            except ImportError:
                print("matplotlib is not installed. Please install it with: pip install matplotlib")
                return

            if len(non_zero_norms) == 0:
                print("All vectors are exact zero; skipping non-zero histogram plot.")
                return

            plots_dir = os.path.join(os.getcwd(), "plots")
            os.makedirs(plots_dir, exist_ok=True)
            base_name = os.path.basename(args.filename)
            out_png = os.path.join(plots_dir, f"{os.path.splitext(base_name)[0]}_norm_hist.png")

            plt.figure()
            plt.hist(non_zero_norms, bins=edges, log=True)
            plt.xscale("log")
            plt.xlabel("L2 Norm Magnitude (Log Scale, non-zero only)")
            plt.ylabel("Frequency (Log Scale)")
            plt.title(f"Distribution of Non-Zero Embedding Norms\n{base_name}")

            ax = plt.gca()
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.text(
                0.98, 0.98,
                f"Perfect zeros: {exact_zeros_count:,}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
            )

            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"✅ Histogram plot saved to {out_png}")

    # Add specific error handling
    except FileNotFoundError:
        print(f"❌ Error: File not found at {args.filename}")
    except ValueError as e:
        print(f"❌ Error processing file: {e}. Check file format, integrity, and selected endianness ('{args.endian}').")
    except Exception as e: # Catch other potential errors
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
