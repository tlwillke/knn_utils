import os
import numpy as np
import argparse
import struct


def stream_check_fvecs(fname, tol_norm=1e-3, tol_zero=1e-6, plot=False, endian='little'):
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
    norms_list = [] if plot else None
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

            # --- Perform checks (norm, zero) ---
            norm_val = np.linalg.norm(vector)
            if plot:
                norms_list.append(norm_val)

            # Update overall normalized status
            if normalized and not (abs(norm_val - 1) < tol_norm):
                normalized = False  # Once false, stays false

            # Count zero vectors
            if norm_val < tol_zero:
                zero_count += 1

    # If no vectors were read at all
    if d == -1 and total_vectors == 0:
        raise ValueError("No valid vector data found in the file.")

    return total_vectors, d, normalized, zero_count, first_embedding, norms_list

def main():
    """Main function to check and validate an .fvecs file using streaming processing."""
    parser = argparse.ArgumentParser(
        description="Check and validate an .fvecs file using streaming processing."
    )
    parser.add_argument("filename", help="Path to the .fvecs file to check")
    parser.add_argument(
        "--endian",
        choices=['little', 'big'], # Allow only these two choices
        default='little',         # Default to little-endian (fvec standard)
        help="Specify the byte order (endianness) of the file. Default: little"
    )
    parser.add_argument("--plot", action="store_true",
                        help="Plot a histogram of vector norms using Plotly")
    args = parser.parse_args()
    try:
        total_vectors, dim, normalized, zero_count, first_embedding, norms_list = stream_check_fvecs(
            args.filename,
            plot=args.plot,
            endian=args.endian  # Pass the selected endianness
        )

        print(f"✅ Successfully processed {total_vectors} embeddings using '{args.endian}' endian format.")
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

        if args.plot:
            try:
                import plotly.express as px
            except ImportError:
                print("Plotly is not installed. Please install plotly and pandas.")
                return

            # Plot the histogram of L2 norms using the accumulated norms_list.
            fig = px.histogram(norms_list, nbins=50, labels={'value': 'L2 Norm'}, title='Histogram of Embedding Norms')
            fig.update_layout(xaxis_title='L2 Norm', yaxis_title='Count')
            output_file = "plot.html"
            fig.write_html(output_file)
            print(f"Plot saved to {output_file}. Open this file in a web browser to view the plot.")
            fig.show()

    # Add specific error handling
    except FileNotFoundError:
        print(f"❌ Error: File not found at {args.filename}")
    except ValueError as e:
        print(f"❌ Error processing file: {e}. Check file format, integrity, and selected endianness ('{args.endian}').")
    except Exception as e: # Catch other potential errors
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
