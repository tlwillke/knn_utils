import os
import numpy as np
import argparse

# tolerance for |L2 norm − 1|
TOL = 1e-3

def read_fvecs(fname, endian='little'):
    """
    Reads an .fvecs file and returns the embeddings as a NumPy array.
    Allows specifying the endianness ('little' or 'big').
    """
    fname = os.path.expanduser(fname)

    if endian == 'little':
        dtype_prefix = '<'
    elif endian == 'big':
        dtype_prefix = '>'
    else:
        # This case should be prevented by argparse choices, but good practice
        raise ValueError("Invalid endian value. Use 'little' or 'big'.")

    int_dtype = dtype_prefix + 'i4'   # e.g., '<i4' or '>i4'
    float_dtype = dtype_prefix + 'f4' # e.g., '<f4' or '>f4'

    with open(fname, "rb") as f:
        # --- Use the determined integer dtype ---
        dim_array = np.fromfile(f, dtype=int_dtype, count=1)
        if len(dim_array) == 0:
            raise ValueError("Cannot read dimension - file might be empty.")
        dim = dim_array[0]  # No .item() needed here, result is scalar

        # Basic sanity check for dimension read
        if dim <= 0:
            raise ValueError(
                f"Invalid dimension read ({dim}) using '{endian}' endian. Check file format/endianness.")

        f.seek(0)  # Reset file pointer

        # --- Use the determined float dtype ---
        data = np.fromfile(f, dtype=float_dtype)

        total_values = len(data)

        # Validation and reshaping logic remains the same
        expected_values_per_record = dim + 1
        if total_values % expected_values_per_record != 0:
            raise ValueError(
                f"File appears corrupted or wrong format/endianness ('{endian}') used: "
                f"total values ({total_values}) not divisible by "
                f"dim+1 ({expected_values_per_record})."
            )

        num_embeddings = total_values // expected_values_per_record
        embeddings = data.reshape(num_embeddings, expected_values_per_record)[:, 1:]

        return embeddings, num_embeddings, dim

def check_normalization(embeddings, tol=1e-3):
    """
    Checks if all embeddings are normalized.
    Returns True if each vector's L2 norm is approximately 1 within the tolerance.
    """
    norms = np.linalg.norm(embeddings, axis=1)
    return np.all(np.abs(norms - 1) < tol)

def count_zero_vectors(embeddings, tol=1e-6):
    """
    Counts the number of embeddings that are near zero vectors.
    A vector is considered a zero vector if its L2 norm is below the specified tolerance.
    """
    norms = np.linalg.norm(embeddings, axis=1)
    return np.sum(norms < tol)

def main():
    """Main function to check and validate an .fvecs file."""
    parser = argparse.ArgumentParser(description="Check and validate an .fvecs file.")
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

    embeddings, num_embeddings, dim = read_fvecs(args.filename, endian=args.endian)

    print(f"✅ Successfully read {num_embeddings} embeddings")
    print(f"🔹 Each embedding has {dim} dimensions")
    print(f"🔍 First embedding: {embeddings[0]}")  # Print first embedding

    def count_unnormalized(embeddings, tol=TOL):
        """
        Counts how many embeddings deviate from unit length by ≥ tol.
        """
        norms = np.linalg.norm(embeddings, axis=1)
        return np.sum(np.abs(norms - 1) >= tol)

    # Report normalization status.
    if check_normalization(embeddings, tol=TOL):
        print(f"✅ All embeddings are within ±{TOL:.1e} of unit length.")
    else:
        bad = count_unnormalized(embeddings, tol=TOL)
        print(f"❌ Found {bad} embeddings with |L2 norm − 1| ≥ {TOL:.1e}")

    # Count and report zero vectors.
    num_zero_vectors = count_zero_vectors(embeddings)
    if num_zero_vectors > 0:
        print(f"⚠️ Warning: Found {num_zero_vectors} ≈zero vectors in the embeddings.")
    else:
        print("✅ No zero vectors found in the embeddings.")

    # Optionally plot the histogram of vector norms using Plotly.
    if args.plot:
        try:
            import plotly.express as px
        except ImportError:
            print("Plotly is not installed. Please install plotly and pandas.")
            return

        norms = np.linalg.norm(embeddings, axis=1)
        fig = px.histogram(norms, nbins=50, labels={'value': 'L2 Norm'}, title='Histogram of Embedding Norms')
        fig.update_layout(xaxis_title='L2 Norm', yaxis_title='Count')
        output_file = "plot.html"
        fig.write_html(output_file)
        print(f"Plot saved to {output_file}. Open this file in a web browser to view the plot.")
        fig.show()

if __name__ == "__main__":
    main()