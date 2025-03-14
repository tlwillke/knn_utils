import numpy as np
import argparse

def read_fvecs(fname):
    """Reads an .fvecs file and returns the embeddings as a NumPy array."""
    with open(fname, "rb") as f:
        data = np.fromfile(f, dtype=np.int32)
        dim = data[0].item()
        f.seek(0)  # Reset file pointer to re-read data correctly
        data = np.fromfile(f, dtype=np.float32)  # Read full data as float32
        total_values = len(data)

        if total_values % (dim + 1) != 0:
            raise ValueError("File appears to be corrupted or incorrectly formatted.")

        num_embeddings = total_values // (dim + 1)

        embeddings = data.reshape(num_embeddings, dim + 1)[:, 1:]  # Remove first column (dim)

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
    parser.add_argument("--plot", action="store_true",
                        help="Plot a histogram of vector norms using Plotly")
    args = parser.parse_args()

    embeddings, num_embeddings, dim = read_fvecs(args.filename)

    print(f"✅ Successfully read {num_embeddings} embeddings")
    print(f"🔹 Each embedding has {dim} dimensions")
    print(f"🔍 First embedding: {embeddings[0]}")  # Print first embedding

    # Report normalization status.
    if check_normalization(embeddings):
        print("✅ Embeddings are normalized (L2 norm ≈ 1).")
    else:
        print("❌ Embeddings are not normalized (L2 norm not ≈ 1).")

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