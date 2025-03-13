import numpy as np
import argparse

def read_fvecs(fname):
    """Reads an .fvecs file and returns the embeddings as a NumPy array."""
    with open(fname, "rb") as f:
        data = np.fromfile(f, dtype=np.int32)
        dim = data[0].item()
        f.seek(0)  # Reset file pointer to re-read data correctly
        total_values = len(data)

        if total_values % (dim + 1) != 0:
            raise ValueError("File appears to be corrupted or incorrectly formatted.")

        num_embeddings = total_values // (dim + 1)

        embeddings = data.reshape(num_embeddings, dim + 1)[:, 1:]  # Remove first column (dim)

        return embeddings, num_embeddings, dim

def main():
    """Main function to check and validate an .fvecs file."""
    parser = argparse.ArgumentParser(description="Check and validate an .fvecs file.")
    parser.add_argument("filename", help="Path to the .fvecs file to check")
    args = parser.parse_args()

    embeddings, num_embeddings, dim = read_fvecs(args.filename)

    print(f"✅ Successfully read {num_embeddings} embeddings")
    print(f"🔹 Each embedding has {dim} dimensions")
    print(f"🔍 First embedding: {embeddings[0]}")  # Print first embedding

if __name__ == "__main__":
    main()