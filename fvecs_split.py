import os
import numpy as np
import argparse

"""
This program splits an fvec file of vector embeddings into a base and query set
of a specified size.  The two sets are mutually exclusive.  The queries are drawn at random.

Example usage:
python split_fvec.py vectors.fvec --normalize --shuffle --num_query 5000 --num_base 20000
"""

def read_fvecs(fname):
    """
    Reads an fvec file and returns a numpy array of shape (n, d).
    Each vector is stored as: [d (int32), float, float, ..., float].
    """
    with open(fname, "rb") as f:
        # Read the dimension (first int32) from the file.
        header = np.fromfile(f, dtype=np.int32, count=1)
        if header.size == 0:
            raise ValueError("Empty file or invalid format.")
        d = header[0].item()
        # Go back to start and read all data as float32.
        f.seek(0)
        data = np.fromfile(f, dtype=np.float32)
    num_vectors = len(data) // (d + 1)
    # Reshape to (num_vectors, d+1) and drop the first column (the dimension).
    return data.reshape(num_vectors, d + 1)[:, 1:]

def write_fvecs(fname, arr):
    """
    Writes a numpy array of shape (n, d) to an fvec file.
    Each vector is written as: [d (int32), float, float, ..., float].
    """
    n, d = arr.shape
    with open(fname, "wb") as f:
        for vec in arr:
            np.array([d], dtype=np.int32).tofile(f)
            vec.astype(np.float32).tofile(f)

def normalize_vectors(arr):
    """Normalize each vector (L2 norm) and avoid division by zero."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return arr / norms

def main():
    parser = argparse.ArgumentParser(
        description="Split an fvec file into query and base vectors."
    )
    parser.add_argument("input", type=str, help="Input fvec file with vector embeddings")
    parser.add_argument("--num_query", type=int, default=10000,
                        help="Number of query vectors to select randomly (default: 10000)")
    parser.add_argument("--num_base", type=int, default=None,
                        help="Number of base vectors to store from the remaining vectors (default: all remaining)")
    parser.add_argument("--normalize", action="store_true",
                        help="Normalize vectors to unit length")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle vectors before splitting")
    args = parser.parse_args()

    # Read the vectors.
    vectors = read_fvecs(args.input)
    total_vectors, dim = vectors.shape
    print(f"Loaded {total_vectors} vectors of dimension {dim}, data type: {vectors.dtype}")

    # Optional normalization.
    if args.normalize:
        vectors = normalize_vectors(vectors)
        print("Vectors normalized.")

    # Optional shuffle.
    if args.shuffle:
        np.random.shuffle(vectors)
        print("Vectors shuffled.")

    # Ensure we don't request more query vectors than available.
    if args.num_query > total_vectors:
        raise ValueError(f"Requested num_query {args.num_query} exceeds available vectors {total_vectors}.")

    # Select query vectors.
    # If vectors were not shuffled already, randomly sample indices.
    if not args.shuffle:
        indices = np.random.choice(total_vectors, args.num_query, replace=False)
        query_vectors = vectors[indices]
        # Create the remaining base set.
        mask = np.ones(total_vectors, dtype=bool)
        mask[indices] = False
        base_vectors = vectors[mask]
    else:
        # If already shuffled, simply split the array.
        query_vectors = vectors[:args.num_query]
        base_vectors = vectors[args.num_query:]

    # Optionally truncate the base vectors if a limit is provided.
    if args.num_base is not None:
        if args.num_base > base_vectors.shape[0]:
            print(f"Warning: Requested num_base {args.num_base} exceeds available base vectors "
                  f"({base_vectors.shape[0]}); using all available.")
        else:
            base_vectors = base_vectors[:args.num_base]

    # Generate output filenames.
    base_name, ext = os.path.splitext(args.input)
    query_file = base_name + "_query" + ext
    base_file = base_name + "_base" + ext

    # Write the query and base vectors.
    write_fvecs(query_file, query_vectors)
    write_fvecs(base_file, base_vectors)

    # Report numbers.
    print(f"Stored {query_vectors.shape[0]} query vectors in '{query_file}'.")
    print(f"Stored {base_vectors.shape[0]} base vectors in '{base_file}'.")

if __name__ == '__main__':
    main()