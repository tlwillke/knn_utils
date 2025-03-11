import os
import numpy as np
import faiss
import argparse

def read_fvecs(fname):
    """
    Reads an fvec file and returns a numpy array of shape (n, d).
    The file is assumed to be in the format where each vector is stored
    as: [d, float, float, ..., float], with 1 integer dimension d and d floats.
    """
    with open(fname, "rb") as f:
        data = np .fromfile(f, dtype=np.int32)
        dim = data[0].item()
        f.seek(0)  # Reset file pointer to re-read data correctly
        data = np.fromfile(f, dtype=np.float32)  # Read full data as float32
        num_vectors = len(data) // (dim + 1)
        return data.reshape(num_vectors, dim + 1)[:, 1:]  # Remove first column (dimension)

def read_hdf5(fname, key="data"):
    """
    Reads an HDF5 file and returns a numpy array from the dataset with the given key.
    """
    import h5py
    with h5py.File(fname, 'r') as hf:
        if key not in hf:
            raise ValueError(f"Key '{key}' not found in HDF5 file: {fname}")
        dset = hf[key]
        return np.array(dset)

def read_vectors(fname):
    """
    Determines whether the file is HDF5 or fvec format.
    For HDF5 files (extension .h5 or .hdf5), it checks for a colon.
    If a colon is found, splits the string into filename and key.
    Otherwise, uses the default key "data".
    """
    fname = os.path.expanduser(fname)
    # If a colon is present, split into file_path and key.
    if ':' in fname:
        file_path, key = fname.split(':', 1)
        if file_path.endswith('.h5') or file_path.endswith('.hdf5'):
            return read_hdf5(file_path, key)
        else:
            raise ValueError("For HDF5, use the format 'file.h5:key'")
    else:
        return read_fvecs(fname)

def write_fvecs(fname, arr):
    """
    Write a numpy array (shape: n x d) to an fvec file.
    Each vector is stored as: [d (int32), float, float, ..., float]
    """
    n, d = arr.shape
    fname = os.path.expanduser(fname)  # Expand tilde to full home directory path
    with open(fname, "wb") as f:
        for i in range(n):
            # Write the dimension as int32
            np.array([d], dtype=np.int32).tofile(f)
            # Write the vector data as float32
            arr[i].astype(np.float32).tofile(f)

def write_ivecs(fname, ivecs):
    """
    Writes an array of integer vectors to an ivec file.
    Each vector is written as: [k, int, int, ..., int] where k is the number
    of elements in the vector.
    """
    n, k = ivecs.shape
    fname = os.path.expanduser(fname)  # Expand tilde to full home directory path
    with open(fname, 'wb') as f:
        for i in range(n):
            # Write the count (k) as an int32
            np.array([k], dtype=np.int32).tofile(f)
            # Write the vector of indices
            ivecs[i].astype(np.int32).tofile(f)

def check_normalization(vecs, tol=1e-3):
    """
    Returns True if all vectors in the array are approximately normalized
    (L2 norm close to 1 within the specified tolerance).
    """
    norms = np.linalg.norm(vecs, axis=1)
    return np.all(np.abs(norms - 1) < tol)

def build_index(base, d, metric, gpu_ids):
    """
    Build a FAISS index for the given base vectors, dimension and metric.
    gpu_ids should be a list of integers.
    """
    if metric == "l2":
        cpu_index = faiss.IndexFlatL2(d)
    elif metric == "ip":
        cpu_index = faiss.IndexFlatIP(d)
    else:
        raise ValueError("Unsupported metric: " + metric)

    if gpu_ids[0] < 0:
        print("Using device: cpu")
        index = cpu_index
    elif len(gpu_ids) == 1:
        print("Using device: cuda({})".format(gpu_ids[0]))
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, gpu_ids[0], cpu_index)
    else:
        print("Using devices:", ", ".join("cuda({})".format(g) for g in gpu_ids))
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.devices = gpu_ids
        index = faiss.index_cpu_to_all_gpus(cpu_index, co=co)
    index.add(base)
    return index

def main():
    parser = argparse.ArgumentParser(
        description='Compute ground truth for nearest neighbor search using a GPU.')
    parser.add_argument('--base', type=str, required=True,
                        help='Path to the base vectors file (fvec or HDF5). For HDF5, use the format "file.h5:key".')
    parser.add_argument('--query', type=str, required=True,
                        help='Path to the query vectors file (fvec or HDF5). For HDF5, use the format "file.h5:key".')
    parser.add_argument('--output', type=str, required=True,
                        help='Output ivec file to write ground truth indices.')
    parser.add_argument('--num_base', type=int, default=0,
                        help='Number of base vectors for truncated dataset (if 0, skip truncation).')
    parser.add_argument('--num_query', type=int, default=0,
                        help='Number of query vectors for truncated dataset (if 0, skip truncation).')
    parser.add_argument('--shuffle', action='store_true', default=False,
                        help='If set, shuffle both base and query vectors.')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='If set, normalize both base and query vectors.')
    parser.add_argument('--processed_base_out', type=str, default="",
                        help='Output file for processed base vectors (fvec file) if truncation or normalization is applied.')
    parser.add_argument('--processed_query_out', type=str, default="",
                        help='Output file for processed query vectors (fvec file) if truncation or normalization is applied.')
    parser.add_argument('--k', type=int, required=True,
                        help='Number of nearest neighbors to compute ground truth indices for.')
    parser.add_argument('--gpus', type=str, default="-1",
                        help='Comma-separated list of GPU ids to use. Use "-1" for CPU.')
    parser.add_argument('--metric', type=str, default='l2', choices=['l2', 'ip'],
                        help='Distance metric to use: "l2" or "ip".')
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpus.split(',')]

    # Load base and query vectors from files
    print("Loading base vectors from:", args.base)
    base = read_vectors(args.base)
    print(f"Loaded {base.shape[0]} base vectors of dimension {base.shape[1]}.")

    print("Loading query vectors from:", args.query)
    query = read_vectors(args.query)
    print(f"Loaded {query.shape[0]} query vectors of dimension {query.shape[1]}.")

    # Ensure dimensions match
    d = base.shape[1]
    if query.shape[1] != d:
        raise ValueError("Dimension mismatch: base vectors have dimension {} but query vectors have dimension {}."
                         .format(d, query.shape[1]))

    # Check normalization of base and query vectors.
    base_normalized = check_normalization(base)
    query_normalized = check_normalization(query)
    print("Base vectors normalized:", "Yes" if base_normalized else "No")
    print("Query vectors normalized:", "Yes" if query_normalized else "No")

    # Optionally shuffle both base and query vectors.
    # Shuffle the full dataset before truncation.
    if args.shuffle:
        print("Shuffling both base and query vectors.")
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(base)
        np.random.shuffle(query)

    # Process datasets if truncation or normalization is requested.
    if args.num_base > 0 or args.num_query > 0 or args.normalize:
        if args.num_base > 0:
            if args.num_base > base.shape[0]:
                raise ValueError("Truncated base size exceeds full dataset size.")
            base = base[:args.num_base]
            print(f"Using truncated base: {args.num_base} vectors.")
        if args.num_query > 0:
            if args.num_query > query.shape[0]:
                raise ValueError("Truncated query size exceeds full dataset size.")
            query = query[:args.num_query]
            print(f"Using truncated query: {args.num_query} vectors.")

        # Apply normalization if requested (to both base and query).
        if args.normalize:
            def normalize_vectors(arr):
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1  # Prevent division by zero.
                return arr / norms

            base = normalize_vectors(base)
            query = normalize_vectors(query)
            print("Normalized both base and query vectors.")

        # Require processed output filenames when processing is applied.
        if args.normalize or args.shuffle or args.num_base > 0 and args.num_query > 0:
            if not args.processed_base_out or not args.processed_query_out:
                raise ValueError(
                    "When normalization, shuffling, or truncation is applied, processed_base_out and processed_query_out must be provided. ")
            print("Writing processed base vectors to:", args.processed_base_out)
            write_fvecs(args.processed_base_out, base)
            print("Writing processed query vectors to:", args.processed_query_out)
            write_fvecs(args.processed_query_out, query)
        elif args.num_base > 0:
            if not args.processed_base_out:
                raise ValueError(
                    "When truncation is applied, processed_base_out must be provided.")
            print("Writing processed base vectors to:", args.processed_base_out)
            write_fvecs(args.processed_base_out, base)
        elif args.num_query > 0:
            if not args.processed_query_out:
                raise ValueError(
                    "When truncation is applied, processed_query_out must be provided.")
            print("Writing processed query vectors to:", args.processed_query_out)
            write_fvecs(args.processed_query_out, query)

    # Creating index and adding base vectors.
    print("Adding base vectors to the index...")
    index = build_index(base, d, args.metric, gpu_ids)

    # Perform the search for each query.
    print("Performing nearest neighbor search for k =", args.k)
    distances, indices = index.search(query, args.k)
    print("Search completed.")

    # Write the ground truth indices to the output ivec file.
    print("Writing results to output file:", args.output)
    write_ivecs(args.output, indices)
    print("Done.")

if __name__ == '__main__':
    main()