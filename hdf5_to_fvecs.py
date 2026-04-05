#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import h5py


def read_hdf5_dataset(hdf5_path: Path, key: str) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as hf:
        if key not in hf:
            raise ValueError(f"Key '{key}' not found in HDF5 file: {hdf5_path}")

        dset = hf[key]

        if not hasattr(dset, "shape") or not hasattr(dset, "dtype"):
            raise ValueError(f"Key '{key}' is not a dataset: {hdf5_path}")

        if len(dset.shape) != 2:
            raise ValueError(
                f"Key '{key}' must be a 2D dataset, got shape {dset.shape} in {hdf5_path}"
            )

        if not np.issubdtype(dset.dtype, np.number):
            raise ValueError(
                f"Key '{key}' must be numeric, got dtype {dset.dtype} in {hdf5_path}"
            )

        return np.asarray(dset, dtype=np.float32)


def write_fvecs(output_path: Path, vectors: np.ndarray) -> None:
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {vectors.shape}")

    count, dim = vectors.shape
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    with open(output_path, "wb") as f:
        dim_prefix = np.array([dim], dtype=np.int32).tobytes()
        for i in range(count):
            f.write(dim_prefix)
            f.write(vectors[i].tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert query and base datasets from an HDF5 file into separate .fvecs files."
        )
    )
    parser.add_argument("hdf5_file", help="Path to the input .hdf5/.h5 file")
    parser.add_argument(
        "--query-key",
        default="test",
        help="HDF5 key for query vectors. Default: test",
    )
    parser.add_argument(
        "--base-key",
        default="train",
        help="HDF5 key for base vectors. Default: train",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output files. Default: same directory as input file",
    )
    args = parser.parse_args()

    hdf5_path = Path(args.hdf5_file).expanduser().resolve()
    if not hdf5_path.exists():
        raise FileNotFoundError(f"Input file not found: {hdf5_path}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else hdf5_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = hdf5_path.stem

    query_vectors = read_hdf5_dataset(hdf5_path, args.query_key)
    base_vectors = read_hdf5_dataset(hdf5_path, args.base_key)

    query_output = output_dir / f"{prefix}_query_{query_vectors.shape[0]}.fvecs"
    base_output = output_dir / f"{prefix}_base_{base_vectors.shape[0]}.fvecs"

    write_fvecs(query_output, query_vectors)
    write_fvecs(base_output, base_vectors)

    print(f"Input file:   {hdf5_path}")
    print(f"Query key:    {args.query_key}")
    print(f"Base key:     {args.base_key}")
    print(f"Query shape:  {query_vectors.shape}")
    print(f"Base shape:   {base_vectors.shape}")
    print(f"Wrote:        {query_output}")
    print(f"Wrote:        {base_output}")


if __name__ == "__main__":
    main()
    