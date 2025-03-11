# knn_utils

knn_utils is a simple set of utilities that are useful for preparing datasets for Approximate Nearest Neighbor search.

## How knn_utils works

knn_utils started as a KNN ground truth generator.  It is built around the [faiss-gpu](https://anaconda.org/pytorch/faiss-gpu) implementation.  This provides fast and scalable  multi-GPU brute force search.  On a pair of Nvidia L40 GPUs it can compute the 100 nearest neighbors for 10M 768 dimensional base vectors and 10k queries in approximately 1 minute.

## Capabilities

- Multi-GPU brute-force KNN search and ground truth generation
- Read .fvec or .hdf5 vector files
- Write ground truth to .ivec
- Supports maximum inner product and L2 metrics
- Vector normalization
- Dataset shuffling

## Usage

```text
usage: knn_utils.py [-h] --base BASE --query QUERY --output OUTPUT 
                    [--num_base NUM_BASE] [--num_query NUM_QUERY] [--shuffle] [--normalize] 
                    [--processed_base_out PROCESSED_BASE_OUT] [--processed_query_out PROCESSED_QUERY_OUT] 
                    --k K [--gpus GPUS] [--metric {l2,ip}]
```


Compute ground truth for nearest neighbor search using a GPU.

options:
```text
  -h, --help            Show this help message and exit.
  --base BASE           Path to the base vectors fvec or hdf5 file.
  --query QUERY         Path to the query vectors fvec or hdf5 file.
  --output OUTPUT       Output ivec file to write ground truth indices.
  --num_base NUM_BASE   Number of base vectors for truncated dataset (if 0, skip truncation).
  --num_query NUM_QUERY
                        Number of query vectors for truncated dataset (if 0, skip truncation).
  --shuffle             If set, shuffle all base and query vectors prior to truncation.
  --normalize           If set, normalize both base and query vectors.
  --processed_base_out PROCESSED_BASE_OUT
                        Output file for processed base vectors (fvec file) if truncation or normalization is
                        applied.
  --processed_query_out PROCESSED_QUERY_OUT
                        Output file for processed query vectors (fvec file) if truncation or normalization is
                        applied.
  --k K                 Number of nearest neighbors to compute ground truth indices for.
  --gpus GPUS           Comma-separated list of GPU ids to use. Use "-1" for CPU.
  --metric {l2,ip}      Distance metric to use: "l2" or "ip".
```
## Requirements

See environment.yml.
