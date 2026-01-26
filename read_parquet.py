#!/usr/bin/env python3
import sys
import pyarrow.parquet as pq

def read_header_and_first_row(parquet_path):
    # Open the file without loading all the data
    pqfile = pq.ParquetFile(parquet_path)
    
    # Schema / headers
    schema = pqfile.schema_arrow
    print("Columns:", schema.names)
    
    # Read exactly one record (batch_size=1)
    batch = next(pqfile.iter_batches(batch_size=1))
    row = batch.to_pydict()  # dict of column -> list
    # Extract the single values
    first_row = {col: vals[0] for col, vals in row.items()}
    print("First row:")
    for col, val in first_row.items():
        print(f"  {col}: {val!r}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} path/to/data.parquet")
        sys.exit(1)
    read_header_and_first_row(sys.argv[1])

