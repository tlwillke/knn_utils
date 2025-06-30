#!/usr/bin/env python3

"""
fvecs_split.py: Parallel, block-buffered splitter for .fvecs files.

Reads a large .fvecs, samples query vectors, and splits into query/base files
in parallel without reinterpreting float bytes.
"""
import os
import argparse
import random
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import shutil

MIN_PARTITION_SIZE     = 1_000_000     # minimum vectors per partition
IO_BUFFER_SIZE_BYTES   = 16 * 1024**2  # 16 MB per read/write block

def read_dim_and_count(path):
    """
    Reads the vector dimension from the first 4 bytes of a .fvecs file
    and computes the total number of records based on file size.

    @param path: filesystem path to the input .fvecs file
    @return: tuple (dim, total_records)
    @raises ValueError: if file is too short or its size is not a multiple of record size
    """
    with open(path, "rb") as f:
        hdr = f.read(4)
        if len(hdr) < 4:
            raise ValueError("Empty or corrupt .fvecs file")
        dim = int.from_bytes(hdr, byteorder="little", signed=True)

    record_size = 4 + 4 * dim
    total_bytes = os.path.getsize(path)
    if total_bytes % record_size != 0:
        raise ValueError("File size not a multiple of record size")
    return dim, total_bytes // record_size

def process_chunk(worker_id, start_idx, end_idx,
                  query_idxs, input_path, record_size,
                  q_part_dir, b_part_dir):
    """
    Reads a contiguous range of raw .fvecs records in large blocks,
    splits each record into query or base output buffers based on the
    sampled indices, and writes them out.

    @param worker_id: integer ID of this worker (for logging)
    @param start_idx: index of the first vector to process
    @param end_idx:   index one past the last vector to process
    @param query_idxs: set of integer indices chosen for the query file
    @param input_path: path to the source .fvecs file
    @param record_size: byte size of one record (4 + 4*dim)
    @param q_part_dir: directory in which to write query_part{worker_id}.fvecs
    @param b_part_dir: directory in which to write base_part{worker_id}.fvecs
    @return: the worker_id on successful completion
    """
    q_out = os.path.join(q_part_dir, f"query_part{worker_id}.fvecs")
    b_out = os.path.join(b_part_dir, f"base_part{worker_id}.fvecs")

    # how many records fit in one I/O block
    recs_per_block = max(1, IO_BUFFER_SIZE_BYTES // record_size)

    with open(input_path, "rb") as fin, \
         open(q_out,     "wb") as fq, \
         open(b_out,     "wb") as fb:

        fin.seek(start_idx * record_size)
        total_recs = end_idx - start_idx
        remaining = total_recs
        curr_idx = start_idx
        processed = 0

        while remaining > 0:
            # read a block of raw bytes
            this_block_recs = min(remaining, recs_per_block)
            block_bytes = this_block_recs * record_size
            data = fin.read(block_bytes)
            if not data:
                break

            # split into two in-memory buffers
            qbuf = bytearray()
            bbuf = bytearray()
            for j in range(this_block_recs):
                offset = j * record_size
                rec = data[offset : offset + record_size]
                if curr_idx in query_idxs:
                    qbuf.extend(rec)
                else:
                    bbuf.extend(rec)
                curr_idx  += 1
                processed += 1

            # one big write per buffer
            if qbuf:
                fq.write(qbuf)
            if bbuf:
                fb.write(bbuf)

            # progress every ~10% of this chunk
            if processed % max(1, total_recs // 10) == 0:
                pct = processed * 100 // total_recs
                print(f"    [worker {worker_id}] {pct}% done")

            remaining -= this_block_recs

    return worker_id

def concat_parts(parts, out_path, label):
    """
    Concatenates a list of intermediate part-files into a single output file,
    removing each part as it’s appended.

    @param parts:   ordered list of filesystem paths to part .fvecs files
    @param out_path: path for the merged output .fvecs
    @param label:   descriptive label ('query' or 'base') for logging
    """
    print(f"→ Concatenating {len(parts)} {label} parts into {out_path}")
    with open(out_path, "wb") as fout:
        for i, part in enumerate(parts, 1):
            with open(part, "rb") as fin:
                shutil.copyfileobj(fin, fout)
            os.remove(part)
            print(f"    {label.capitalize()} concat: {i}/{len(parts)}")
    print(f"✔ Finished {label} concatenation\n")

def main():
    """
    Orchestrates the parallel split of a single .fvecs file into
    query and base outputs:

      1. Parses command-line arguments.
      2. Reads the dimension header and total record count.
      3. Reservoir-samples k query indices at random.
      4. Partitions the full record range into P worker chunks
         (respecting MIN_PARTITION_SIZE and CPU count).
      5. Uses a ThreadPoolExecutor to process each chunk in parallel
         with block-buffered raw-byte I/O.
      6. Concatenates the per-worker query and base parts.
      7. Optionally truncates the base output if --num_base was set.

    Usage: splitter.py <input.fvecs> [--num_query N] [--num_base M]
    """
    p = argparse.ArgumentParser(
        description=(
            "Parallel, block-buffered raw-byte split of an .fvecs file\n"
            f"(min partition size={MIN_PARTITION_SIZE}, I/O buffer={IO_BUFFER_SIZE_BYTES//(1024**2)} MB)"
        )
    )
    p.add_argument("input", help="Input .fvecs file")
    p.add_argument("--num_query", type=int, default=10_000,
                   help="How many query vectors to sample")
    p.add_argument("--num_base", type=int, default=None,
                   help="Max number of base vectors (default: all remaining)")
    args = p.parse_args()

    # Phase 1: header + count
    dim, total_records = read_dim_and_count(args.input)
    record_size = 4 + 4 * dim

    if args.num_query > total_records:
        raise ValueError(f"num_query={args.num_query} > {total_records} available")

    # Phase 2: sampling
    print(f"→ Sampling {args.num_query} queries out of {total_records} vectors...")
    query_idxs = set(random.sample(range(total_records), args.num_query))
    print(f"✔ Sampled {len(query_idxs)} unique query indices\n")

    remaining = total_records - args.num_query
    base_limit = remaining if args.num_base is None else min(remaining, args.num_base)

    base_name, ext = os.path.splitext(args.input)
    q_final = f"{base_name}_query{ext}"
    b_final = f"{base_name}_base{ext}"
    q_part_dir = f"{base_name}_qparts"
    b_part_dir = f"{base_name}_bparts"
    os.makedirs(q_part_dir, exist_ok=True)
    os.makedirs(b_part_dir, exist_ok=True)

    # Phase 3: partition decision
    min_chunk = MIN_PARTITION_SIZE
    max_parts = total_records // min_chunk or 1
    cpu_cores = os.cpu_count() or 1
    P = min(cpu_cores, max_parts)
    print(f"→ Splitting into {P} partition(s) (min chunk = {min_chunk} vectors)...")
    chunk = math.ceil(total_records / P)
    work = [(wid, wid*chunk, min((wid+1)*chunk, total_records))
            for wid in range(P) if wid*chunk < total_records]

    # Phase 4: parallel block‐buffered split
    print(f"→ Launching {len(work)} worker threads with {IO_BUFFER_SIZE_BYTES//(1024**2)} MB buffers...")
    with ThreadPoolExecutor(max_workers=len(work)) as exe:
        futures = {
            exe.submit(process_chunk, wid, start, end,
                       query_idxs, args.input, record_size,
                       q_part_dir, b_part_dir): wid
            for wid, start, end in work
        }
        for completed, fut in enumerate(as_completed(futures), 1):
            wid = futures[fut]
            fut.result()  # raise if error
            print(f"  ✔ Worker {wid} finished ({completed}/{len(work)})")
    print("✔ All workers done\n")

    # Phase 5: concat query parts
    q_parts = sorted(glob.glob(f"{q_part_dir}/query_part*.fvecs"),
                     key=lambda fn: int(fn.split("query_part")[1].split(".")[0]))
    concat_parts(q_parts, q_final, "query")

    # Phase 6: concat base parts
    b_parts = sorted(glob.glob(f"{b_part_dir}/base_part*.fvecs"),
                     key=lambda fn: int(fn.split("base_part")[1].split(".")[0]))
    concat_parts(b_parts, b_final, "base")

    # Phase 7: optional truncation of base
    if args.num_base is not None:
        keep = min(remaining, args.num_base)
        with open(b_final, "r+b") as fb:
            fb.truncate(keep * record_size)
        print(f"✔ Truncated base to {keep}/{remaining} vectors\n")

    print(f"All done!\n • Queries → {q_final}\n • Base    → {b_final}")

if __name__ == "__main__":
    main()
