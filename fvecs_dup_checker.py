#!/usr/bin/env python3
"""
fvecs_dup_checker.py

Split a large .fvecs file into sorted runs using a reader thread,
a main‐thread sorter, and a writer thread to overlap I/O and sorting.
Then k-way merge the runs into a single sorted file, reporting any
vector that appears more than N times.
"""

import os
import sys
import argparse
import struct
import threading
import queue
import heapq
import tempfile
import shutil

def reader_thread(fname, raw_q, chunk_size, endian_prefix):
    """Read chunks of up to chunk_size vectors, push lists of tuples to raw_q."""
    fmt_int = endian_prefix + 'i'
    with open(fname, 'rb') as f:
        run = 0
        while True:
            chunk = []
            for _ in range(chunk_size):
                d_bytes = f.read(4)
                if not d_bytes:
                    break
                d = struct.unpack(fmt_int, d_bytes)[0]
                vec_bytes = f.read(4 * d)
                if len(vec_bytes) < 4*d:
                    raise IOError(f"Incomplete record at chunk {run}")
                vec = struct.unpack(endian_prefix + f'{d}f', vec_bytes)
                chunk.append(vec)
            if not chunk:
                break
            print(f"[Reader] Read chunk {run} ({len(chunk)} vectors)")
            raw_q.put(chunk)
            run += 1
        raw_q.put(None)
    print("[Reader] Done")

def writer_thread(temp_dir, sorted_q, endian_prefix, dim):
    """Take sorted (run_id, chunk) from sorted_q and write each run out."""
    fmt_int = endian_prefix + 'i'
    run_files = []
    while True:
        item = sorted_q.get()
        if item is None:
            break
        run_id, sorted_chunk = item
        run_path = os.path.join(temp_dir, f"run_{run_id:04d}.fvecs")
        print(f"[Writer] Writing run {run_id:04d} ({len(sorted_chunk)} vectors) → {run_path}")
        with open(run_path, 'wb') as out:
            for vec in sorted_chunk:
                out.write(struct.pack(fmt_int, dim))
                out.write(struct.pack(endian_prefix + f'{dim}f', *vec))
        run_files.append(run_path)
        sorted_q.task_done()
    sorted_q.task_done()
    print("[Writer] Done")

def merge_runs(temp_dir, run_count, threshold, output_path, endian_prefix, report_path=None):
    """K-way merge all the run files and then write sorted output and report duplicates."""
    report_f = open(report_path, 'w') if report_path else None
    fmt_int = endian_prefix + 'i'
    run_paths = [os.path.join(temp_dir, f"run_{i:04d}.fvecs") for i in range(run_count)]
    readers = [open(p, 'rb') for p in run_paths]

    # Initialize heap
    heap = []
    for i, f in enumerate(readers):
        d_bytes = f.read(4)
        if not d_bytes:
            continue
        d = struct.unpack(fmt_int, d_bytes)[0]
        vec_bytes = f.read(4 * d)
        vec = struct.unpack(endian_prefix + f'{d}f', vec_bytes)
        heapq.heappush(heap, (vec, i))
    print(f"[Merge] Initialized heap with {len(heap)} runs")

    last_vec = None
    dup_count = 0
    total = 0

    with open(output_path, 'wb') as out:
        print(f"[Merge] Writing merged output to {output_path}")
        while heap:
            vec, rid = heapq.heappop(heap)
            if vec == last_vec:
                dup_count += 1
            else:
                # report if the last vector was too frequent
                if last_vec is not None and dup_count > threshold:
                    msg = f"Vector {last_vec} appears {dup_count} times"
                    print(f"[Dup] {msg}")
                    if report_f:
                        report_f.write(msg + "\n")
                last_vec = vec
                dup_count = 1
                # write the new vector
                out.write(struct.pack(fmt_int, len(vec)))
                out.write(struct.pack(endian_prefix + f'{len(vec)}f', *vec))

            # refill from run rid
            d_bytes = readers[rid].read(4)
            if d_bytes:
                d = struct.unpack(fmt_int, d_bytes)[0]
                vec_bytes = readers[rid].read(4 * d)
                next_vec = struct.unpack(endian_prefix + f'{d}f', vec_bytes)
                heapq.heappush(heap, (next_vec, rid))

            total += 1
            if total % 100_000 == 0:
                print(f"[Merge] Processed {total:,} vectors")

        # final duplicate check
        if last_vec is not None and dup_count > threshold:
            print(f"[Dup] Vector {last_vec} appears {dup_count} times")

    for f in readers:
        f.close()
    print(f"[Merge] Completed, total vectors merged: {total}")

    if report_f:
        report_f.close()
        print(f"[Merge] Duplicate report written to {report_path}")

def main():
    p = argparse.ArgumentParser(
        description="External mergesort for .fvecs with I/O overlap and duplicate reporting."
    )
    p.add_argument("input", help="Input .fvecs file")
    p.add_argument("-n", "--threshold", type=int, default=1,
                   help="Report vectors appearing more than N times (default 1)")
    p.add_argument("-c", "--chunk_size", type=int, default=200_000,
                   help="Number of vectors per in-memory chunk (default 200k)")
    p.add_argument("-e", "--endian", choices=['little','big'], default='little',
                   help="File endianness (default little)")
    p.add_argument("-t", "--temp_dir", default=None,
                   help="Directory for run files (default: auto temp)")
    p.add_argument("-o", "--output", default=None,
                   help="Final merged output filename (default sorted_<input>)")
    p.add_argument("-r", "--report_file", default=None,
                   help="Path to write duplicate report (one line per vector)")
    args = p.parse_args()

    # derive temp_dir & output
    temp_dir = args.temp_dir or tempfile.mkdtemp(prefix="fvecs_runs_")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
        print(f"[Main] Created temporary directory {temp_dir}")
    else:
        print(f"[Main] temp_dir = {temp_dir}")
    output = args.output or f"sorted_{os.path.basename(args.input)}"
    print(f"[Main] output = {output}")

    # queue sizes tuned for 1-deep pipeline
    raw_q    = queue.Queue(maxsize=2)
    sorted_q = queue.Queue(maxsize=2)

    endian_prefix = '<' if args.endian=='little' else '>'

    # 1) start reader
    rt = threading.Thread(
        target=reader_thread,
        args=(args.input, raw_q, args.chunk_size, endian_prefix),
        daemon=True)
    rt.start()

    # 2) start writer
    #    we need to know dimension; read it from the first chunk in-line
    first_chunk = raw_q.get()
    if first_chunk is None:
        print("⚠️  Input file empty. Exiting.")
        sys.exit(1)
    dim = len(first_chunk[0])
    # put it back for main sorter
    raw_q.put(first_chunk)

    wt = threading.Thread(
        target=writer_thread,
        args=(temp_dir, sorted_q, endian_prefix, dim),
        daemon=True)
    wt.start()

    # 3) main thread: sort each chunk and hand off
    run_id = 0
    while True:
        chunk = raw_q.get()
        if chunk is None:
            break
        print(f"[Main] Sorting chunk {run_id:04d} ({len(chunk)} vectors)…")
        chunk.sort()  # lexicographic sort of tuples
        sorted_q.put((run_id, chunk))
        run_id += 1

    # signal writer we're done
    sorted_q.put(None)

    # wait for writer to finish
    wt.join()

    # 4) merge runs and report duplicates
    print("[Main] Merging runs and preparing duplicate report…")
    merge_runs(temp_dir, run_id, args.threshold, output, endian_prefix, args.report_file)

    # 5) cleanup
    print(f"[Main] Removing temporary run files at {temp_dir}")
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
