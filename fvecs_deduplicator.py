#!/usr/bin/env python3
"""
fvecs_deduplicator.py

Split a large .fvecs file into sorted runs using a reader thread,
a main‐thread sorter, and a writer thread to overlap I/O and sorting.
Then k-way merge the runs into a single sorted file, reporting any
vector that appears more than the reporting_threshold.  The duplicate
report now records the input-file line number of the first occurrence,
the first 4 floats of that vector, and shows a Top-100 table
(including duplicate count) of the most frequent duplicates.

If --presorted is set, skips the chunk/sort pipeline and instead
streams the sorted .fvecs input directly for deduplication.
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
from collections import Counter

def reader_thread(fname, raw_q, chunk_size, endian_prefix):
    fmt_int = endian_prefix + 'i'
    line_num = 1
    run = 0
    with open(fname, 'rb') as f:
        while True:
            chunk = []
            for _ in range(chunk_size):
                d_bytes = f.read(4)
                if not d_bytes:
                    break
                d = struct.unpack(fmt_int, d_bytes)[0]
                vec_bytes = f.read(4 * d)
                if len(vec_bytes) < 4 * d:
                    raise IOError(f"Incomplete record at chunk {run}")
                vec = struct.unpack(endian_prefix + f'{d}f', vec_bytes)
                chunk.append((vec, line_num))
                line_num += 1
            if not chunk:
                break
            print(f"[Reader] Read chunk {run} ({len(chunk)} vectors)")
            raw_q.put(chunk)
            run += 1
        raw_q.put(None)
    print("[Reader] Done")

def writer_thread(temp_dir, sorted_q, endian_prefix, dim):
    fmt_int  = endian_prefix + 'i'
    line_fmt = endian_prefix + 'q'
    while True:
        item = sorted_q.get()
        if item is None:
            break
        run_id, sorted_chunk = item
        run_path = os.path.join(temp_dir, f"run_{run_id:04d}.fvecs")
        print(f"[Writer] Writing run {run_id:04d} ({len(sorted_chunk)} vectors) → {run_path}")
        with open(run_path, 'wb') as out:
            for vec, line_no in sorted_chunk:
                out.write(struct.pack(fmt_int, dim))
                out.write(struct.pack(endian_prefix + f'{dim}f', *vec))
                out.write(struct.pack(line_fmt, line_no))
        sorted_q.task_done()
    sorted_q.task_done()
    print("[Writer] Done")

def merge_runs(temp_dir, run_count, reporting_threshold, output_path, endian_prefix, report_path=None):
    report_f = open(report_path, 'w') if report_path else None
    fmt_int  = endian_prefix + 'i'
    line_fmt = endian_prefix + 'q'

    run_paths = [os.path.join(temp_dir, f"run_{i:04d}.fvecs") for i in range(run_count)]
    readers   = [open(p, 'rb') for p in run_paths]

    # track totals for summary
    total_processed = 0
    total_written = 0

    heap = []
    for i, f in enumerate(readers):
        d_bytes = f.read(4)
        if not d_bytes:
            continue
        d = struct.unpack(fmt_int, d_bytes)[0]
        vec_bytes = f.read(4 * d)
        vec = struct.unpack(endian_prefix + f'{d}f', vec_bytes)
        line_no = struct.unpack(line_fmt, f.read(8))[0]
        heapq.heappush(heap, (vec, line_no, i))
    print(f"[Merge] Initialized heap with {len(heap)} runs")

    dup_hist = Counter()
    dup_recs = []

    last_vec     = None
    last_line    = None
    dup_count    = 0
    total        = 0
    run_line_nos = []

    with open(output_path, 'wb') as out:
        print(f"[Merge] Writing merged output to {output_path}")
        while heap:
            vec, line_no, rid = heapq.heappop(heap)
            total_processed += 1

            if last_vec is not None and vec == last_vec:
                dup_count += 1
                run_line_nos.append(line_no)
            else:
                if last_vec is not None and dup_count > reporting_threshold:
                    msg = f"Line {last_line}: {last_vec[:4]} appears {dup_count} times"
                    print(f"[Dup] {msg}")
                    if report_f:
                        report_f.write(msg + "\n")
                    dup_hist[dup_count] += 1
                    dup_recs.append((dup_count, last_line, last_vec[:4], run_line_nos[1:11]))
                last_vec     = vec
                last_line    = line_no
                dup_count    = 1
                run_line_nos = [line_no]
                total_written += 1
                out.write(struct.pack(fmt_int, len(vec)))
                out.write(struct.pack(endian_prefix + f'{len(vec)}f', *vec))

            d_bytes = readers[rid].read(4)
            if d_bytes:
                d = struct.unpack(fmt_int, d_bytes)[0]
                vec_bytes = readers[rid].read(4 * d)
                next_vec = struct.unpack(endian_prefix + f'{d}f', vec_bytes)
                next_line = struct.unpack(line_fmt, readers[rid].read(8))[0]
                heapq.heappush(heap, (next_vec, next_line, rid))

            total += 1
            if total % 100_000 == 0:
                print(f"[Merge] Processed {total:,} vectors")

        if last_vec is not None and dup_count > reporting_threshold:
            msg = f"Line {last_line}: {last_vec[:4]} appears {dup_count} times"
            print(f"[Dup] {msg}")
            if report_f:
                report_f.write(msg + "\n")
            dup_hist[dup_count] += 1
            dup_recs.append((dup_count, last_line, last_vec[:4], run_line_nos[1:11]))

    for f in readers:
        f.close()

    removed = total_processed - total_written
    print(f"[Merge] Total unique written: {total_written}, removed: {removed}")

    if report_f:
        report_f.write(f"Total unique written: {total_written}, removed: {removed}\n")
        report_f.close()
        print(f"[Merge] Duplicate report written to {report_path}")

    print(f"[Merge] Completed, total vectors merged: {total}")

    if dup_hist:
        print("\nDuplicate-count histogram (count → distinct vectors):")
        for count, freq in sorted(dup_hist.items(), key=lambda x: x[0], reverse=True):
            print(f"{count:>5} → {freq}")

    top100 = sorted(dup_recs, key=lambda x: x[0], reverse=True)[:100]
    if top100:
        headers = ["Count", "1st Line", "Value", "Other Lines"]
        w0 = max(len(headers[0]), *(len(str(r[0])) for r in top100))
        w1 = max(len(headers[1]), *(len(str(r[1])) for r in top100))
        w2 = max(len(headers[2]), *(len(str(r[2])) for r in top100))
        w3 = max(len(headers[3]), *(len(str(r[3])) for r in top100))

        print("\nTop 100 duplications:")
        hdr = (
            f"{headers[0]:<{w0}}  "
            f"{headers[1]:<{w1}}  "
            f"{headers[2]:<{w2}}  "
            f"{headers[3]:<{w3}}"
        )
        print(hdr)
        print("-" * len(hdr))
        for cnt, first_ln, vec4, others in top100:
            print(
                f"{cnt:<{w0}}  "
                f"{first_ln:<{w1}}  "
                f"{str(vec4):<{w2}}  "
                f"{str(others):<{w3}}"
            )

def dedup_presorted(input_path, reporting_threshold, output_path, endian_prefix, report_path=None):
    report_f = open(report_path, 'w') if report_path else None
    fmt_int = endian_prefix + 'i'
    dup_hist = Counter()
    dup_recs = []

    last_vec = None
    last_line = None
    dup_count = 0
    other_lines = []
    total = 0
    line_num = 1

    total_processed = 0
    total_written = 0

    with open(input_path, 'rb') as fin, open(output_path, 'wb') as out:
        print(f"[Dedup] Streaming dedup on presorted input → {output_path}")
        while True:
            d_bytes = fin.read(4)
            if not d_bytes:
                break
            d = struct.unpack(fmt_int, d_bytes)[0]
            vec_bytes = fin.read(4 * d)
            if len(vec_bytes) < 4 * d:
                raise IOError(f"Incomplete record at line {line_num}")
            vec = struct.unpack(endian_prefix + f'{d}f', vec_bytes)
            total_processed += 1

            if last_vec is not None and vec == last_vec:
                dup_count += 1
                if len(other_lines) < 10:
                    other_lines.append(line_num)
            else:
                total_written += 1
                if last_vec is not None and dup_count > reporting_threshold:
                    msg = f"Line {last_line}: {last_vec[:4]} appears {dup_count} times"
                    print(f"[Dup] {msg}")
                    if report_f:
                        report_f.write(msg + "\n")
                    dup_hist[dup_count] += 1
                    dup_recs.append((dup_count, last_line, last_vec[:4], other_lines[:]))
                last_vec = vec
                last_line = line_num
                dup_count = 1
                other_lines = []
                out.write(struct.pack(fmt_int, d))
                out.write(struct.pack(endian_prefix + f'{d}f', *vec))

            total += 1
            line_num += 1
            if total % 100_000 == 0:
                print(f"[Dedup] Processed {total:,} vectors")

        if last_vec is not None and dup_count > reporting_threshold:
            msg = f"Line {last_line}: {last_vec[:4]} appears {dup_count} times"
            print(f"[Dup] {msg}")
            if report_f:
                report_f.write(msg + "\n")
            dup_hist[dup_count] += 1
            dup_recs.append((dup_count, last_line, last_vec[:4], other_lines[:]))

    removed = total_processed - total_written
    print(f"[Dedup] Total unique written: {total_written}, removed: {removed}")

    if report_f:
        report_f.write("\nDuplicate-count histogram (count → distinct vectors):\n")
        for count, freq in sorted(dup_hist.items(), key=lambda x: x[0], reverse=True):
            report_f.write(f"{count:>5} → {freq}\n")
        report_f.write(f"Total unique written: {total_written}, removed: {removed}\n")

        report_f.close()
    print(f"[Dedup] Completed, total vectors processed: {total}")

    if dup_hist:
        print("\nDuplicate-count histogram (count → distinct vectors):")
        for count, freq in sorted(dup_hist.items(), key=lambda x: x[0], reverse=True):
            print(f"{count:>5} → {freq}")

    top100 = sorted(dup_recs, key=lambda x: x[0], reverse=True)[:100]
    if top100:
        headers = ["Count", "1st Line", "Value", "Other Lines"]
        w0 = max(len(headers[0]), *(len(str(r[0])) for r in top100))
        w1 = max(len(headers[1]), *(len(str(r[1])) for r in top100))
        w2 = max(len(headers[2]), *(len(str(r[2])) for r in top100))
        w3 = max(len(headers[3]), *(len(str(r[3])) for r in top100))

        print("\nTop 100 duplications:")
        hdr = f"{headers[0]:<{w0}}  {headers[1]:<{w1}}  {headers[2]:<{w2}}  {headers[3]:<{w3}}"
        print(hdr)
        print("-" * len(hdr))
        for cnt, first_ln, vec4, others in top100:
            print(f"{cnt:<{w0}}  {first_ln:<{w1}}  {str(vec4):<{w2}}  {str(others):<{w3}}")

def main():
    p = argparse.ArgumentParser(
        description="External mergesort for .fvecs with I/O overlap and duplicate reporting."
    )
    p.add_argument("input", help="Input .fvecs file")
    p.add_argument("-n", "--reporting_threshold", type=int, default=1,
                   help="Report vectors appearing more than this many times (default 1)")
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
    p.add_argument("--presorted", action='store_true',
                   help="Skip chunking/sorting and dedupe presorted input file")
    args = p.parse_args()

    endian_prefix = '<' if args.endian == 'little' else '>'
    output = args.output or f"sorted_{os.path.basename(args.input)}"

    if args.presorted:
        dedup_presorted(
            args.input,
            args.reporting_threshold,
            output,
            endian_prefix,
            args.report_file
        )
        sys.exit(0)

    temp_dir = args.temp_dir or tempfile.mkdtemp(prefix="fvecs_runs_")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
        print(f"[Main] Created temporary directory {temp_dir}")
    else:
        print(f"[Main] temp_dir = {temp_dir}")

    print(f"[Main] output = {output}")

    raw_q    = queue.Queue(maxsize=2)
    sorted_q = queue.Queue(maxsize=2)

    rt = threading.Thread(
        target=reader_thread,
        args=(args.input, raw_q, args.chunk_size, endian_prefix),
        daemon=True)
    rt.start()

    first_chunk = raw_q.get()
    if first_chunk is None:
        print("⚠️  Input file empty. Exiting.")
        sys.exit(1)
    dim = len(first_chunk[0][0])
    raw_q.put(first_chunk)

    wt = threading.Thread(
        target=writer_thread,
        args=(temp_dir, sorted_q, endian_prefix, dim),
        daemon=True)
    wt.start()

    run_id = 0
    while True:
        chunk = raw_q.get()
        if chunk is None:
            break
        print(f"[Main] Sorting chunk {run_id:04d} ({len(chunk)} vectors)…")
        chunk.sort(key=lambda x: x[0])
        sorted_q.put((run_id, chunk))
        run_id += 1

    sorted_q.put(None)
    wt.join()

    print("[Main] Merging runs and preparing duplicate report…")
    merge_runs(
        temp_dir,
        run_id,
        args.reporting_threshold,
        output,
        endian_prefix,
        args.report_file
    )

    # shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()


