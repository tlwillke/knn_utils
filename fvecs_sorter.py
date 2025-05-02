#!/usr/bin/env python3
""" Uses external merge sort to sort larger-than-memory fvecs files.
It's a lexicographic numeric sort in ascending order.

Usage:
python fvecs_sorter.py
  bigdata.fvecs
  --chunk_size 120000
  --endian little
  --output sorted_bigdata.fvecs
"""

import os, sys, argparse, struct, threading, queue, heapq, tempfile, shutil

def reader_thread(fname, raw_q, chunk_size, endian_prefix):
    fmt_int = endian_prefix + 'i'
    with open(fname,'rb') as f:
        while True:
            chunk = []
            for _ in range(chunk_size):
                d_bytes = f.read(4)
                if not d_bytes: break
                d = struct.unpack(fmt_int, d_bytes)[0]
                vec_bytes = f.read(4*d)
                if len(vec_bytes) < 4*d:
                    raise IOError("Incomplete record")
                chunk.append(struct.unpack(endian_prefix+f'{d}f', vec_bytes))
            if not chunk: break
            raw_q.put(chunk)
        raw_q.put(None)

def writer_thread(temp_dir, sorted_q, endian_prefix, dim):
    fmt_int = endian_prefix + 'i'
    while True:
        item = sorted_q.get()
        if item is None: break
        run_id, chunk = item
        run_path = os.path.join(temp_dir, f"run_{run_id:04d}.fvecs")
        with open(run_path,'wb') as out:
            for vec in chunk:
                out.write(struct.pack(fmt_int, dim))
                out.write(struct.pack(endian_prefix+f'{dim}f', *vec))
        sorted_q.task_done()

def merge_runs(temp_dir, run_count, output_path, endian_prefix):
    fmt_int = endian_prefix + 'i'
    run_paths = [os.path.join(temp_dir,f"run_{i:04d}.fvecs")
                 for i in range(run_count)]
    readers = [open(p,'rb') for p in run_paths]
    heap = []
    # init heap
    for i,f in enumerate(readers):
        d_bytes = f.read(4)
        if not d_bytes: continue
        d = struct.unpack(fmt_int, d_bytes)[0]
        vec = struct.unpack(endian_prefix+f'{d}f', f.read(4*d))
        heapq.heappush(heap,(vec,i))
    with open(output_path,'wb') as out:
        while heap:
            vec, rid = heapq.heappop(heap)
            out.write(struct.pack(fmt_int, len(vec)))
            out.write(struct.pack(endian_prefix+f'{len(vec)}f', *vec))
            # refill
            d_bytes = readers[rid].read(4)
            if d_bytes:
                d = struct.unpack(fmt_int, d_bytes)[0]
                vec = struct.unpack(endian_prefix+f'{d}f', readers[rid].read(4*d))
                heapq.heappush(heap,(vec,rid))
    for f in readers: f.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Input .fvecs file")
    p.add_argument("-c", "--chunk_size", type=int, default=200_000,
                   help="Number of vectors per in-memory chunk")
    p.add_argument("-e", "--endian", choices=['little','big'], default='little',
                   help="File endianness")
    p.add_argument("-t", "--temp_dir", default=None,
                   help="Where to write intermediate runs")
    p.add_argument("-o", "--output", default=None,
                   help="Final sorted filename (overrides default)")
    args = p.parse_args()

    # --- new default-output logic: append "_sorted" before extension ---
    if args.output:
        output = args.output
    else:
        base, ext = os.path.splitext(os.path.basename(args.input))
        output = f"{base}_sorted{ext}"

    temp_dir = args.temp_dir or tempfile.mkdtemp(prefix="fvecs_runs_")
    print(f"[Main] temp_dir = {temp_dir}")
    print(f"[Main] output   = {output}")

    # Queues for the pipeline
    raw_q = queue.Queue(maxsize=2)
    sorted_q = queue.Queue(maxsize=2)

    # 1) Start reader thread
    rt = threading.Thread(
        target=reader_thread,
        args=(args.input, raw_q, args.chunk_size, '<' if args.endian == 'little' else '>'),
        daemon=True
    )
    rt.start()

    # 2) Peek at first chunk to learn dim, then put it back
    first_chunk = raw_q.get()
    if first_chunk is None:
        sys.exit("[Main] Empty input file—nothing to sort.")
    dim = len(first_chunk[0])
    raw_q.put(first_chunk)

    # 3) Start writer thread
    wt = threading.Thread(
        target=writer_thread,
        args=(temp_dir, sorted_q, '<' if args.endian == 'little' else '>', dim),
        daemon=True
    )
    wt.start()

    # 4) Main thread: pull, sort, and hand off each chunk
    run_id = 0
    while True:
        chunk = raw_q.get()
        if chunk is None:
            break
        print(f"[Main] Sorting chunk {run_id} ({len(chunk)} vectors)…")
        chunk.sort()
        sorted_q.put((run_id, chunk))
        run_id += 1

    # signal writer no more runs coming
    sorted_q.put(None)
    wt.join()

    # 5) Merge all runs into the final sorted file
    merge_runs(temp_dir, run_id, output, '<' if args.endian == 'little' else '>')

    # cleanup
    shutil.rmtree(temp_dir)
    print(f"[Main] Sorted file written to {output}")

if __name__ == "__main__":
    main()
