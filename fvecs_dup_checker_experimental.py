#!/usr/bin/env python3
"""
fvecs_dup_checker.py

Verifies that a sorted .fvecs file is truly sorted, then deduplicates it and reports duplicates.

Split a large .fvecs file into sorted runs using a reader thread,
a main‐thread sorter, and a writer thread to overlap I/O and sorting.
Then k-way merge the runs into a single sort, reporting any
vector that appears more than N times.

Verifies that a sorted .fvecs file is truly sorted, using a two-stage approach:
  1) Fast raw-bytes comparison
  2) Fallback numeric lexicographic comparison if raw check fails
Then deduplicates it and reports duplicates.

WARNING: Input file must be sorted. The script will abort at the first unsorted record.

Usage:
    ./fvecs_dup_checker.py /path/to/your/index_sorted.fvecs
"""

import os
import sys
import struct
import argparse
from collections import Counter

# Progress interval for sortedness checks
PROGRESS_INTERVAL = 1_000_000


def check_sorted_bytes(input_path: str, record_size: int) -> bool:
    """
    Fast raw-bytes pass: compares each record blob for non-decreasing order.
    Prints progress every PROGRESS_INTERVAL records.
    If the final chunk is smaller, prints the actual total checked.
    Returns True if sorted, False on first failure.
    """
    with open(input_path, 'rb') as f:
        prev = f.read(record_size)
        if len(prev) != record_size:
            return True  # empty or single-record is sorted
        line_num = 1
        while True:
            rec = f.read(record_size)
            if not rec:
                break
            line_num += 1
            if len(rec) != record_size:
                raise EOFError(f"Unexpected EOF during raw sortedness check at line {line_num}.")
            if rec < prev:
                print(f"ERROR [raw check]: record {line_num} < record {line_num-1}", file=sys.stderr)
                return False
            prev = rec
            if line_num % PROGRESS_INTERVAL == 0:
                print(f"Raw check: {line_num:,} records OK...", file=sys.stderr)
        # Final partial interval
        if line_num % PROGRESS_INTERVAL != 0:
            print(f"Raw check: checked {line_num:,} records (final).", file=sys.stderr)
    return True


def check_sorted_numeric(input_path: str, dim: int, record_size: int) -> bool:
    """
    Detailed numeric pass: unpacks floats and compares lexicographically.
    Prints progress every PROGRESS_INTERVAL records.
    If the final chunk is smaller, prints the actual total checked.
    Returns True if sorted, False on first failure (with debug floats).
    """
    fmt = '<' + 'f' * dim
    with open(input_path, 'rb') as f:
        first = f.read(record_size)
        if len(first) != record_size:
            return True
        prev_vals = struct.unpack(fmt, first[4:])
        line_num = 1
        while True:
            rec = f.read(record_size)
            if not rec:
                break
            line_num += 1
            if len(rec) != record_size:
                raise EOFError(f"Unexpected EOF during numeric sortedness check at line {line_num}.")
            vals = struct.unpack(fmt, rec[4:])
            if prev_vals > vals:
                print(f"ERROR [numeric check]: not sorted at line {line_num}", file=sys.stderr)
                print(f"  prev(line {line_num-1}) first4: {prev_vals[:4]}", file=sys.stderr)
                print(f"  rec (line {line_num})    first4: {vals[:4]}", file=sys.stderr)
                return False
            prev_vals = vals
            if line_num % PROGRESS_INTERVAL == 0:
                print(f"Numeric check: {line_num:,} records OK...", file=sys.stderr)
        # Final partial interval
        if line_num % PROGRESS_INTERVAL != 0:
            print(f"Numeric check: checked {line_num:,} records (final).", file=sys.stderr)
    return True


def report_duplicates(input_path: str, dim: int, record_size: int, report_path: str):
    """
    Single-pass duplicate detection on a sorted file.
    Builds a histogram of run-lengths >1 and collects top-100 runs.
    Writes a text report at report_path.
    """
    histogram = Counter()
    dup_infos = []  # (first_line, first4floats, count, first10lines)
    fmt = '<' + 'f' * dim

    with open(input_path, 'rb') as f:
        prev = f.read(record_size)
        if len(prev) != record_size:
            print("No vectors to process.")
            return
        prev_vals = struct.unpack(fmt, prev[4:])
        line_num = 1
        run_count = 1
        run_first = 1
        run_lines = [1]

        while True:
            rec = f.read(record_size)
            if not rec:
                break
            line_num += 1
            vals = struct.unpack(fmt, rec[4:])

            if vals == prev_vals:
                run_count += 1
                if len(run_lines) < 10:
                    run_lines.append(line_num)
            else:
                if run_count > 1:
                    histogram[run_count] += 1
                    dup_infos.append((run_first, prev_vals[:4], run_count, run_lines.copy()))
                run_count = 1
                run_first = line_num
                run_lines = [line_num]
                prev_vals = vals

        # handle final run
        if run_count > 1:
            histogram[run_count] += 1
            dup_infos.append((run_first, prev_vals[:4], run_count, run_lines.copy()))

    # If no duplicates found, report and exit
    if not dup_infos:
        print("No duplicates found.", file=sys.stdout)
        with open(report_path, 'w') as rpt:
            rpt.write('No duplicates found.\n')
        print(f"Report written to {report_path}")
        return

    # write report with histogram and top-100
    with open(report_path, 'w') as rpt:
        rpt.write('Histogram of duplicate counts → # distinct vectors\n')
        for count, freq in sorted(histogram.items()):
            rpt.write(f'{count:5d} → {freq:,}\n')

        rpt.write('\nTop 100 duplicate runs:\n')
        rpt.write('Line\tFirst4Floats\tCount\tFirst10Lines\n')
        rpt.write('----\t------------\t-----\t----------------\n')
        for first_line, floats4, cnt, lines in sorted(dup_infos, key=lambda x: x[2], reverse=True)[:100]:
            fp4 = ', '.join(f'{v:.6g}' for v in floats4)
            lines_str = ', '.join(str(ln) for ln in lines)
            rpt.write(f'{first_line:4d}\t[{fp4}]\t{cnt:5d}\t[{lines_str}]\n')

    print(f"Report written to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Verify sortedness (two-stage) and report duplicates in a .fvecs file.'
    )
    parser.add_argument('input_file', help='Path to sorted .fvecs file (read-only)')
    args = parser.parse_args()

    in_path = args.input_file
    if not os.path.isfile(in_path):
        print(f'ERROR: No such file {in_path!r}', file=sys.stderr)
        sys.exit(1)

    # read header
    with open(in_path, 'rb') as f:
        hdr = f.read(4)
        if len(hdr) < 4:
            print('ERROR: File too small or empty.', file=sys.stderr)
            sys.exit(1)
        dim = struct.unpack('<i', hdr)[0]
    record_size = 4 + dim * 4

    # Stage 1: raw-bytes check
    print('Stage 1: raw-bytes sortedness check...', file=sys.stderr)
    if check_sorted_bytes(in_path, record_size):
        print('Raw-bytes check passed.', file=sys.stderr)
    else:
        print('Raw-bytes check failed.', file=sys.stderr)
        # Stage 2: numeric check
        print('Stage 2: numeric lexicographic sortedness check...', file=sys.stderr)
        if not check_sorted_numeric(in_path, dim, record_size):
            print('ERROR: Input file not sorted. Aborting.', file=sys.stderr)
            sys.exit(1)
        print('Numeric check passed.', file=sys.stderr)

    # Report duplicates
    dirn, fname = os.path.split(in_path)
    base, ext = os.path.splitext(fname)
    report_path = os.path.join(dirn, f'{base}_dup_report.txt')
    report_duplicates(in_path, dim, record_size, report_path)

if __name__ == '__main__':
    main()

