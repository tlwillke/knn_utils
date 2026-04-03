#!/usr/bin/env python3

import os
import struct
import sys
from collections import Counter


def write_report(report_path, report_lines):
    """Write the validation report to a text file in the current working directory."""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")


def validate_ivecs_file(filepath, required_k):
    """
    Validate an .ivecs ground-truth file.

    Checks performed:
      - duplicate ordinals within each row (error)
      - negative entries within each row (error; valid values are >= 0)
      - row length differs from required_k (error)
      - malformed / truncated file while parsing (error)

    Returns a dict containing both parsed info and validation findings.
    Rows are reported with 0-based row numbers.
    """
    result = {
        "first_row": None,
        "num_rows": 0,                 # fully read rows
        "first_row_length": None,      # first fully read row length
        "max_row_length": 0,           # max declared row length seen
        "max_ordinal": None,           # largest integer found in fully read rows
        "duplicate_rows": [],          # [(row_num, {ordinal: count, ...}), ...]
        "negative_rows": [],           # [(row_num, negative_count), ...]
        "truncation_rows": [],         # [(row_num, row_len), ...] where row_len < required_k
        "overlong_rows": [],           # [(row_num, row_len), ...] where row_len > required_k
        "fatal_error": None,           # parse error string, if any
    }

    if not os.path.exists(filepath):
        result["fatal_error"] = f"File not found: '{filepath}'"
        return result

    bytes_per_int = 4

    try:
        with open(filepath, "rb") as f:
            while True:
                row_num = result["num_rows"]

                dim_bytes = f.read(bytes_per_int)
                if not dim_bytes:
                    break

                if len(dim_bytes) < bytes_per_int:
                    result["fatal_error"] = (
                        f"Error reading dimension at row {row_num}: unexpected end of file."
                    )
                    break

                try:
                    dim = struct.unpack("<i", dim_bytes)[0]
                except struct.error as e:
                    result["fatal_error"] = f"Error unpacking dimension at row {row_num}: {e}"
                    break

                if dim <= 0:
                    result["fatal_error"] = (
                        f"Error: non-positive vector dimension {dim} at row {row_num}."
                    )
                    break

                result["max_row_length"] = max(result["max_row_length"], dim)

                if dim < required_k:
                    result["truncation_rows"].append((row_num, dim))
                elif dim > required_k:
                    result["overlong_rows"].append((row_num, dim))

                expected_bytes = dim * bytes_per_int
                vector_bytes = f.read(expected_bytes)
                if len(vector_bytes) < expected_bytes:
                    result["fatal_error"] = (
                        f"Error reading vector data at row {row_num}: "
                        f"expected {expected_bytes} bytes, got {len(vector_bytes)}. "
                        "Unexpected end of file."
                    )
                    break

                format_string = f"<{dim}i"
                try:
                    vector = struct.unpack(format_string, vector_bytes)
                except struct.error as e:
                    result["fatal_error"] = (
                        f"Error unpacking vector data at row {row_num} (dim={dim}): {e}"
                    )
                    break

                if result["first_row"] is None:
                    result["first_row"] = list(vector)
                    result["first_row_length"] = dim

                row_max = max(vector)
                if result["max_ordinal"] is None or row_max > result["max_ordinal"]:
                    result["max_ordinal"] = row_max

                negative_count = sum(1 for x in vector if x < 0)
                if negative_count > 0:
                    result["negative_rows"].append((row_num, negative_count))

                counts = Counter(vector)
                dup_counts = {ordinal: count for ordinal, count in counts.items() if count > 1}
                if dup_counts:
                    result["duplicate_rows"].append((row_num, dup_counts))

                result["num_rows"] += 1

    except OSError as e:
        result["fatal_error"] = f"OS error while reading file: {e}"

    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {os.path.basename(sys.argv[0])} <groundtruth.ivecs> <required_k>", file=sys.stderr)
        sys.exit(2)

    ivecs_file_path = sys.argv[1]
    required_k_input = sys.argv[2]

    report_prefix = os.path.splitext(os.path.basename(ivecs_file_path))[0]
    report_path = f"{report_prefix}_report.txt"
    report_lines = []

    def emit(line=""):
        print(line)
        report_lines.append(line)

    exit_code = 0

    try:
        required_k = int(required_k_input)
        if required_k <= 0:
            raise ValueError("required ground-truth k must be positive")

        info = validate_ivecs_file(ivecs_file_path, required_k)

        first_vector = info["first_row"]
        total_rows = info["num_rows"]
        first_row_length = info["first_row_length"]
        max_row_length = info["max_row_length"]
        max_ordinal = info["max_ordinal"]
        duplicate_rows = info["duplicate_rows"]
        negative_rows = info["negative_rows"]
        truncation_rows = info["truncation_rows"]
        overlong_rows = info["overlong_rows"]
        fatal_error = info["fatal_error"]

        max_rows_to_report = 20
        max_dups_per_row_to_report = 10

        emit(f"--- File Information for: {ivecs_file_path} ---")
        emit(f"Fully read rows: {total_rows}")
        emit(f"Maximum declared row length observed: {max_row_length}")
        emit(f"Largest integer found in fully read rows: {max_ordinal if max_ordinal is not None else 'N/A'}")

        if first_vector is not None:
            emit(f"First fully read row length: {first_row_length}")
            emit("")
            emit("First row (vector):")
            max_elements_to_print = 20
            if len(first_vector) > max_elements_to_print:
                emit(f"  {first_vector[:max_elements_to_print]} ... (truncated)")
            else:
                emit(f"  {first_vector}")
        else:
            emit("No complete row was successfully read.")

        emit("")
        emit("--- Validation Summary ---")

        if fatal_error is None:
            emit("PASS: File parsing completed.")
        else:
            emit(f"FAIL: File parsing did not complete: {fatal_error}")
            exit_code = 1

        if not duplicate_rows:
            emit("PASS: No duplicate ordinals found within any fully read row.")
        else:
            emit(f"FAIL: {len(duplicate_rows)} row(s) contain duplicate ordinals.")
            for row_num, dup_counts in duplicate_rows[:max_rows_to_report]:
                items = sorted(dup_counts.items())[:max_dups_per_row_to_report]
                detail = ", ".join(f"{ordinal} x{count}" for ordinal, count in items)
                if len(dup_counts) > max_dups_per_row_to_report:
                    detail += ", ..."
                emit(f"  Row {row_num}: {detail}")
            if len(duplicate_rows) > max_rows_to_report:
                emit(f"  ... and {len(duplicate_rows) - max_rows_to_report} more row(s)")
            exit_code = 1

        if not negative_rows:
            emit("PASS: No invalid entries found (all values are >= 0 in fully read rows).")
        else:
            emit(f"FAIL: {len(negative_rows)} row(s) contain invalid entries (values < 0).")
            for row_num, negative_count in negative_rows[:max_rows_to_report]:
                emit(f"  Row {row_num}: invalid_count={negative_count}")
            if len(negative_rows) > max_rows_to_report:
                emit(f"  ... and {len(negative_rows) - max_rows_to_report} more row(s)")
            exit_code = 1

        if not truncation_rows and not overlong_rows:
            emit(f"PASS: Every declared row length matched required k={required_k}.")
        else:
            if truncation_rows:
                emit(f"FAIL: {len(truncation_rows)} row(s) are shorter than required k ({required_k}).")
                for row_num, row_len in truncation_rows[:max_rows_to_report]:
                    emit(f"  Row {row_num}: length={row_len}")
                if len(truncation_rows) > max_rows_to_report:
                    emit(f"  ... and {len(truncation_rows) - max_rows_to_report} more row(s)")
                exit_code = 1

            if overlong_rows:
                emit(f"FAIL: {len(overlong_rows)} row(s) exceed required k ({required_k}).")
                for row_num, row_len in overlong_rows[:max_rows_to_report]:
                    emit(f"  Row {row_num}: length={row_len}")
                if len(overlong_rows) > max_rows_to_report:
                    emit(f"  ... and {len(overlong_rows) - max_rows_to_report} more row(s)")
                exit_code = 1

        emit("")
        emit(f"OVERALL: {'PASS' if exit_code == 0 else 'FAIL'}")

    except ValueError as e:
        emit(f"--- File Information for: {ivecs_file_path} ---")
        emit(f"FAIL: Invalid required ground-truth k: {e}")
        emit("")
        emit("OVERALL: FAIL")
        exit_code = 1

    try:
        write_report(report_path, report_lines)
        print(f"\nReport written to: {report_path}")
    except OSError as e:
        print(f"Failed to write report file '{report_path}': {e}", file=sys.stderr)
        exit_code = 1

    sys.exit(exit_code)
