#!/usr/bin/env python3
import argparse
import struct
import sys

# Example usage:
#   python fvecs_dedup_sorted.py \
#     --input sorted_bigdata.fvecs \
#     --output unique_bigdata.fvecs \
#     --endian little

def dedup_sorted_fvecs(input_path, output_path, endian):
    # Choose struct prefixes
    prefix = '<' if endian=='little' else '>'
    int_fmt = prefix + 'i'
    float_fmt = lambda d: prefix + f'{d}f'

    with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
        # --- Read first vector ---
        hdr = fin.read(4)
        if not hdr:
            print("❌ Input file is empty.", file=sys.stderr)
            return
        dim = struct.unpack(int_fmt, hdr)[0]
        data = fin.read(4*dim)
        if len(data) < 4*dim:
            raise ValueError("Incomplete first vector data.")
        prev = struct.unpack(float_fmt(dim), data)

        # Write it out
        fout.write(hdr)
        fout.write(data)
        unique_count = 1
        idx = 1

        # --- Process remaining vectors ---
        while True:
            hdr = fin.read(4)
            if not hdr:
                break
            d_cur = struct.unpack(int_fmt, hdr)[0]
            if d_cur != dim:
                raise ValueError(
                    f"✖ Dimension mismatch at vector {idx}: "
                    f"expected {dim}, got {d_cur}"
                )
            data = fin.read(4*dim)
            if len(data) < 4*dim:
                raise ValueError(f"Incomplete vector at index {idx}.")
            curr = struct.unpack(float_fmt(dim), data)

            # 1) sorted check
            if curr < prev:
                raise ValueError(
                    f"✖ File not sorted at vector {idx}: "
                    f"{curr} < {prev}"
                )

            # 2) dedupe
            if curr != prev:
                fout.write(hdr)
                fout.write(data)
                unique_count += 1

            prev = curr
            idx += 1

    print(f"✅ Deduplicated vector count: {unique_count}")

def main():
    p = argparse.ArgumentParser(
        description="Confirm a sorted .fvecs file and write only unique vectors."
    )
    p.add_argument("--input",  "-i", required=True,
                   help="Path to sorted input .fvecs file")
    p.add_argument("--output", "-o", required=True,
                   help="Path to write deduplicated .fvecs file")
    p.add_argument("--endian", "-e", choices=['little','big'], default='little',
                   help="Endianness of the fvecs file (default: little)")
    args = p.parse_args()

    try:
        dedup_sorted_fvecs(args.input, args.output, args.endian)
    except Exception as e:
        print(f"⚠️  Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
