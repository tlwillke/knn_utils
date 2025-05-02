
#!/usr/bin/env python3
import os
import struct
import argparse
import sys

# Usage:
# ./shard_fvecs.py mydata.fvecs 1000000 --output-dir /shards
# Produces:
# mydata_1000000_0.fvecs
# mydata_1000000_1.fvecs
# …
# mydata_1000000_{N-1}.fvecs
# mydata_{R}_{N}.fvecs   # R < 1000000 is the final remainder
#
# Must specify absolute output path to write shards to (which will be created if it doesn't exist):

def shard_fvecs(input_path: str, shard_size: int, output_dir: str):
    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # verify write permission
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"Cannot write to output directory {output_dir!r}: permission denied")

    # use only the filename for naming shards
    filename = os.path.basename(input_path)
    base, ext = os.path.splitext(filename)

    # --- read dimension from header
    with open(input_path, "rb") as f:
        hdr = f.read(4)
        if len(hdr) < 4:
            raise ValueError("Input file is empty or too small.")
        dim = struct.unpack("<i", hdr)[0]

    record_size = 4 + dim * 4  # 4 bytes for int32 + 4 bytes per float

    # --- compute total vectors
    total_bytes = os.path.getsize(input_path)
    if total_bytes % record_size != 0:
        raise ValueError(
            f"File size ({total_bytes}) is not a multiple of record size ({record_size})."
        )
    total_vectors = total_bytes // record_size

    full_shards = total_vectors // shard_size
    remainder   = total_vectors % shard_size

    print(f"Input has {total_vectors:,} vectors (dim={dim}).")
    print(f"Creating {full_shards} full shards of {shard_size} vectors each", end="")
    if remainder:
        print(f", plus 1 final shard of {remainder} vectors.")
    else:
        print(".")

    # --- stream through and write out each shard
    with open(input_path, "rb") as fin:
        for shard_id in range(full_shards):
            out_name = f"{base}_{shard_size}_{shard_id}{ext}"
            out_path = os.path.join(output_dir, out_name)
            with open(out_path, "wb") as fout:
                for _ in range(shard_size):
                    data = fin.read(record_size)
                    if len(data) != record_size:
                        raise EOFError("Unexpected EOF in a full shard.")
                    fout.write(data)
            print(f"  ▸ Wrote shard {shard_id}: {out_path}")

        if remainder:
            out_name = f"{base}_{remainder}_{full_shards}{ext}"
            out_path = os.path.join(output_dir, out_name)
            with open(out_path, "wb") as fout:
                for _ in range(remainder):
                    data = fin.read(record_size)
                    if len(data) != record_size:
                        raise EOFError("Unexpected EOF in remainder shard.")
                    fout.write(data)
            print(f"  ▸ Wrote final shard {full_shards}: {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Split a .fvecs file into shards of S vectors each."
    )
    parser.add_argument("input_file", help="path to the input .fvecs file")
    parser.add_argument("shard_size", type=int, help="number of vectors per shard (S)")
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="absolute path to the directory to write shards into"
    )
    args = parser.parse_args()

    # ensure shard_size is positive
    if args.shard_size <= 0:
        print("ERROR: shard_size must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    # check input file
    if not os.path.isfile(args.input_file):
        print(f"ERROR: Input file {args.input_file!r} does not exist.", file=sys.stderr)
        sys.exit(1)

    # enforce absolute output directory
    out_dir = args.output_dir
    if not os.path.isabs(out_dir):
        print(f"ERROR: --output-dir must be an absolute path, got {out_dir!r}.", file=sys.stderr)
        sys.exit(1)

    try:
        shard_fvecs(args.input_file, args.shard_size, out_dir)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
