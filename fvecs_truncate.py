import os
import argparse
import struct

def stream_truncate_fvecs(input_file, output_file, num_base):
    """
    Reads the input fvec file vector by vector and writes the first num_base vectors
    to the output file.
    Each vector is stored as: [d (int32), float, float, ..., float].
    """
    count = 0
    with open(input_file, "rb") as fin, open(output_file, "wb") as fout:
        while count < num_base:
            # Read 4 bytes for the dimension (int32).
            d_bytes = fin.read(4)
            if not d_bytes:
                # End of file reached before we got num_base vectors.
                break
            # Unpack the dimension.
            d = struct.unpack('i', d_bytes)[0]
            # Read the next d float32 numbers (4 bytes each).
            vec_bytes = fin.read(4 * d)
            if len(vec_bytes) < 4 * d:
                raise ValueError("Unexpected EOF or incomplete vector encountered.")
            # Write the dimension and the vector bytes to the output file.
            fout.write(d_bytes)
            fout.write(vec_bytes)
            count += 1
    return count

def main():
    parser = argparse.ArgumentParser(
        description="Stream process and truncate a base fvec file to a specified number of vectors."
    )
    parser.add_argument("input", type=str, help="Input fvec file with base vectors")
    parser.add_argument("num_base", type=int, help="Number of base vectors to keep (from the beginning)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output fvec file name. If not provided, a default name will be used.")
    args = parser.parse_args()

    # Determine the output file name.
    if args.output is not None:
        output_file = args.output
    else:
        base_name, ext = os.path.splitext(args.input)
        output_file = base_name + "_truncated" + ext

    num_written = stream_truncate_fvecs(args.input, output_file, args.num_base)
    if num_written < args.num_base:
        print(f"Warning: Only {num_written} vectors were found in the input file.")
    print(f"Stored {num_written} base vectors in '{output_file}'.")

if __name__ == '__main__':
    main()


