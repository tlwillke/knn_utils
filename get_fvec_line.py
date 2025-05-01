import os
import numpy as np
import argparse
import struct

def get_fvec_at_line(fname, target_index, endian='little'):
    """
    Returns the vector at the given zero-based index from a .fvecs file using streaming.
    """
    fname = os.path.expanduser(fname)

    if endian == 'little':
        struct_prefix = '<'
        numpy_fmt_float = '<f4'
    elif endian == 'big':
        struct_prefix = '>'
        numpy_fmt_float = '>f4'
    else:
        raise ValueError(f"Invalid endianness specified: {endian}")

    struct_fmt_int = struct_prefix + 'i'

    with open(fname, 'rb') as f:
        line_num = 0
        while True:
            d_bytes = f.read(4)
            if not d_bytes:
                raise IndexError(f"Reached EOF before index {target_index}.")

            if len(d_bytes) < 4:
                raise ValueError(f"Incomplete dimension header at line {line_num}.")

            d = struct.unpack(struct_fmt_int, d_bytes)[0]
            vec_bytes = f.read(4 * d)
            if len(vec_bytes) < 4 * d:
                raise ValueError(f"Incomplete vector data at line {line_num}.")

            if line_num == target_index:
                vector = np.frombuffer(vec_bytes, dtype=numpy_fmt_float)
                return vector

            # Skip to next line
            line_num += 1

def main():
    parser = argparse.ArgumentParser(
        description="Print the vector at a given line number from an .fvecs file (streamed)."
    )
    parser.add_argument("filename", help="Path to the .fvecs file.")
    parser.add_argument("line_number", type=int, help="0-based index of the line/vector to read.")
    parser.add_argument(
        "--endian",
        choices=["little", "big"],
        default="little",
        help="Byte order of the .fvecs file (default: little-endian)."
    )
    args = parser.parse_args()

    try:
        vec = get_fvec_at_line(args.filename, args.line_number, endian=args.endian)
        print(f"Vector at line {args.line_number}:")
        print(vec)
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
