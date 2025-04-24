#!/usr/bin/env python
import os
import numpy as np
import argparse
import struct
import sys

def swap_fvecs_endianness(input_fname, output_fname, input_endian, chunk_size=10000):
    """
    Reads an .fvecs file with specified input endianness and writes a new
    .fvecs file with the opposite endianness. Processes in chunks for memory efficiency.

    Args:
        input_fname (str): Path to the input .fvecs file.
        output_fname (str): Path for the output .fvecs file.
        input_endian (str): Endianness of the input file ('little' or 'big').
        chunk_size (int): Number of vectors to process before printing progress status.

    Returns:
        tuple: (total_vectors_processed, dimension) on success, or (None, None) on failure.
    """
    input_fname = os.path.expanduser(input_fname)
    output_fname = os.path.expanduser(output_fname)

    # Determine input and output format specifiers based on the input endianness
    if input_endian == 'little':
        in_struct_prefix = '<'  # Little-endian for struct
        in_numpy_prefix = '<'   # Little-endian for numpy
        out_struct_prefix = '>' # Big-endian for struct (output)
        output_endian = 'big'
    elif input_endian == 'big':
        in_struct_prefix = '>'  # Big-endian for struct
        in_numpy_prefix = '>'   # Big-endian for numpy
        out_struct_prefix = '<' # Little-endian for struct (output)
        output_endian = 'little'
    else:
        # This should be caught by argparse, but defensive check is good
        raise ValueError("Invalid input_endian value. Use 'little' or 'big'.")

    # Construct the full format strings
    in_struct_fmt_int = in_struct_prefix + 'i'   # Format for reading input dimension (e.g., '<i' or '>i')
    in_numpy_fmt_float = in_numpy_prefix + 'f4' # Format for reading input vector floats (e.g., '<f4' or '>f4')
    out_struct_fmt_int = out_struct_prefix + 'i'   # Format for writing output dimension (e.g., '>i' or '<i')

    print(f"Input file: {input_fname} (Endian: {input_endian})")
    print(f"Output file: {output_fname} (Endian: {output_endian})")

    total_vectors = 0
    dim = -1 # Dimension, determined from the first vector read

    try:
        # Open input file for binary reading, output file for binary writing
        with open(input_fname, "rb") as f_in, open(output_fname, "wb") as f_out:
            while True:
                # --- Step 1: Read the 4-byte dimension from the input file ---
                d_bytes = f_in.read(4)
                if not d_bytes:
                    # No more bytes to read, indicates clean end of file
                    break

                # Check if we read a full 4 bytes for the dimension
                if len(d_bytes) < 4:
                    raise IOError(f"Incomplete dimension read at vector offset {total_vectors}. File might be truncated.")

                # --- Step 2: Unpack the dimension using the specified input endianness ---
                d_current = struct.unpack(in_struct_fmt_int, d_bytes)[0]

                # --- Step 3: Validate the dimension ---
                if total_vectors == 0:
                    # This is the first vector, store its dimension
                    dim = d_current
                    if dim <= 0:
                         raise ValueError(f"Invalid dimension ({dim}) read from the first vector using '{input_endian}' endianness. Check file format or specified input endianness.")
                    print(f"Detected dimension: {dim}")
                elif d_current != dim:
                    # Check if subsequent vectors have the same dimension
                     raise ValueError(f"Inconsistent dimension found at vector offset {total_vectors}. Expected {dim}, but read {d_current}.")

                # --- Step 4: Pack the dimension using the *output* endianness and write to output file ---
                f_out.write(struct.pack(out_struct_fmt_int, d_current))

                # --- Step 5: Read the vector data (dim * 4 bytes) from the input file ---
                vec_byte_count = dim * 4
                vec_bytes = f_in.read(vec_byte_count)

                # Check if we read the full vector data
                if len(vec_bytes) < vec_byte_count:
                     raise IOError(f"Incomplete vector data read at vector offset {total_vectors}. Expected {vec_byte_count} bytes, but got {len(vec_bytes)}. File might be truncated.")

                # --- Step 6: Convert vector bytes to the opposite endianness and write to output file ---
                # Method:
                # 1. Load the raw bytes into a NumPy array, interpreting them with the INPUT endianness.
                # 2. Use the `.byteswap(inplace=False)` method, which returns a *new* array with the byte order reversed for each element.
                # 3. Use `.tobytes()` to get the raw bytes from the byteswapped array.
                vec_array = np.frombuffer(vec_bytes, dtype=in_numpy_fmt_float)
                swapped_vec_bytes = vec_array.byteswap(inplace=False).tobytes()
                f_out.write(swapped_vec_bytes)

                total_vectors += 1

                # --- Optional: Print progress periodically ---
                if total_vectors > 0 and total_vectors % chunk_size == 0:
                    # Print progress update, using '\r' to overwrite the previous line
                    print(f"Processed {total_vectors} vectors...", end='\r', flush=True)
                    sys.stdout.flush() # Ensure the output buffer is flushed immediately

    except FileNotFoundError:
        print(f"\nError: Input file not found at '{input_fname}'", file=sys.stderr)
        # Attempt to clean up the potentially partially written output file
        if os.path.exists(output_fname):
             try:
                 os.remove(output_fname)
                 print(f"Cleaned up partially written output file '{output_fname}'.")
             except OSError as e:
                 print(f"Warning: Could not remove partial output file '{output_fname}': {e}", file=sys.stderr)
        return None, None # Indicate failure

    except (IOError, ValueError) as e:
        print(f"\nError processing file: {e}", file=sys.stderr)
        # Attempt to clean up the potentially partially written output file
        if os.path.exists(output_fname):
             try:
                 os.remove(output_fname)
                 print(f"Cleaned up partially written output file '{output_fname}'.")
             except OSError as e:
                 print(f"Warning: Could not remove partial output file '{output_fname}': {e}", file=sys.stderr)
        return None, None # Indicate failure

    except Exception as e:
        # Catch any other unexpected errors
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        if os.path.exists(output_fname):
             try:
                 os.remove(output_fname)
                 print(f"Cleaned up partially written output file '{output_fname}'.")
             except OSError as e:
                 print(f"Warning: Could not remove partial output file '{output_fname}': {e}", file=sys.stderr)
        return None, None # Indicate failure


    # Ensure the final status message is on a new line after progress updates
    print(f"\nSuccessfully processed {total_vectors} vectors.")
    return total_vectors, dim

def main():
    """Parses command-line arguments and runs the endianness swapping function."""
    parser = argparse.ArgumentParser(
        description="Swap the endianness of an .fvecs file vector by vector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help message
    )
    parser.add_argument("input_file", help="Path to the input .fvecs file.")
    parser.add_argument("output_file", help="Path for the output .fvecs file.")
    parser.add_argument(
        "--input-endian",
        choices=['little', 'big'],
        default='little', # Default to the common fvecs standard (little-endian)
        help="Endianness of the INPUT file."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000, # Process 10k vectors before printing progress
        help="Number of vectors to process before printing a progress update."
    )
    args = parser.parse_args()

    # Prevent accidentally overwriting the input file
    if os.path.abspath(args.input_file) == os.path.abspath(args.output_file):
        print("Error: Input and output file paths cannot be the same.", file=sys.stderr)
        sys.exit(1) # Exit with a non-zero status code indicating failure

    # Call the main swapping function
    total_vectors, dim = swap_fvecs_endianness(
        args.input_file,
        args.output_file,
        args.input_endian,
        args.chunk_size
    )

    # Check the return value to determine success or failure
    if total_vectors is not None:
        output_endian = 'big' if args.input_endian == 'little' else 'little'
        print(f"Conversion complete. Output file '{args.output_file}' written with {output_endian}-endian format ({total_vectors} vectors, dimension {dim}).")
        sys.exit(0) # Exit with status code 0 indicating success
    else:
        print("Conversion failed due to errors.", file=sys.stderr)
        sys.exit(1) # Exit with status code 1 indicating failure

# Standard Python entry point check
if __name__ == "__main__":
    main()
