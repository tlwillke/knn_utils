

import struct
import sys
import os

def read_ivecs_info(filepath):
    """
    Reads an .ivecs file to get the first row, number of rows, and vector length.

    Args:
        filepath (str): The path to the .ivecs file.

    Returns:
        tuple: A tuple containing:
            - list: The first vector (row) as a list of integers, or None if empty.
            - int: The total number of vectors (rows) in the file.
            - int: The dimensionality (length) of the vectors, or None if empty.
    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there's an error reading the file (e.g., malformed).
        ValueError: If dimensions are inconsistent or non-positive.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: File not found at '{filepath}'")

    first_row = None
    num_rows = 0
    row_length = None
    bytes_per_int = 4 # Standard size for int32

    try:
        with open(filepath, 'rb') as f:
            while True:
                # Read the dimension (4 bytes)
                dim_bytes = f.read(bytes_per_int)

                # Check if end of file reached
                if not dim_bytes:
                    break # Successfully reached end of file

                if len(dim_bytes) < bytes_per_int:
                     raise IOError(f"Error reading dimension at row {num_rows}: Unexpected end of file.")

                # Unpack dimension (assuming little-endian 32-bit integer)
                # Use '<i' for little-endian signed integer
                # Use '>i' for big-endian signed integer
                # Use '<I' or '>I' for unsigned if needed, but signed is common
                try:
                    dim = struct.unpack('<i', dim_bytes)[0]
                except struct.error as e:
                    raise IOError(f"Error unpacking dimension at row {num_rows}: {e}")

                # --- Validation and Storing Information ---
                if row_length is None: # First vector
                    if dim <= 0:
                        raise ValueError(f"Error: First vector dimension is non-positive ({dim}).")
                    row_length = dim
                elif dim != row_length:
                    # Check consistency
                    raise ValueError(f"Inconsistent dimension found at row {num_rows}. "
                                     f"Expected {row_length}, but got {dim}.")

                # --- Read the Vector Data ---
                expected_bytes = row_length * bytes_per_int
                vector_bytes = f.read(expected_bytes)

                if len(vector_bytes) < expected_bytes:
                    raise IOError(f"Error reading vector data at row {num_rows}: "
                                  f"Expected {expected_bytes} bytes, got {len(vector_bytes)}. "
                                  "Unexpected end of file.")

                # --- Unpack Vector Data ---
                # Format string like '<128i' for a 128-dim vector, little-endian
                format_string = f'<{row_length}i'
                try:
                    vector = struct.unpack(format_string, vector_bytes)
                except struct.error as e:
                     raise IOError(f"Error unpacking vector data at row {num_rows} (dim={row_length}): {e}")

                # --- Store First Row ---
                if first_row is None:
                    first_row = list(vector) # Store as a list

                # --- Increment Row Count ---
                num_rows += 1

    except Exception as e:
        # Re-raise exceptions for clarity or handle them as needed
        print(f"An error occurred: {e}", file=sys.stderr)
        raise # Re-raise the caught exception

    return first_row, num_rows, row_length

# --- Example Usage ---
if __name__ == "__main__":
    # Replace with the actual path to your .ivecs file
    # Example: SIFT1M ground truth file
    # ivecs_file_path = 'sift/sift_groundtruth.ivecs'
    ivecs_file_path = input("Enter the path to the .ivecs file: ")

    try:
        first_vector, total_rows, vector_dim = read_ivecs_info(ivecs_file_path)

        print(f"\n--- File Information for: {ivecs_file_path} ---")

        if total_rows > 0:
            print(f"Total number of rows (vectors): {total_rows}")
            print(f"Vector dimensionality (length): {vector_dim}")
            print(f"\nFirst row (vector):")
            # Print only the first few elements if it's very long
            max_elements_to_print = 20
            if len(first_vector) > max_elements_to_print:
                 print(f"  {first_vector[:max_elements_to_print]} ... (truncated)")
            else:
                 print(f"  {first_vector}")
        else:
            print("The file appears to be empty or could not be read correctly.")

    except FileNotFoundError as e:
        print(e, file=sys.stderr)
    except (IOError, ValueError) as e:
        print(f"Error processing file: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
