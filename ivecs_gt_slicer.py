import struct
import sys
import os
import time

def filter_ground_truth_subset(input_filepath, output_filepath, max_base_index, target_neighbors_k):
    """
    Filters an .ivecs ground truth file for a subset database.

    Reads an .ivecs file containing nearest neighbor indices (sorted by distance)
    for a large database. Filters each row to include only indices less than
    max_base_index, keeping up to target_neighbors_k results per row.
    Writes the filtered results to a new .ivecs file.

    Args:
        input_filepath (str): Path to the original .ivecs ground truth file.
        output_filepath (str): Path where the filtered .ivecs file will be saved.
        max_base_index (int): The upper limit (exclusive) for valid base vector indices.
                                (e.g., 250_000_000 for the first 250M vectors).
        target_neighbors_k (int): The maximum number of neighbors to keep for each query
                                  after filtering.

    Returns:
        int: The number of queries that had fewer than target_neighbors_k
             valid neighbors after filtering.
    Raises:
        FileNotFoundError: If the input file does not exist.
        IOError: If there's an error reading/writing files.
        ValueError: If dimensions are inconsistent or non-positive in the input.
    """
    if not os.path.exists(input_filepath):
        raise FileNotFoundError(f"Error: Input file not found at '{input_filepath}'")

    print(f"\nStarting filtering process:")
    print(f"  Input file: {input_filepath}")
    print(f"  Output file: {output_filepath}")
    print(f"  Base index threshold: < {max_base_index:,}")
    print(f"  Target neighbors per query: {target_neighbors_k}")

    bytes_per_int = 4
    processed_queries = 0
    queries_below_target = 0
    start_time = time.time()

    try:
        with open(input_filepath, 'rb') as infile, open(output_filepath, 'wb') as outfile:
            original_row_length = None # Store the dimension read from the first row

            while True:
                # Read dimension from input file
                dim_bytes = infile.read(bytes_per_int)
                if not dim_bytes:
                    break # End of input file

                if len(dim_bytes) < bytes_per_int:
                    raise IOError(f"Error reading dimension at input row {processed_queries}: Unexpected end of file.")

                try:
                    # Assuming little-endian signed int for dimension
                    original_dim = struct.unpack('<i', dim_bytes)[0]
                except struct.error as e:
                    raise IOError(f"Error unpacking dimension at input row {processed_queries}: {e}")

                if original_dim <= 0:
                     raise ValueError(f"Error: Invalid dimension ({original_dim}) found at input row {processed_queries}.")

                # Store the dimension from the first row for consistency checks (optional but good practice)
                if original_row_length is None:
                    original_row_length = original_dim
                    print(f"  Detected original dimension (neighbors per query): {original_row_length}")
                elif original_dim != original_row_length:
                    # Warning or error if dimensions are inconsistent in input
                    print(f"Warning: Inconsistent dimension at input row {processed_queries}. "
                          f"Expected {original_row_length}, got {original_dim}. Processing anyway.", file=sys.stderr)
                    # Or raise ValueError if strict consistency is required

                # Read the original vector data
                expected_bytes = original_dim * bytes_per_int
                vector_bytes = infile.read(expected_bytes)
                if len(vector_bytes) < expected_bytes:
                    raise IOError(f"Error reading vector data at input row {processed_queries}: Unexpected end of file.")

                try:
                    # Unpack original neighbors
                    format_string = f'<{original_dim}i'
                    original_neighbors = struct.unpack(format_string, vector_bytes)
                except struct.error as e:
                     raise IOError(f"Error unpacking vector data at input row {processed_queries} (dim={original_dim}): {e}")

                # --- Filtering Logic ---
                filtered_neighbors = []
                for neighbor_index in original_neighbors:
                    if neighbor_index < max_base_index:
                        filtered_neighbors.append(neighbor_index)
                        if len(filtered_neighbors) == target_neighbors_k:
                            break # Stop once we have enough valid neighbors

                # --- Check and Count ---
                if len(filtered_neighbors) < target_neighbors_k:
                    queries_below_target += 1
                    # You might want to log which query index this was if needed
                    # print(f"Warning: Query {processed_queries} has only {len(filtered_neighbors)} neighbors < {max_base_index}")

                # --- Write Filtered Data to Output File ---
                output_dim = len(filtered_neighbors)
                try:
                    # Write the new dimension (number of filtered neighbors)
                    outfile.write(struct.pack('<i', output_dim))
                    # Write the filtered neighbor indices
                    if output_dim > 0:
                        output_format_string = f'<{output_dim}i'
                        outfile.write(struct.pack(output_format_string, *filtered_neighbors))
                except struct.error as e:
                    raise IOError(f"Error packing data for output row {processed_queries}: {e}")
                except Exception as e:
                    raise IOError(f"Error writing data for output row {processed_queries}: {e}")


                processed_queries += 1
                if processed_queries % 1000 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Processed {processed_queries:,} queries... [{elapsed:.2f}s]")

    except Exception as e:
        print(f"\nAn error occurred during processing: {e}", file=sys.stderr)
        # Clean up partially written output file if an error occurs
        if 'outfile' in locals() and not outfile.closed:
            outfile.close()
        if os.path.exists(output_filepath):
             print(f"Attempting to remove potentially incomplete output file: {output_filepath}", file=sys.stderr)
             try:
                 os.remove(output_filepath)
             except OSError as rm_err:
                 print(f"Error removing file: {rm_err}", file=sys.stderr)
        raise # Re-raise the exception


    end_time = time.time()
    print(f"\n--- Filtering Complete ---")
    print(f"Total queries processed: {processed_queries:,}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print(f"Output file generated: {output_filepath}")
    if queries_below_target > 0:
        print(f"\nWarning: {queries_below_target:,} queries had fewer than {target_neighbors_k} neighbors "
              f"with index < {max_base_index:,}.")
    else:
        print(f"\nAll {processed_queries:,} queries had at least {target_neighbors_k} neighbors "
              f"with index < {max_base_index:,} (or as many as available if the original row was shorter).")

    return queries_below_target


# --- Configuration (Fixed Parameters) ---
MAX_BASE_INDEX = 250_000_000  # Keep only neighbors whose index is LESS THAN this
TARGET_NEIGHBORS = 10        # Keep at most this many neighbors per query

# --- Execution ---
if __name__ == "__main__":
    # --- Get File Paths from User ---
    input_filepath_from_user = input("Enter the path to the INPUT .ivecs ground truth file: ")
    output_filepath_from_user = input("Enter the path for the OUTPUT filtered .ivecs file: ")

    # Basic validation to check if input path is not empty
    if not input_filepath_from_user:
        print("Error: Input file path cannot be empty.", file=sys.stderr)
        sys.exit(1)
    if not output_filepath_from_user:
        print("Error: Output file path cannot be empty.", file=sys.stderr)
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_filepath_from_user)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"Error creating output directory '{output_dir}': {e}", file=sys.stderr)
            sys.exit(1)

    try:
        num_short = filter_ground_truth_subset(
            input_filepath_from_user,      # Use variable from input()
            output_filepath_from_user,     # Use variable from input()
            MAX_BASE_INDEX,
            TARGET_NEIGHBORS
        )
        # You can use the 'num_short' value if needed
    except (FileNotFoundError, IOError, ValueError, Exception) as main_e:
        print(f"\nScript failed: {main_e}", file=sys.stderr)
        sys.exit(1) # Exit with a non-zero code to indicate failure

    print("\nScript finished successfully.")
