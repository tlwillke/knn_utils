#!/usr/bin/env python3
"""
fvecs_merge_directory.py

Autodetect pairs of .fvec/.fvecs files in an input directory and merge each
pair into a single .fvecs file in an output directory.

This script:
1. Considers only .fvec and .fvecs files.
2. Pairs files by strongest shared underscore-delimited filename prefix.
3. Uses fvecs_merge.py to scan, copy, and verify the merged file.
4. Names outputs as:

    <dataset_prefix>_<total_vector_count>.fvecs

Notes:
- The merge order is deterministic: the larger file (by vector count) is
  written first, then the smaller file.
- Files that cannot be paired unambiguously are skipped and reported.
"""

import argparse
from pathlib import Path

import fvecs_merge

VALID_SUFFIXES = {".fvec", ".fvecs"}


def list_candidate_files(input_dir: Path):
    """Return all candidate .fvec/.fvecs files in sorted order."""
    return [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    ]


def shared_prefix_tokens(a: Path, b: Path):
    """
    Return the shared underscore-delimited stem tokens for two filenames.

    Example:
      ada_002_1000000_base_vectors
      ada_002_1000000_query_vectors_10000

    -> ["ada", "002", "1000000"]
    """
    a_tokens = a.stem.split("_")
    b_tokens = b.stem.split("_")

    shared = []
    for left, right in zip(a_tokens, b_tokens):
        if left != right:
            break
        shared.append(left)
    return shared


def canonical_dataset_prefix(shared_tokens):
    """
    Convert shared filename tokens into the dataset prefix.

    Heuristic:
    - If the last shared token is a large integer (>= 10000), treat it as an
      embedded vector-count token and drop it from the output prefix.
    - Keep smaller numeric tokens such as model/version/dimension markers
      like 002, 128, 1536, 3072.
    """
    tokens = list(shared_tokens)
    if tokens and tokens[-1].isdigit() and int(tokens[-1]) >= 10000:
        tokens.pop()
    return "_".join(tokens)


def pair_score(a: Path, b: Path):
    """
    Return a sortable score for a candidate pair, or None if too weak.

    Higher is better.
    """
    shared = shared_prefix_tokens(a, b)
    prefix = canonical_dataset_prefix(shared)
    if not prefix:
        return None
    return (len(shared), len("_".join(shared)), prefix)


def find_pairs(input_dir: Path):
    """
    Find pairs by mutual best shared-prefix match.

    Returns:
      complete_pairs: dict[prefix] = (file1, file2)
      unmatched: list[Path]
    """
    files = list_candidate_files(input_dir)
    if not files:
        return {}, []

    best_match = {}

    for path in files:
        best_other = None
        best_score = None

        for other in files:
            if other == path:
                continue

            score = pair_score(path, other)
            if score is None:
                continue

            if best_score is None or score > best_score:
                best_other = other
                best_score = score

        if best_other is not None:
            best_match[path] = (best_other, best_score)

    complete_pairs = {}
    used = set()

    for path in files:
        if path in used or path not in best_match:
            continue

        other, score = best_match[path]
        if other in used or other not in best_match:
            continue

        reverse_other, reverse_score = best_match[other]
        if reverse_other != path:
            continue

        prefix = score[2]
        reverse_prefix = reverse_score[2]
        if prefix != reverse_prefix:
            raise ValueError(
                f"Internal pairing error for {path.name} and {other.name}: "
                f"derived prefixes differ ({prefix!r} vs {reverse_prefix!r})"
            )

        if prefix in complete_pairs:
            raise ValueError(f"Ambiguous pair detection for prefix '{prefix}'")

        complete_pairs[prefix] = (path, other)
        used.add(path)
        used.add(other)

    unmatched = [path for path in files if path not in used]
    return complete_pairs, unmatched


def merge_pair(prefix: str, path1: Path, path2: Path, output_dir: Path) -> Path:
    """
    Merge one detected pair using fvecs_merge.py helpers.

    The larger file by vector count is written first.
    """
    print(f"→ Dataset: {prefix}")

    dim1, count1 = fvecs_merge.scan_fvecs(path1)
    print(f"  {path1.name}: {count1} vectors, dimension {dim1}")

    dim2, count2 = fvecs_merge.scan_fvecs(path2)
    print(f"  {path2.name}: {count2} vectors, dimension {dim2}")

    if dim1 != dim2:
        raise ValueError(
            f"Dimension mismatch for prefix '{prefix}': "
            f"{path1.name} dim={dim1}, {path2.name} dim={dim2}"
        )

    if count1 > count2 or (count1 == count2 and path1.name <= path2.name):
        first_path, first_count = path1, count1
        second_path, second_count = path2, count2
    else:
        first_path, first_count = path2, count2
        second_path, second_count = path1, count1

    total_count = first_count + second_count
    output_path = output_dir / f"{prefix}_{total_count}.fvecs"

    print(f"  Writing: {output_path.name}")
    print(f"    First : {first_path.name}")
    print(f"    Second: {second_path.name}")

    with output_path.open("wb") as fout:
        fvecs_merge.copy_file(first_path, fout)
        fvecs_merge.copy_file(second_path, fout)

    out_dim, out_count = fvecs_merge.scan_fvecs(output_path)
    if out_dim != dim1 or out_count != total_count:
        raise ValueError(
            f"Verification failed for '{output_path}': "
            f"expected {total_count} vectors dim {dim1}, "
            f"found {out_count} vectors dim {out_dim}"
        )

    print(f"  ✔ Verified output: {out_count} vectors, dimension {out_dim}\n")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Autodetect and merge paired .fvec/.fvecs files in a directory."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing input files")
    parser.add_argument("output_dir", type=Path, help="Directory for merged output files")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    complete_pairs, unmatched = find_pairs(args.input_dir)

    if unmatched:
        print("Skipping unmatched files:")
        for path in unmatched:
            print(f"  {path.name}")
        print()

    if not complete_pairs:
        raise ValueError(f"No complete pairs found in {args.input_dir}")

    print(f"Found {len(complete_pairs)} complete pair(s)\n")

    written = []
    for prefix, (path1, path2) in sorted(complete_pairs.items()):
        written.append(merge_pair(prefix, path1, path2, args.output_dir))

    print("Done.")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
    