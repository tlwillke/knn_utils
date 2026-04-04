import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import fvecs_merge
import fvecs_merge_directory


def write_fvecs(path: Path, rows):
    """Write rows to an .fvecs or .fvec file."""
    with open(path, "wb") as f:
        for row in rows:
            f.write(struct.pack("<i", len(row)))
            f.write(struct.pack(f"<{len(row)}f", *row))


def read_fvecs(path: Path):
    """Read all rows from an .fvecs or .fvec file."""
    rows = []
    with open(path, "rb") as f:
        while True:
            hdr = f.read(4)
            if not hdr:
                break
            if len(hdr) != 4:
                raise ValueError("Incomplete dimension header")
            dim = struct.unpack("<i", hdr)[0]
            data = f.read(4 * dim)
            if len(data) != 4 * dim:
                raise ValueError("Incomplete vector payload")
            rows.append(list(struct.unpack(f"<{dim}f", data)))
    return rows


class FindPairsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_find_pairs_ignores_non_fvec_files_and_reports_unmatched(self):
        matched_a = self.tmp_path / "demo_128_alpha_100000.fvec"
        matched_b = self.tmp_path / "demo_128_beta_10000.fvecs"
        orphan = self.tmp_path / "orphan_256_only.fvec"
        ignored = self.tmp_path / "demo_128.hdf5"

        write_fvecs(matched_a, [[1.0, 2.0]])
        write_fvecs(matched_b, [[3.0, 4.0]])
        write_fvecs(orphan, [[5.0, 6.0]])
        ignored.write_bytes(b"not an fvec file")

        pairs, unmatched = fvecs_merge_directory.find_pairs(self.tmp_path)

        self.assertEqual(set(pairs.keys()), {"demo_128"})
        self.assertEqual(pairs["demo_128"], (matched_a, matched_b))
        self.assertEqual(unmatched, [orphan])

    def test_canonical_dataset_prefix_drops_large_trailing_count_token(self):
        prefix = fvecs_merge_directory.canonical_dataset_prefix(["ada", "002", "1000000"])
        self.assertEqual(prefix, "ada_002")

    def test_canonical_dataset_prefix_keeps_small_numeric_tokens(self):
        prefix = fvecs_merge_directory.canonical_dataset_prefix(["text-embedding-3-large", "1536"])
        self.assertEqual(prefix, "text-embedding-3-large_1536")


class FvecsMergeDirectoryCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.input_dir = self.tmp_path / "input"
        self.output_dir = self.tmp_path / "output"
        self.run_dir = self.tmp_path / "run"
        self.input_dir.mkdir()
        self.output_dir.mkdir()
        self.run_dir.mkdir()
        self.script_path = Path(__file__).resolve().parent.parent / "fvecs_merge_directory.py"

    def tearDown(self):
        self.tmp.cleanup()

    def test_cli_merges_autodetected_pair_and_uses_total_count_only_in_name(self):
        smaller = self.input_dir / "demo_128_alpha_100000.fvec"
        larger = self.input_dir / "demo_128_beta_10000.fvecs"
        ignored = self.input_dir / "demo_128.hdf5"
        orphan = self.input_dir / "lonely_256_gamma.fvec"

        rows_smaller = [[1.0, 2.0], [3.0, 4.0]]
        rows_larger = [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]

        write_fvecs(smaller, rows_smaller)
        write_fvecs(larger, rows_larger)
        write_fvecs(orphan, [[11.0, 12.0]])
        ignored.write_bytes(b"ignore me")

        result = subprocess.run(
            [sys.executable, str(self.script_path), str(self.input_dir), str(self.output_dir)],
            cwd=self.run_dir,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)

        output_file = self.output_dir / "demo_128_5.fvecs"
        self.assertTrue(output_file.exists())
        self.assertFalse((self.output_dir / "demo_128_100005.fvecs").exists())

        merged_rows = read_fvecs(output_file)
        self.assertEqual(merged_rows, rows_larger + rows_smaller)

        out_dim, out_count = fvecs_merge.scan_fvecs(output_file)
        self.assertEqual(out_dim, 2)
        self.assertEqual(out_count, 5)

        self.assertIn("Found 1 complete pair(s)", result.stdout)
        self.assertIn("Skipping unmatched files:", result.stdout)
        self.assertIn("lonely_256_gamma.fvec", result.stdout)
        self.assertIn("demo_128_5.fvecs", result.stdout)

    def test_cli_rejects_dimension_mismatch(self):
        left = self.input_dir / "bad_128_left_100000.fvec"
        right = self.input_dir / "bad_128_right_10000.fvec"

        write_fvecs(left, [[1.0, 2.0]])
        write_fvecs(right, [[3.0, 4.0, 5.0]])

        result = subprocess.run(
            [sys.executable, str(self.script_path), str(self.input_dir), str(self.output_dir)],
            cwd=self.run_dir,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Dimension mismatch for prefix 'bad_128'", result.stderr)

    def test_cli_fails_when_no_complete_pairs_exist(self):
        only_file = self.input_dir / "solo_128_alpha.fvec"
        write_fvecs(only_file, [[1.0, 2.0]])

        result = subprocess.run(
            [sys.executable, str(self.script_path), str(self.input_dir), str(self.output_dir)],
            cwd=self.run_dir,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No complete pairs found", result.stderr)


if __name__ == "__main__":
    unittest.main()
