import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import fvecs_merge


def write_fvecs(path: Path, rows):
    """Write rows to an .fvecs file."""
    with open(path, "wb") as f:
        for row in rows:
            f.write(struct.pack("<i", len(row)))
            f.write(struct.pack(f"<{len(row)}f", *row))


def read_fvecs(path: Path):
    """Read all rows from an .fvecs file."""
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


class ScanFvecsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_scan_valid_file(self):
        path = self.tmp_path / "valid.fvecs"
        write_fvecs(path, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        dim, count = fvecs_merge.scan_fvecs(path)

        self.assertEqual(dim, 2)
        self.assertEqual(count, 3)

    def test_scan_rejects_empty_file(self):
        path = self.tmp_path / "empty.fvecs"
        path.write_bytes(b"")

        with self.assertRaisesRegex(ValueError, "empty \\.fvecs file"):
            fvecs_merge.scan_fvecs(path)

    def test_scan_rejects_inconsistent_dimension(self):
        path = self.tmp_path / "bad_dim.fvecs"
        with open(path, "wb") as f:
            f.write(struct.pack("<i", 2))
            f.write(struct.pack("<2f", 1.0, 2.0))
            f.write(struct.pack("<i", 3))
            f.write(struct.pack("<3f", 3.0, 4.0, 5.0))

        with self.assertRaisesRegex(ValueError, "inconsistent dimension"):
            fvecs_merge.scan_fvecs(path)

    def test_scan_rejects_truncated_payload(self):
        path = self.tmp_path / "truncated.fvecs"
        with open(path, "wb") as f:
            f.write(struct.pack("<i", 2))
            f.write(struct.pack("<f", 1.0))  # one float missing

        with self.assertRaisesRegex(ValueError, "truncated payload"):
            fvecs_merge.scan_fvecs(path)


class FvecsMergeCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.input_dir = self.tmp_path / "input"
        self.run_dir = self.tmp_path / "run"
        self.input_dir.mkdir()
        self.run_dir.mkdir()
        self.script_path = Path(__file__).resolve().parent.parent / "fvecs_merge.py"

    def tearDown(self):
        self.tmp.cleanup()

    def test_cli_merges_inputs_and_verifies_output(self):
        input_a = self.input_dir / "base.fvecs"
        input_b = self.input_dir / "query.fvecs"
        output = self.run_dir / "merged.fvecs"

        rows_a = [[1.0, 2.0], [3.0, 4.0]]
        rows_b = [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
        write_fvecs(input_a, rows_a)
        write_fvecs(input_b, rows_b)

        result = subprocess.run(
            [sys.executable, str(self.script_path), str(input_a), str(input_b), str(output)],
            cwd=self.run_dir,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertTrue(output.exists())

        merged_rows = read_fvecs(output)
        self.assertEqual(merged_rows, rows_a + rows_b)

        out_dim, out_count = fvecs_merge.scan_fvecs(output)
        self.assertEqual(out_dim, 2)
        self.assertEqual(out_count, 5)

        self.assertIn("Found 2 vectors, dimension 2", result.stdout)
        self.assertIn("Found 3 vectors, dimension 2", result.stdout)
        self.assertIn("✔ Input dimensionality matches: 2", result.stdout)
        self.assertIn("✔ Output verified: 5 vectors, dimension 2", result.stdout)

    def test_cli_rejects_dimension_mismatch(self):
        input_a = self.input_dir / "a.fvecs"
        input_b = self.input_dir / "b.fvecs"
        output = self.run_dir / "merged.fvecs"

        write_fvecs(input_a, [[1.0, 2.0]])
        write_fvecs(input_b, [[1.0, 2.0, 3.0]])

        result = subprocess.run(
            [sys.executable, str(self.script_path), str(input_a), str(input_b), str(output)],
            cwd=self.run_dir,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Input dimensionality mismatch", result.stderr)
        self.assertFalse(output.exists())

    def test_cli_rejects_malformed_input(self):
        input_a = self.input_dir / "a.fvecs"
        input_b = self.input_dir / "b.fvecs"
        output = self.run_dir / "merged.fvecs"

        with open(input_a, "wb") as f:
            f.write(struct.pack("<i", 2))
            f.write(struct.pack("<f", 1.0))  # truncated payload

        write_fvecs(input_b, [[3.0, 4.0]])

        result = subprocess.run(
            [sys.executable, str(self.script_path), str(input_a), str(input_b), str(output)],
            cwd=self.run_dir,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("truncated payload", result.stderr)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
