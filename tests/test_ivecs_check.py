import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import ivecs_check


def write_ivecs(path: Path, rows):
    """Write rows to an .ivecs file."""
    with open(path, "wb") as f:
        for row in rows:
            f.write(struct.pack("<i", len(row)))
            f.write(struct.pack(f"<{len(row)}i", *row))


class ValidateIvecsFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_file(self):
        path = self.tmp_path / "valid.ivecs"
        write_ivecs(path, [[0, 1, 2], [3, 4, 5]])

        info = ivecs_check.validate_ivecs_file(str(path), required_k=3)

        self.assertIsNone(info["fatal_error"])
        self.assertEqual(info["num_rows"], 2)
        self.assertEqual(info["first_row"], [0, 1, 2])
        self.assertEqual(info["first_row_length"], 3)
        self.assertEqual(info["max_row_length"], 3)
        self.assertEqual(info["max_ordinal"], 5)
        self.assertEqual(info["duplicate_rows"], [])
        self.assertEqual(info["negative_rows"], [])
        self.assertEqual(info["truncation_rows"], [])
        self.assertEqual(info["overlong_rows"], [])

    def test_duplicate_detection(self):
        path = self.tmp_path / "dups.ivecs"
        write_ivecs(path, [[0, 1, 1, 2, 2, 2], [3, 4, 5, 6, 7, 8]])

        info = ivecs_check.validate_ivecs_file(str(path), required_k=6)

        self.assertIsNone(info["fatal_error"])
        self.assertEqual(info["num_rows"], 2)
        self.assertEqual(len(info["duplicate_rows"]), 1)
        row_num, dup_counts = info["duplicate_rows"][0]
        self.assertEqual(row_num, 0)
        self.assertEqual(dup_counts, {1: 2, 2: 3})

    def test_negative_value_detection(self):
        path = self.tmp_path / "negative.ivecs"
        write_ivecs(path, [[0, -1, 2], [3, 4, -5]])

        info = ivecs_check.validate_ivecs_file(str(path), required_k=3)

        self.assertIsNone(info["fatal_error"])
        self.assertEqual(info["num_rows"], 2)
        self.assertEqual(info["negative_rows"], [(0, 1), (1, 1)])

    def test_truncation_and_overlong_detection(self):
        path = self.tmp_path / "sizes.ivecs"
        write_ivecs(path, [[0, 1], [2, 3, 4], [5, 6, 7, 8]])

        info = ivecs_check.validate_ivecs_file(str(path), required_k=3)

        self.assertIsNone(info["fatal_error"])
        self.assertEqual(info["num_rows"], 3)
        self.assertEqual(info["truncation_rows"], [(0, 2)])
        self.assertEqual(info["overlong_rows"], [(2, 4)])
        self.assertEqual(info["max_row_length"], 4)

    def test_file_not_found(self):
        path = self.tmp_path / "missing.ivecs"

        info = ivecs_check.validate_ivecs_file(str(path), required_k=3)

        self.assertEqual(info["num_rows"], 0)
        self.assertIsNotNone(info["fatal_error"])
        self.assertIn("File not found", info["fatal_error"])

    def test_malformed_truncated_vector_data(self):
        path = self.tmp_path / "malformed.ivecs"
        with open(path, "wb") as f:
            f.write(struct.pack("<i", 3))
            f.write(struct.pack("<3i", 0, 1, 2))
            f.write(struct.pack("<i", 3))
            f.write(struct.pack("<2i", 7, 8))  # one int missing

        info = ivecs_check.validate_ivecs_file(str(path), required_k=3)

        self.assertEqual(info["num_rows"], 1)
        self.assertIsNotNone(info["fatal_error"])
        self.assertIn("Error reading vector data at row 1", info["fatal_error"])
        self.assertEqual(info["first_row"], [0, 1, 2])
        self.assertEqual(info["max_ordinal"], 2)

    def test_malformed_dimension_header(self):
        path = self.tmp_path / "bad_header.ivecs"
        with open(path, "wb") as f:
            f.write(b"\x01\x02")  # incomplete dimension header

        info = ivecs_check.validate_ivecs_file(str(path), required_k=3)

        self.assertEqual(info["num_rows"], 0)
        self.assertIsNotNone(info["fatal_error"])
        self.assertIn("Error reading dimension at row 0", info["fatal_error"])

    def test_non_positive_dimension_is_fatal(self):
        path = self.tmp_path / "zero_dim.ivecs"
        with open(path, "wb") as f:
            f.write(struct.pack("<i", 0))

        info = ivecs_check.validate_ivecs_file(str(path), required_k=3)

        self.assertEqual(info["num_rows"], 0)
        self.assertIsNotNone(info["fatal_error"])
        self.assertIn("non-positive vector dimension 0", info["fatal_error"])


class IvecsCheckCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.input_dir = self.tmp_path / "input"
        self.run_dir = self.tmp_path / "run"
        self.input_dir.mkdir()
        self.run_dir.mkdir()
        self.script_path = Path(__file__).resolve().parent.parent / "ivecs_check.py"

    def tearDown(self):
        self.tmp.cleanup()

    def test_cli_writes_report_to_launch_directory_on_success(self):
        input_file = self.input_dir / "sample.ivecs"
        write_ivecs(input_file, [[0, 1, 2], [3, 4, 5]])

        result = subprocess.run(
            [sys.executable, str(self.script_path), str(input_file), "3"],
            cwd=self.run_dir,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        report_file = self.run_dir / "sample_report.txt"
        self.assertTrue(report_file.exists())
        self.assertFalse((self.input_dir / "sample_report.txt").exists())

        report = report_file.read_text(encoding="utf-8")
        self.assertIn("--- File Information for:", report)
        self.assertIn("PASS: File parsing completed.", report)
        self.assertIn("PASS: No duplicate ordinals found within any fully read row.", report)
        self.assertIn("PASS: No invalid entries found (all values are >= 0 in fully read rows).", report)
        self.assertIn("PASS: Every declared row length matched required k=3.", report)
        self.assertIn("OVERALL: PASS", report)

    def test_cli_writes_report_to_launch_directory_on_failure(self):
        input_file = self.input_dir / "broken.ivecs"
        with open(input_file, "wb") as f:
            f.write(struct.pack("<i", 3))
            f.write(struct.pack("<2i", 10, 11))  # truncated row

        result = subprocess.run(
            [sys.executable, str(self.script_path), str(input_file), "3"],
            cwd=self.run_dir,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 1)
        report_file = self.run_dir / "broken_report.txt"
        self.assertTrue(report_file.exists())

        report = report_file.read_text(encoding="utf-8")
        self.assertIn("FAIL: File parsing did not complete:", report)
        self.assertIn("OVERALL: FAIL", report)

    def test_cli_rejects_bad_argument_count(self):
        result = subprocess.run(
            [sys.executable, str(self.script_path)],
            cwd=self.run_dir,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage:", result.stderr)

    def test_cli_rejects_non_positive_required_k_and_still_writes_report(self):
        input_file = self.input_dir / "sample.ivecs"
        write_ivecs(input_file, [[0, 1, 2]])

        result = subprocess.run(
            [sys.executable, str(self.script_path), str(input_file), "0"],
            cwd=self.run_dir,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 1)
        report_file = self.run_dir / "sample_report.txt"
        self.assertTrue(report_file.exists())

        report = report_file.read_text(encoding="utf-8")
        self.assertIn("FAIL: Invalid required ground-truth k:", report)
        self.assertIn("OVERALL: FAIL", report)


if __name__ == "__main__":
    unittest.main()