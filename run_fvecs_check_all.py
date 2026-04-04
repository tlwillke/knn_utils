#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def find_input_files(input_dir: Path, recursive: bool) -> list[Path]:
    pattern_iter = input_dir.rglob("*") if recursive else input_dir.glob("*")
    files = [
        path for path in pattern_iter
        if path.is_file() and path.suffix.lower() in {".fvec", ".fvecs"}
    ]
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run fvecs_check.py on every .fvec/.fvecs file in a directory."
    )
    parser.add_argument("input_dir", help="Directory containing .fvec/.fvecs files")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively",
    )
    parser.add_argument(
        "--endian",
        choices=["little", "big"],
        default="little",
        help="Byte order to pass to fvecs_check.py",
    )
    parser.add_argument(
        "--checker",
        default=str(Path(__file__).with_name("fvecs_check.py")),
        help="Path to fvecs_check.py",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    checker = Path(args.checker).expanduser().resolve()

    if not input_dir.is_dir():
        print(f"❌ Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    if not checker.is_file():
        print(f"❌ Checker script not found: {checker}", file=sys.stderr)
        return 1

    files = find_input_files(input_dir, args.recursive)
    if not files:
        print(f"⚠️ No .fvec or .fvecs files found in {input_dir}")
        return 0

    repo_root = checker.parent
    plots_dir = repo_root / "plots"
    plots_dir.mkdir(exist_ok=True)

    failures = 0

    for i, path in enumerate(files, start=1):
        print()
        print("=" * 100)
        print(f"[{i}/{len(files)}] Checking {path}")
        print("=" * 100)

        cmd = [
            sys.executable,
            str(checker),
            str(path),
            "--endian",
            args.endian,
            "--plot",
        ]

        result = subprocess.run(
            cmd,
            cwd=repo_root,
            text=True,
            capture_output=True,
        )

        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")

        log_path = plots_dir / f"{path.stem}_fvecs_check.log"
        log_path.write_text(result.stdout + result.stderr, encoding="utf-8")

        if result.returncode != 0:
            failures += 1
            print(f"❌ Failed: {path}")
        else:
            print(f"✅ Done: {path}")
            print(f"📝 Log saved to {log_path}")

    print()
    print(f"Finished. Files checked: {len(files)}, failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())