#!/usr/bin/env python3
"""
data_pipeline/crisis_lines/run_all.py

Runs the full crisis_lines_full.json build pipeline, in order:
  1. extract_crisis_lines.py    -> extracts from Wikipedia
  2. apply_manual_fixes.py      -> fixes cases automatic extraction can't resolve
  3. categorize_crisis_lines.py -> classifies by audience (elderly, children, etc.)

Usage:
    python3 run_all.py --live          # fetches the current Wikipedia page (needs internet)
    python3 run_all.py --from-file raw_table.txt   # uses a local snapshot
"""
import argparse
import subprocess
import sys

STEPS = [
    ("1. Extract from Wikipedia", "extract_crisis_lines.py"),
    ("2. Apply manual fixes", "apply_manual_fixes.py"),
    ("3. Classify by audience", "categorize_crisis_lines.py"),
]


def run(cmd):
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Failed at: {' '.join(cmd)}")
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--from-file", type=str)
    args = ap.parse_args()

    print(STEPS[0][0])
    extract_cmd = [sys.executable, "extract_crisis_lines.py", "--out", "crisis_lines_full.json"]
    extract_cmd += ["--live"] if args.live else ["--from-file", args.from_file]
    run(extract_cmd)

    print(STEPS[1][0])
    run([sys.executable, "apply_manual_fixes.py"])

    print(STEPS[2][0])
    run([sys.executable, "categorize_crisis_lines.py"])

    print("\nDone: crisis_lines_full.json is up to date.")
