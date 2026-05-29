#!/usr/bin/env python3
"""
load_dataset.py

Loads the UI-PRMD rehabilitation dataset, verifies structure, extracts labels,
and writes a manifest CSV.
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_ROOT = Path("./UI-PRMD/OpenDataLab___UI-PRMD/raw")
MANIFEST_DIR = Path("./data")
MANIFEST_CSV = MANIFEST_DIR / "manifest.csv"

EXERCISE_PATTERN = re.compile(r"[eE](\d{1,2})")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_quality_label(filepath: Path) -> str:
    """
    Infer quality label from parent folder name.
    movements/ or segmented_movements/ -> correct
    incorrect_movements/ or incorrect_segmented_movements/ -> incorrect
    """
    # Walk up the path to find which top-level subfolder the file lives under
    parts = [p.lower().replace(" ", "_") for p in filepath.parts]
    for part in parts:
        if part in ("movements", "segmented_movements"):
            return "correct"
        if part in ("incorrect_movements", "incorrect_segmented_movements"):
            return "incorrect"
    return "unknown"


def extract_exercise_id(filename: str) -> int:
    """
    Search filename for E01, E02, ... E10 (case-insensitive).
    Returns the integer exercise id, or 0 if not found.
    """
    match = EXERCISE_PATTERN.search(filename)
    if match:
        return int(match.group(1))
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not DATA_ROOT.exists():
        print(f"ERROR: Data root does not exist: {DATA_ROOT.resolve()}", file=sys.stderr)
        sys.exit(1)

    # Collect all .txt files recursively
    txt_files = sorted(DATA_ROOT.rglob("*.txt"))

    if not txt_files:
        print("WARNING: No .txt files found under data root.")
        sys.exit(0)

    # Print first 10 filenames for verification
    print("First 10 filenames found:")
    for i, f in enumerate(txt_files[:10], 1):
        print(f"  {i}. {f.relative_to(DATA_ROOT)}")
    if len(txt_files) > 10:
        print(f"  ... ({len(txt_files) - 10} more)")
    print()

    # Prepare manifest directory
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    label_distribution = {"correct": 0, "incorrect": 0}
    exercise_distribution = {}

    for filepath in txt_files:
        # a. Load with numpy (dataset files are comma-separated)
        try:
            data = np.loadtxt(filepath)
        except Exception:
            try:
                data = np.loadtxt(filepath, delimiter=",")
            except Exception as e:
                print(f"WARNING: Failed to load {filepath}: {e}")
                continue

        # Handle single-row files (1D array)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # b. Verify exactly 117 columns (39 markers × 3 coords)
        if data.shape[1] != 117:
            print(
                f"WARNING: {filepath} has {data.shape[1]} columns, expected 117. Skipping."
            )
            continue

        frames = data.shape[0]

        # c. Extract exercise id
        exercise_id = extract_exercise_id(filepath.name)
        if exercise_id == 0:
            print(
                f"WARNING: No exercise pattern found in filename '{filepath.name}'. "
                f"Setting exercise_id=0 for manual inspection."
            )

        # d. Infer quality label
        quality_label = get_quality_label(filepath)
        if quality_label == "unknown":
            print(
                f"WARNING: Could not infer quality label for {filepath}. "
                f"Setting quality_label='unknown'."
            )

        # e. Record metadata
        record = {
            "filepath": str(filepath),
            "frames": frames,
            "exercise_id": exercise_id,
            "quality_label": quality_label,
        }
        records.append(record)

        # Update distributions
        label_distribution[quality_label] = label_distribution.get(quality_label, 0) + 1
        exercise_distribution[exercise_id] = exercise_distribution.get(exercise_id, 0) + 1

    # -----------------------------------------------------------------------
    # 4. Print summary table
    # -----------------------------------------------------------------------
    print("Summary table:")
    print(f"{'filepath':<60} | {'frames':>6} | {'exercise_id':>11} | {'quality':>7}")
    print("-" * 100)
    for r in records:
        fp = r["filepath"]
        if len(fp) > 58:
            fp = "..." + fp[-55:]
        print(
            f"{fp:<60} | {r['frames']:>6} | {r['exercise_id']:>11} | {r['quality_label']:>7}"
        )
    print()

    # -----------------------------------------------------------------------
    # 5. Print label distribution
    # -----------------------------------------------------------------------
    print("Label distribution:")
    print(f"  Total files processed: {len(records)}")
    print("  By quality_label:")
    for label, count in sorted(label_distribution.items()):
        print(f"    {label:<10}: {count}")
    print("  By exercise_id:")
    for eid, count in sorted(exercise_distribution.items()):
        print(f"    E{eid:02d}: {count}")
    print()

    # -----------------------------------------------------------------------
    # 6. Save manifest.csv
    # -----------------------------------------------------------------------
    with open(MANIFEST_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=["filepath", "frames", "exercise_id", "quality_label"]
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"Manifest saved to: {MANIFEST_CSV.resolve()}")


if __name__ == "__main__":
    main()
