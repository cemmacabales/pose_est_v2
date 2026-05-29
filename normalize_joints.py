#!/usr/bin/env python3
"""
normalize_joints.py

Reads ./data/manifest.csv, normalizes UI-PRMD Vicon data to MoveNet format,
and saves .npy files plus a labels.csv.
"""

import csv
import sys
from pathlib import Path

import numpy as np

from joint_map import VICON_TO_MOVENET, LEFT_HIP_MARKER, RIGHT_HIP_MARKER

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MANIFEST_CSV = Path("./data/manifest.csv")
NORMALIZED_DIR = Path("./data/normalized")
LABELS_CSV = Path("./data/labels.csv")

# Sort Vicon markers by MoveNet joint index (ascending)
SORTED_MARKERS = sorted(VICON_TO_MOVENET.items(), key=lambda item: item[1])


def main() -> None:
    if not MANIFEST_CSV.exists():
        print(f"ERROR: Manifest not found: {MANIFEST_CSV.resolve()}", file=sys.stderr)
        sys.exit(1)

    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)

    labels = []

    with open(MANIFEST_CSV, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filepath = Path(row["filepath"])
            stem = filepath.stem

            if "positions" not in filepath.name.lower():
                print(f"SKIP (not positions): {filepath.name}")
                continue

            if not filepath.exists():
                print(f"WARNING: File not found, skipping: {filepath}")
                continue

            # Load data (UI-PRMD files are usually space-separated;
            # fall back to comma-separated if that fails.)
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

            if data.shape[1] != 117:
                print(
                    f"WARNING: {filepath} has {data.shape[1]} columns, expected 117. Skipping."
                )
                continue

            num_frames = data.shape[0]

            # --- Hip midpoint (in mm) --------------------------------------
            hip_x = (data[:, LEFT_HIP_MARKER * 3] + data[:, RIGHT_HIP_MARKER * 3]) / 2.0
            hip_y = (
                data[:, LEFT_HIP_MARKER * 3 + 1] + data[:, RIGHT_HIP_MARKER * 3 + 1]
            ) / 2.0

            # --- Extract & normalize joints --------------------------------
            normalized = np.empty((num_frames, 12, 2), dtype=np.float64)

            for joint_idx, (vicon_marker, _movenet_index) in enumerate(SORTED_MARKERS):
                # Convert mm to approximate 0-1 scale by dividing by a body-scale factor.
                # A standing person in Vicon is roughly 1700mm tall.
                # Dividing by 1700 puts values in roughly -0.5 to 0.5 range,
                # matching MoveNet 0-1 normalized coordinates after hip subtraction.
                normalized[:, joint_idx, 0] = (
                    data[:, vicon_marker * 3] - hip_x
                ) / 1700.0
                normalized[:, joint_idx, 1] = (
                    data[:, vicon_marker * 3 + 1] - hip_y
                ) / 1700.0

            # Save normalized trial
            out_path = NORMALIZED_DIR / f"{stem}.npy"
            np.save(out_path, normalized)

            exercise_id = row.get("exercise_id", "0")
            quality_label = row.get("quality_label", "unknown")
            labels.append([stem, exercise_id, quality_label])

            print(f"{stem} | {num_frames} | {exercise_id} | {quality_label}")

    # Save labels.csv
    with open(LABELS_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["stem", "exercise_id", "quality_label"])
        writer.writerows(labels)

    print(f"\nSaved {len(labels)} normalized trials to {NORMALIZED_DIR}")
    print(f"Labels saved to {LABELS_CSV}")


if __name__ == "__main__":
    main()
