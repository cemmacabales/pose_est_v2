#!/usr/bin/env python3
"""
extract_video_keypoints.py — Extract BlazePose keypoints from labeled .mp4 files
and feed them into the existing training pipeline.

Writes:
  ./data/normalized/<stem>.npy   — normalized keypoints, shape (frames, 12, 2)
  ./data/labels.csv              — metadata appended (idempotent)

After running this script the user can run build_windows.py and
train_classifier.py exactly as before.
"""

import argparse
import csv
import os
import re
import sys

import cv2
import numpy as np

from pose_estimator import PoseEstimator

OUTPUT_DIR = "./data/normalized"
LABELS_CSV = "./data/labels.csv"
VIDEO_PATTERN = re.compile(r"^(\d{2})_([a-zA-Z]+)_(?:P\d+_)?(\d+)\.mp4$")


def parse_filename(filename):
    """
    Parse filenames like 06_correct_1.mp4 or 01_correct_P1_1.mp4.
    Returns (stem, exercise_id_int, quality_label) or None if no match.
    """
    match = VIDEO_PATTERN.match(filename)
    if not match:
        return None
    exercise_id_str, quality, _n = match.groups()
    stem = os.path.splitext(filename)[0]
    exercise_id = int(exercise_id_str)
    return stem, exercise_id, quality


def extract_keypoints(video_path, pose_estimator):
    """
    Run BlazePose on every frame of the video.
    Returns an array of shape (num_frames, 12, 2) with normalized keypoints.
    Frames without a detected person are skipped.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lm = pose_estimator.process_frame(frame)
        if lm is None:
            continue

        joints = PoseEstimator.extract_mapped_joints(lm)
        frames.append(joints)

    cap.release()
    if not frames:
        raise RuntimeError(f"No person detected in any frame of {video_path}")
    return np.array(frames, dtype=np.float32)


def load_existing_stems(labels_path):
    """Return a set of stems already present in labels.csv."""
    if not os.path.exists(labels_path):
        return set()
    stems = set()
    with open(labels_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = row.get("stem", "").strip()
            if stem:
                stems.add(stem)
    return stems


def write_label(labels_path, stem, exercise_id, quality_label):
    """Append a single row to labels.csv, creating header if needed."""
    file_exists = os.path.exists(labels_path)
    with open(labels_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists or os.path.getsize(labels_path) == 0:
            writer.writerow(["stem", "exercise_id", "quality_label"])
        writer.writerow([stem, exercise_id, quality_label])


def main():
    parser = argparse.ArgumentParser(
        description="Extract BlazePose keypoints from labeled .mp4 files."
    )
    parser.add_argument(
        "--video_dir",
        required=True,
        help="Directory containing <exercise_id>_<quality>_<n>.mp4 files.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be processed without writing any files.",
    )
    args = parser.parse_args()

    video_dir = args.video_dir
    if not os.path.isdir(video_dir):
        print(f"Error: not a directory: {video_dir}", file=sys.stderr)
        sys.exit(1)

    candidates = []
    for fname in sorted(os.listdir(video_dir)):
        parsed = parse_filename(fname)
        if parsed:
            candidates.append((fname, parsed))

    if not candidates:
        print(f"No matching .mp4 files found in {video_dir}")
        sys.exit(0)

    existing_stems = load_existing_stems(LABELS_CSV)

    to_process = []
    skipped = []
    for fname, (stem, exercise_id, quality) in candidates:
        if stem in existing_stems:
            skipped.append((fname, stem))
        else:
            to_process.append((fname, stem, exercise_id, quality))

    all_seen_ids = {ex_id for _, (_, ex_id, _) in candidates}
    missing_ids = sorted(set(range(1, 11)) - all_seen_ids)

    if args.dry_run:
        print("=== DRY RUN ===")
        print(f"Videos found:      {len(candidates)}")
        print(f"Already in CSV:    {len(skipped)}")
        print(f"Would process:     {len(to_process)}")
        print("\nWould process:")
        for fname, stem, exercise_id, quality in to_process:
            print(f"  {fname}  ->  stem={stem}, exercise={exercise_id}, quality={quality}")
        if skipped:
            print("\nSkipped (already in labels.csv):")
            for fname, stem in skipped:
                print(f"  {fname}  (stem={stem})")
        print(f"\nExercise IDs seen: {sorted(all_seen_ids) if all_seen_ids else 'None'}")
        print(f"Missing IDs 1-10:  {missing_ids if missing_ids else 'None'}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LABELS_CSV), exist_ok=True)

    print("Loading BlazePose...")
    pose_estimator = PoseEstimator(model_complexity=1)

    processed_count = 0

    for fname, stem, exercise_id, quality in to_process:
        video_path = os.path.join(video_dir, fname)
        try:
            keypoints = extract_keypoints(video_path, pose_estimator)
        except Exception as exc:
            print(f"Warning: failed to process {fname}: {exc}")
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{stem}.npy")
        np.save(out_path, keypoints)
        write_label(LABELS_CSV, stem, exercise_id, quality)

        processed_count += 1
        print(f"Processed {fname}  ->  {out_path}  ({keypoints.shape[0]} frames)")

    for fname, stem in skipped:
        print(f"Skipped {fname}  (stem '{stem}' already in {LABELS_CSV})")

    print("\n=== Summary ===")
    print(f"Videos processed:  {processed_count}")
    print(f"Videos skipped:    {len(skipped)}")
    print(f"Exercise IDs seen: {sorted(all_seen_ids) if all_seen_ids else 'None'}")
    print(f"Missing IDs 1-10:  {missing_ids if missing_ids else 'None'}")


if __name__ == "__main__":
    main()
