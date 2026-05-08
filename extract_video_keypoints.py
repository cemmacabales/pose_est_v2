#!/usr/bin/env python3
"""
extract_video_keypoints.py — Extract MoveNet keypoints from labeled .mp4 files
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

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from ai_edge_litert.interpreter import Interpreter

from joint_map import VICON_TO_MOVENET

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH = "./models/movenet_thunder_int8.tflite"
OUTPUT_DIR = "./data/normalized"
LABELS_CSV = "./data/labels.csv"
VIDEO_PATTERN = re.compile(r"^(\d{2})_([a-zA-Z]+)_(\d+)\.mp4$")

MAPPED_INDICES = sorted(VICON_TO_MOVENET.values())  # [5, 6, ..., 16]
MOVENET_INPUT_SIZE = 256


def load_movenet():
    """Load the MoveNet TFLite model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MoveNet model not found: {MODEL_PATH}")
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def parse_filename(filename):
    """
    Parse filenames like 06_correct_1.mp4.
    Returns (stem, exercise_id_int, quality_label) or None if no match.
    """
    match = VIDEO_PATTERN.match(filename)
    if not match:
        return None
    exercise_id_str, quality, _n = match.groups()
    stem = os.path.splitext(filename)[0]
    exercise_id = int(exercise_id_str)
    return stem, exercise_id, quality


def extract_keypoints(video_path, interpreter, input_details, output_details):
    """
    Run MoveNet on every frame of the video.
    Returns an array of shape (num_frames, 12, 2) with normalized keypoints.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_h, orig_w = frame.shape[:2]
        mn_frame = cv2.resize(frame, (MOVENET_INPUT_SIZE, MOVENET_INPUT_SIZE))
        mn_input = np.expand_dims(mn_frame, axis=0).astype(np.uint8)

        interpreter.set_tensor(input_details[0]["index"], mn_input)
        interpreter.invoke()
        kps = interpreter.get_tensor(output_details[0]["index"])[0][0]  # (17, 3) [y, x, conf]

        # Normalize by hip midpoint (same as gui.py inference path)
        left_hip = kps[11]
        right_hip = kps[12]
        hip_mid_x = (left_hip[1] + right_hip[1]) / 2.0
        hip_mid_y = (left_hip[0] + right_hip[0]) / 2.0

        mapped_kps = []
        for m in MAPPED_INDICES:
            y, x, _conf = kps[m]
            nx = x - hip_mid_x
            ny = y - hip_mid_y
            mapped_kps.append([nx, ny])

        frames.append(mapped_kps)

    cap.release()
    return np.array(frames, dtype=np.float32)  # (num_frames, 12, 2)


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
        description="Extract MoveNet keypoints from labeled .mp4 files."
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

    # Gather candidate videos
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

    # Compute seen / missing from all candidates in the directory
    all_seen_ids = {ex_id for _, (_, ex_id, _) in candidates}
    missing_ids = sorted(set(range(1, 11)) - all_seen_ids)

    # ── Dry-run preview ──────────────────────────────────────────────────────
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

    # ── Normal run ───────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LABELS_CSV), exist_ok=True)

    interpreter, input_details, output_details = load_movenet()

    processed_count = 0

    for fname, stem, exercise_id, quality in to_process:
        video_path = os.path.join(video_dir, fname)
        try:
            keypoints = extract_keypoints(video_path, interpreter, input_details, output_details)
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
