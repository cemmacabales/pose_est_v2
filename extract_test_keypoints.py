#!/usr/bin/env python3
"""
extract_test_keypoints.py — Extract BlazePose keypoints from test videos
in eval/TEST/ and add them to the training pipeline.

Test videos are assumed to demonstrate correct form (quality_label=correct)
and are mapped to exercise IDs based on filename patterns.

Usage:
    source .venv/bin/activate
    python extract_test_keypoints.py

Writes:
    ./data/normalized/TEST_<stem>.npy
    Appends to ./data/labels.csv
"""

import csv
import os
import re
import sys

import cv2
import numpy as np

from pose_estimator import PoseEstimator

TEST_DIR = "./eval/TEST"
OUTPUT_DIR = "./data/normalized"
LABELS_CSV = "./data/labels.csv"

FILENAME_TO_EXERCISE = [
    ("deep squats", 1, "DeepSquat"),
    ("squat", 1, "DeepSquat"),
    ("hurdle step", 2, "HurdleStep"),
    ("inline lunge", 3, "InlineLunge"),
    ("side lunges", 4, "SideLunge"),
    ("side lunge", 4, "SideLunge"),
    ("sit to stand", 5, "SitToStand"),
    ("standing leg raise", 6, "StandingLegRaise"),
    ("leg raises", 6, "StandingLegRaise"),
    ("shoulder abduction", 7, "ShoulderAbduction"),
    ("shoulder extension", 8, "ShoulderExtension"),
    ("shoulder scaption", 10, "ShoulderScaption"),
]


def filename_to_exercise_id(fname):
    lowered = fname.lower()
    for substring, ex_id, ex_name in FILENAME_TO_EXERCISE:
        if substring in lowered:
            return ex_id, ex_name
    return None


def safe_stem(fname):
    base = os.path.splitext(fname)[0]
    stem = re.sub(r"[^A-Za-z0-9]", "_", base)
    stem = re.sub(r"_+", "_", stem)
    stem = stem.strip("_")
    return f"TEST_{stem}"


def load_existing_stems(labels_path):
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


def extract_keypoints(video_path, pose_estimator):
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


def write_label(labels_path, stem, exercise_id, quality_label):
    file_exists = os.path.exists(labels_path)
    with open(labels_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists or os.path.getsize(labels_path) == 0:
            writer.writerow(["stem", "exercise_id", "quality_label"])
        writer.writerow([stem, exercise_id, quality_label])


def main():
    if not os.path.isdir(TEST_DIR):
        print(f"Error: test directory not found: {TEST_DIR}", file=sys.stderr)
        sys.exit(1)

    existing_stems = load_existing_stems(LABELS_CSV)

    candidates = []
    for fname in sorted(os.listdir(TEST_DIR)):
        if not fname.lower().endswith(".mp4"):
            continue
        mapped = filename_to_exercise_id(fname)
        if mapped is None:
            print(f"Warning: could not map {fname} to an exercise, skipping")
            continue
        ex_id, ex_name = mapped
        stem = safe_stem(fname)
        candidates.append((fname, stem, ex_id, ex_name))

    if not candidates:
        print("No test videos found.")
        sys.exit(0)

    to_process = []
    skipped = []
    for fname, stem, ex_id, ex_name in candidates:
        if stem in existing_stems:
            skipped.append((fname, stem))
        else:
            to_process.append((fname, stem, ex_id, ex_name))

    print(f"Test videos found:   {len(candidates)}")
    print(f"Already in CSV:      {len(skipped)}")
    print(f"Will process:        {len(to_process)}")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading BlazePose (Full model)...")
    pose_estimator = PoseEstimator(model_complexity=1)

    processed_count = 0
    for fname, stem, ex_id, ex_name in to_process:
        video_path = os.path.join(TEST_DIR, fname)
        try:
            keypoints = extract_keypoints(video_path, pose_estimator)
        except Exception as exc:
            print(f"Warning: failed to process {fname}: {exc}")
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{stem}.npy")
        np.save(out_path, keypoints)
        write_label(LABELS_CSV, stem, ex_id, "correct")

        processed_count += 1
        print(f"  {fname:40s} -> {stem}  ({keypoints.shape[0]:4d} frames)  {ex_name}")

    for fname, stem in skipped:
        print(f"  Skipped {fname}  (stem '{stem}' already in {LABELS_CSV})")

    print(f"\n=== Summary ===")
    print(f"Processed:  {processed_count}")
    print(f"Skipped:    {len(skipped)}")
    print(f"Total test: {len(candidates)}")


if __name__ == "__main__":
    main()
