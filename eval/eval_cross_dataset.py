#!/usr/bin/env python3
"""
Cross-dataset evaluation of the 9-class retrained model.
Tests on the older eval/test_videos/ (15 clips) that were NOT in training.
"""

import os
import re
import sys
from collections import deque, defaultdict

import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

from pose_estimator import PoseEstimator
from joint_map import EXERCISE_NAMES
from joint_angles import batch_keypoints_to_angles

CLASSIFIER_PATH = "./models/classifier.tflite"
TEST_VIDEO_DIR = "./eval/test_videos"

VIDEO_PATTERN = re.compile(r"^(\d{2})_correct_\d+\.mp4$")


def load_model(path):
    interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp, interp.get_input_details(), interp.get_output_details()


def extract_keypoints(video_path, pose_est, classifier,
                      classifier_in, classifier_out, exercise_out_idx):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frame_buffer = deque(maxlen=30)
    predictions = []
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lm = pose_est.process_frame(frame)
        if lm is None:
            continue

        joints = PoseEstimator.extract_mapped_joints(lm)
        frame_buffer.append(joints)
        frame_counter += 1

        if len(frame_buffer) == 30 and frame_counter % 5 == 0:
            window = np.array(frame_buffer, dtype=np.float32)  # (30, 12, 2)
            angles = batch_keypoints_to_angles(window)         # (30, 16)
            window = angles[np.newaxis, :, :]                  # (1, 30, 16)

            classifier.set_tensor(classifier_in[0]["index"], window)
            classifier.invoke()

            ex_out = classifier.get_tensor(classifier_out[exercise_out_idx]["index"])[0]
            ex_idx = int(np.argmax(ex_out))
            ex_conf = float(np.max(ex_out))
            predictions.append((ex_idx, ex_conf))

    cap.release()
    return predictions


def main():
    print("Loading BlazePose...")
    pose_est = PoseEstimator(model_complexity=1)

    print("Loading classifier...")
    classifier, classifier_in, classifier_out = load_model(CLASSIFIER_PATH)

    exercise_out_idx = None
    quality_out_idx = None
    for idx, detail in enumerate(classifier_out):
        if detail["shape"][1] == 9:
            exercise_out_idx = idx
        elif detail["shape"][1] == 2:
            quality_out_idx = idx

    if exercise_out_idx is None:
        print("ERROR: Could not find exercise output with shape (1, 9)")
        sys.exit(1)

    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    per_class_conf = defaultdict(list)
    confusion = defaultdict(lambda: defaultdict(int))
    all_results = []

    for fname in sorted(os.listdir(TEST_VIDEO_DIR)):
        match = VIDEO_PATTERN.match(fname)
        if not match:
            continue

        ex_id_str = match.group(1)
        ex_id = int(ex_id_str)
        expected_cls = ex_id - 1 if ex_id != 10 else 8

        video_path = os.path.join(TEST_VIDEO_DIR, fname)
        predictions = extract_keypoints(
            video_path, pose_est, classifier,
            classifier_in, classifier_out, exercise_out_idx,
        )

        if not predictions:
            print(f"WARNING: No predictions for {fname}")
            continue

        pred_indices = [p[0] for p in predictions]
        pred_confidences = [p[1] for p in predictions]
        majority_pred = max(set(pred_indices), key=pred_indices.count)
        avg_conf = sum(pred_confidences) / len(pred_confidences)
        correct = (majority_pred == expected_cls)

        per_class_total[expected_cls] += 1
        if correct:
            per_class_correct[expected_cls] += 1
        per_class_conf[expected_cls].append(avg_conf)
        confusion[expected_cls][majority_pred] += 1

        all_results.append({
            "file": fname,
            "expected": EXERCISE_NAMES.get(expected_cls, "Unknown"),
            "predicted": EXERCISE_NAMES.get(majority_pred, "Unknown"),
            "confidence": avg_conf,
            "correct": correct,
        })

    print("\n" + "=" * 70)
    print("CROSS-DATASET EVALUATION RESULTS")
    print("=" * 70)

    print(f"\n{'File':<25} {'Expected':<20} {'Predicted':<20} {'Conf':>6} {'OK':>4}")
    print("-" * 70)
    total_correct = 0
    total_videos = 0
    for r in all_results:
        ok = "OK" if r["correct"] else "FAIL"
        print(f"{r['file']:<25} {r['expected']:<20} {r['predicted']:<20} {r['confidence']:>6.2f} {ok:>4}")
        total_correct += r["correct"]
        total_videos += 1

    print("-" * 70)
    overall_acc = total_correct / total_videos if total_videos > 0 else 0
    print(f"\nOVERALL ACCURACY: {total_correct}/{total_videos} = {overall_acc:.1%}")

    print("\nPER-CLASS BREAKDOWN:")
    print(f"{'Class':<25} {'Correct':>8} {'Total':>8} {'Acc':>8} {'Avg Conf':>10}")
    print("-" * 70)
    for cls in range(9):
        name = EXERCISE_NAMES.get(cls, "Unknown")
        c = per_class_correct[cls]
        t = per_class_total[cls]
        acc = c / t if t > 0 else 0
        conf = sum(per_class_conf[cls]) / len(per_class_conf[cls]) if per_class_conf[cls] else 0
        print(f"{name:<25} {c:>8} {t:>8} {acc:>7.1%} {conf:>10.2f}")

    print("\nCONFUSION MATRIX (rows = expected, cols = predicted):")
    header = "Expected \\ Pred".ljust(20)
    for c in range(9):
        header += f" {c:>3}"
    print(header)
    print("-" * (20 + 4 * 9))
    for expected in range(9):
        row = f"{EXERCISE_NAMES.get(expected, '?')[:18]:<20}"
        for pred in range(9):
            row += f" {confusion[expected][pred]:>3}"
        print(row)


if __name__ == "__main__":
    main()
