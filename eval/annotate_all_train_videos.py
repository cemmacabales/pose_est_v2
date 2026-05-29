#!/usr/bin/env python3
"""Batch-annotate all 174 training videos with BlazePose keypoints + classifier predictions + ground truth comparison."""

import sys
import pathlib
import csv
import re
import argparse
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import cv2
import numpy as np
from collections import deque

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

from pose_estimator import PoseEstimator
from joint_map import EXERCISE_NAMES
from joint_angles import batch_keypoints_to_angles

_SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
CLASSIFIER_PATH = str(_SCRIPT_DIR.parent / "models" / "classifier.tflite")

EXERCISE_ID_MAP = {
    "01": 0, "02": 1, "03": 2, "04": 3, "05": 4,
    "06": 5, "07": 6, "08": 7, "10": 8,
}

VIDEO_PATTERN = re.compile(
    r"(\d+)_(correct|incorrect)_P(\d+)_(\d+)$"
)

TRAIN_VIDEOS_DIR = _SCRIPT_DIR / "train_videos_flat"
OUTPUT_DIR = _SCRIPT_DIR / "results" / "train_annotated"
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"


def parse_video_path(video_path):
    stem = pathlib.Path(video_path).stem
    match = VIDEO_PATTERN.search(stem)
    if not match:
        return None
    ex_id, quality_str, person_str, clip_str = match.groups()
    return {
        "exercise_id": ex_id,
        "quality": quality_str,
        "person": int(person_str),
        "clip": int(clip_str),
    }


def draw_ground_truth_overlay(frame, expected_exercise_name, expected_quality_str,
                               predicted_exercise_name, predicted_quality_str,
                               exercise_match, quality_match, confidence):
    h, w = frame.shape[:2]

    panel_x = w - 340
    panel_w = 330

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, 0), (panel_x + panel_w, 250), (33, 33, 33), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    def put_text(text, y_offset, size=0.45, color=(136, 136, 136), thickness=1):
        cv2.putText(frame, text, (panel_x + 15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)

    put_text("GROUND TRUTH", 28, 0.45, (136, 136, 136), 1)
    gt_color = (105, 240, 174) if exercise_match else (255, 145, 145)
    put_text(f"{expected_exercise_name} / {expected_quality_str}", 55, 0.6, gt_color, 2)

    put_text("PREDICTION", 82, 0.45, (136, 136, 136), 1)
    pred_color = (105, 240, 174) if exercise_match else (145, 145, 255)
    put_text(f"{predicted_exercise_name} / {predicted_quality_str}", 109, 0.6, pred_color, 2)

    if exercise_match:
        put_text("Exercise correct", 136, 0.4, (105, 240, 174), 1)
    else:
        put_text("Exercise MISMATCH", 136, 0.4, (145, 145, 255), 1)

    put_text("CONFIDENCE", 162, 0.45, (136, 136, 136), 1)

    bar_y1, bar_y2 = 172, 184
    bar_x1, bar_x2 = panel_x + 15, panel_x + panel_w - 15
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (80, 80, 80), -1)
    fill_w = int(confidence * (bar_x2 - bar_x1))
    if fill_w > 0:
        if confidence >= 0.75:
            fill_clr = (105, 240, 174)
        elif confidence >= 0.5:
            fill_clr = (255, 215, 64)
        else:
            fill_clr = (64, 110, 255)
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), fill_clr, -1)
    put_text(f"{int(confidence * 100)}%", 195, 0.5, (204, 204, 204), 1)


def annotate_video(video_path, pose_est, classifier,
                   classifier_in, classifier_out,
                   exercise_out_idx, quality_out_idx,
                   output_height=720):
    info = parse_video_path(video_path)
    if info is None:
        print(f"  Skipping {video_path.name}: unrecognized filename pattern")
        return None

    ex_id = info["exercise_id"]
    expected_quality = info["quality"]
    person = info["person"]
    clip = info["clip"]

    expected_exercise_idx = EXERCISE_ID_MAP.get(ex_id, -1)
    expected_exercise_name = EXERCISE_NAMES.get(expected_exercise_idx, "Unknown")
    expected_quality_str = "CORRECT" if expected_quality == "correct" else "INCORRECT"

    ex_dir_name = f"{ex_id}_{expected_exercise_name.replace(' ', '')}"
    out_dir = OUTPUT_DIR / ex_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / (video_path.stem + "_annotated.mp4")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Error: Could not open {video_path}")
        return None

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_height > 0:
        out_h = output_height
        out_w = int(frame_w * out_h / frame_h)
    else:
        out_h, out_w = frame_h, frame_w

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    frame_buffer = deque(maxlen=30)
    prediction_buffer = deque(maxlen=10)
    frame_counter = 0

    predicted_exercise_name = "Detecting..."
    predicted_quality_str = "DETECTING..."
    exercise_match = False
    quality_match = False
    confidence = 0.0
    predictions_made = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if output_height > 0:
            display = cv2.resize(frame, (out_w, out_h))
            disp_w, disp_h = out_w, out_h
        else:
            display = frame.copy()
            disp_w, disp_h = frame_w, frame_h

        lm = pose_est.process_frame(frame)

        if lm is not None:
            pose_est.draw_landmarks(display, lm)

        if lm is not None:
            joints = PoseEstimator.extract_mapped_joints(lm)
            frame_buffer.append(joints)

        frame_counter += 1
        if len(frame_buffer) == 30 and frame_counter % 5 == 0:
            window = np.array(frame_buffer, dtype=np.float32)  # (30, 12, 2)
            angles = batch_keypoints_to_angles(window)         # (30, 16)
            window = angles[np.newaxis, :, :]                  # (1, 30, 16)
            classifier.set_tensor(classifier_in[0]["index"], window)
            classifier.invoke()

            exercise_out = classifier.get_tensor(classifier_out[exercise_out_idx]["index"])[0]
            quality_out = classifier.get_tensor(classifier_out[quality_out_idx]["index"])[0]

            pred_ex_idx = int(np.argmax(exercise_out))
            pred_q_idx = int(np.argmax(quality_out))
            ex_conf = float(np.max(exercise_out))

            prediction_buffer.append((pred_ex_idx, pred_q_idx, ex_conf))

            if len(prediction_buffer) >= 5:
                ex_idxs = [p[0] for p in prediction_buffer]
                q_idxs = [p[1] for p in prediction_buffer]
                confs = [p[2] for p in prediction_buffer]

                stable_ex_idx = max(set(ex_idxs), key=ex_idxs.count)
                stable_q_idx = 1 if sum(q_idxs) > len(q_idxs) / 2 else 0
                stable_conf = sum(confs) / len(confs)

                predicted_exercise_name = EXERCISE_NAMES.get(stable_ex_idx, "Unknown")
                predicted_quality_str = "CORRECT" if stable_q_idx == 1 else "INCORRECT"
                exercise_match = (stable_ex_idx == expected_exercise_idx)
                quality_match = (stable_q_idx == (1 if expected_quality == "correct" else 0))
                confidence = stable_conf
                predictions_made += 1

        draw_ground_truth_overlay(
            display,
            expected_exercise_name, expected_quality_str,
            predicted_exercise_name, predicted_quality_str,
            exercise_match, quality_match, confidence,
        )

        cv2.putText(display, f"frame {frame_counter}",
                    (10, disp_h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (136, 136, 136), 1, cv2.LINE_AA)

        if frame_counter % 100 == 0 and frame_counter < total_frames:
            print(f"    frame {frame_counter}/{total_frames}")

        writer.write(display)

    cap.release()
    writer.release()

    print(f"  Saved {output_path}")

    return {
        "filename": video_path.name,
        "expected_exercise": expected_exercise_name,
        "predicted_exercise": predicted_exercise_name,
        "exercise_match": "Y" if exercise_match else "N",
        "expected_quality": expected_quality_str,
        "predicted_quality": predicted_quality_str,
        "quality_match": "Y" if quality_match else "N",
        "avg_confidence": f"{confidence:.2f}",
        "predictions": predictions_made,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch-annotate all 174 training videos")
    parser.add_argument("--video_dir", default=None,
                        help="Directory with training videos (default: eval/train_videos_flat)")
    parser.add_argument("--force", action="store_true",
                        help="Re-annotate even if output already exists")
    parser.add_argument("--output-height", type=int, default=720,
                        help="Output video height in pixels (default: 720; set to 0 for original size)")
    args = parser.parse_args()

    video_dir = pathlib.Path(args.video_dir) if args.video_dir else TRAIN_VIDEOS_DIR
    if not video_dir.exists():
        print(f"Error: Video directory not found: {video_dir}")
        sys.exit(1)

    mp4_files = sorted(video_dir.glob("*.mp4"))
    if not mp4_files:
        print(f"Error: No .mp4 files found in {video_dir}")
        sys.exit(1)

    print(f"Found {len(mp4_files)} videos in {video_dir}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading BlazePose...")
    pose_est = PoseEstimator(model_complexity=1)

    print("Loading classifier...")
    classifier = Interpreter(model_path=CLASSIFIER_PATH)
    classifier.allocate_tensors()
    classifier_in = classifier.get_input_details()
    classifier_out = classifier.get_output_details()

    exercise_out_idx = None
    quality_out_idx = None
    for idx, detail in enumerate(classifier_out):
        if detail["shape"][1] == 9:
            exercise_out_idx = idx
        elif detail["shape"][1] == 2:
            quality_out_idx = idx

    if exercise_out_idx is None or quality_out_idx is None:
        print("Error: Could not identify classifier outputs")
        sys.exit(1)

    total = len(mp4_files)
    results = []
    for i, video_path in enumerate(mp4_files, 1):
        print(f"[{i}/{total}] {video_path.name}")

        info = parse_video_path(video_path)
        if info:
            ex_name = EXERCISE_NAMES.get(EXERCISE_ID_MAP.get(info["exercise_id"], -1), "Unknown")
            ex_dir = f"{info['exercise_id']}_{ex_name.replace(' ', '')}"
            out_path = OUTPUT_DIR / ex_dir / (video_path.stem + "_annotated.mp4")
            if out_path.exists() and not args.force:
                print(f"  Already exists, skipping (use --force to re-annotate)")
                continue

        result = annotate_video(
            video_path, pose_est, classifier,
            classifier_in, classifier_out,
            exercise_out_idx, quality_out_idx,
            output_height=args.output_height,
        )
        if result:
            results.append(result)

    fieldnames = [
        "filename", "expected_exercise", "predicted_exercise", "exercise_match",
        "expected_quality", "predicted_quality", "quality_match",
        "avg_confidence", "predictions",
    ]
    with open(SUMMARY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    total_proc = len(results)
    ex_correct = sum(1 for r in results if r["exercise_match"] == "Y")
    q_correct = sum(1 for r in results if r["quality_match"] == "Y")
    print(f"\n{'='*60}")
    print(f"Done! {total_proc} videos annotated.")
    if total_proc > 0:
        print(f"Exercise accuracy:  {ex_correct}/{total_proc} ({ex_correct/total_proc*100:.1f}%)")
        print(f"Quality accuracy:   {q_correct}/{total_proc} ({q_correct/total_proc*100:.1f}%)")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
