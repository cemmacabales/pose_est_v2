#!/usr/bin/env python3
"""annotate_test_videos.py — Annotate all test videos with exercise + quality predictions."""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import cv2
import numpy as np
from collections import deque

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

from pose_estimator import PoseEstimator
from joint_map import EXERCISE_NAMES
from joint_angles import batch_keypoints_to_angles

TEST_DIR = pathlib.Path(__file__).parent / "TEST"
OUTPUT_DIR = pathlib.Path(__file__).parent / "results" / "test_annotated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CLASSIFIER_PATH = str(pathlib.Path(__file__).parent.parent / "models" / "classifier.tflite")

videos = sorted(TEST_DIR.glob("*.mp4"))
print(f"Found {len(videos)} test videos in {TEST_DIR}")

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

for vi, video_path in enumerate(videos, 1):
    stem = video_path.stem
    output_path = OUTPUT_DIR / f"{stem}_annotated.mp4"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [{vi}/{len(videos)}] ERROR: {stem}")
        continue

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    frame_buffer = deque(maxlen=30)
    prediction_buffer = deque(maxlen=10)
    frame_counter = 0
    exercise_name = "Detecting..."
    quality_state = "DETECTING..."
    confidence = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lm = pose_est.process_frame(frame)

        if lm is not None:
            pose_est.draw_landmarks(frame, lm)

        if lm is not None:
            joints = PoseEstimator.extract_mapped_joints(lm)
            frame_buffer.append(joints)

        frame_counter += 1
        if len(frame_buffer) == 30 and frame_counter % 5 == 0:
            window = np.array(frame_buffer, dtype=np.float32)
            angles = batch_keypoints_to_angles(window)
            window = angles[np.newaxis, :, :]
            classifier.set_tensor(classifier_in[0]["index"], window)
            classifier.invoke()

            exercise_out = classifier.get_tensor(classifier_out[exercise_out_idx]["index"])[0]
            quality_out = classifier.get_tensor(classifier_out[quality_out_idx]["index"])[0]

            exercise_idx = int(np.argmax(exercise_out))
            quality_idx = int(np.argmax(quality_out))
            exercise_conf = float(np.max(exercise_out))

            prediction_buffer.append((exercise_idx, quality_idx, exercise_conf))

            if len(prediction_buffer) >= 5:
                exercise_indices = [p[0] for p in prediction_buffer]
                quality_indices = [p[1] for p in prediction_buffer]
                confidences = [p[2] for p in prediction_buffer]

                stable_exercise_idx = max(set(exercise_indices), key=exercise_indices.count)
                stable_quality_idx = 1 if sum(quality_indices) > len(quality_indices) / 2 else 0
                stable_confidence = sum(confidences) / len(confidences)

                exercise_name = EXERCISE_NAMES.get(stable_exercise_idx, "Unknown")
                quality_state = "CORRECT" if stable_quality_idx == 1 else "INCORRECT"
                confidence = stable_confidence

        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 320, 0), (w, 200), (33, 33, 33), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cv2.putText(frame, "EXERCISE", (w - 305, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (136, 136, 136), 1, cv2.LINE_AA)
        cv2.putText(frame, exercise_name, (w - 305, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, "QUALITY", (w - 305, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (136, 136, 136), 1, cv2.LINE_AA)

        if quality_state == "CORRECT":
            badge_color, text_color, badge_text = (32, 94, 27), (174, 240, 105), "CORRECT"
        elif quality_state == "INCORRECT":
            badge_color, text_color, badge_text = (28, 28, 183), (128, 138, 255), "INCORRECT"
        else:
            badge_color, text_color, badge_text = (51, 51, 51), (136, 136, 136), "DETECTING..."

        cv2.rectangle(frame, (w - 305, 88), (w - 15, 118), badge_color, -1)
        cv2.putText(frame, badge_text, (w - 297, 109),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)

        cv2.putText(frame, "CONFIDENCE", (w - 305, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (136, 136, 136), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (w - 305, 145), (w - 25, 157), (80, 80, 80), -1)

        fill_width = int(confidence * 280)
        if confidence >= 0.75:
            fill_color = (174, 240, 105)
        elif confidence >= 0.50:
            fill_color = (64, 215, 255)
        else:
            fill_color = (64, 110, 255)
        if fill_width > 0:
            cv2.rectangle(frame, (w - 305, 145), (w - 305 + fill_width, 157), fill_color, -1)

        cv2.putText(frame, f"{int(confidence * 100)}%", (w - 305, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204, 204, 204), 1, cv2.LINE_AA)

        cv2.putText(frame, f"frame {frame_counter}/{total}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (136, 136, 136), 1, cv2.LINE_AA)

        writer.write(frame)

        if frame_counter % 100 == 0:
            print(f"    frame {frame_counter}/{total}")

    cap.release()
    writer.release()

    tag = f"OK" if confidence > 0 else "no predictions"
    print(f"  [{vi:3d}/{len(videos)}] {stem}  ({tag})")

pose_est.close()
print(f"\nDone! {len(videos)} videos annotated → {OUTPUT_DIR}")
