#!/usr/bin/env python3
"""annotate_video.py — Headless pose estimation and exercise classification on a video file."""

import sys
import pathlib
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

CLASSIFIER_PATH = "../models/classifier.tflite"

if len(sys.argv) < 2:
    print("Usage: python annotate_video.py <video_path>")
    sys.exit(1)

video_path = sys.argv[1]
input_stem = pathlib.Path(video_path).stem

results_dir = pathlib.Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)
output_path = results_dir / (input_stem + "_annotated.mp4")

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

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    sys.exit(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

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
        window = np.array(frame_buffer, dtype=np.float32)  # (30, 12, 2)
        angles = batch_keypoints_to_angles(window)         # (30, 16)
        window = angles[np.newaxis, :, :]                  # (1, 30, 16)
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
            if stable_quality_idx == 1:
                quality_state = "CORRECT"
            else:
                quality_state = "INCORRECT"
            confidence = stable_confidence

    overlay = frame.copy()
    cv2.rectangle(overlay, (frame_width - 320, 0), (frame_width, 200), (33, 33, 33), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "EXERCISE",
                (frame_width - 305, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (136, 136, 136), 1, cv2.LINE_AA)

    cv2.putText(frame, exercise_name,
                (frame_width - 305, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, "QUALITY",
                (frame_width - 305, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (136, 136, 136), 1, cv2.LINE_AA)

    if quality_state == "CORRECT":
        badge_color = (32, 94, 27)
        text_color = (174, 240, 105)
        badge_text = "CORRECT"
    elif quality_state == "INCORRECT":
        badge_color = (28, 28, 183)
        text_color = (128, 138, 255)
        badge_text = "INCORRECT"
    else:
        badge_color = (51, 51, 51)
        text_color = (136, 136, 136)
        badge_text = "DETECTING..."

    cv2.rectangle(frame,
                  (frame_width - 305, 88), (frame_width - 15, 118),
                  badge_color, -1)
    cv2.putText(frame, badge_text,
                (frame_width - 297, 109), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, text_color, 1, cv2.LINE_AA)

    cv2.putText(frame, "CONFIDENCE",
                (frame_width - 305, 135), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (136, 136, 136), 1, cv2.LINE_AA)

    cv2.rectangle(frame,
                  (frame_width - 305, 145), (frame_width - 25, 157), (80, 80, 80), -1)
    fill_width = int(confidence * 280)
    if confidence >= 0.75:
        fill_color = (174, 240, 105)
    elif confidence >= 0.50:
        fill_color = (64, 215, 255)
    else:
        fill_color = (64, 110, 255)
    if fill_width > 0:
        cv2.rectangle(frame,
                      (frame_width - 305, 145), (frame_width - 305 + fill_width, 157),
                      fill_color, -1)

    cv2.putText(frame, f"{int(confidence * 100)}%",
                (frame_width - 305, 175), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (204, 204, 204), 1, cv2.LINE_AA)

    cv2.putText(frame, f"frame {frame_counter}",
                (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (136, 136, 136), 1, cv2.LINE_AA)

    writer.write(frame)

    if frame_counter % 30 == 0:
        print(f"Frame {frame_counter} / {total_frames}")

cap.release()
writer.release()

print(f"Saved to eval/results/{input_stem}_annotated.mp4")
