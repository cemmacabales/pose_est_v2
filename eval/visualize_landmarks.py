#!/usr/bin/env python3
"""visualize_landmarks.py — BlazePose skeleton overlay on video (no classifier needed)."""

import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import cv2
from pose_estimator import PoseEstimator

if len(sys.argv) < 2:
    print("Usage: python visualize_landmarks.py <video_path> [output_dir]")
    sys.exit(1)

video_path = sys.argv[1]
output_dir = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path(__file__).parent / "results"
output_dir.mkdir(exist_ok=True)

stem = pathlib.Path(video_path).stem
output_path = output_dir / f"{stem}_landmarks.mp4"

print(f"Video: {video_path}")
print("Loading BlazePose...")
pose = PoseEstimator(model_complexity=1)

cap = cv2.VideoCapture(video_path)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Resolution: {w}x{h}, FPS: {fps:.0f}, Frames: {total}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

frame_idx = 0
dropped = 0
t0 = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = pose.process_frame(frame)

    if landmarks:
        pose.draw_landmarks(frame, landmarks)
        visible = PoseEstimator.count_visible(landmarks)
        color = (0, 255, 0) if visible >= 25 else (0, 165, 255) if visible >= 15 else (0, 0, 255)
    else:
        visible = 0
        color = (0, 0, 255)
        dropped += 1

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (220, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, f"Landmarks: {visible}/33", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Frame: {frame_idx}/{total}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    writer.write(frame)
    frame_idx += 1

    if frame_idx % 100 == 0:
        elapsed = time.time() - t0
        fps_proc = frame_idx / elapsed
        print(f"  Frame {frame_idx}/{total} ({100*frame_idx/total:.0f}%), "
              f"{fps_proc:.1f} fps, landmarks={visible}/33")

elapsed = time.time() - t0
cap.release()
writer.release()
pose.close()

print(f"\nDone! {frame_idx} frames in {elapsed:.0f}s ({frame_idx/elapsed:.1f} fps)")
print(f"Frames without person: {dropped} ({100*dropped/max(frame_idx,1):.1f}%)")
print(f"Output: {output_path}")
