#!/usr/bin/env python3
"""
test_rep_detection.py — Test RepCounter on train_videos_flat training videos.

Selects up to N correct-form videos per exercise, runs the full live-equivalent
pipeline (BlazePose → classifier → RepCounter), and writes annotated output
videos with the rep count overlaid.

Usage:
    python eval/test_rep_detection.py             # default: 5 per exercise
    python eval/test_rep_detection.py --max 1      # quick test: 1 per exercise
    python eval/test_rep_detection.py --max 1 --video_dir /path/to/videos

Output:
    eval/results/rep_test/<stem>_rep_test.mp4   — annotated video
    eval/results/rep_test/summary.csv            — per-video results
"""

import argparse
import csv
import os
import re
import sys
import pathlib
import time
from collections import deque, defaultdict

import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_root))

from pose_estimator import PoseEstimator
from joint_map import EXERCISE_NAMES
from joint_angles import batch_keypoints_to_angles, compute_motion, keypoints_to_angles
from rep_counter import RepCounter

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
CLASSIFIER_PATH = str(_root / "models" / "classifier.tflite")
VIDEO_DIR = str(_root / "eval" / "train_videos_flat")
OUTPUT_DIR = str(pathlib.Path(__file__).parent / "results" / "rep_test")
VIDEOS_PER_EXERCISE = 5
IDLE_THRESHOLD = 0.03
IDLE_CONFIRM_COUNT = 2
MAX_PREDICTION_BUFFER = 10

# Map filename exercise ID → EXERCISE_NAMES class index
#  01 → 0 (Deep Squat), 02 → 1 (Hurdle Step), …, 08 → 7, 10 → 8
def _id_to_class(ex_id: int) -> int:
    if ex_id == 10:
        return 8
    return ex_id - 1


# ---------------------------------------------------------------------------
# Video selection
# ---------------------------------------------------------------------------
def select_videos(video_dir: str, max_per_exercise: int) -> list[tuple[str, str, str]]:
    """Return list of (filename, exercise_name, expected_class)."""
    available = defaultdict(list)
    pattern = re.compile(r"^(\d{2})_correct.*\.mp4$")

    for fname in sorted(os.listdir(video_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        ex_id = int(m.group(1))
        cls_idx = _id_to_class(ex_id)
        if cls_idx < 0 or cls_idx > 8:
            continue
        name = EXERCISE_NAMES.get(cls_idx, "Unknown")
        available[name].append(fname)

    selected = []
    for name in sorted(available):
        chosen = available[name][:max_per_exercise]
        for fname in chosen:
            selected.append((fname, name, _id_to_class(int(re.match(r"^(\d{2})_", fname).group(1)))))

    return selected


# ---------------------------------------------------------------------------
# Classifier setup
# ---------------------------------------------------------------------------
def load_classifier(path: str):
    interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()

    ex_idx = None
    ql_idx = None
    for idx, d in enumerate(out):
        if d["shape"][1] == 9:
            ex_idx = idx
        elif d["shape"][1] == 2:
            ql_idx = idx

    if ex_idx is None:
        raise RuntimeError("Classifier missing 9-class exercise head")
    return interp, inp, out, ex_idx, ql_idx


# ---------------------------------------------------------------------------
# Video processing (single)
# ---------------------------------------------------------------------------
def process_video(
    video_path: str,
    out_path: str,
    pose_est: PoseEstimator,
    classifier,
    classifier_in,
    classifier_out,
    ex_out_idx: int,
    ql_out_idx: int,
    exercise_name: str,
) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (fw, fh))

    frame_buffer = deque(maxlen=30)
    prediction_buffer = deque(maxlen=MAX_PREDICTION_BUFFER)
    frame_counter = 0
    idle_count = 0
    current_exercise = None

    rep_counter = RepCounter()

    # Overlay state
    display_ex = "Detecting..."
    display_quality = "DETECTING..."
    display_conf = 0.0
    display_reps = 0
    display_rep_str = "—"

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lm = pose_est.process_frame(frame)

        if lm is not None:
            pose_est.draw_landmarks(frame, lm)
            joints = PoseEstimator.extract_mapped_joints(lm)
            frame_buffer.append(joints)
            if current_exercise is not None:
                rep_counter.update(current_exercise, keypoints_to_angles(joints))

        frame_counter += 1

        # ---- classifier + rep logic (every 5 frames, once buffer full) ----
        if len(frame_buffer) == 30 and frame_counter % 5 == 0:
            window = np.array(frame_buffer, dtype=np.float32)
            angles = batch_keypoints_to_angles(window)
            motion = compute_motion(angles)

            # Idle detection
            prev_idle = idle_count
            if motion < IDLE_THRESHOLD:
                idle_count = min(idle_count + 1, IDLE_CONFIRM_COUNT)
            else:
                idle_count = 0

            if idle_count >= IDLE_CONFIRM_COUNT:
                if prev_idle < IDLE_CONFIRM_COUNT:
                    prediction_buffer.clear()
                    current_exercise = None
                display_ex = "Idle"
                display_quality = "WAITING"
                display_conf = 0.0
                display_rep_str = "—"
            else:
                # Run classifier
                win_in = angles[np.newaxis, :, :]  # (1, 30, 16)
                classifier.set_tensor(classifier_in[0]["index"], win_in)
                classifier.invoke()
                ex_raw = classifier.get_tensor(classifier_out[ex_out_idx]["index"])[0]
                ex_idx_pred = int(np.argmax(ex_raw))
                ex_conf = float(np.max(ex_raw))

                if ql_out_idx is not None:
                    ql_raw = classifier.get_tensor(classifier_out[ql_out_idx]["index"])[0]
                    ql_idx_pred = int(np.argmax(ql_raw))
                else:
                    ql_idx_pred = 0

                prediction_buffer.append((ex_idx_pred, ql_idx_pred, ex_conf))

                # Start counting reps on the very first prediction — don't wait for 5
                instant_ex = EXERCISE_NAMES.get(ex_idx_pred)
                if instant_ex is not None:
                    current_exercise = instant_ex

                if len(prediction_buffer) >= 5:
                    ex_indices = [p[0] for p in prediction_buffer]
                    ql_indices = [p[1] for p in prediction_buffer]
                    confs = [p[2] for p in prediction_buffer]

                    stable_ex = max(set(ex_indices), key=ex_indices.count)
                    stable_ql = 1 if sum(ql_indices) > len(ql_indices) / 2 else 0
                    stable_conf = sum(confs) / len(confs)

                    display_ex = EXERCISE_NAMES.get(stable_ex, "Unknown")
                    display_quality = "CORRECT" if stable_ql == 1 else "INCORRECT"
                    display_conf = stable_conf

                    # Rep counter is driven per-frame above; just read the current count
                    display_reps = rep_counter.get_counts().get(display_ex, 0)
                    display_rep_str = str(display_reps)

        # ---- draw overlay ----
        _draw_overlay(frame, fw, display_ex, display_quality, display_conf,
                      display_rep_str, frame_counter)

        writer.write(frame)

        if frame_counter % 90 == 0:
            elapsed = time.time() - start_time
            pct = (frame_counter / total_frames * 100) if total_frames > 0 else 0
            print(f"  frame {frame_counter}/{total_frames} ({pct:.0f}%) — {elapsed:.1f}s")

    cap.release()
    writer.release()

    total_reps = rep_counter.get_counts().get(exercise_name, 0)

    predicted = display_ex
    if display_ex in ("Detecting...", "Idle"):
        if len(prediction_buffer) >= 1:
            last_ex = prediction_buffer[-1][0]
            predicted = EXERCISE_NAMES.get(last_ex, "Unknown")
        else:
            predicted = "Unknown"

    return {
        "exercise": exercise_name,
        "predicted": predicted,
        "rep_count": total_reps,
        "frames": frame_counter,
        "duration_sec": frame_counter / fps if fps > 0 else 0,
        "output": out_path,
    }


# ---------------------------------------------------------------------------
# Overlay drawing (replicates annotate_video.py + adds REPS)
# ---------------------------------------------------------------------------
def _draw_overlay(frame, fw, exercise, quality, confidence, reps_str, frame_num):
    overlay = frame.copy()
    cv2.rectangle(overlay, (fw - 350, 0), (fw, 280), (33, 33, 33), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    y = 28
    # EXERCISE
    cv2.putText(frame, "EXERCISE", (fw - 335, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (136, 136, 136), 1, cv2.LINE_AA)
    y += 32
    cv2.putText(frame, exercise, (fw - 335, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # QUALITY
    y += 32
    cv2.putText(frame, "QUALITY", (fw - 335, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (136, 136, 136), 1, cv2.LINE_AA)
    y += 22
    if quality == "CORRECT":
        badge_color = (32, 94, 27)
        text_color = (174, 240, 105)
    elif quality == "INCORRECT":
        badge_color = (28, 28, 183)
        text_color = (128, 138, 255)
    else:
        badge_color = (51, 51, 51)
        text_color = (136, 136, 136)
    cv2.rectangle(frame, (fw - 335, y), (fw - 15, y + 30), badge_color, -1)
    cv2.putText(frame, quality, (fw - 327, y + 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, text_color, 1, cv2.LINE_AA)

    # REPS (new)
    y += 46
    cv2.putText(frame, "REPS", (fw - 335, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (136, 136, 136), 1, cv2.LINE_AA)
    y += 28
    try:
        reps_val = int(reps_str)
    except (ValueError, TypeError):
        reps_val = -1

    if reps_val < 0:
        rep_color = (136, 136, 136)
        rep_size = 0.65
    elif reps_val > 0:
        rep_color = (174, 240, 105)
        rep_size = 0.75
    else:
        rep_color = (255, 255, 255)
        rep_size = 0.75
    cv2.putText(frame, reps_str, (fw - 335, y), cv2.FONT_HERSHEY_SIMPLEX,
                rep_size, rep_color, 2, cv2.LINE_AA)

    # CONFIDENCE
    y += 24
    cv2.putText(frame, "CONFIDENCE", (fw - 335, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (136, 136, 136), 1, cv2.LINE_AA)
    y += 12
    cv2.rectangle(frame, (fw - 335, y), (fw - 25, y + 12), (80, 80, 80), -1)
    fill_width = int(confidence * 310)
    if confidence >= 0.75:
        fill_color = (174, 240, 105)
    elif confidence >= 0.50:
        fill_color = (64, 215, 255)
    else:
        fill_color = (64, 110, 255)
    if fill_width > 0:
        cv2.rectangle(frame, (fw - 335, y), (fw - 335 + fill_width, y + 12), fill_color, -1)
    y += 20
    cv2.putText(frame, f"{int(confidence * 100)}%", (fw - 335, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204, 204, 204), 1, cv2.LINE_AA)

    # Frame counter
    cv2.putText(frame, f"frame {frame_num}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (136, 136, 136), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Test RepCounter on training videos")
    parser.add_argument("--max", type=int, default=VIDEOS_PER_EXERCISE,
                        metavar="N", help=f"Max videos per exercise (default: {VIDEOS_PER_EXERCISE})")
    parser.add_argument("--video_dir", default=VIDEO_DIR,
                        help="Path to video directory")
    parser.add_argument("--output_dir", default=OUTPUT_DIR,
                        help="Path for annotated output videos")
    args = parser.parse_args()

    print("=" * 60)
    print("REP DETECTION TEST — train_videos_flat")
    print(f"Videos per exercise: {args.max}")
    print(f"Video dir: {args.video_dir}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 60)

    # Select videos
    selected = select_videos(args.video_dir, args.max)
    n_videos = len(selected)
    if n_videos == 0:
        print("No matching videos found!")
        return
    print(f"\nSelected {n_videos} videos across {len(set(e for _, e, _ in selected))} exercises.\n")

    # Load models
    print("Loading BlazePose (full model)...")
    pose_est = PoseEstimator(model_complexity=1)

    print(f"Loading classifier ({CLASSIFIER_PATH})...")
    classifier, c_in, c_out, ex_out_idx, ql_out_idx = load_classifier(CLASSIFIER_PATH)

    # Ensure output dir
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    results = []
    start_all = time.time()

    for i, (fname, ex_name, ex_cls) in enumerate(selected):
        video_path = os.path.join(args.video_dir, fname)
        stem = pathlib.Path(fname).stem
        out_path = os.path.join(args.output_dir, f"{stem}_rep_test.mp4")

        print(f"\n[{i+1}/{n_videos}] {fname}  ({ex_name})")
        try:
            res = process_video(
                video_path, out_path, pose_est,
                classifier, c_in, c_out, ex_out_idx, ql_out_idx,
                ex_name,
            )
            res["file"] = fname
            results.append(res)
            print(f"  -> Reps: {res['rep_count']} | Duration: {res['duration_sec']:.1f}s"
                  f" | Predicted: {res['predicted']}")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append({
                "file": fname,
                "exercise": ex_name,
                "predicted": "ERROR",
                "rep_count": None,
                "frames": 0,
                "duration_sec": 0,
                "output": "",
                "error": str(exc),
            })

    elapsed = time.time() - start_all
    print(f"\n{'=' * 60}")
    print(f"Done. {len(results)} videos processed in {elapsed:.0f}s.")
    print(f"Output: {args.output_dir}/")
    print(f"{'=' * 60}")

    # Write summary CSV
    csv_path = os.path.join(args.output_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "exercise", "predicted", "rep_count", "frames", "duration_sec"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})

    # Print summary table
    print(f"\n{'File':<35} {'Expected':<20} {'Predicted':<20} {'Reps':>5}")
    print("-" * 85)
    for r in results:
        reps = str(r.get("rep_count", "ERR")) if r.get("rep_count") is not None else "ERR"
        print(f"{r['file']:<35} {r['exercise']:<20} {r.get('predicted', '?')[:18]:<20} {reps:>5}")

    print(f"\nSummary written to {csv_path}")


if __name__ == "__main__":
    main()
