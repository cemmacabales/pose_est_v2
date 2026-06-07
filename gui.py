#!/usr/bin/env python3
"""
gui.py — Real-time pose estimation and exercise classification GUI for Raspberry Pi 5.

Usage:
    python gui.py              # default: lite model (RPi 5 optimized)
    python gui.py --model lite # same as above
    python gui.py --model full # full model (desktop/M2)
"""

import argparse
import queue
import tkinter as tk
from tkinter import ttk
from collections import deque
import time

import cv2
import numpy as np
from PIL import Image, ImageTk
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
from joint_angles import batch_keypoints_to_angles, compute_motion

from tts_engine import TTSEngine
from session_logger import SessionLogger
from session_chat.app import start_server
import qrcode
import socket
import threading

parser = argparse.ArgumentParser(description="Pose Estimation + Exercise Classification GUI")
parser.add_argument("--model", choices=["lite", "full"], default="lite",
                    help="BlazePose model complexity: lite (RPi) or full (desktop)")
args, _ = parser.parse_known_args()

MODEL_COMPLEXITY = 0 if args.model == "lite" else 1
print(f"Model: {args.model} (complexity={MODEL_COMPLEXITY})")

pose_est = PoseEstimator(model_complexity=MODEL_COMPLEXITY, running_mode="live_stream")

print("Loading classifier...")
classifier = Interpreter(model_path="./models/classifier.tflite")
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

_classifier_input_queue = queue.Queue(maxsize=1)
_classifier_output_queue = queue.Queue(maxsize=1)


def _classifier_worker():
    while True:
        window_data = _classifier_input_queue.get()
        if window_data is None:
            break
        classifier.set_tensor(classifier_in[0]["index"], window_data)
        classifier.invoke()
        exercise_out = classifier.get_tensor(classifier_out[exercise_out_idx]["index"])[0]
        quality_out = classifier.get_tensor(classifier_out[quality_out_idx]["index"])[0]
        exercise_idx = int(np.argmax(exercise_out))
        quality_idx = int(np.argmax(quality_out))
        exercise_conf = float(np.max(exercise_out))
        while not _classifier_output_queue.empty():
            try:
                _classifier_output_queue.get_nowait()
            except queue.Empty:
                break
        _classifier_output_queue.put((exercise_idx, quality_idx, exercise_conf))


_classifier_thread = threading.Thread(target=_classifier_worker, daemon=True)
_classifier_thread.start()

tts = TTSEngine()
logger = SessionLogger()
tts.announce_session_start()

_session_ended = False

SIDEBAR_WIDTH = 380


def add_spacer(parent, height):
    s = tk.Frame(parent, height=height, bg="#212121")
    s.pack(fill=tk.X, pady=0)
    return s


def _get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def on_closing():
    import tkinter.messagebox as mb
    if mb.askyesno("End Session", "Are you sure you want to end the session?"):
        if _session_ended:
            root.destroy()
        else:
            _end_session(from_closing=True)


def _end_session(from_closing=False):
    global _session_ended
    if _session_ended:
        return
    _session_ended = True
    end_session_btn.config(state=tk.DISABLED)
    tts.announce_session_end()
    session_data = logger.end_session()
    start_server(session_data)
    ip = _get_local_ip()
    url = f"http://{ip}:5000"
    hostname = socket.gethostname()
    mdns_url = f"http://{hostname}.local:5000"
    img = qrcode.make(url)
    photo = ImageTk.PhotoImage(img)
    top = tk.Toplevel(root)
    top.title("Session Complete")
    top.configure(bg="#1a1a1a")
    top.resizable(False, False)
    qr_label = tk.Label(top, image=photo, bg="#1a1a1a")
    qr_label.image = photo
    qr_label.pack(pady=10)
    tk.Label(top, text=f"mDNS: {mdns_url}", fg="#FFFFFF", bg="#1a1a1a").pack()
    tk.Label(top, text=f"IP: {url}", fg="#FFFFFF", bg="#1a1a1a").pack()
    close_btn = tk.Button(top, text="Close", command=top.destroy)
    close_btn.pack(pady=10)
    if from_closing:
        def _on_top_closing():
            top.destroy()
            root.destroy()
        top.protocol("WM_DELETE_WINDOW", _on_top_closing)
        close_btn.config(command=_on_top_closing)
    top.update_idletasks()
    w = top.winfo_width()
    h = top.winfo_height()
    x = (top.winfo_screenwidth() // 2) - (w // 2)
    y = (top.winfo_screenheight() // 2) - (h // 2)
    top.geometry(f"+{x}+{y}")


def detect_cameras():
    cameras = []
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append((i, f"Camera {i}"))
        cap.release()
    return cameras


cameras = detect_cameras()

if not cameras:
    import tkinter.messagebox as mb
    mb.showerror("No Camera", "No cameras detected. Please connect a camera and restart.")
    exit(1)

selected_index = cameras[0][0]
_current_camera_index = selected_index
_pending_camera_index = None
_camera_values = [label for _, label in cameras]

cap = cv2.VideoCapture(selected_index)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

_actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
_actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera {selected_index}: {_actual_w}x{_actual_h}")

root = tk.Tk()
root.title("Pose Estimation")
root.geometry(f"{_actual_w + SIDEBAR_WIDTH}x{_actual_h}")
root.configure(bg="#1a1a1a")
root.resizable(False, False)
root.protocol("WM_DELETE_WINDOW", on_closing)

_running = True

frame_buffer = deque(maxlen=30)
prediction_buffer = deque(maxlen=10)

IDLE_THRESHOLD = 0.03      # mean std below this → idle (tune up if slow exercises trigger idle)
IDLE_CONFIRM_COUNT = 2     # consecutive idle windows required before switching to idle
_idle_count = 0

fps_times = deque(maxlen=30)
frame_counter = 0

left_panel = tk.Frame(root, width=_actual_w, height=_actual_h, bg="#1a1a1a")
left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
left_panel.pack_propagate(False)

camera_label = tk.Label(left_panel, bg="#1a1a1a")
camera_label.pack(expand=True, fill=tk.BOTH)

right_panel = tk.Frame(root, width=SIDEBAR_WIDTH, height=_actual_h, bg="#212121")
right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
right_panel.pack_propagate(False)

sidebar = tk.Frame(right_panel, bg="#212121", padx=20, pady=20)
sidebar.pack(fill=tk.BOTH, expand=True)

camera_frame = tk.Frame(sidebar, bg="#212121")
camera_frame.pack(fill=tk.X, pady=(0, 4))

camera_selector_label = tk.Label(
    camera_frame, text="CAMERA", font=("Helvetica", 11),
    fg="#888888", bg="#212121", anchor="w"
)
camera_selector_label.pack(side=tk.LEFT)

camera_selector = ttk.Combobox(
    camera_frame, values=_camera_values,
    state="readonly", width=12
)
camera_selector.set(f"Camera {_current_camera_index}")
camera_selector.pack(side=tk.RIGHT)
camera_selector.bind("<<ComboboxSelected>>", lambda e: _on_camera_selected())

camera_status_label = tk.Label(
    sidebar, text="", font=("Helvetica", 10),
    fg="#FF6E40", bg="#212121", anchor="w"
)
camera_status_label.pack(fill=tk.X, pady=(0, 12))


def _update_layout(w, h):
    root.geometry(f"{w + SIDEBAR_WIDTH}x{h}")
    left_panel.config(width=w, height=h)
    right_panel.config(height=h)
    camera_label.config(width=w, height=h)


def _on_camera_selected():
    global _pending_camera_index
    selection = camera_selector.get()
    try:
        new_idx = int(selection.split()[-1])
    except (ValueError, IndexError):
        return
    if new_idx != _current_camera_index:
        _pending_camera_index = new_idx


def _attempt_camera_switch():
    global cap, _current_camera_index, _pending_camera_index, _actual_w, _actual_h
    if _pending_camera_index is None:
        return

    new_index = _pending_camera_index
    _pending_camera_index = None

    if new_index == _current_camera_index:
        return

    new_cap = cv2.VideoCapture(new_index)
    if not new_cap.isOpened():
        camera_status_label.config(text=f"Camera {new_index} unavailable")
        camera_selector.set(f"Camera {_current_camera_index}")
        return

    cap.release()
    cap = new_cap
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    _current_camera_index = new_index
    _actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    _actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {new_index}: {_actual_w}x{_actual_h}")
    camera_status_label.config(text="")
    _update_layout(_actual_w, _actual_h)


exercise_header = tk.Label(
    sidebar, text="EXERCISE", font=("Helvetica", 11),
    fg="#888888", bg="#212121", anchor="w"
)
exercise_header.pack(fill=tk.X)

exercise_label = tk.Label(
    sidebar, text="Detecting...", font=("Helvetica", 22, "bold"),
    fg="#FFFFFF", bg="#212121", anchor="w"
)
exercise_label.pack(fill=tk.X)

add_spacer(sidebar, 16)

quality_header = tk.Label(
    sidebar, text="QUALITY", font=("Helvetica", 11),
    fg="#888888", bg="#212121", anchor="w"
)
quality_header.pack(fill=tk.X)

quality_badge = tk.Label(
    sidebar, text="WAITING", font=("Helvetica", 16, "bold"),
    fg="#888888", bg="#333333", padx=16, pady=8, anchor="w"
)
quality_badge.pack(fill=tk.X)

add_spacer(sidebar, 16)

confidence_header = tk.Label(
    sidebar, text="CONFIDENCE", font=("Helvetica", 11),
    fg="#888888", bg="#212121", anchor="w"
)
confidence_header.pack(fill=tk.X)

confidence_canvas = tk.Canvas(
    sidebar, width=320, height=24,
    bg="#212121", highlightthickness=0
)
confidence_canvas.pack(fill=tk.X, pady=(4, 0))

confidence_label = tk.Label(
    sidebar, text="0%", font=("Helvetica", 13),
    fg="#CCCCCC", bg="#212121", anchor="w"
)
confidence_label.pack(fill=tk.X, pady=(2, 0))

add_spacer(sidebar, 24)

keypoints_header = tk.Label(
    sidebar, text="KEYPOINTS", font=("Helvetica", 11),
    fg="#888888", bg="#212121", anchor="w"
)
keypoints_header.pack(fill=tk.X)

keypoints_label = tk.Label(
    sidebar, text="0 / 33 detected", font=("Helvetica", 14),
    fg="#FFFFFF", bg="#212121", anchor="w"
)
keypoints_label.pack(fill=tk.X)

add_spacer(sidebar, 24)

fps_header = tk.Label(
    sidebar, text="FPS", font=("Helvetica", 11),
    fg="#888888", bg="#212121", anchor="w"
)
fps_header.pack(fill=tk.X)

fps_label = tk.Label(
    sidebar, text="0", font=("Helvetica", 14),
    fg="#FFFFFF", bg="#212121", anchor="w"
)
fps_label.pack(fill=tk.X)

quit_btn = tk.Button(
    sidebar, text="QUIT", width=20,
    bg="#333333", fg="#FFFFFF", command=on_closing
)
quit_btn.pack(side=tk.BOTTOM, pady=(20, 0))

end_session_btn = tk.Button(
    sidebar, text="End Session", width=20,
    bg="#B71C1C", fg="#FFFFFF", command=_end_session
)
end_session_btn.pack(side=tk.BOTTOM, pady=(20, 0))


def draw_confidence_bar(confidence):
    confidence_canvas.delete("all")
    confidence_canvas.create_rectangle(0, 0, 320, 24, fill="#333333", outline="")
    fill_w = int(confidence * 320)
    if confidence >= 0.75:
        color = "#69F0AE"
    elif confidence >= 0.50:
        color = "#FFD740"
    else:
        color = "#FF6E40"
    if fill_w > 0:
        confidence_canvas.create_rectangle(0, 0, fill_w, 24, fill=color, outline="")


def update():
    global frame_counter, _idle_count

    if not _running:
        root.after(33, update)
        return

    _attempt_camera_switch()

    ret, raw_frame = cap.read()
    if not ret:
        root.after(33, update)
        return

    now = time.time()
    fps_times.append(now)

    timestamp_ms = int(now * 1000)
    pose_est.submit_frame(raw_frame, timestamp_ms)
    lm = pose_est.latest_landmarks

    display_frame = raw_frame.copy()
    disp_h, disp_w = display_frame.shape[:2]

    if lm is not None:
        pose_est.draw_landmarks(display_frame, lm)

    visible = PoseEstimator.count_visible(lm, threshold=0.5)

    if lm is not None:
        joints = PoseEstimator.extract_mapped_joints(lm)
        frame_buffer.append(joints)

    frame_counter += 1
    if len(frame_buffer) == 30 and frame_counter % 5 == 0:
        window = np.array(frame_buffer, dtype=np.float32)
        angles = batch_keypoints_to_angles(window)
        motion = compute_motion(angles)

        prev_idle = _idle_count
        if motion < IDLE_THRESHOLD:
            _idle_count = min(_idle_count + 1, IDLE_CONFIRM_COUNT)
        else:
            _idle_count = 0

        if _idle_count >= IDLE_CONFIRM_COUNT:
            if prev_idle < IDLE_CONFIRM_COUNT:  # just became idle
                prediction_buffer.clear()
                while not _classifier_output_queue.empty():
                    try:
                        _classifier_output_queue.get_nowait()
                    except queue.Empty:
                        break
            exercise_label.config(text="Idle")
            quality_badge.config(text="WAITING", fg="#888888", bg="#333333")
            draw_confidence_bar(0.0)
            confidence_label.config(text="0%")
        else:
            window_in = angles[np.newaxis, :, :]
            if _classifier_input_queue.empty():
                try:
                    _classifier_input_queue.put_nowait(window_in)
                except queue.Full:
                    pass

    if _idle_count < IDLE_CONFIRM_COUNT and not _classifier_output_queue.empty():
        try:
            result = _classifier_output_queue.get_nowait()
            stable_exercise_idx, stable_quality_idx, stable_confidence = result
            prediction_buffer.append(result)
        except queue.Empty:
            result = None

        if result is not None and len(prediction_buffer) >= 5:
            exercise_indices = [p[0] for p in prediction_buffer]
            quality_indices = [p[1] for p in prediction_buffer]
            confidences = [p[2] for p in prediction_buffer]

            stable_exercise_idx = max(set(exercise_indices), key=exercise_indices.count)
            stable_quality_idx = 1 if sum(quality_indices) > len(quality_indices) / 2 else 0
            stable_confidence = sum(confidences) / len(confidences)

            ex_name = EXERCISE_NAMES.get(stable_exercise_idx, "Unknown")
            exercise_label.config(text=ex_name)

            if stable_quality_idx == 1:
                quality_badge.config(text="CORRECT", fg="#69F0AE", bg="#1B5E20")
            else:
                quality_badge.config(text="INCORRECT", fg="#FF8A80", bg="#B71C1C")

            draw_confidence_bar(stable_confidence)
            confidence_label.config(text=f"{int(stable_confidence * 100)}%")

            exercise_name = EXERCISE_NAMES.get(stable_exercise_idx, "Unknown")
            tts.update(exercise_name, stable_quality_idx, time.time())
            logger.log_frame(stable_exercise_idx, stable_quality_idx, stable_confidence)

    keypoints_label.config(text=f"{visible} / 33 detected")

    if len(fps_times) > 1:
        fps = len(fps_times) / (fps_times[-1] - fps_times[0])
    else:
        fps = 0.0
    fps_label.config(text=f"{fps:.1f}")

    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    imgtk = ImageTk.PhotoImage(image=pil_img)
    camera_label.imgtk = imgtk
    camera_label.config(image=imgtk)

    root.after(10, update)


root.after(0, update)
root.mainloop()

cap.release()
