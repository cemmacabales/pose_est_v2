#!/usr/bin/env python3
"""
gui.py — Real-time pose estimation and exercise classification GUI for Raspberry Pi 5.
"""

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

from joint_map import VICON_TO_MOVENET, MOVENET_JOINT_NAMES, EXERCISE_NAMES, COCO_SKELETON_EDGES

from tts_engine import TTSEngine
from session_logger import SessionLogger
from session_chat.app import start_server
import qrcode
import socket
import threading

# ── Load models (before window appears) ─────────────────────────────────────
print("Loading MoveNet...")
movenet = Interpreter(model_path="./models/movenet_thunder_int8.tflite")
movenet.allocate_tensors()

print("Loading classifier...")
classifier = Interpreter(model_path="./models/classifier.tflite")
classifier.allocate_tensors()

movenet_in = movenet.get_input_details()
movenet_out = movenet.get_output_details()

classifier_in = classifier.get_input_details()
classifier_out = classifier.get_output_details()

# Determine which classifier output is exercise / quality by shape
exercise_out_idx = None
quality_out_idx = None
for idx, detail in enumerate(classifier_out):
    if detail["shape"][1] == 10:
        exercise_out_idx = idx
    elif detail["shape"][1] == 2:
        quality_out_idx = idx

# Initialize TTS and session logger
tts = TTSEngine()
logger = SessionLogger()
tts.announce_session_start()

_session_ended = False

# ── Tkinter setup ───────────────────────────────────────────────────────────
root = tk.Tk()
root.title("Pose Estimation")
root.geometry("1100x620")
root.configure(bg="#1a1a1a")
root.resizable(False, False)


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


root.protocol("WM_DELETE_WINDOW", on_closing)

# ── Camera selection ────────────────────────────────────────────────────────
cameras = detect_cameras()

if not cameras:
    import tkinter.messagebox as mb
    mb.showerror("No Camera", "No cameras detected. Please connect a camera and restart.")
    exit(1)

selected_index = cameras[0][0]
_current_camera_index = selected_index
_pending_camera_index = None
_camera_values = [label for _, label in cameras]

# ── Camera ──────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(selected_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

_running = True

# ── Rolling buffer ──────────────────────────────────────────────────────────
frame_buffer = deque(maxlen=30)
prediction_buffer = deque(maxlen=10)
MAPPED_INDICES = sorted(VICON_TO_MOVENET.values())

# ── FPS ─────────────────────────────────────────────────────────────────────
fps_times = deque(maxlen=30)
frame_counter = 0

# Left panel (camera feed)
left_panel = tk.Frame(root, width=720, height=620, bg="#1a1a1a")
left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
left_panel.pack_propagate(False)

camera_label = tk.Label(left_panel, bg="#1a1a1a")
camera_label.pack(expand=True, fill=tk.BOTH)

# Right panel (stats sidebar)
right_panel = tk.Frame(root, width=380, height=620, bg="#212121")
right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
right_panel.pack_propagate(False)

sidebar = tk.Frame(right_panel, bg="#212121", padx=20, pady=20)
sidebar.pack(fill=tk.BOTH, expand=True)

# 0. CAMERA selector
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
    global cap, _current_camera_index, _pending_camera_index
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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    _current_camera_index = new_index
    camera_status_label.config(text="")


# 1. EXERCISE header
exercise_header = tk.Label(
    sidebar, text="EXERCISE", font=("Helvetica", 11),
    fg="#888888", bg="#212121", anchor="w"
)
exercise_header.pack(fill=tk.X)

# 2. Exercise name
exercise_label = tk.Label(
    sidebar, text="Detecting...", font=("Helvetica", 22, "bold"),
    fg="#FFFFFF", bg="#212121", anchor="w"
)
exercise_label.pack(fill=tk.X)

# 3. Spacer 16px
add_spacer(sidebar, 16)

# 4. QUALITY header
quality_header = tk.Label(
    sidebar, text="QUALITY", font=("Helvetica", 11),
    fg="#888888", bg="#212121", anchor="w"
)
quality_header.pack(fill=tk.X)

# 5. Quality badge
quality_badge = tk.Label(
    sidebar, text="● WAITING", font=("Helvetica", 16, "bold"),
    fg="#888888", bg="#333333", padx=16, pady=8, anchor="w"
)
quality_badge.pack(fill=tk.X)

# 6. Spacer 16px
add_spacer(sidebar, 16)

# 7. CONFIDENCE header
confidence_header = tk.Label(
    sidebar, text="CONFIDENCE", font=("Helvetica", 11),
    fg="#888888", bg="#212121", anchor="w"
)
confidence_header.pack(fill=tk.X)

# 8. Confidence bar canvas
confidence_canvas = tk.Canvas(
    sidebar, width=320, height=24,
    bg="#212121", highlightthickness=0
)
confidence_canvas.pack(fill=tk.X, pady=(4, 0))

# 9. Confidence percentage
confidence_label = tk.Label(
    sidebar, text="0%", font=("Helvetica", 13),
    fg="#CCCCCC", bg="#212121", anchor="w"
)
confidence_label.pack(fill=tk.X, pady=(2, 0))

# 10. Spacer 24px
add_spacer(sidebar, 24)

# 11. KEYPOINTS header
keypoints_header = tk.Label(
    sidebar, text="KEYPOINTS", font=("Helvetica", 11),
    fg="#888888", bg="#212121", anchor="w"
)
keypoints_header.pack(fill=tk.X)

# 12. Keypoints count
keypoints_label = tk.Label(
    sidebar, text="0 / 17 detected", font=("Helvetica", 14),
    fg="#FFFFFF", bg="#212121", anchor="w"
)
keypoints_label.pack(fill=tk.X)

# 13. Spacer 24px
add_spacer(sidebar, 24)

# 14. FPS header
fps_header = tk.Label(
    sidebar, text="FPS", font=("Helvetica", 11),
    fg="#888888", bg="#212121", anchor="w"
)
fps_header.pack(fill=tk.X)

# 15. FPS value
fps_label = tk.Label(
    sidebar, text="0", font=("Helvetica", 14),
    fg="#FFFFFF", bg="#212121", anchor="w"
)
fps_label.pack(fill=tk.X)

# 16. QUIT button at bottom
quit_btn = tk.Button(
    sidebar, text="QUIT", width=20,
    bg="#333333", fg="#FFFFFF", command=on_closing
)
quit_btn.pack(side=tk.BOTTOM, pady=(20, 0))

# 17. End Session button
end_session_btn = tk.Button(
    sidebar, text="End Session", width=20,
    bg="#B71C1C", fg="#FFFFFF", command=_end_session
)
end_session_btn.pack(side=tk.BOTTOM, pady=(20, 0))


# ── Helper: redraw confidence bar ───────────────────────────────────────────
def draw_confidence_bar(confidence):
    confidence_canvas.delete("all")
    # Background
    confidence_canvas.create_rectangle(0, 0, 320, 24, fill="#333333", outline="")
    # Fill
    fill_w = int(confidence * 320)
    if confidence >= 0.75:
        color = "#69F0AE"
    elif confidence >= 0.50:
        color = "#FFD740"
    else:
        color = "#FF6E40"
    if fill_w > 0:
        confidence_canvas.create_rectangle(0, 0, fill_w, 24, fill=color, outline="")


# ── Main update loop ────────────────────────────────────────────────────────
def update():
    global frame_counter

    if not _running:
        root.after(33, update)
        return

    # Safe camera switch between frames
    _attempt_camera_switch()

    ret, frame = cap.read()
    if not ret:
        root.after(33, update)
        return

    # FPS timestamp
    now = time.time()
    fps_times.append(now)

    # Original dimensions
    orig_h, orig_w = frame.shape[:2]

    # MoveNet input
    mn_frame = cv2.resize(frame, (256, 256))
    mn_input = np.expand_dims(mn_frame, axis=0).astype(np.uint8)
    movenet.set_tensor(movenet_in[0]["index"], mn_input)
    movenet.invoke()
    kps = movenet.get_tensor(movenet_out[0]["index"])[0][0]  # (17, 3) [y, x, conf]

    # Resize to 640x480 for display
    display_frame = cv2.resize(frame, (640, 480))
    disp_h, disp_w = display_frame.shape[:2]

    # Count detected keypoints
    detected_count = 0
    for i in range(17):
        if kps[i][2] > 0.3:
            detected_count += 1

    # Draw skeleton edges (white, thickness 2)
    for edge in COCO_SKELETON_EDGES:
        i, j = edge
        y1, x1, c1 = kps[i]
        y2, x2, c2 = kps[j]
        if c1 > 0.3 and c2 > 0.3:
            px1 = int(x1 * disp_w)
            py1 = int(y1 * disp_h)
            px2 = int(x2 * disp_w)
            py2 = int(y2 * disp_h)
            cv2.line(display_frame, (px1, py1), (px2, py2), (255, 255, 255), 2)

    # Draw keypoints
    for i in range(17):
        y, x, conf = kps[i]
        if conf > 0.3:
            px = int(x * disp_w)
            py = int(y * disp_h)
            if i in MOVENET_JOINT_NAMES:
                # Blue #4FC3F7  -> BGR (247, 195, 79)
                cv2.circle(display_frame, (px, py), 7, (247, 195, 79), -1)
            else:
                # Gray #888888 -> BGR (136, 136, 136)
                cv2.circle(display_frame, (px, py), 4, (136, 136, 136), -1)

    # Extract 12 mapped keypoints, normalize by hip midpoint
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

    mapped_kps_arr = np.array(mapped_kps, dtype=np.float32)  # (12, 2)
    frame_buffer.append(mapped_kps_arr)

    # Run classifier every 5th frame when buffer is full
    frame_counter += 1
    if len(frame_buffer) == 30 and frame_counter % 5 == 0:
        window = np.array(frame_buffer, dtype=np.float32)  # (30, 12, 2)
        window = window.reshape(1, 30, 24)
        classifier.set_tensor(classifier_in[0]["index"], window)
        classifier.invoke()

        exercise_out = classifier.get_tensor(classifier_out[exercise_out_idx]["index"])[0]  # (10,)
        quality_out = classifier.get_tensor(classifier_out[quality_out_idx]["index"])[0]    # (2,)

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

            ex_name = EXERCISE_NAMES.get(stable_exercise_idx + 1, "Unknown")
            exercise_label.config(text=ex_name)

            if stable_quality_idx == 1:
                quality_badge.config(text="● CORRECT", fg="#69F0AE", bg="#1B5E20")
            else:
                quality_badge.config(text="● INCORRECT", fg="#FF8A80", bg="#B71C1C")

            draw_confidence_bar(stable_confidence)
            confidence_label.config(text=f"{int(stable_confidence * 100)}%")

            exercise_name = EXERCISE_NAMES.get(stable_exercise_idx + 1, "Unknown")
            tts.update(exercise_name, stable_quality_idx, time.time())
            logger.log_frame(stable_exercise_idx, stable_quality_idx, stable_confidence)

    # Update keypoints count
    keypoints_label.config(text=f"{detected_count} / 17 detected")

    # Update FPS
    if len(fps_times) > 1:
        fps = len(fps_times) / (fps_times[-1] - fps_times[0])
    else:
        fps = 0.0
    fps_label.config(text=f"{fps:.1f}")

    # Convert to PIL and resize to fill left panel
    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    pil_img = pil_img.resize((720, 620), Image.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=pil_img)
    camera_label.imgtk = imgtk
    camera_label.config(image=imgtk)

    root.after(33, update)


# Start
root.after(0, update)
root.mainloop()

# Cleanup
cap.release()
