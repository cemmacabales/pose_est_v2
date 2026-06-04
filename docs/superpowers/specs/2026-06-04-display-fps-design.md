# Display Pipeline + Camera Capture FPS Design

**Date:** 2026-06-04
**Status:** Approved
**Goal:** Improve person-present FPS on Raspberry Pi 5 from 15 FPS to 25-30 FPS

---

## Problem

After the BlazePose LIVE_STREAM change (see `2026-06-04-blazepose-livestream-design.md`), the main loop now runs at 20-25 FPS without a person present, but drops to 15 FPS when a person is detected. Two remaining bottlenecks:

1. **Display pipeline at 1080p** — `draw_landmarks()` (33 circles + 35 lines via OpenCV), `cv2.cvtColor()`, `Image.fromarray()`, and `ImageTk.PhotoImage()` all operate on a full 1920×1080 frame. Combined cost: ~25-35ms per frame.
2. **`cap.read()` blocking** — the main loop stalls waiting for the camera to deliver the next frame (~33ms at 30fps). A camera capture thread eliminates this.

Display downscaling does **not** affect classification accuracy. BlazePose receives `raw_frame` at full 1080p; only `display_frame` is resized.

---

## Solution

Two independent changes to `gui.py`, both targeting the main loop:

### Change 1: 720p Display Pipeline

Resize `raw_frame` to 720p immediately after capture, before any display operation. All downstream rendering — skeleton drawing, color conversion, PIL, ImageTk — operates on the smaller frame. `raw_frame` remains full resolution for BlazePose.

```
cap.read() → raw_frame (1080p)
                ├─→ submit_frame(raw_frame)    [BlazePose, full res, unchanged]
                └─→ display_frame = resize(raw_frame, 1280×720)
                        ├─→ draw_landmarks()   [on 720p]
                        ├─→ cv2.cvtColor()     [on 720p]
                        ├─→ Image.fromarray()  [on 720p]
                        └─→ ImageTk.PhotoImage()[on 720p]
```

A single constant `_DISPLAY_W, _DISPLAY_H = 1280, 720` controls display dimensions. The Tkinter window geometry and left panel size use these constants.

### Change 2: Camera Capture Thread

A daemon background thread runs `cap.read()` continuously and stores the latest frame in `_latest_frame` (protected by `_frame_lock`). The main loop reads the shared variable instantly rather than blocking on the camera.

Same producer/consumer pattern as the existing classifier thread.

---

## Component Changes

### `gui.py` only — no other files touched

**New constants (near top of file):**
```python
_DISPLAY_W, _DISPLAY_H = 1280, 720
```

**New module-level state:**
```python
_latest_frame = None
_frame_lock = threading.Lock()
```

**New capture thread:**
```python
def _capture_worker():
    while True:
        ret, frame = cap.read()
        if ret:
            with _frame_lock:
                global _latest_frame
                _latest_frame = frame

_capture_thread = threading.Thread(target=_capture_worker, daemon=True)
_capture_thread.start()
```

**`update()` changes:**
- Replace `ret, raw_frame = cap.read()` with a non-blocking read from `_latest_frame`
- Add `display_frame = cv2.resize(raw_frame, (_DISPLAY_W, _DISPLAY_H))` before any display operation
- Pass `raw_frame` to `submit_frame()`, `display_frame` to everything else

**Window geometry:**
- Initial geometry: `_DISPLAY_W + SIDEBAR_WIDTH` × `_DISPLAY_H`
- `_update_layout()` uses `_DISPLAY_W`, `_DISPLAY_H` instead of camera-reported resolution

---

## Edge Cases

| Case | Behaviour |
|------|-----------|
| Startup: `_latest_frame` is `None` | `update()` skips and reschedules (`root.after(10, update)`) until first frame arrives |
| Camera switch | Capture thread automatically reads from new `cap`; one stale frame possible — imperceptible |
| Thread cleanup | Capture thread is `daemon=True` — exits automatically with the process, no explicit shutdown needed |

---

## Out of Scope

- Skeleton throttling (every N frames) — not needed if display downscaling achieves target FPS
- Changes to `pose_estimator.py`, `joint_angles.py`, or any other file
- Resolution configurability (720p is hardcoded — no CLI flag needed)
