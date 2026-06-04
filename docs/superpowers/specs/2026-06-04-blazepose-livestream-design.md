# BlazePose LIVE_STREAM Mode — Performance Design

**Date:** 2026-06-04
**Status:** Approved
**Goal:** Improve real-time FPS on Raspberry Pi 5 from 10-15 FPS to 25-30 FPS

---

## Problem

BlazePose (`pose_est.process_frame()`) runs synchronously in the Tkinter main loop, blocking the UI thread on every frame until inference completes. On RPi 5 with the lite model, each inference takes ~40-70ms, directly capping FPS at 14-25.

Prior attempts that did **not** help:
- Downscaling BlazePose input to 480p
- Downscaling the display frame
- Reducing classifier frequency from every 5th to every 8th frame

This rules out resolution and display rendering as bottlenecks. The synchronous inference block is the root cause.

---

## Solution: MediaPipe LIVE_STREAM Mode

Switch BlazePose from `VIDEO` (synchronous, blocking) to `LIVE_STREAM` (async, callback-based). MediaPipe manages its own internal inference thread. The main loop submits frames fire-and-forget and reads back the latest landmarks from the previous inference.

### Data Flow

```
Tkinter main loop                     MediaPipe internal thread
─────────────────                     ─────────────────────────
cap.read() → raw_frame
pose_est.submit_frame(frame, ts) ──►  detect_async() runs...
lm = pose_est.latest_landmarks   ◄──  callback stores result
draw skeleton + display frame
schedule root.after(10, update)
```

Landmarks are at most 1 frame stale — imperceptible at 25+ FPS.

---

## Component Changes

### `pose_estimator.py`

- Add `running_mode="live_stream"` support in `__init__`
- Wire an internal `_result_callback` that stores `result.pose_landmarks[0]` (or `None`) into `self.latest_landmarks`
- Add `submit_frame(frame, timestamp_ms)` method calling `detect_async()` — returns immediately
- Track `_last_submitted_ts` to guard against duplicate timestamps (LIVE_STREAM raises on non-monotonic timestamps)

`self.latest_landmarks` is a plain Python attribute. CPython GIL makes single reference assignment atomic — no explicit lock needed.

### `gui.py`

- Change `PoseEstimator` init: `running_mode="live_stream"`
- In `update()`, replace:
  ```python
  lm = pose_est.process_frame(raw_frame, timestamp_ms=timestamp_ms)
  ```
  with:
  ```python
  pose_est.submit_frame(raw_frame, timestamp_ms)
  lm = pose_est.latest_landmarks
  ```
- No other changes — classifier thread, joint angles, TTS, session logger all unaffected

---

## Edge Cases

| Case | Behavior |
|------|----------|
| Startup (no inference completed yet) | `latest_landmarks` is `None`; existing `lm is None` guards handle this everywhere |
| Duplicate timestamp (two frames in same ms) | Skip `detect_async()` call; reuse previous landmarks; frame still displays |

---

## Out of Scope

- Classifier threading (already done)
- Camera capture threading (no evidence camera is the bottleneck)
- Model complexity changes
- Resolution changes
