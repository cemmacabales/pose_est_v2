# Idle Detection via Motion Variance Threshold — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show "Idle" and skip exercise classification whenever joint motion is below a tunable threshold, suppressing phantom detections during stillness and micro-pauses.

**Architecture:** Add `compute_motion(angles)` to `joint_angles.py` (mean per-feature std deviation across 30 frames). In `gui.py`, gate the classifier push: if motion is below `IDLE_THRESHOLD` for `IDLE_CONFIRM_COUNT` consecutive windows, flush stale buffers and display "Idle" in the UI. TDD throughout — `joint_angles.py` is importable without side effects so it is fully unit-testable.

**Tech Stack:** NumPy, Python 3, tkinter, existing TFLite classifier pipeline

---

## File Map

| File | Change |
|------|--------|
| `joint_angles.py` | Add `compute_motion(angles)` helper |
| `tests/test_joint_angles.py` | Create — unit tests for `compute_motion` |
| `gui.py` | Import `compute_motion`; add constants + state; gate classifier push; idle UI update |

---

## Task 1: Add `compute_motion()` to `joint_angles.py` (TDD)

**Files:**
- Create: `tests/test_joint_angles.py`
- Modify: `joint_angles.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_joint_angles.py`:

```python
import numpy as np
import pytest
from joint_angles import compute_motion, batch_keypoints_to_angles

IDLE_THRESHOLD = 0.03


def test_compute_motion_zero_for_constant_input():
    # 30 identical frames — no motion at all
    base = np.ones((12, 2), dtype=np.float32) * 0.1
    window = np.tile(base, (30, 1, 1))          # (30, 12, 2)
    angles = batch_keypoints_to_angles(window)
    assert compute_motion(angles) == pytest.approx(0.0, abs=1e-6)


def test_compute_motion_positive_for_varying_input():
    rng = np.random.default_rng(42)
    window = rng.uniform(-0.5, 0.5, size=(30, 12, 2)).astype(np.float32)
    angles = batch_keypoints_to_angles(window)
    assert compute_motion(angles) > 0.0


def test_compute_motion_below_threshold_for_still_pose():
    # Tiny jitter around a fixed pose — should read as idle
    rng = np.random.default_rng(0)
    base = rng.uniform(-0.3, 0.3, size=(12, 2)).astype(np.float32)
    noise = np.random.default_rng(1).normal(0, 0.001, size=(30, 12, 2)).astype(np.float32)
    window = (base[np.newaxis] + noise).astype(np.float32)
    angles = batch_keypoints_to_angles(window)
    assert compute_motion(angles) < IDLE_THRESHOLD


def test_compute_motion_above_threshold_for_active_motion():
    # Simulate a knee-flexion exercise with large amplitude
    t = np.linspace(0, np.pi, 30)
    window = np.zeros((30, 12, 2), dtype=np.float32)
    window[:, 8, 1] = (np.sin(t) * 0.4).astype(np.float32)   # left knee y
    window[:, 9, 1] = (np.sin(t) * 0.4).astype(np.float32)   # right knee y
    angles = batch_keypoints_to_angles(window)
    assert compute_motion(angles) > IDLE_THRESHOLD


def test_compute_motion_returns_scalar():
    window = np.zeros((30, 12, 2), dtype=np.float32)
    angles = batch_keypoints_to_angles(window)
    result = compute_motion(angles)
    assert isinstance(result, float)
```

- [ ] **Step 2: Run tests — expect ImportError on `compute_motion`**

```bash
cd /Users/cemmacabales/pose_est_v2 && python -m pytest tests/test_joint_angles.py -v
```

Expected output contains: `ImportError: cannot import name 'compute_motion'`

- [ ] **Step 3: Add `compute_motion` to `joint_angles.py`**

Append to the bottom of `joint_angles.py`:

```python


def compute_motion(angles: np.ndarray) -> float:
    """Mean per-feature std deviation across frames. Near 0 = idle, higher = active."""
    return float(np.std(angles, axis=0).mean())
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
cd /Users/cemmacabales/pose_est_v2 && python -m pytest tests/test_joint_angles.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add joint_angles.py tests/test_joint_angles.py
git commit -m "feat: add compute_motion helper to joint_angles"
```

---

## Task 2: Integrate idle gating into `gui.py`

**Files:**
- Modify: `gui.py`

### Step 2a — Update import line

- [ ] **Step 1: Update the `joint_angles` import**

In `gui.py`, find this line (near the top of the file):

```python
from joint_angles import batch_keypoints_to_angles
```

Replace with:

```python
from joint_angles import batch_keypoints_to_angles, compute_motion
```

### Step 2b — Add constants and state variable

- [ ] **Step 2: Add idle constants and counter after the `prediction_buffer` deque**

In `gui.py`, find:

```python
frame_buffer = deque(maxlen=30)
prediction_buffer = deque(maxlen=10)
```

Replace with:

```python
frame_buffer = deque(maxlen=30)
prediction_buffer = deque(maxlen=10)

IDLE_THRESHOLD = 0.03      # mean std below this → idle (tune up if slow exercises trigger idle)
IDLE_CONFIRM_COUNT = 2     # consecutive idle windows required before switching to idle
_idle_count = 0
```

### Step 2c — Declare `_idle_count` as global in `update()`

- [ ] **Step 3: Add `_idle_count` to the global declaration inside `update()`**

In `gui.py`, inside `def update():`, find:

```python
    global frame_counter
```

Replace with:

```python
    global frame_counter, _idle_count
```

### Step 2d — Replace the classifier-push block with idle-gated version

- [ ] **Step 4: Replace the window/classifier push block**

In `gui.py`, find this block inside `update()`:

```python
    frame_counter += 1
    if len(frame_buffer) == 30 and frame_counter % 5 == 0:
        window = np.array(frame_buffer, dtype=np.float32)
        angles = batch_keypoints_to_angles(window)
        window = angles[np.newaxis, :, :]
        if _classifier_input_queue.empty():
            try:
                _classifier_input_queue.put_nowait(window)
            except queue.Full:
                pass
```

Replace with:

```python
    frame_counter += 1
    if len(frame_buffer) == 30 and frame_counter % 5 == 0:
        window = np.array(frame_buffer, dtype=np.float32)
        angles = batch_keypoints_to_angles(window)
        motion = compute_motion(angles)

        if motion < IDLE_THRESHOLD:
            _idle_count += 1
        else:
            _idle_count = 0

        if _idle_count >= IDLE_CONFIRM_COUNT:
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
```

- [ ] **Step 5: Commit**

```bash
git add gui.py
git commit -m "feat: suppress classifier during idle via motion variance gate"
```

---

## Task 3: Manual Verification

- [ ] **Step 1: Run the app**

```bash
cd /Users/cemmacabales/pose_est_v2 && python gui.py
```

- [ ] **Step 2: Verify idle state**

Stand still in front of the camera. After ~1 second (2 windows × 5-frame stride), the exercise label should change to **"Idle"**, the quality badge should show **"WAITING"** (grey), and the confidence bar should clear to 0%.

- [ ] **Step 3: Verify exercise detection resumes**

Begin performing any exercise (e.g., Deep Squat). Within ~0.5 seconds of starting motion, the classifier should resume and show the predicted exercise name and quality.

- [ ] **Step 4: Verify micro-pause suppression**

Pause mid-exercise for 2+ seconds. The label should revert to "Idle". Resume the exercise and confirm it re-detects.

- [ ] **Step 5: Tune threshold if needed**

If slow exercises (e.g., a very slow Sit to Stand) are incorrectly triggering idle: lower `IDLE_THRESHOLD` in `gui.py` to `0.02`.

If standing still still triggers an exercise prediction: raise `IDLE_THRESHOLD` to `0.04` or `0.05`.

---

## Tuning Reference

| Symptom | Fix |
|---------|-----|
| Slow exercises classified as idle | Lower `IDLE_THRESHOLD` (e.g. `0.02`) |
| Standing still still shows exercise | Raise `IDLE_THRESHOLD` (e.g. `0.05`) |
| Flickering between idle/active | Raise `IDLE_CONFIRM_COUNT` to `3` or `4` |
| Sluggish return to idle after stillness | Lower `IDLE_CONFIRM_COUNT` to `1` |
