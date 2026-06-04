# Display Pipeline + Camera Capture FPS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve person-present FPS on Raspberry Pi 5 from 15 to 25-30 by moving camera capture off the main thread and reducing display pipeline cost from 1080p to 720p.

**Architecture:** A daemon capture thread runs `cap.read()` continuously and stores frames in `_latest_frame`; `update()` reads this shared variable non-blocking. `raw_frame` (1080p) goes to BlazePose unchanged; `display_frame = cv2.resize(raw_frame, (1280, 720))` is used for all rendering. No changes outside `gui.py` and its integration test.

**Tech Stack:** Python `threading.Lock`, OpenCV `cv2.resize`, Tkinter, existing `threading` import in `gui.py`.

---

## File Map

| File | Change |
|------|--------|
| `gui.py` | Add `_DISPLAY_W/_DISPLAY_H` constants, `_latest_frame`/`_frame_lock`, `_capture_worker` thread, modify `update()` and window layout |
| `tests/test_gui_integration.py` | Add `cv2.resize` mock, expose `cv2` mock in `_test_mocks`, set `_latest_frame` in helper, add 3 new tests |

---

## Task 1: Camera Capture Thread

**Files:**
- Modify: `gui.py`
- Modify: `tests/test_gui_integration.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_gui_integration.py`, add `mock_cv2.resize.side_effect` and expose `cv2` in `_test_mocks`. Find the `mock_cv2` block (around line 77) and add one line after `mock_cv2.COLOR_BGR2RGB = 4`:

```python
    mock_cv2.resize.side_effect = lambda img, size: img
```

Find the `gui._test_mocks = {` block (around line 131) and add `"cv2": mock_cv2` to it:

```python
        gui._test_mocks = {
            "tts_class": mock_tts_class,
            "tts_instance": mock_tts_instance,
            "logger_class": mock_logger_class,
            "logger_instance": mock_logger_instance,
            "start_server": mock_start_server,
            "qrcode": mock_qrcode,
            "qr_img": mock_qr_img,
            "pose_estimator_cls": mock_pose_estimator_cls,
            "pose_estimator_inst": mock_pose_estimator_inst,
            "cv2": mock_cv2,
        }
```

In `_run_one_inference_iteration`, add one line before `gui_mod.update()` to seed `_latest_frame` (the capture thread may not have run yet in tests):

```python
def _run_one_inference_iteration(gui_mod):
    gui_mod._latest_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    dummy_frame = np.zeros((12, 2), dtype=np.float32)
    for _ in range(30):
        gui_mod.frame_buffer.append(dummy_frame)

    for _ in range(5):
        gui_mod.prediction_buffer.append((0, 1, 0.9))

    gui_mod._classifier_output_queue.put((0, 1, 0.9))

    gui_mod.frame_counter = 4

    gui_mod.tts.update.reset_mock()
    gui_mod.logger.log_frame.reset_mock()

    gui_mod.update()
```

Add two new tests at the bottom of the file:

```python
def test_capture_thread_is_alive(gui_mod):
    assert gui_mod._capture_thread.is_alive()


def test_update_skips_when_no_frame_available(gui_mod):
    gui_mod._latest_frame = None
    gui_mod.tts.update.reset_mock()
    gui_mod.update()
    gui_mod.tts.update.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source .venv/bin/activate
pytest tests/test_gui_integration.py::test_capture_thread_is_alive tests/test_gui_integration.py::test_update_skips_when_no_frame_available -v
```

Expected: both FAIL — `gui` module has no `_capture_thread` or `_latest_frame` yet.

- [ ] **Step 3: Add the capture thread to gui.py**

After line 197 (`print(f"Camera {selected_index}: {_actual_w}x{_actual_h}")`), add:

```python
_latest_frame = None
_frame_lock = threading.Lock()


def _capture_worker():
    global _latest_frame
    while True:
        ret, frame = cap.read()
        if ret:
            with _frame_lock:
                _latest_frame = frame


_capture_thread = threading.Thread(target=_capture_worker, daemon=True)
_capture_thread.start()
```

- [ ] **Step 4: Replace `cap.read()` in `update()` with non-blocking read**

In `update()`, replace lines 409-412:

```python
    ret, raw_frame = cap.read()
    if not ret:
        root.after(33, update)
        return
```

with:

```python
    with _frame_lock:
        raw_frame = _latest_frame
    if raw_frame is None:
        root.after(10, update)
        return
```

- [ ] **Step 5: Run new tests to verify they pass**

```bash
pytest tests/test_gui_integration.py::test_capture_thread_is_alive tests/test_gui_integration.py::test_update_skips_when_no_frame_available -v
```

Expected: both PASS.

- [ ] **Step 6: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all 58 tests pass.

- [ ] **Step 7: Commit**

```bash
git add gui.py tests/test_gui_integration.py
git commit -m "perf: move camera capture to background thread, non-blocking update loop"
```

---

## Task 2: 720p Display Pipeline

**Files:**
- Modify: `gui.py`
- Modify: `tests/test_gui_integration.py`

- [ ] **Step 1: Write the failing tests**

Add two new tests at the bottom of `tests/test_gui_integration.py`:

```python
def test_display_dimensions_are_720p(gui_mod):
    assert gui_mod._DISPLAY_W == 1280
    assert gui_mod._DISPLAY_H == 720


def test_update_resizes_frame_for_display(gui_mod):
    _run_one_inference_iteration(gui_mod)
    gui_mod._test_mocks["cv2"].resize.assert_called()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source .venv/bin/activate
pytest tests/test_gui_integration.py::test_display_dimensions_are_720p tests/test_gui_integration.py::test_update_resizes_frame_for_display -v
```

Expected: both FAIL — no `_DISPLAY_W`/`_DISPLAY_H` or `cv2.resize` call yet.

- [ ] **Step 3: Add display dimension constants**

In `gui.py`, after line 99 (`SIDEBAR_WIDTH = 380`), add:

```python
_DISPLAY_W = 1280
_DISPLAY_H = 720
```

- [ ] **Step 4: Update window geometry and layout to use display constants**

Replace line 201:
```python
root.geometry(f"{_actual_w + SIDEBAR_WIDTH}x{_actual_h}")
```
with:
```python
root.geometry(f"{_DISPLAY_W + SIDEBAR_WIDTH}x{_DISPLAY_H}")
```

Replace line 215:
```python
left_panel = tk.Frame(root, width=_actual_w, height=_actual_h, bg="#1a1a1a")
```
with:
```python
left_panel = tk.Frame(root, width=_DISPLAY_W, height=_DISPLAY_H, bg="#1a1a1a")
```

Replace line 222:
```python
right_panel = tk.Frame(root, width=SIDEBAR_WIDTH, height=_actual_h, bg="#212121")
```
with:
```python
right_panel = tk.Frame(root, width=SIDEBAR_WIDTH, height=_DISPLAY_H, bg="#212121")
```

Replace line 296 (inside `_attempt_camera_switch`):
```python
    _update_layout(_actual_w, _actual_h)
```
with:
```python
    _update_layout(_DISPLAY_W, _DISPLAY_H)
```

- [ ] **Step 5: Resize display_frame in update()**

In `update()`, replace lines 421-422:
```python
    display_frame = raw_frame.copy()
    disp_h, disp_w = display_frame.shape[:2]
```
with:
```python
    display_frame = cv2.resize(raw_frame, (_DISPLAY_W, _DISPLAY_H))
```

- [ ] **Step 6: Run new tests to verify they pass**

```bash
pytest tests/test_gui_integration.py::test_display_dimensions_are_720p tests/test_gui_integration.py::test_update_resizes_frame_for_display -v
```

Expected: both PASS.

- [ ] **Step 7: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all 60 tests pass.

- [ ] **Step 8: Commit**

```bash
git add gui.py tests/test_gui_integration.py
git commit -m "perf: downscale display frame to 720p, reduce PIL/Tkinter rendering cost"
```

---

## Verification on RPi 5

After deploying (`git pull` on RPi 5, then `git checkout feat/improve-fps`):

```bash
source .venv/bin/activate
python gui.py --model lite
```

Expected:
- FPS without person: 25-30 (up from 20-25)
- FPS with person present: 25-30 (up from 15)
- Skeleton overlay visible at 720p display resolution
- Exercise classification and quality scoring unchanged
