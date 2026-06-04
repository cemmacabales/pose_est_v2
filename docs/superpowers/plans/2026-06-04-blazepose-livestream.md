# BlazePose LIVE_STREAM Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch BlazePose from synchronous VIDEO mode to async LIVE_STREAM mode so pose inference no longer blocks the Tkinter main loop, targeting 25-30 FPS on RPi 5 (currently 10-15).

**Architecture:** MediaPipe's LIVE_STREAM running mode manages its own inference thread internally. The main loop calls `submit_frame()` (fire-and-forget) and reads `latest_landmarks` set by a result callback on the previous frame. Landmarks are at most 1 frame stale — imperceptible at 25+ FPS.

**Tech Stack:** MediaPipe Tasks API (`vision.RunningMode.LIVE_STREAM`, `detect_async`), Python threading (GIL-safe attribute write), Tkinter, OpenCV.

---

## File Map

| File | Change |
|------|--------|
| `pose_estimator.py` | Add LIVE_STREAM mode: `latest_landmarks`, `_last_submitted_ts`, `_result_callback()`, `submit_frame()` |
| `gui.py` | Line 49: `running_mode="live_stream"`; lines 415-416: swap `process_frame` → `submit_frame` + `latest_landmarks` |
| `tests/test_pose_estimator_live_stream.py` | New file: unit tests for the new PoseEstimator behaviour |
| `tests/test_gui_integration.py` | Add `submit_frame` + `latest_landmarks` to pose estimator mock |

---

## Task 1: Unit-test and implement LIVE_STREAM support in PoseEstimator

**Files:**
- Create: `tests/test_pose_estimator_live_stream.py`
- Modify: `pose_estimator.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pose_estimator_live_stream.py`:

```python
from unittest.mock import MagicMock, patch
import numpy as np
import pytest


@pytest.fixture
def live_estimator():
    fake_lm = MagicMock()
    fake_lm.detect_async = MagicMock()

    with patch("pose_estimator._resolve_model_path", return_value="/fake.task"), \
         patch("pose_estimator.vision.PoseLandmarkerOptions", MagicMock()), \
         patch("pose_estimator.vision.PoseLandmarker.create_from_options", return_value=fake_lm):
        from pose_estimator import PoseEstimator
        est = PoseEstimator(running_mode="live_stream")
        est._test_landmarker = fake_lm
        return est


def test_latest_landmarks_starts_none(live_estimator):
    assert live_estimator.latest_landmarks is None


def test_last_submitted_ts_starts_negative(live_estimator):
    assert live_estimator._last_submitted_ts < 0


def test_submit_frame_advances_timestamp(live_estimator):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    live_estimator.submit_frame(frame, 1000)
    assert live_estimator._last_submitted_ts == 1000


def test_submit_frame_calls_detect_async(live_estimator):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    live_estimator.submit_frame(frame, 1000)
    assert live_estimator._test_landmarker.detect_async.call_count == 1


def test_submit_frame_skips_duplicate_timestamp(live_estimator):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    live_estimator.submit_frame(frame, 1000)
    live_estimator.submit_frame(frame, 1000)
    assert live_estimator._test_landmarker.detect_async.call_count == 1


def test_submit_frame_skips_older_timestamp(live_estimator):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    live_estimator.submit_frame(frame, 2000)
    live_estimator.submit_frame(frame, 1999)
    assert live_estimator._test_landmarker.detect_async.call_count == 1


def test_result_callback_stores_landmarks(live_estimator):
    result = MagicMock()
    fake_landmark = MagicMock()
    result.pose_landmarks = [fake_landmark]
    live_estimator._result_callback(result, MagicMock(), 1000)
    assert live_estimator.latest_landmarks is fake_landmark


def test_result_callback_stores_none_on_empty_detection(live_estimator):
    live_estimator.latest_landmarks = MagicMock()
    result = MagicMock()
    result.pose_landmarks = []
    live_estimator._result_callback(result, MagicMock(), 1000)
    assert live_estimator.latest_landmarks is None
```

- [ ] **Step 2: Run tests to verify they all fail**

```bash
source .venv/bin/activate
pytest tests/test_pose_estimator_live_stream.py -v
```

Expected: all 8 tests FAIL with `AttributeError` (no `latest_landmarks`, `submit_frame`, or `_result_callback` yet).

- [ ] **Step 3: Implement LIVE_STREAM support in pose_estimator.py**

Replace lines 44–63 (`__init__` and just before `process_frame`) with:

```python
class PoseEstimator:
    def __init__(self, model_complexity=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5, static_image_mode=False,
                 running_mode="image"):
        model_path = _resolve_model_path(model_complexity)
        if running_mode == "video":
            mode = vision.RunningMode.VIDEO
        elif running_mode == "live_stream":
            mode = vision.RunningMode.LIVE_STREAM
        else:
            mode = vision.RunningMode.IMAGE

        self.latest_landmarks = None
        self._last_submitted_ts = -1

        kwargs = dict(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mode,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        if running_mode == "live_stream":
            kwargs["result_callback"] = self._result_callback
        options = vision.PoseLandmarkerOptions(**kwargs)
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self._video_mode = (running_mode == "video")

    def _result_callback(self, result, output_image, timestamp_ms):
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            self.latest_landmarks = result.pose_landmarks[0]
        else:
            self.latest_landmarks = None

    def submit_frame(self, frame, timestamp_ms):
        """Submit a BGR frame for async LIVE_STREAM inference. Returns immediately."""
        ts = int(timestamp_ms)
        if ts <= self._last_submitted_ts:
            return
        self._last_submitted_ts = ts
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.landmarker.detect_async(mp_image, ts)
```

- [ ] **Step 4: Run tests to verify they all pass**

```bash
pytest tests/test_pose_estimator_live_stream.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
pytest tests/ -v
```

Expected: all tests pass (47+ tests).

- [ ] **Step 6: Commit**

```bash
git add tests/test_pose_estimator_live_stream.py pose_estimator.py
git commit -m "feat: add LIVE_STREAM mode to PoseEstimator (submit_frame + result callback)"
```

---

## Task 2: Update gui.py and integration tests to use LIVE_STREAM mode

**Files:**
- Modify: `gui.py:49` and `gui.py:415`
- Modify: `tests/test_gui_integration.py:33-44`

- [ ] **Step 1: Update the pose estimator mock in the integration test**

In `tests/test_gui_integration.py`, replace the `_make_pose_estimator_mock` function (lines 33–44):

```python
def _make_pose_estimator_mock():
    """Return a mock PoseEstimator class with fake BlazePose behavior."""
    mock_inst = MagicMock()
    mock_inst.process_frame.return_value = MagicMock()
    mock_inst.submit_frame = MagicMock()
    mock_inst.latest_landmarks = MagicMock()  # non-None so joints get extracted
    mock_inst.draw_landmarks = MagicMock()

    mock_cls = MagicMock(return_value=mock_inst)
    mock_cls.count_visible = MagicMock(return_value=33)
    mock_cls.extract_mapped_joints = MagicMock(
        return_value=np.zeros((12, 2), dtype=np.float32)
    )
    return mock_cls, mock_inst
```

- [ ] **Step 2: Run integration tests to verify they still pass before touching gui.py**

```bash
source .venv/bin/activate
pytest tests/test_gui_integration.py -v
```

Expected: all 6 integration tests PASS (mock is backward-compatible because `process_frame` is still present but unused).

- [ ] **Step 3: Switch gui.py to LIVE_STREAM mode**

In `gui.py` line 49, change:

```python
pose_est = PoseEstimator(model_complexity=MODEL_COMPLEXITY, running_mode="video")
```

to:

```python
pose_est = PoseEstimator(model_complexity=MODEL_COMPLEXITY, running_mode="live_stream")
```

- [ ] **Step 4: Replace process_frame call in gui.py update loop**

In `gui.py` around line 415, replace:

```python
    lm = pose_est.process_frame(raw_frame, timestamp_ms=timestamp_ms)
```

with:

```python
    pose_est.submit_frame(raw_frame, timestamp_ms)
    lm = pose_est.latest_landmarks
```

- [ ] **Step 5: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add gui.py tests/test_gui_integration.py
git commit -m "perf: switch BlazePose to LIVE_STREAM mode for async inference"
```

---

## Verification on RPi 5

After deploying (`git pull` on RPi 5):

```bash
source .venv/bin/activate
python gui.py --model lite
```

Expected: FPS display shows 25-30 (up from 10-15). Skeleton overlay and exercise classification continue to work. At startup, skeleton may lag by 1 frame before the first callback fires — this is normal.
