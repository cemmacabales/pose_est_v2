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
