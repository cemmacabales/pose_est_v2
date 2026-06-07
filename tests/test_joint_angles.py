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
