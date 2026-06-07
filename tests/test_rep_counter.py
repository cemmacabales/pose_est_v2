import numpy as np
import pytest
from rep_counter import RepCounter, EXERCISE_REP_CONFIG


def _angles(overrides: dict) -> np.ndarray:
    a = np.zeros(16, dtype=np.float32)
    for idx, val in overrides.items():
        a[idx] = val
    return a


def _drive(counter, ex, feature_vals, n=25):
    a = _angles({f: v for f, v in zip(EXERCISE_REP_CONFIG[ex]["features"], feature_vals)})
    for _ in range(n):
        counter.update(ex, a)


def _rep_low(counter, ex):
    cfg = EXERCISE_REP_CONFIG[ex]
    neutral_val = cfg["exit"] + 0.1
    peak_val = cfg["enter"] - 0.1
    n_feats = len(cfg["features"])
    _drive(counter, ex, [neutral_val] * n_feats, n=10)
    _drive(counter, ex, [peak_val] * n_feats, n=25)
    _drive(counter, ex, [neutral_val] * n_feats, n=25)


def _rep_high(counter, ex):
    cfg = EXERCISE_REP_CONFIG[ex]
    neutral_val = cfg["exit"] - 0.1
    peak_val = cfg["enter"] + 0.1
    n_feats = len(cfg["features"])
    _drive(counter, ex, [neutral_val] * n_feats, n=10)
    _drive(counter, ex, [peak_val] * n_feats, n=25)
    _drive(counter, ex, [neutral_val] * n_feats, n=25)


def test_deep_squat_one_rep():
    counter = RepCounter()
    _rep_low(counter, "Deep Squat")
    assert counter.get_counts()["Deep Squat"] == 1


def test_deep_squat_two_reps():
    counter = RepCounter()
    _rep_low(counter, "Deep Squat")
    _rep_low(counter, "Deep Squat")
    assert counter.get_counts()["Deep Squat"] == 2


def test_inline_lunge_one_rep():
    counter = RepCounter()
    _rep_low(counter, "Inline Lunge")
    assert counter.get_counts()["Inline Lunge"] == 1


def test_shoulder_abduction_one_rep():
    counter = RepCounter()
    _rep_high(counter, "Shoulder Abduction")
    assert counter.get_counts()["Shoulder Abduction"] == 1


def test_sit_to_stand_one_rep():
    counter = RepCounter()
    _rep_high(counter, "Sit to Stand")
    assert counter.get_counts()["Sit to Stand"] == 1


def test_staying_at_neutral_counts_zero():
    counter = RepCounter()
    cfg = EXERCISE_REP_CONFIG["Deep Squat"]
    neutral_val = cfg["exit"] + 0.1
    _drive(counter, "Deep Squat", [neutral_val] * len(cfg["features"]), n=50)
    assert counter.get_counts().get("Deep Squat", 0) == 0


def test_partial_rep_no_return_counts_zero():
    counter = RepCounter()
    cfg = EXERCISE_REP_CONFIG["Deep Squat"]
    peak_val = cfg["enter"] - 0.1
    _drive(counter, "Deep Squat", [peak_val] * len(cfg["features"]), n=50)
    assert counter.get_counts().get("Deep Squat", 0) == 0


def test_unknown_exercise_returns_zero():
    counter = RepCounter()
    result = counter.update("Not A Real Exercise", np.zeros(16, dtype=np.float32))
    assert result == 0
    assert counter.get_counts() == {}


def test_get_counts_tracks_multiple_exercises_independently():
    counter = RepCounter()
    _rep_low(counter, "Deep Squat")
    _rep_low(counter, "Deep Squat")
    _rep_low(counter, "Inline Lunge")
    counts = counter.get_counts()
    assert counts["Deep Squat"] == 2
    assert counts["Inline Lunge"] == 1


def test_get_counts_empty_for_new_counter():
    counter = RepCounter()
    assert counter.get_counts() == {}


def test_all_nine_exercises_have_config():
    from joint_map import EXERCISE_NAMES
    for name in EXERCISE_NAMES.values():
        assert name in EXERCISE_REP_CONFIG, f"Missing config for: {name}"


def test_update_returns_current_count():
    counter = RepCounter()
    _rep_low(counter, "Deep Squat")
    result = counter.update("Deep Squat", np.zeros(16, dtype=np.float32))
    assert isinstance(result, int)
    assert result >= 0
