import os
import json
import pytest
from session_logger import SessionLogger


def test_single_exercise_all_correct(tmp_path):
    logger = SessionLogger(log_dir=str(tmp_path))
    for _ in range(100):
        logger.log_frame(exercise_idx=0, quality=1, confidence=0.9)
    result = logger.end_session()
    assert result["exercises"][0]["name"] == "Deep Squat"
    assert result["exercises"][0]["frames_correct"] == 100
    assert result["exercises"][0]["frames_incorrect"] == 0
    assert result["exercises"][0]["form_score_pct"] == 100
    assert result["exercises"][0]["avg_confidence"] == 0.9


def test_mixed_quality_form_score(tmp_path):
    logger = SessionLogger(log_dir=str(tmp_path))
    for _ in range(75):
        logger.log_frame(exercise_idx=0, quality=1, confidence=0.9)
    for _ in range(25):
        logger.log_frame(exercise_idx=0, quality=0, confidence=0.9)
    result = logger.end_session()
    assert result["exercises"][0]["form_score_pct"] == 75


def test_exercise_transition_creates_two_segments(tmp_path):
    logger = SessionLogger(log_dir=str(tmp_path))
    for _ in range(50):
        logger.log_frame(exercise_idx=0, quality=1, confidence=0.9)
    for _ in range(50):
        logger.log_frame(exercise_idx=1, quality=1, confidence=0.9)
    result = logger.end_session()
    assert len(result["exercises"]) == 2
    assert result["exercises"][0]["name"] == "Deep Squat"
    assert result["exercises"][1]["name"] == "Hurdle Step"
    assert result["total_exercises_detected"] == 2


def test_overall_score_is_mean_of_segments(tmp_path):
    logger = SessionLogger(log_dir=str(tmp_path))
    for _ in range(80):
        logger.log_frame(exercise_idx=0, quality=1, confidence=0.9)
    for _ in range(20):
        logger.log_frame(exercise_idx=0, quality=0, confidence=0.9)
    for _ in range(60):
        logger.log_frame(exercise_idx=1, quality=1, confidence=0.9)
    for _ in range(40):
        logger.log_frame(exercise_idx=1, quality=0, confidence=0.9)
    result = logger.end_session()
    assert result["overall_form_score_pct"] == 70


def test_log_file_written_and_matches_result(tmp_path):
    logger = SessionLogger(log_dir=str(tmp_path))
    for _ in range(10):
        logger.log_frame(exercise_idx=0, quality=1, confidence=0.9)
    result = logger.end_session()
    assert os.path.exists(result["log_file"])
    with open(result["log_file"]) as f:
        on_disk = json.load(f)
    assert on_disk["date"] == result["date"]
    assert on_disk["overall_form_score_pct"] == result["overall_form_score_pct"]


def test_empty_session_does_not_crash(tmp_path):
    logger = SessionLogger(log_dir=str(tmp_path))
    result = logger.end_session()
    assert result["exercises"] == []
    assert result["overall_form_score_pct"] == 0
    assert result["duration_seconds"] >= 0


def test_unknown_exercise_idx_stored_as_unknown(tmp_path):
    logger = SessionLogger(log_dir=str(tmp_path))
    for _ in range(10):
        logger.log_frame(exercise_idx=99, quality=1, confidence=0.9)
    result = logger.end_session()
    assert result["exercises"][0]["name"] == "Unknown"


def test_duration_is_non_negative(tmp_path):
    logger = SessionLogger(log_dir=str(tmp_path))
    for _ in range(10):
        logger.log_frame(exercise_idx=0, quality=1, confidence=0.9)
    result = logger.end_session()
    assert result["duration_seconds"] >= 0


def test_end_session_with_rep_counts_adds_reps_field(tmp_path):
    logger = SessionLogger(log_dir=str(tmp_path))
    for _ in range(10):
        logger.log_frame(exercise_idx=0, quality=1, confidence=0.9)
    result = logger.end_session(rep_counts={"Deep Squat": 5})
    assert result["exercises"][0]["reps"] == 5


def test_end_session_rep_counts_defaults_to_zero_when_missing(tmp_path):
    logger = SessionLogger(log_dir=str(tmp_path))
    for _ in range(10):
        logger.log_frame(exercise_idx=0, quality=1, confidence=0.9)
    result = logger.end_session(rep_counts={})
    assert result["exercises"][0]["reps"] == 0


def test_end_session_no_rep_counts_arg_defaults_to_zero(tmp_path):
    logger = SessionLogger(log_dir=str(tmp_path))
    for _ in range(10):
        logger.log_frame(exercise_idx=0, quality=1, confidence=0.9)
    result = logger.end_session()
    assert result["exercises"][0]["reps"] == 0


def test_end_session_rep_counts_written_to_json(tmp_path):
    import json
    logger = SessionLogger(log_dir=str(tmp_path))
    for _ in range(10):
        logger.log_frame(exercise_idx=0, quality=1, confidence=0.9)
    result = logger.end_session(rep_counts={"Deep Squat": 3})
    with open(result["log_file"]) as f:
        on_disk = json.load(f)
    assert on_disk["exercises"][0]["reps"] == 3
