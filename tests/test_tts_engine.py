import time
import threading
from unittest.mock import patch, MagicMock

import pytest

from tts_engine import TTSEngine


@pytest.fixture
def engine():
    """Provide a TTSEngine with socket.create_connection patched to succeed."""
    with patch("tts_engine.socket.create_connection"):
        yield TTSEngine()


def test_speak_skips_if_already_speaking(engine):
    engine._speaking = True
    with patch("tts_engine.gTTS") as mock_gtts, patch("tts_engine.subprocess.run") as mock_run:
        engine.speak("hello")
        assert mock_gtts.call_count == 0
        mock_run.assert_not_called()


def test_speaking_flag_resets_after_audio(engine):
    def mock_play_audio(text):
        engine._speaking = True
        time.sleep(0.05)
        engine._speaking = False

    engine._play_audio = mock_play_audio

    created_threads = []
    original_thread = threading.Thread

    def tracking_thread(*args, **kwargs):
        t = original_thread(*args, **kwargs)
        created_threads.append(t)
        return t

    with patch("tts_engine.threading.Thread", side_effect=tracking_thread):
        engine.speak("hello")
        if created_threads:
            created_threads[-1].join(timeout=2)

    assert engine._speaking is False


def test_exercise_announced_after_1_5s_stable(engine):
    engine.speak = MagicMock()
    engine.update("Deep Squat", 1, 0.0)
    engine.update("Deep Squat", 1, 1.0)
    assert engine.speak.call_count == 0
    engine.update("Deep Squat", 1, 1.6)
    assert engine.speak.call_count == 1
    assert engine.speak.call_args[0][0] == "Deep Squat detected"


def test_exercise_not_announced_if_changed_before_stable(engine):
    engine.speak = MagicMock()
    engine.update("Deep Squat", 1, 0.0)
    engine.update("Hurdle Step", 1, 1.0)
    engine.update("Hurdle Step", 1, 2.0)
    spoken_texts = [c[0][0] for c in engine.speak.call_args_list]
    assert "Deep Squat detected" not in spoken_texts


def test_incorrect_cue_fires_after_3s(engine):
    engine.speak = MagicMock()
    engine._last_exercise = "Deep Squat"  # skip exercise-announcement logic
    engine.update("Deep Squat", 0, 0.0)
    engine.update("Deep Squat", 0, 2.9)
    assert engine.speak.call_count == 0
    engine.update("Deep Squat", 0, 3.1)
    assert engine.speak.call_count == 1
    assert engine.speak.call_args[0][0] == "Check your form"


def test_incorrect_cue_respects_5s_cooldown(engine):
    engine.speak = MagicMock()
    engine._last_exercise = "Deep Squat"  # skip exercise-announcement logic
    engine.update("Deep Squat", 0, 0.0)
    engine.update("Deep Squat", 0, 3.1)
    assert engine.speak.call_count == 1
    engine.update("Deep Squat", 0, 6.0)
    assert engine.speak.call_count == 1
    engine.update("Deep Squat", 0, 8.5)
    assert engine.speak.call_count == 2


def test_recovery_cue_fires_after_long_incorrect_streak(engine):
    engine.speak = MagicMock()
    engine.update("Deep Squat", 0, 0.0)
    engine.update("Deep Squat", 0, 3.5)
    engine.update("Deep Squat", 1, 4.0)
    spoken_texts = [c[0][0] for c in engine.speak.call_args_list]
    assert "Good job, form restored" in spoken_texts


def test_recovery_cue_not_fired_if_streak_too_short(engine):
    engine.speak = MagicMock()
    engine.update("Deep Squat", 0, 0.0)
    engine.update("Deep Squat", 0, 1.5)
    engine.update("Deep Squat", 1, 2.0)
    spoken_texts = [c[0][0] for c in engine.speak.call_args_list]
    assert "Good job, form restored" not in spoken_texts
