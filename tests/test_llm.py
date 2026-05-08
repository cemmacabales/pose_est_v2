import os
import importlib
from unittest.mock import patch, MagicMock

# Set fake project BEFORE importing llm module so EnvironmentError is not raised.
os.environ["VERTEX_PROJECT"] = "test-fake-project"
os.environ["VERTEX_LOCATION"] = "us-central1"

import pytest
from session_chat.llm import build_system_prompt, ChatSession

FAKE_SESSION = {
    "date": "2026-05-08",
    "start_time": "10:00:00",
    "end_time": "10:15:00",
    "duration_seconds": 900,
    "exercises": [
        {
            "name": "Deep Squat",
            "duration_seconds": 300,
            "frames_correct": 240,
            "frames_incorrect": 60,
            "form_score_pct": 80,
            "avg_confidence": 0.88,
        },
        {
            "name": "Side Lunge",
            "duration_seconds": 600,
            "frames_correct": 400,
            "frames_incorrect": 200,
            "form_score_pct": 67,
            "avg_confidence": 0.75,
        },
    ],
    "overall_form_score_pct": 73,
    "total_exercises_detected": 2,
    "log_file": "logs/session_test.json",
}


def test_build_system_prompt_contains_exercise_names():
    prompt = build_system_prompt(FAKE_SESSION)
    assert "Deep Squat" in prompt
    assert "Side Lunge" in prompt


def test_build_system_prompt_contains_scores():
    prompt = build_system_prompt(FAKE_SESSION)
    assert "73" in prompt
    assert "80" in prompt
    assert "2026-05-08" in prompt


def test_build_system_prompt_returns_non_empty_string():
    prompt = build_system_prompt(FAKE_SESSION)
    assert isinstance(prompt, str)
    assert len(prompt) > 100


def test_chat_returns_mocked_reply():
    mock_response = MagicMock()
    mock_response.text = "You did great!"

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("session_chat.llm.client", mock_client):
        session = ChatSession(FAKE_SESSION)
        reply = session.chat("How did I do?")
    assert reply == "You did great!"


def test_chat_appends_to_history():
    mock_response = MagicMock()
    mock_response.text = "Nice work!"

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("session_chat.llm.client", mock_client):
        session = ChatSession(FAKE_SESSION)
        session.chat("Question one")
        session.chat("Question two")
    assert len(session.history) == 4


def test_chat_returns_fallback_on_api_error():
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("network error")

    with patch("session_chat.llm.client", mock_client):
        session = ChatSession(FAKE_SESSION)
        reply = session.chat("Hello")
    assert "Sorry" in reply or "couldn't" in reply


def test_empty_message_returns_prompt_to_ask():
    session = ChatSession(FAKE_SESSION)
    reply = session.chat("")
    assert "Please ask" in reply or len(reply) > 0


def test_missing_api_key_raises_environment_error(monkeypatch):
    import session_chat.llm as llm_module
    monkeypatch.delenv("VERTEX_PROJECT", raising=False)
    with patch("google.genai.Client", return_value=MagicMock()):
        with patch("dotenv.load_dotenv", lambda *a, **k: None):
            with pytest.raises(EnvironmentError):
                importlib.reload(llm_module)
    # Restore module state for subsequent tests.
    monkeypatch.setenv("VERTEX_PROJECT", "test-fake-project")
    llm_module.VERTEX_PROJECT = "test-fake-project"
    with patch("google.genai.Client", return_value=MagicMock()):
        importlib.reload(llm_module)
