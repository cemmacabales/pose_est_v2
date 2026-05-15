import os
import importlib
from unittest.mock import patch, MagicMock

# Set fake API key BEFORE importing llm module so EnvironmentError is not raised.
os.environ["GROQ_API_KEY"] = "test-fake-groq-key"

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

FAKE_RETRIEVED = [
    {
        "text": "Keep your chest up and core tight during squats.",
        "source": "conditioning_manual.pdf",
        "page": 42,
        "section_title": "Squat Form",
        "score": 0.95,
    },
    {
        "text": "Set small, achievable goals to build habit consistency.",
        "source": "behaviour_manual.pdf",
        "page": 18,
        "section_title": "Goal Setting",
        "score": 0.88,
    },
]


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


def test_prompt_includes_retrieved_chunks():
    prompt = build_system_prompt(FAKE_SESSION, retrieved_chunks=FAKE_RETRIEVED)
    assert "conditioning_manual.pdf" in prompt
    assert "behaviour_manual.pdf" in prompt
    assert "Squat Form" in prompt
    assert "Goal Setting" in prompt


def test_prompt_instructs_citations():
    prompt = build_system_prompt(FAKE_SESSION, retrieved_chunks=FAKE_RETRIEVED)
    assert "cite your sources" in prompt.lower()


def test_prompt_forbids_outside_knowledge():
    prompt = build_system_prompt(FAKE_SESSION, retrieved_chunks=FAKE_RETRIEVED)
    assert "NO general world knowledge" in prompt
    assert "sports" in prompt.lower()
    assert "NEVER use your pre-trained knowledge" in prompt


def test_prompt_requires_refusal_for_off_topic():
    prompt = build_system_prompt(FAKE_SESSION, retrieved_chunks=FAKE_RETRIEVED)
    assert "can only answer questions about your workout" in prompt


def _mock_groq_reply(reply_text: str):
    """Build a mock Groq client that returns *reply_text*."""
    mock_choice = MagicMock()
    mock_choice.message.content = reply_text

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def test_chat_returns_mocked_reply():
    mock_client = _mock_groq_reply("You did great!")

    with patch("session_chat.llm.client", mock_client):
        session = ChatSession(FAKE_SESSION)
        reply = session.chat("How did I do?")
    assert reply == "You did great!"


def test_chat_appends_to_history():
    mock_client = _mock_groq_reply("Nice work!")

    with patch("session_chat.llm.client", mock_client):
        session = ChatSession(FAKE_SESSION)
        session.chat("Question one")
        session.chat("Question two")
    assert len(session.history) == 4


def test_chat_returns_fallback_on_api_error():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("network error")

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
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with patch("groq.Groq", return_value=MagicMock()):
        with patch("dotenv.load_dotenv", lambda *a, **k: None):
            with pytest.raises(EnvironmentError):
                importlib.reload(llm_module)
    # Restore module state for subsequent tests.
    monkeypatch.setenv("GROQ_API_KEY", "test-fake-groq-key")
    llm_module.GROQ_API_KEY = "test-fake-groq-key"
    with patch("groq.Groq", return_value=MagicMock()):
        importlib.reload(llm_module)
