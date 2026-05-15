import os

# Set fake key BEFORE importing llm-related modules so EnvironmentError is not raised.
os.environ["GROQ_API_KEY"] = "test-fake-groq-key"

import pytest
from unittest.mock import patch
from session_chat.app import init_app, app

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


@pytest.fixture
def client():
    with patch("session_chat.app.ChatSession") as MockChatSession:
        MockChatSession.return_value.chat.return_value = "Mocked reply"
        init_app(FAKE_SESSION)
        app.config["TESTING"] = True
        yield app.test_client()


def test_index_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200


def test_index_contains_session_date(client):
    response = client.get("/")
    assert b"2026-05-08" in response.data


def test_index_contains_overall_score(client):
    response = client.get("/")
    assert b"73" in response.data


def test_chat_endpoint_returns_reply(client):
    import session_chat.app

    session_chat.app._chat_session.chat.return_value = "You did great!"
    response = client.post("/chat", json={"message": "How did I do?"})
    assert response.status_code == 200
    assert response.get_json()["reply"] == "You did great!"


def test_chat_endpoint_missing_message_key_returns_400(client):
    response = client.post("/chat", json={})
    assert response.status_code == 400


def test_chat_endpoint_empty_message_returns_400(client):
    response = client.post("/chat", json={"message": "   "})
    assert response.status_code == 400


def test_chat_endpoint_uninitialized_returns_500(client):
    import session_chat.app

    session_chat.app._chat_session = None
    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 500
