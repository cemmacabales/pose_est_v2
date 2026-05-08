from flask import Flask, jsonify, request, render_template_string
from session_chat.llm import ChatSession
import threading

# Module-level state
_chat_session = None
_session_data = None

app = Flask(__name__)


def init_app(session_data: dict):
    global _chat_session, _session_data
    _session_data = session_data
    _chat_session = ChatSession(session_data)


@app.route("/")
def index():
    if _session_data is None:
        return "Session not initialized", 500

    duration_seconds = _session_data.get("duration_seconds", 0)
    minutes = duration_seconds // 60
    seconds = duration_seconds % 60

    return render_template_string(
        open("session_chat/templates/chat.html").read(),
        session_date=_session_data["date"],
        session_duration=f"{minutes} min {seconds} sec",
        overall_score=_session_data["overall_form_score_pct"],
    )


@app.route("/chat", methods=["POST"])
def chat():
    if _chat_session is None:
        return jsonify({"error": "Not initialized"}), 500

    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing message"}), 400

    message = data["message"].strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    reply = _chat_session.chat(message)
    return jsonify({"reply": reply})


def start_server(session_data: dict):
    init_app(session_data)
    t = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=5000, use_reloader=False),
        daemon=True,
    )
    t.start()
