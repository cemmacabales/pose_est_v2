import os
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
if not VERTEX_PROJECT:
    raise EnvironmentError("VERTEX_PROJECT not set. Set it to your Google Cloud project ID.")
client = genai.Client(
    vertexai=True,
    project=VERTEX_PROJECT,
    location=VERTEX_LOCATION
)


def build_system_prompt(session_data: dict) -> str:
    date = session_data.get("date", "")
    duration_seconds = session_data.get("duration_seconds", 0)
    minutes = duration_seconds // 60
    seconds = duration_seconds % 60
    overall_form_score = session_data.get("overall_form_score_pct", 0)
    total_exercises = session_data.get("total_exercises_detected", 0)
    exercises = session_data.get("exercises", [])

    lines = [
        "You are a friendly fitness coaching assistant.",
        "The user just completed a workout session. Here is their data:",
        "",
        f"Date: {date}",
        f"Duration: {minutes} min {seconds} sec",
        f"Overall Form Score: {overall_form_score}%",
        f"Total exercises detected: {total_exercises}",
        "",
        "Exercise breakdown:",
    ]

    for ex in exercises:
        name = ex.get("name", "")
        dur = ex.get("duration_seconds", 0)
        form = ex.get("form_score_pct", 0)
        conf = ex.get("avg_confidence", 0)
        lines.append(f"- {name}: {dur}s | Form: {form}% | Confidence: {conf}")

    lines.extend([
        "",
        "Instructions:",
        "- Answer questions specifically about this session only.",
        "- Be concise, friendly, and encouraging.",
        "- If asked something not covered by this data, say so honestly.",
        "- Do not fabricate numbers not present in the data above.",
    ])

    return "\n".join(lines)


class ChatSession:
    def __init__(self, session_data: dict):
        self.system_prompt = build_system_prompt(session_data)
        self.history = []

    def chat(self, user_message: str) -> str:
        if not user_message or not user_message.strip():
            return "Please ask me something about your session."

        self.history.append({"role": "user", "parts": [user_message]})

        try:
            contents = [
                types.Content(
                    role=turn["role"],
                    parts=[types.Part(text=turn["parts"][0])]
                )
                for turn in self.history
            ]

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    max_output_tokens=512
                )
            )
            reply = response.text
        except Exception as e:
            print(f"[LLM ERROR] {type(e).__name__}: {e}")
            return "Sorry, I couldn't reach the AI service. Check your internet connection."

        self.history.append({"role": "model", "parts": [reply]})
        return reply
