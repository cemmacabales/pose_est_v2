import os
from dotenv import load_dotenv

from groq import Groq

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not set. Get a free key at https://console.groq.com"
    )

client = Groq(api_key=GROQ_API_KEY)
DEFAULT_MODEL = "llama-3.1-8b-instant"


def _format_retrieved_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant reference material was retrieved for this query."

    lines = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "Unknown")
        page = chunk.get("page", "?")
        section = chunk.get("section_title", "")
        text = chunk.get("text", "")
        header = f"[{i}] Source: {source}, Page {page}"
        if section:
            header += f", Section: {section}"
        lines.append(header)
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


def build_system_prompt(session_data: dict, retrieved_chunks: list[dict] = None) -> str:
    date = session_data.get("date", "")
    duration_seconds = session_data.get("duration_seconds", 0)
    minutes = duration_seconds // 60
    seconds = duration_seconds % 60
    overall_form_score = session_data.get("overall_form_score_pct", 0)
    total_exercises = session_data.get("total_exercises_detected", 0)
    exercises = session_data.get("exercises", [])

    lines = [
        "You are a friendly fitness coaching assistant with access to three knowledge sources:",
        "1. The user's current workout session data",
        "2. Conditioning manual — exercise science, programming, and form guidance",
        "3. Behaviour manual — psychology, habit building, and motivation",
        "",
        "=" * 40,
        "SESSION DATA",
        "=" * 40,
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
        "=" * 40,
        "RETRIEVED KNOWLEDGE",
        "=" * 40,
        "",
    ])

    lines.append(_format_retrieved_chunks(retrieved_chunks or []))

    lines.extend([
        "",
        "=" * 40,
        "INSTRUCTIONS",
        "=" * 40,
        "- Use the session data to comment on the user's workout performance.",
        "- Use the retrieved knowledge to suggest specific improvements or answer general fitness questions.",
        "- Always cite your sources when using retrieved knowledge: e.g., [Source: conditioning_manual.pdf, Page 42].",
        "- If the user asks something not in the session data or retrieved knowledge, say so honestly.",
        "- Be concise, friendly, and encouraging.",
        "- Do not fabricate numbers or facts.",
    ])

    return "\n".join(lines)


class ChatSession:
    def __init__(self, session_data: dict, retrieval_engine=None, model: str = DEFAULT_MODEL):
        self.session_data = session_data
        self.retrieval_engine = retrieval_engine
        self.model = model
        self.history = []

    def chat(self, user_message: str) -> str:
        if not user_message or not user_message.strip():
            return "Please ask me something about your session or fitness in general."

        # Retrieve relevant knowledge for this query
        retrieved_chunks = []
        if self.retrieval_engine is not None:
            try:
                retrieved_chunks = self.retrieval_engine.search(user_message.strip(), top_k=4)
            except Exception as e:
                print(f"[Retrieval ERROR] {type(e).__name__}: {e}")
                # Continue without retrieved knowledge rather than failing entirely

        system_prompt = build_system_prompt(self.session_data, retrieved_chunks)

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Rebuild message history in Groq/OpenAI format
        for turn in self.history:
            role = turn["role"]
            # Groq uses "assistant" not "model"
            if role == "model":
                role = "assistant"
            messages.append({"role": role, "content": turn["parts"][0]})

        messages.append({"role": "user", "content": user_message.strip()})

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=512,
                temperature=0.7,
            )
            reply = response.choices[0].message.content
        except Exception as e:
            print(f"[LLM ERROR] {type(e).__name__}: {e}")
            return "Sorry, I couldn't reach the AI service. Check your internet connection."

        # Store history in internal format ("user" / "model")
        self.history.append({"role": "user", "parts": [user_message.strip()]})
        self.history.append({"role": "model", "parts": [reply]})
        return reply
