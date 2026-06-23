import json
import os
from pathlib import Path

import numpy as np

# Optional: only needed when actually running inference
# We lazy-load them in the class so tests can import without them installed.
_onnxruntime = None
_tokenizers = None

# Small per-source nudge applied before ranking. The conditioning manual
# (broad NSCA content) shares enough vocabulary with exercise-specific
# questions to crowd out the more precise exercise form guide chunks even
# when it isn't actually answering the question — e.g. "how often should I
# train" pulls in unrelated speed-drill/interval-training paragraphs from
# the conditioning manual alongside the one chunk that actually answers it.
_SOURCE_BIAS = {
    "exercise_form_guide.pdf": 0.03,
    "behaviour_manual.pdf": 0.0,
    "conditioning_manual.pdf": -0.02,
}

# Maps each named exercise to substrings that indicate a chunk or query is
# about it. Used to stop cross-exercise leaks within exercise_form_guide.pdf
# itself — e.g. a Sit to Stand query pulling in an Inline Lunge wobble-
# correction chunk just because both mention "wobble"/"imbalance".
_EXERCISE_TRIGGERS = {
    "deep squat": ["squat"],
    "hurdle step": ["hurdle step", "hurdle"],
    "inline lunge": ["inline lunge"],
    "side lunge": ["side lunge"],
    "sit to stand": ["sit to stand", "sit-to-stand"],
    "standing leg raise": ["leg raise"],
    "shoulder abduction": ["shoulder abduction", "lateral raise"],
    "shoulder scaption": ["scaption"],
    "shoulder extension": ["shoulder extension"],
}

_EXERCISE_MATCH_BOOST = 0.05
# Must outweigh _SOURCE_BIAS["exercise_form_guide.pdf"] (0.03), otherwise a
# wrong-exercise chunk from that same source nets to zero penalty.
_EXERCISE_MISMATCH_PENALTY = 0.08


def _detect_exercises(text: str) -> set[str]:
    """Return the set of named exercises mentioned in *text* (case-insensitive)."""
    lowered = text.lower()
    return {
        name
        for name, triggers in _EXERCISE_TRIGGERS.items()
        if any(t in lowered for t in triggers)
    }


def _get_onnxruntime():
    global _onnxruntime
    if _onnxruntime is None:
        import onnxruntime as ort
        _onnxruntime = ort
    return _onnxruntime


def _get_tokenizers():
    global _tokenizers
    if _tokenizers is None:
        import tokenizers
        _tokenizers = tokenizers
    return _tokenizers


class RetrievalEngine:
    """
    Lightweight retrieval engine for RPi 5.

    Loads a pre-built knowledge base (JSON) and an ONNX embedding model.
    Embeds user queries locally and finds the most relevant chunks via
    cosine similarity (pure numpy).
    """

    def __init__(
        self,
        kb_path="data/knowledge_base.json",
        model_dir="data/embedding_model",
        max_seq_length=512,
    ):
        self.kb_path = Path(kb_path)
        self.model_dir = Path(model_dir)
        self.max_seq_length = max_seq_length

        self.chunks = []
        self.embeddings = None  # shape: (num_chunks, embedding_dim)
        self.embedding_dim = 0

        self._session = None
        self._tokenizer = None

        self._load_kb()
        self._load_onnx_model()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_kb(self):
        if not self.kb_path.exists():
            raise FileNotFoundError(
                f"Knowledge base not found: {self.kb_path}\n"
                "Run build_knowledge_base.py on your Mac/Colab first."
            )

        with open(self.kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)

        self.chunks = kb["chunks"]
        self.embedding_dim = kb["embedding_dim"]

        if not self.chunks:
            raise ValueError("Knowledge base contains no chunks.")

        self.embeddings = np.array(
            [c["embedding"] for c in self.chunks], dtype=np.float32
        )
        # Precomputed once at load time so search() doesn't re-scan chunk
        # text on every call. Includes section_title so sub-section chunks
        # (e.g. "Inline Lunge — Coaching Cues") are still tagged with their
        # exercise even when the body never names it.
        self._chunk_exercises = [
            _detect_exercises(c.get("section_title", "") + " " + c["text"])
            for c in self.chunks
        ]
        print(f"[RetrievalEngine] Loaded {len(self.chunks)} chunks, dim={self.embedding_dim}")

    def _load_onnx_model(self):
        model_path = self.model_dir / "model.onnx"
        tokenizer_path = self.model_dir / "tokenizer.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {model_path}\n"
                "Run build_knowledge_base.py on your Mac/Colab first."
            )

        ort = _get_onnxruntime()
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

        if tokenizer_path.exists():
            tok_lib = _get_tokenizers()
            self._tokenizer = tok_lib.Tokenizer.from_file(str(tokenizer_path))
        else:
            raise FileNotFoundError(
                f"Tokenizer not found: {tokenizer_path}\n"
                "Run build_knowledge_base.py on your Mac/Colab first."
            )

        print(f"[RetrievalEngine] ONNX model loaded from {model_path}")

    # ── Query embedding ───────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query string using the local ONNX model."""
        if self._tokenizer is None or self._session is None:
            raise RuntimeError("ONNX model or tokenizer not loaded.")

        encoded = self._tokenizer.encode(query)
        input_ids = encoded.ids
        attention_mask = [1] * len(input_ids)

        # Truncate
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[: self.max_seq_length]
            attention_mask = attention_mask[: self.max_seq_length]

        # Pad
        pad_len = self.max_seq_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [0] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        # To numpy arrays with batch dimension
        input_ids_np = np.array([input_ids], dtype=np.int64)
        attention_mask_np = np.array([attention_mask], dtype=np.int64)

        outputs = self._session.run(
            None,
            {"input_ids": input_ids_np, "attention_mask": attention_mask_np},
        )
        last_hidden_state = outputs[0]  # (1, seq_len, hidden_dim)

        # Mean pooling (excluding padding)
        mask = np.expand_dims(attention_mask_np, -1)  # (1, seq_len, 1)
        sum_embeddings = np.sum(last_hidden_state * mask, axis=1)  # (1, hidden_dim)
        sum_mask = np.clip(np.sum(mask, axis=1), a_min=1e-9, a_max=None)  # (1, 1)
        mean_pooled = sum_embeddings / sum_mask  # (1, hidden_dim)

        # L2 normalize
        norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        return mean_pooled / norms  # (1, hidden_dim)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 6) -> list[dict]:
        """
        Search the knowledge base for chunks relevant to *query*.

        Returns a list of dicts with keys:
            text, source, page, section_title, score
        """
        if not query or not query.strip():
            return []

        query_emb = self._embed_query(query.strip())  # (1, dim)

        # Cosine similarity = dot product because both are L2-normalized
        similarities = np.dot(self.embeddings, query_emb.T).flatten()  # (num_chunks,)

        # Source-aware + exercise-aware re-ranking. Adjusted scores are only
        # used to pick and order the top_k; the *reported* score stays the
        # raw cosine similarity so it keeps its normal [-1, 1] meaning.
        query_exercises = _detect_exercises(query)
        adjusted = similarities.copy()
        for i, chunk in enumerate(self.chunks):
            adjusted[i] += _SOURCE_BIAS.get(chunk["source"], 0.0)
            if query_exercises:
                chunk_exercises = self._chunk_exercises[i]
                if chunk_exercises & query_exercises:
                    adjusted[i] += _EXERCISE_MATCH_BOOST
                elif chunk_exercises:
                    adjusted[i] -= _EXERCISE_MISMATCH_PENALTY

        top_indices = np.argsort(adjusted)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "section_title": chunk.get("section_title", ""),
                "score": round(float(similarities[idx]), 4),
            })

        return results
