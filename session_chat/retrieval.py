import json
import os
from pathlib import Path

import numpy as np

# Optional: only needed when actually running inference
# We lazy-load them in the class so tests can import without them installed.
_onnxruntime = None
_tokenizers = None


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

    def search(self, query: str, top_k: int = 4) -> list[dict]:
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

        top_indices = np.argsort(similarities)[::-1][:top_k]

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
