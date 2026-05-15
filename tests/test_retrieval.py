import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from session_chat.retrieval import RetrievalEngine


@pytest.fixture
def fake_kb(tmp_path):
    """Create a minimal fake knowledge base for testing."""
    chunks = [
        {
            "text": "Keep chest up during squats.",
            "source": "conditioning_manual.pdf",
            "page": 10,
            "section_title": "Squat",
            "embedding": [1.0, 0.0, 0.0],
        },
        {
            "text": "Set weekly goals for consistency.",
            "source": "behaviour_manual.pdf",
            "page": 5,
            "section_title": "Goals",
            "embedding": [0.0, 1.0, 0.0],
        },
        {
            "text": "Warm up before heavy lifts.",
            "source": "conditioning_manual.pdf",
            "page": 3,
            "section_title": "Warm-up",
            "embedding": [0.0, 0.0, 1.0],
        },
    ]
    kb = {
        "model_name": "fake-model",
        "embedding_dim": 3,
        "num_chunks": len(chunks),
        "chunks": chunks,
    }
    kb_path = tmp_path / "knowledge_base.json"
    with open(kb_path, "w") as f:
        json.dump(kb, f)
    return kb_path


@pytest.fixture
def engine_no_onnx(fake_kb, tmp_path):
    """RetrievalEngine with ONNX loading skipped (no real model needed)."""
    with patch.object(RetrievalEngine, "_load_onnx_model"):
        engine = RetrievalEngine(
            kb_path=str(fake_kb),
            model_dir=str(tmp_path / "model"),
        )
    return engine


def test_load_kb(engine_no_onnx):
    """RetrievalEngine loads the knowledge base correctly."""
    assert len(engine_no_onnx.chunks) == 3
    assert engine_no_onnx.embedding_dim == 3
    assert engine_no_onnx.embeddings.shape == (3, 3)


def test_search_returns_relevant_chunks(engine_no_onnx):
    """Search finds chunks closest to the query embedding."""
    # Monkey-patch the ONNX embedder so we don't need a real model.
    # We force the query embedding to align with chunk 0 (squats).
    engine_no_onnx._embed_query = lambda q: np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    results = engine_no_onnx.search("How do I squat better?", top_k=2)
    assert len(results) == 2
    assert results[0]["source"] == "conditioning_manual.pdf"
    assert results[0]["page"] == 10
    assert results[0]["section_title"] == "Squat"
    assert results[0]["score"] > 0.9


def test_search_respects_top_k(engine_no_onnx):
    engine_no_onnx._embed_query = lambda q: np.array([[0.5, 0.5, 0.0]], dtype=np.float32)

    results = engine_no_onnx.search("mixed query", top_k=1)
    assert len(results) == 1


def test_search_empty_query_returns_empty(engine_no_onnx):
    assert engine_no_onnx.search("") == []
    assert engine_no_onnx.search("   ") == []


def test_load_kb_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        RetrievalEngine(kb_path="data/nonexistent_kb.json")
