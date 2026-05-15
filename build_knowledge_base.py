#!/usr/bin/env python3
"""
build_knowledge_base.py — Run on Mac or Colab (NOT on RPi 5).

Processes PDFs in references/ into:
  data/knowledge_base.json      — text chunks + pre-computed embeddings
  data/embedding_model/         — ONNX model + tokenizer for runtime query embedding

Dependencies (install before running):
    pip install pymupdf sentence-transformers torch transformers

Usage:
    python build_knowledge_base.py
"""

import json
import os
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

# ── Config ──────────────────────────────────────────────────────────────────
REFERENCES_DIR = Path("references")
DATA_DIR = Path("data")
MODEL_DIR = DATA_DIR / "embedding_model"
KB_PATH = DATA_DIR / "knowledge_base.json"

CHUNK_SIZE = 500          # chars per chunk
CHUNK_OVERLAP = 50        # overlap between chunks
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 512


def table_to_markdown(rows):
    """Convert PyMuPDF table rows to Markdown."""
    if not rows:
        return ""
    def fmt(cell):
        return str(cell) if cell is not None else ""
    header = "| " + " | ".join(fmt(c) for c in rows[0]) + " |"
    sep = "|" + "|".join([" --- "] * len(rows[0])) + "|"
    body = "\n".join("| " + " | ".join(fmt(c) for c in row) + " |" for row in rows[1:])
    return f"{header}\n{sep}\n{body}"


def extract_page_text(page):
    """Extract text + tables from a PyMuPDF page in reading order."""
    elements = []

    # Text blocks
    blocks = page.get_text("blocks")
    for b in blocks:
        x0, y0, x1, y1, text, block_no, block_type = b
        text = text.strip()
        if text:
            elements.append((y0, text))

    # Tables
    try:
        tables = page.find_tables()
        for t in tables:
            y0 = t.bbox[1]  # top y-coordinate
            rows = t.extract()
            if rows:
                md = table_to_markdown(rows)
                elements.append((y0, md))
    except Exception:
        pass  # Older PyMuPDF or no tables found

    elements.sort(key=lambda x: x[0])
    return "\n\n".join(e[1] for e in elements)


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks at paragraph/word boundaries."""
    # First split into paragraphs
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = current + "\n\n" + para if current else para
        else:
            if current:
                chunks.append(current)
            # If a single paragraph exceeds chunk_size, split it harder
            if len(para) > chunk_size:
                start = 0
                while start < len(para):
                    end = min(start + chunk_size, len(para))
                    if end < len(para):
                        # Try to break at space
                        while end > start and para[end] not in " \n":
                            end -= 1
                        if end == start:
                            end = min(start + chunk_size, len(para))
                    chunks.append(para[start:end])
                    start = end - overlap
                    if start < 0:
                        start = 0
            else:
                current = para

    if current:
        chunks.append(current)

    return [c.strip() for c in chunks if c.strip()]


def extract_chunks_from_pdf(pdf_path):
    """Extract all chunks from a single PDF with metadata."""
    doc = fitz.open(pdf_path)
    filename = pdf_path.name
    all_chunks = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = extract_page_text(page)
        if not page_text.strip():
            continue

        # Detect section headers heuristically: short, all-caps or title-case lines
        # We just chunk the whole page text
        page_chunks = chunk_text(page_text)

        for chunk_text_content in page_chunks:
            all_chunks.append({
                "text": chunk_text_content,
                "source": filename,
                "page": page_num + 1,
                "section_title": "",  # Could be improved with heading detection
            })

    doc.close()
    return all_chunks


def build_knowledge_base():
    """Main entry point: parse PDFs, embed chunks, export ONNX + tokenizer."""
    print("=" * 60)
    print("Building Knowledge Base")
    print("=" * 60)

    if not REFERENCES_DIR.exists():
        print(f"ERROR: {REFERENCES_DIR} does not exist.")
        sys.exit(1)

    pdf_files = sorted(REFERENCES_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"ERROR: No PDFs found in {REFERENCES_DIR}")
        sys.exit(1)

    print(f"Found PDFs: {[p.name for p in pdf_files]}")

    # ── Step 1: Extract chunks ───────────────────────────────────────────────
    print("\n[1/4] Extracting text & tables from PDFs...")
    all_chunks = []
    for pdf_path in pdf_files:
        print(f"  → {pdf_path.name}")
        chunks = extract_chunks_from_pdf(pdf_path)
        print(f"      {len(chunks)} chunks")
        all_chunks.extend(chunks)
    print(f"Total chunks: {len(all_chunks)}")

    # ── Step 2: Compute embeddings ───────────────────────────────────────────
    print(f"\n[2/4] Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("  Embedding chunks...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Store embeddings in chunks
    for i, emb in enumerate(embeddings):
        all_chunks[i]["embedding"] = emb.tolist()

    # ── Step 3: Export ONNX model + tokenizer ────────────────────────────────
    print("\n[3/4] Exporting ONNX query encoder...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    base_model = AutoModel.from_pretrained(EMBEDDING_MODEL)
    base_model.eval()

    # Save tokenizer for runtime loading by `tokenizers` library
    tokenizer.save_pretrained(MODEL_DIR)
    # Also save a single tokenizer.json that `tokenizers.Tokenizer.from_file` can load
    if hasattr(tokenizer, "backend_tokenizer"):
        tokenizer.backend_tokenizer.save(str(MODEL_DIR / "tokenizer.json"))

    dummy_text = "Generate an embedding for this query."
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )

    onnx_path = MODEL_DIR / "model.onnx"
    torch.onnx.export(
        base_model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14,
    )
    print(f"  ONNX model saved to {onnx_path}")

    # ── Step 4: Save knowledge base ──────────────────────────────────────────
    print("\n[4/4] Saving knowledge base...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    kb = {
        "model_name": EMBEDDING_MODEL,
        "embedding_dim": int(embeddings.shape[1]),
        "num_chunks": len(all_chunks),
        "chunks": all_chunks,
    }

    with open(KB_PATH, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)
    print(f"  Knowledge base saved to {KB_PATH}")

    # Stats
    total_text_bytes = sum(len(c["text"].encode("utf-8")) for c in all_chunks)
    print(f"\nStats:")
    print(f"  Total chunks:     {len(all_chunks)}")
    print(f"  Embedding dim:    {embeddings.shape[1]}")
    print(f"  Total text size:  {total_text_bytes / 1024 / 1024:.2f} MB")
    print(f"  ONNX model size:  {onnx_path.stat().st_size / 1024 / 1024:.2f} MB")
    print("=" * 60)
    print("Done! Copy the following to your RPi 5:")
    print("  data/knowledge_base.json")
    print("  data/embedding_model/")
    print("=" * 60)


if __name__ == "__main__":
    build_knowledge_base()
