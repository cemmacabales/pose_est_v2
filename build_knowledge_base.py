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

CHUNK_SIZE = 1000         # chars per chunk (~250 tokens, well under MAX_SEQ_LENGTH)
CHUNK_OVERLAP = 100       # overlap between chunks
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 512

# The exercise form guide nests each exercise's body under a fixed set of
# sub-section headings that never name the exercise themselves. We thread the
# parent "Exercise N: <name>" heading onto those sub-sections so their chunks
# (and embeddings) aren't orphaned from the exercise name — e.g. a "Coaching
# Cues" chunk for the Inline Lunge that, without this, contains no "inline
# lunge" text and so matches that query weakly and dodges the exercise re-ranker.
# Short words that title case leaves lowercase; ignored when judging whether a
# short line is a heading (see is_heading).
_HEADING_CONNECTORS = {
    "to", "of", "and", "the", "a", "an", "for", "in", "on", "at",
    "by", "or", "vs", "per", "with", "from", "it",
}

_EXERCISE_HEADING_RE = re.compile(r"^\s*Exercise\s+\d+\s*[:\-]", re.IGNORECASE)
_SUBSECTION_HEADINGS = {
    "overview",
    "setup",
    "coaching cues",
    "common errors and corrections",
    "rep guidance",
}


def embedding_text(text, section_title):
    """Text actually fed to the embedder: a contextual header (the section
    title, which carries the exercise name) prepended to the chunk body.

    The header is only used to compute the embedding — the stored/displayed
    chunk text stays raw, so LLM context and citations are unchanged.
    """
    section_title = (section_title or "").strip()
    if not section_title:
        return text
    return f"{section_title}\n{text}"


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


def is_heading(para):
    """Heuristic: is *para* a section heading rather than body text?

    Headings are short single-line fragments that are either ALL CAPS
    ("FREQUENCY") or title-case ("Overview", "Exercise 3: Inline Lunge") and
    do not end in sentence punctuation. Keeping these attached to the body
    that follows them is what stops e.g. a "FREQUENCY" heading being orphaned
    from its "Three sessions per week..." answer across a chunk boundary.
    """
    s = para.strip()
    if not s or "\n" in s or len(s) > 60:
        return False
    if s[-1] in ".!?":
        return False
    if not any(c.isalpha() for c in s):
        return False
    if s.isupper():
        return True
    words = [w for w in re.split(r"\s+", s) if w]
    if len(words) > 8:
        return False
    # Title case lowercases short connector words ("Sit to Stand", "Putting It
    # Together"), so judge only the significant (alphabetic, non-connector)
    # words — otherwise a heading like "Exercise 5: Sit to Stand" is misread as
    # body text and its sub-sections get orphaned from the exercise name.
    significant = [
        w for w in words
        if any(c.isalpha() for c in w) and w.lower() not in _HEADING_CONNECTORS
    ]
    if not significant:
        return False
    capitalized = sum(1 for w in significant if w[0].isupper())
    return capitalized >= len(significant) - 1


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, start_parent=""):
    """Split text into chunks, keeping section headings attached to the body
    that follows them and carrying a little overlap across chunk boundaries.

    *start_parent* seeds the active "Exercise N: <name>" heading so the parent
    can be threaded across page boundaries (each PDF page is chunked separately).

    Returns ``(chunks, parent_title)`` where chunks is a list of
    ``(chunk_text, section_title)`` tuples and parent_title is the exercise
    heading still in effect at the end (to feed the next page's start_parent).
    """
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    chunks = []            # list of (text, section_title)
    current = []           # paragraphs in the chunk being built
    current_len = 0
    section_title = ""     # most recent heading, threaded onto chunks
    parent_title = start_parent  # active "Exercise N: <name>", threaded onto sub-sections

    def flush(carry_overlap=True):
        nonlocal current, current_len
        if not current:
            return
        chunks.append(("\n\n".join(current), section_title))
        # Seed the next chunk with the trailing paragraph so content near a
        # boundary stays recoverable from both chunks. Never carry a heading
        # this way — headings are placed explicitly so they lead their body.
        tail = current[-1]
        if carry_overlap and overlap > 0 and not is_heading(tail) and len(tail) <= overlap * 4:
            current, current_len = [tail], len(tail)
        else:
            current, current_len = [], 0

    for para in paragraphs:
        if is_heading(para):
            # A heading starts a fresh chunk and stays with what follows it,
            # never dangling at the tail of the previous chunk.
            flush(carry_overlap=False)
            if _EXERCISE_HEADING_RE.match(para):
                # New exercise: becomes the parent threaded onto its sub-sections.
                parent_title = para
                section_title = para
            elif para.strip().lower() in _SUBSECTION_HEADINGS:
                # Sub-section under an exercise: keep the exercise name attached.
                section_title = f"{parent_title} — {para}" if parent_title else para
            else:
                # Any other heading is a new chapter and ends the exercise context.
                parent_title = ""
                section_title = para
            current, current_len = [para], len(para)
            continue

        if len(para) > chunk_size:
            # Oversized paragraph: flush, then hard-split at word boundaries.
            flush(carry_overlap=False)
            start = 0
            while start < len(para):
                end = min(start + chunk_size, len(para))
                if end < len(para):
                    while end > start and para[end] not in " \n":
                        end -= 1
                    if end == start:
                        end = min(start + chunk_size, len(para))
                chunks.append((para[start:end].strip(), section_title))
                start = max(end - overlap, start + 1)
            continue

        if current and current_len + len(para) + 2 > chunk_size:
            flush()
        current.append(para)
        current_len += len(para) + 2

    flush(carry_overlap=False)

    return [(c.strip(), st) for c, st in chunks if c.strip()], parent_title


def extract_chunks_from_pdf(pdf_path):
    """Extract all chunks from a single PDF with metadata."""
    doc = fitz.open(pdf_path)
    filename = pdf_path.name
    all_chunks = []
    parent_title = ""  # threaded across pages so an exercise that spills onto the
                       # next page keeps tagging its sub-sections with the exercise name

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = extract_page_text(page)
        if not page_text.strip():
            continue

        # chunk_text keeps section headings attached to their body and returns
        # the active heading alongside each chunk as section_title, plus the
        # parent exercise heading still active at the page end.
        page_chunks, parent_title = chunk_text(page_text, start_parent=parent_title)

        for chunk_body, section_title in page_chunks:
            all_chunks.append({
                "text": chunk_body,
                "source": filename,
                "page": page_num + 1,
                "section_title": section_title,
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
    # Embed each chunk with its section title as a contextual header so
    # orphaned sub-sections carry the exercise name into the vector.
    texts = [embedding_text(c["text"], c.get("section_title", "")) for c in all_chunks]
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
