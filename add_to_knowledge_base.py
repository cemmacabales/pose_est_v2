#!/usr/bin/env python3
"""
add_to_knowledge_base.py — Incrementally add a PDF to the existing knowledge base.

Extracts text chunks from the given PDF, embeds them with the same model already
used for the KB, and appends them to data/knowledge_base.json. Existing chunks
from other sources are preserved. Chunks from the same source filename are replaced.

Usage:
    python add_to_knowledge_base.py references/exercise_form_guide.pdf

Run on Mac or Colab (NOT on RPi 5). Requires:
    pip install pymupdf sentence-transformers torch transformers
"""

import argparse
import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
KB_PATH = DATA_DIR / "knowledge_base.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def table_to_markdown(rows):
    if not rows:
        return ""
    def fmt(cell):
        return str(cell) if cell is not None else ""
    header = "| " + " | ".join(fmt(c) for c in rows[0]) + " |"
    sep = "|" + "|".join([" --- "] * len(rows[0])) + "|"
    body = "\n".join("| " + " | ".join(fmt(c) for c in row) + " |" for row in rows[1:])
    return f"{header}\n{sep}\n{body}"


def extract_page_text(page):
    elements = []
    blocks = page.get_text("blocks")
    for b in blocks:
        x0, y0, x1, y1, text, block_no, block_type = b
        text = text.strip()
        if text:
            elements.append((y0, text))
    try:
        tables = page.find_tables()
        for t in tables:
            y0 = t.bbox[1]
            rows = t.extract()
            if rows:
                elements.append((y0, table_to_markdown(rows)))
    except Exception:
        pass
    elements.sort(key=lambda x: x[0])
    return "\n\n".join(e[1] for e in elements)


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = current + "\n\n" + para if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) > chunk_size:
                start = 0
                while start < len(para):
                    end = min(start + chunk_size, len(para))
                    if end < len(para):
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
    doc = fitz.open(pdf_path)
    filename = pdf_path.name
    all_chunks = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = extract_page_text(page)
        if not page_text.strip():
            continue
        for chunk in chunk_text(page_text):
            all_chunks.append({
                "text": chunk,
                "source": filename,
                "page": page_num + 1,
                "section_title": "",
            })
    doc.close()
    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Incrementally add a PDF to the knowledge base.")
    parser.add_argument("pdf", help="Path to the PDF file to add")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        sys.exit(1)

    if not KB_PATH.exists():
        print(f"ERROR: Knowledge base not found at {KB_PATH}. Run build_knowledge_base.py first.")
        sys.exit(1)

    print(f"Loading existing knowledge base from {KB_PATH}...")
    with open(KB_PATH, encoding="utf-8") as f:
        kb = json.load(f)

    existing_sources = {}
    for c in kb["chunks"]:
        existing_sources.setdefault(c["source"], 0)
        existing_sources[c["source"]] += 1
    print(f"  Existing chunks by source: {existing_sources}")

    # Drop any existing chunks from the same source (full replace for that source)
    source_name = pdf_path.name
    kept_chunks = [c for c in kb["chunks"] if c["source"] != source_name]
    dropped = len(kb["chunks"]) - len(kept_chunks)
    if dropped:
        print(f"  Replacing {dropped} existing chunks from {source_name}")

    print(f"\nExtracting chunks from {pdf_path.name}...")
    new_chunks = extract_chunks_from_pdf(pdf_path)
    print(f"  {len(new_chunks)} new chunks extracted")

    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("  Embedding new chunks...")
    texts = [c["text"] for c in new_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    for i, emb in enumerate(embeddings):
        new_chunks[i]["embedding"] = emb.tolist()

    all_chunks = kept_chunks + new_chunks
    embedding_dim = embeddings.shape[1]

    print(f"\nSaving updated knowledge base...")
    updated_kb = {
        "model_name": kb.get("model_name", EMBEDDING_MODEL),
        "embedding_dim": embedding_dim,
        "num_chunks": len(all_chunks),
        "chunks": all_chunks,
    }
    with open(KB_PATH, "w", encoding="utf-8") as f:
        json.dump(updated_kb, f, ensure_ascii=False, indent=2)

    new_sources = {}
    for c in all_chunks:
        new_sources.setdefault(c["source"], 0)
        new_sources[c["source"]] += 1

    print(f"\nDone. Updated knowledge base:")
    for src, count in new_sources.items():
        print(f"  {src}: {count} chunks")
    print(f"  Total: {len(all_chunks)} chunks")


if __name__ == "__main__":
    main()
