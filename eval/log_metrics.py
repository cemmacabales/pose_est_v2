#!/usr/bin/env python3
"""
Append a RAG eval run to rag_metrics.md (the running results log for the paper).

Shared by both runners so every run is recorded the same way, automatically:
  - eval/eval_rag.py   (local, Groq/OpenAI/Ollama judge)
  - eval/colab_rag_eval.ipynb  (Colab, local gpt-oss:20b judge)

Each call appends a dated section with an aggregate table (mean / n / nan per
metric, NaN-safe) and a per-query table. Only the metrics actually present in the
rows are logged, so a 3-metric Groq run and a 5-metric Colab run both work.

CLI:
  python eval/log_metrics.py eval/rag_results_colab.json \
      --label "Colab local judge" --generator llama-3.1-8b --judge gpt-oss:20b \
      --top-k 6 --engine ollama --kb "heading-aware (rebuilt)"
"""

import argparse
import json
import math
from datetime import date
from pathlib import Path

# Logged in this order when present; ragas first, then the reference-based extras.
METRIC_ORDER = [
    "ContextRecall",
    "Faithfulness",
    "AnswerRelevancy",
    "SemanticSimilarity",
    "RougeL",
]

DEFAULT_MD = Path(__file__).resolve().parent.parent / "rag_metrics.md"


def _is_num(v):
    return isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))


def summarize(rows):
    """Return {metric: (mean, n_valid, n_nan)} for every metric present in rows."""
    present = [m for m in METRIC_ORDER if any(m in r for r in rows)]
    out = {}
    for m in present:
        vals = [r.get(m) for r in rows if m in r]
        valid = [v for v in vals if _is_num(v)]
        mean = sum(valid) / len(valid) if valid else float("nan")
        out[m] = (mean, len(valid), len(vals) - len(valid))
    return out


def _cell(v):
    return "—" if not _is_num(v) else f"{v:.2f}"


def render_section(rows, meta):
    agg = summarize(rows)
    metrics = list(agg.keys())
    run_date = meta.get("date") or date.today().isoformat()
    label = meta.get("label", "run")

    lines = [f"\n## {run_date} — {label} (auto-logged)\n"]

    cfg = []
    for k in ("generator", "judge", "kb", "top_k", "engine"):
        if meta.get(k) not in (None, ""):
            cfg.append(f"**{k}**: {meta[k]}")
    if cfg:
        lines.append(" · ".join(cfg) + "\n")

    # Aggregate table
    lines.append("| Metric | Mean | n | NaN |")
    lines.append("|--------|:----:|:-:|:---:|")
    for m in metrics:
        mean, n, nan = agg[m]
        mean_s = "nan" if not _is_num(mean) else f"{mean:.4f}"
        lines.append(f"| {m} | {mean_s} | {n} | {nan} |")

    # Per-query table
    lines.append("")
    header = "| # | Query | " + " | ".join(metrics) + " |"
    sep = "|---|-------|" + "|".join([":---:"] * len(metrics)) + "|"
    lines.append(header)
    lines.append(sep)
    for i, r in enumerate(rows, 1):
        q = r.get("query", "")[:60].replace("|", "\\|")
        cells = " | ".join(_cell(r.get(m)) for m in metrics)
        lines.append(f"| {i} | {q} | {cells} |")

    return "\n".join(lines) + "\n"


def append_run(rows, meta, md_path=DEFAULT_MD):
    """Append a run section to md_path. Returns the section text."""
    section = render_section(rows, meta)
    md_path = Path(md_path)
    # Create with a minimal header if missing (normally it already exists).
    if not md_path.exists():
        md_path.write_text("# RAG Evaluation Metrics Log\n", encoding="utf-8")
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(section)
    return section


def _load_rows(path):
    data = json.load(open(path, encoding="utf-8"))
    # Accept either a bare list of per-sample rows or {"rows": [...]}.
    return data["rows"] if isinstance(data, dict) and "rows" in data else data


def main():
    ap = argparse.ArgumentParser(description="Append a RAG eval run to rag_metrics.md.")
    ap.add_argument("results", help="Per-sample results JSON (list of rows with metric keys).")
    ap.add_argument("--label", default="run", help="Short run label (engine/judge summary).")
    ap.add_argument("--generator", default=None)
    ap.add_argument("--judge", default=None)
    ap.add_argument("--kb", default=None)
    ap.add_argument("--top-k", dest="top_k", default=None)
    ap.add_argument("--engine", default=None)
    ap.add_argument("--date", default=None, help="Override the run date (YYYY-MM-DD).")
    ap.add_argument("--md", default=str(DEFAULT_MD), help="Path to rag_metrics.md.")
    args = ap.parse_args()

    rows = _load_rows(args.results)
    meta = {
        "label": args.label, "generator": args.generator, "judge": args.judge,
        "kb": args.kb, "top_k": args.top_k, "engine": args.engine, "date": args.date,
    }
    section = append_run(rows, meta, args.md)
    print(f"Appended run '{args.label}' ({len(rows)} samples) to {args.md}")
    print(section)


if __name__ == "__main__":
    main()
