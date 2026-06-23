#!/usr/bin/env python3
"""
Ragas-based RAG evaluation for the exercise session chatbot.

Metrics computed:
  ContextRecall    - can the ground truth be attributed to retrieved contexts?
  Faithfulness     - does the generated answer stay within the retrieved context?
  AnswerRelevancy  - is the generated answer relevant to the question?

Faithfulness and AnswerRelevancy require a generated response (--generate-responses).
ContextRecall runs without one and is the most informative metric for retriever quality.

Requirements:
  pip install -r eval/requirements-eval.txt

Ragas uses an LLM internally for scoring. Defaults to Groq (GROQ_API_KEY must be set
in .env or environment). Pass --ragas-llm openai to use OpenAI instead.

Usage:
  # Context Recall + live retriever (default, most useful):
  python eval/eval_rag.py

  # Full three-metric evaluation (retriever + LLM responses):
  python eval/eval_rag.py --generate-responses

  # Skip live retrieval (uses ground truth reference_contexts — scores will be inflated):
  python eval/eval_rag.py --no-live-retrieval

  # Use OpenAI instead of Groq for Ragas scoring:
  python eval/eval_rag.py --ragas-llm openai

  # Use a free Ollama instance (e.g. tunneled from Google Colab) for Ragas scoring:
  #   set OLLAMA_BASE_URL to the tunnel URL + /v1 before running:
  python eval/eval_rag.py --generate-responses --ragas-llm ollama
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import ModuleType

from dotenv import load_dotenv
load_dotenv()

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

GROUND_TRUTH_PATH = REPO_ROOT / "eval" / "rag_ground_truth.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "eval" / "rag_results.json"

# ── Ragas compatibility shim ───────────────────────────────────────────────────
# ragas hard-imports langchain_community.chat_models.vertexai.ChatVertexAI at startup.
# That class was removed from langchain-community 0.3+ (moved to langchain-google-vertexai).
# Pre-populating sys.modules with a stub prevents the ImportError without requiring
# Google Cloud packages.
def _stub_vertexai() -> None:
    key = "langchain_community.chat_models.vertexai"
    if key not in sys.modules:
        stub = ModuleType(key)
        stub.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[attr-defined]
        sys.modules[key] = stub

_stub_vertexai()

# ── Ragas ─────────────────────────────────────────────────────────────────────

try:
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics.collections import AnswerRelevancy, ContextRecall, Faithfulness
except ImportError as _e:
    print(f"ragas import failed: {_e}")
    print("Install dependencies with:  pip install -r eval/requirements-eval.txt")
    sys.exit(1)


# Default ragas *judge* model per backend. Deliberately NOT the same as the
# generator (llama-3.1-8b-instant, the RPi production model under test): an 8B
# judge parrots the JSON schema back to instructor instead of an instance, so
# AnswerRelevancy raises and returns nan on every sample, and ContextRecall is
# unstable (intermittent NaNs). A stronger judge fixes both. The generator stays
# 8B — only the scoring LLM changes. Override with --judge-model.
_DEFAULT_JUDGE_MODEL = {
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "ollama": None,  # resolved from OLLAMA_MODEL below
}


def build_ragas_llm(backend: str, judge_model: str | None = None):
    """Return a Ragas InstructorLLM via llm_factory using an async client (required by ragas 0.4.x)."""
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    if backend == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("GROQ_API_KEY not set. Add it to your .env file.")
            sys.exit(1)
        model = judge_model or _DEFAULT_JUDGE_MODEL["groq"]
        client = AsyncOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        print(f"  ragas judge: groq/{model}")
        return llm_factory(model, client=client)

    if backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print(
                "OPENAI_API_KEY not set. Switch to --ragas-llm groq (uses your existing key),\n"
                "or set OPENAI_API_KEY in your .env file."
            )
            sys.exit(1)
        model = judge_model or _DEFAULT_JUDGE_MODEL["openai"]
        client = AsyncOpenAI(api_key=api_key)
        print(f"  ragas judge: openai/{model}")
        return llm_factory(model, client=client)

    if backend == "ollama":
        import httpx

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        model = judge_model or os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
        # Long-running requests through the ngrok tunnel (multi-thousand-token
        # generations) can outlast ngrok's idle keep-alive window. httpx then
        # reuses a connection ngrok already closed, which hangs forever with no
        # error. max_keepalive_connections=0 forces a fresh connection per
        # request instead of reusing a possibly-dead one; the explicit timeout
        # is a backstop so any unexpected hang fails loudly instead of forever.
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=0, max_connections=5),
            timeout=httpx.Timeout(120.0, connect=15.0),
        )
        # Ollama's OpenAI-compatible endpoint ignores the API key but the client requires one.
        # ngrok's free tier serves an HTML interstitial instead of proxying through unless
        # this header is present on every request.
        client = AsyncOpenAI(
            api_key="ollama",
            base_url=base_url,
            default_headers={"ngrok-skip-browser-warning": "true"},
            http_client=http_client,
            timeout=120.0,
        )
        # llama3.1:8b is far more verbose than Groq's hosted instance at default
        # settings, padding every classification with long-winded reasoning and
        # blowing through max_tokens on samples with many ground-truth statements.
        # temperature=0 makes it answer more directly instead of elaborating.
        return llm_factory(model, client=client, temperature=0.0)

    raise ValueError(f"Unknown ragas-llm backend: {backend!r}")


# ── Retrieval ─────────────────────────────────────────────────────────────────

def load_retriever():
    from session_chat.retrieval import RetrievalEngine
    return RetrievalEngine(
        kb_path=str(REPO_ROOT / "data" / "knowledge_base.json"),
        model_dir=str(REPO_ROOT / "data" / "embedding_model"),
    )


def build_ragas_embeddings():
    """Load the same all-MiniLM-L6-v2 model used for the knowledge base, locally."""
    from ragas.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        use_api=False,
        normalize_embeddings=True,
    )


# ── Response generation ───────────────────────────────────────────────────────

# Realistic session stub so the LLM can answer session-specific questions
# (e.g. sets/reps for these exercises) without refusing due to an empty exercise list.
_EVAL_SESSION_STUB = {
    "date": "2026-01-15",
    "duration_seconds": 1800,
    "overall_form_score_pct": 71,
    "total_exercises_detected": 5,
    "exercises": [
        {"name": "Deep Squat", "duration_seconds": 240, "form_score_pct": 72},
        {"name": "Hurdle Step", "duration_seconds": 210, "form_score_pct": 65},
        {"name": "Inline Lunge", "duration_seconds": 240, "form_score_pct": 70},
        {"name": "Standing Leg Raise", "duration_seconds": 180, "form_score_pct": 68},
        {"name": "Side Lunge", "duration_seconds": 200, "form_score_pct": 75},
    ],
}


def generate_response(query: str, retrieved_contexts: list[str]) -> str | None:
    """Generate a chatbot response using Groq. Returns None on failure."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("  GROQ_API_KEY not set; cannot generate responses.")
        return None

    try:
        from groq import Groq
        from session_chat.llm import build_system_prompt

        chunks = [
            {"text": ctx, "source": "knowledge_base", "page": "?", "section_title": ""}
            for ctx in retrieved_contexts
        ]
        system_prompt = build_system_prompt(_EVAL_SESSION_STUB, chunks)

        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            max_tokens=512,
            temperature=0.0,
        )
        return completion.choices[0].message.content
    except Exception as exc:
        print(f"  Response generation failed: {exc}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ragas RAG evaluation for the exercise session chatbot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ground-truth",
        default=str(GROUND_TRUTH_PATH),
        help="Path to the ground truth JSON file (default: eval/rag_ground_truth.json).",
    )
    parser.add_argument(
        "--no-live-retrieval",
        dest="live_retrieval",
        action="store_false",
        help=(
            "Use reference_contexts from the ground truth file instead of running the "
            "actual RetrievalEngine. Context Recall scores will be inflated and meaningless. "
            "Only useful for offline smoke-testing without the ONNX model loaded."
        ),
    )
    parser.set_defaults(live_retrieval=True)
    parser.add_argument(
        "--generate-responses",
        action="store_true",
        help=(
            "Generate chatbot responses via Groq for each query. "
            "Required for Faithfulness and AnswerRelevancy metrics. "
            "Requires GROQ_API_KEY in environment or .env."
        ),
    )
    parser.add_argument(
        "--ragas-llm",
        choices=["openai", "groq", "ollama"],
        default="groq",
        help=(
            "LLM backend for Ragas scoring (default: groq). GROQ_API_KEY must be set. "
            "ollama requires OLLAMA_BASE_URL (e.g. an ngrok tunnel to a Colab instance) "
            "and optionally OLLAMA_MODEL (default: llama3.1:8b)."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help=(
            "Override the ragas judge model (the scoring LLM, NOT the generator). "
            "Defaults per backend: groq=llama-3.3-70b-versatile, openai=gpt-4o-mini, "
            "ollama=$OLLAMA_MODEL. A weak 8B judge returns nan for AnswerRelevancy "
            "(schema echo); the 70B default fixes that. The generated responses are "
            "always from llama-3.1-8b-instant (the RPi production model under test)."
        ),
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help=(
            "Comma-separated subset of ragas metrics to run (e.g. 'AnswerRelevancy'). "
            "Default: all applicable. Use to iterate on one metric without spending the "
            "judge's daily token budget on the others."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to write per-sample results JSON (default: eval/rag_results.json).",
    )
    parser.add_argument(
        "--no-log",
        dest="log_metrics",
        action="store_false",
        help="Do not append this run to rag_metrics.md (logging is on by default).",
    )
    parser.set_defaults(log_metrics=True)
    args = parser.parse_args()

    # ── Load ground truth ──────────────────────────────────────────────────────
    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        print(f"Ground truth file not found: {gt_path}")
        sys.exit(1)

    with open(gt_path, encoding="utf-8") as f:
        gt_data = json.load(f)

    samples_raw = gt_data["samples"]
    print(f"Loaded {len(samples_raw)} samples from {gt_path.name}")

    # ── Initialise optional components ────────────────────────────────────────
    retriever = None
    if args.live_retrieval:
        print("Initialising RetrievalEngine...")
        try:
            retriever = load_retriever()
        except FileNotFoundError as exc:
            print(f"Cannot load retriever: {exc}")
            print("Run build_knowledge_base.py first, or pass --no-live-retrieval to skip retrieval.")
            sys.exit(1)

    ragas_llm = build_ragas_llm(args.ragas_llm, args.judge_model)

    # AnswerRelevancy needs embeddings — load the same model used for the knowledge base.
    ragas_embeddings = None
    if args.generate_responses:
        print("Loading sentence-transformers model for AnswerRelevancy embeddings...")
        try:
            ragas_embeddings = build_ragas_embeddings()
        except Exception as exc:
            print(f"Warning: embeddings unavailable ({exc}). AnswerRelevancy will be skipped.")

    # ── Decide which metrics to run ───────────────────────────────────────────
    # ContextRecall: LLM only, runs even without a generated response.
    # Faithfulness: LLM + response.
    # AnswerRelevancy: LLM + embeddings + response.
    if args.generate_responses:
        metrics = [
            ContextRecall(llm=ragas_llm),
            Faithfulness(llm=ragas_llm),
        ]
        if ragas_embeddings:
            metrics.append(AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings))
        else:
            print("Note: AnswerRelevancy skipped (ONNX model unavailable for embeddings).")
    else:
        metrics = [ContextRecall(llm=ragas_llm)]
        print(
            "ContextRecall only.\n"
            "Pass --generate-responses to also compute Faithfulness and AnswerRelevancy."
        )

    # Optional metric subset — lets you iterate on one metric (e.g. AnswerRelevancy)
    # without spending the judge's daily token budget on all three.
    if args.metrics:
        wanted = {m.strip().lower() for m in args.metrics.split(",")}
        metrics = [m for m in metrics if m.__class__.__name__.lower() in wanted]
        if not metrics:
            print(f"No metrics matched --metrics {args.metrics!r}. "
                  f"Choices: ContextRecall, Faithfulness, AnswerRelevancy.")
            sys.exit(1)
    print(f"Metrics: {[m.__class__.__name__ for m in metrics]}")

    if not args.live_retrieval:
        print(
            "Note: --no-live-retrieval set — using reference_contexts from ground truth.\n"
            "      Context Recall scores will be inflated and do not reflect real retriever quality."
        )

    # ── Build Ragas samples ───────────────────────────────────────────────────
    eval_samples: list[SingleTurnSample] = []

    for item in samples_raw:
        query = item["query"]
        ground_truth = item["ground_truth"]

        # Retrieve contexts
        if retriever is not None:
            raw_chunks = retriever.search(query, top_k=6)
            retrieved_contexts = [c["text"] for c in raw_chunks]
        else:
            retrieved_contexts = item["reference_contexts"]

        # Generate response (optional)
        response = "N/A"
        if args.generate_responses:
            print(f"  Generating response: {query[:70]}...")
            generated = generate_response(query, retrieved_contexts)
            if generated is None:
                print(f"  Skipping sample {item['id']} (response generation failed).")
                continue
            response = generated

        eval_samples.append(
            SingleTurnSample(
                user_input=query,
                retrieved_contexts=retrieved_contexts,
                response=response,
                reference=ground_truth,
            )
        )

    if not eval_samples:
        print("No valid samples — nothing to evaluate.")
        sys.exit(1)

    print(f"\nEvaluating {len(eval_samples)} samples with {[m.__class__.__name__ for m in metrics]}...")

    # ── Run Ragas ─────────────────────────────────────────────────────────────
    # ragas 0.4.x evaluate() only accepts the old Metric subclasses; the collection
    # metrics (ContextRecall, Faithfulness, AnswerRelevancy) are BaseMetric subclasses
    # and must be called via .batch_score() directly.
    inputs = [
        {
            "user_input": s.user_input,
            "retrieved_contexts": s.retrieved_contexts,
            "response": s.response,
            "reference": s.reference,
        }
        for s in eval_samples
    ]

    # Score sequentially to stay within Groq free-tier TPM limits.
    # batch_score fires all samples concurrently; .score() runs one at a time.
    import inspect, time
    all_scores: dict[str, list] = {m.__class__.__name__: [] for m in metrics}
    per_sample_rows = []

    # Groq's free tier needs a gap to stay under 6k TPM; Ollama/OpenAI don't.
    inter_sample_delay = 2 if args.ragas_llm == "groq" else 0

    for metric in metrics:
        print(f"  Scoring {metric.__class__.__name__} (sequential)...")
        required = set(inspect.signature(metric.ascore).parameters) - {"self"}
        scores = []
        for idx, inp in enumerate(inputs):
            metric_input = {k: v for k, v in inp.items() if k in required}
            try:
                results = metric.batch_score([metric_input])
                r = results[0]
                scores.append(r.value if r.value is not None else float("nan"))
            except Exception as exc:
                print(f"    Sample {idx} failed: {exc}")
                scores.append(float("nan"))
            if inter_sample_delay:
                time.sleep(inter_sample_delay)
        all_scores[metric.__class__.__name__] = scores

    for i, sample in enumerate(eval_samples):
        row = {"query": sample.user_input}
        for name, scores in all_scores.items():
            row[name] = scores[i]
        # Persist the generated answer too — needed for the paper and for
        # diagnosing AnswerRelevancy (noncommittal / off-topic answers).
        if args.generate_responses:
            row["response"] = sample.response
        per_sample_rows.append(row)

    print("\n=== Aggregate Results ===")
    for name, scores in all_scores.items():
        valid = [s for s in scores if not (s != s)]  # filter NaN
        avg = sum(valid) / len(valid) if valid else float("nan")
        print(f"  {name}: {avg:.4f}  (n={len(valid)})")

    # ── Save per-sample results ───────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(per_sample_rows, f, indent=2, ensure_ascii=False)
    print(f"\nPer-sample results saved to {output_path}")

    # ── Auto-log to rag_metrics.md (the paper's running record) ───────────────
    if args.log_metrics:
        if args.ragas_llm == "groq":
            judge_label = "groq/" + (args.judge_model or _DEFAULT_JUDGE_MODEL["groq"])
        elif args.ragas_llm == "openai":
            judge_label = "openai/" + (args.judge_model or _DEFAULT_JUDGE_MODEL["openai"])
        else:
            judge_label = "ollama/" + (args.judge_model or os.environ.get("OLLAMA_MODEL", "llama3.1:8b"))
        try:
            from eval.log_metrics import append_run
        except ImportError:
            from log_metrics import append_run
        append_run(
            per_sample_rows,
            {
                "label": f"eval_rag.py · judge {judge_label}",
                "generator": "llama-3.1-8b-instant" if args.generate_responses else "N/A (retrieval only)",
                "judge": judge_label,
                "kb": "data/knowledge_base.json",
                "top_k": 6,
                "engine": args.ragas_llm,
            },
        )
        print("Run appended to rag_metrics.md")


if __name__ == "__main__":
    main()
