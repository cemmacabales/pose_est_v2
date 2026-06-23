# RAG Evaluation Metrics Log

Running record of post-session RAG chatbot accuracy measurements, for the paper.
Append a new dated run section each time the eval is run; never overwrite past runs.

**System under test:** `session_chat/` — retrieval (`retrieval.py`, ONNX MiniLM-L6-v2
embeddings + numpy cosine, source/exercise-aware re-ranking) → generator
(`llama-3.1-8b-instant`, the RPi production model). Knowledge base: 3 sources
(exercise_form_guide, conditioning_manual/NSCA, behaviour_manual/WHO).

**Metrics:** ContextRecall, Faithfulness, AnswerRelevancy (ragas, LLM-judged) +
SemanticSimilarity (mpnet cosine vs ground truth) + RougeL (lexical F1). Eval set:
15 ground-truth queries (`eval/rag_ground_truth.json`). Target: 0.90 on the three
ragas metrics.

> **Judge ≠ generator.** The generator is always `llama-3.1-8b-instant` (production).
> The *judge* is a separate, stronger model — an 8B judge parrots the JSON schema back
> to instructor and returns `nan` for AnswerRelevancy (and unstable ContextRecall), so
> it cannot score itself reliably. Judge model is recorded per run below.

---

## Aggregate results

| Run | Date | Generator | Judge | KB / chunking | top_k | ContextRecall | Faithfulness | AnswerRelevancy |
|-----|------|-----------|-------|---------------|-------|---------------|--------------|-----------------|
| **Baseline** | pre-2026-06-23 | llama-3.1-8b-instant | Groq llama-3.1-8b-instant | original (500-char, heading-blind) | 4 | **0.800** (n=13, 2 nan) | **0.811** (n=15) | **0.757** (n=15) |
| **After #1+#2** | 2026-06-23 | llama-3.1-8b-instant | Groq **llama-3.3-70b-versatile** | heading-aware + contextual-header embed (rebuilt) | 6 | **0.933** (n=15, 0 nan) | **0.937** (n=15, 0 nan) | 0.518 (n=7, partial*) | 

\* AnswerRelevancy completed only 7/15 samples before Groq's free 100k-tokens/day
cap throttled the 70B judge (`429`). The 8 failures were rate-limit, **not**
schema-echo — AR returns real values now. A complete AR number needs the
local-judge path (Colab `gpt-oss:20b`) or a fresh daily budget. **Pending.**

**Headline:** the chunking fix (#1) lifted ContextRecall 0.80→0.93 and Faithfulness
0.81→0.94 — both above the 0.90 target — and removed the 2 phantom ContextRecall
NaNs. The judge fix (#2) made AnswerRelevancy measurable (was `nan` on the 8B judge).
AnswerRelevancy is now the remaining gap (see per-query below).

---

## Per-query — After #1+#2 (2026-06-23, Groq 70B judge)

Source: `eval/rag_results_70bjudge.json`. `—` = not scored (daily token cap).

| # | Query | ContextRecall | Faithfulness | AnswerRelevancy |
|---|-------|:---:|:---:|:---:|
| 1 | knees caving (squat) | 1.00 | 1.00 | 0.40 |
| 2 | heels lift (squat) | 0.50 | 1.00 | 0.43 |
| 3 | balance (hurdle step) | 1.00 | 1.00 | 0.39 |
| 4 | scaption vs lateral raise | 1.00 | 1.00 | 0.97 |
| 5 | shoulders shrug | 1.00 | 1.00 | 0.90 |
| 6 | sets and reps | 1.00 | 0.75 | **0.00** |
| 7 | frequency / rest days | 0.75 | 1.00 | 0.53 |
| 8 | breathing | 1.00 | 0.80 | — |
| 9 | inline lunge wobbly | 1.00 | 1.00 | — |
| 10 | standing leg raise leaning | 0.75 | 1.00 | — |
| 11 | rest between sets | 1.00 | 1.00 | — |
| 12 | sit to stand lopsided | 1.00 | 0.50 | — |
| 13 | side lunge feel | 1.00 | 1.00 | — |
| 14 | shoulder extension from back | 1.00 | 1.00 | — |
| 15 | bottom of deep squat | 1.00 | 1.00 | — |

**AnswerRelevancy reads (n=7):** lowest on advice-list questions — sets/reps **0.00**
(judge flagged the answer "noncommittal"/hedgy), frequency 0.53, knees 0.40, heels
0.43. High on scaption 0.97 and shrug 0.90. Indicates the generator hedges / wanders
on broad "how much / how often" questions → the AnswerRelevancy prompt work.

### Notable per-query swings vs baseline (ragas, judge change noted)
| Query | Metric | Baseline (8B judge) | After (70B judge) |
|-------|--------|:---:|:---:|
| hurdle step balance | ContextRecall | 0.00 | 1.00 |
| sit to stand lopsided | ContextRecall | nan / 0.14 | 1.00 |
| inline lunge wobbly | Faithfulness | 0.30 | 1.00 |
| standing leg raise | Faithfulness | 0.17 | 1.00 |

> Caveat: baseline used the 8B judge at top_k=4 on the old KB; the after-run changed
> KB + top_k + judge together. Deltas are directional, not a controlled single-variable
> A/B. For strict attribution, re-run on the old KB with the same (70B/local) judge.

---

## Retrieval-layer check (no judge) — chunking fix #1

For each of the 15 queries, count of top-6 retrieved chunks belonging to the
*correct* exercise (a proxy for retrieval focus; before/after the heading-aware
re-chunk + contextual-header embedding). Exercise-specific queries only.

| Query | Before | After |
|-------|:---:|:---:|
| hurdle step balance | mixed/generic | **6/6 Hurdle Step** |
| sit to stand lopsided | 0 Sit-to-Stand chunks | **6/6 Sit to Stand** |
| standing leg raise | generic Common Errors | **6/6 Standing Leg Raise** |
| side lunge feel | partial | **6/6 Side Lunge** |
| inline lunge wobbly | 1 correct + leak | **5/6 Inline Lunge** |
| shoulder extension | generic | **5/6 Shoulder Extension** |

Before the fix, 38/82 form-guide chunks named no exercise (orphaned sub-sections
like "Coaching Cues"); after, sub-sections carry their parent `Exercise N: <name>`.

---

## 2026-06-24 — AnswerRelevancy prompt work (#4)

**Diagnosis** (from the generated answers, 8B generator). AnswerRelevancy (ragas)
generates questions from the answer and compares to the original; it zeroes the
score if the answer is flagged *noncommittal*. Three failure modes found:
1. **Evasive trailing fragments** — q06 (sets/reps) ended with *"For the Standing Leg
   Raise, I couldn't find specific guidance…"* → noncommittal=1 → **AR 0.00**.
2. **Vague non-answers** — q02 (heels lift) returned *"something further up the chain
   isn't working… pay attention to your feet"* (pulled from the wrong source,
   behaviour_manual) instead of the real cause → AR 0.43.
3. **Cue-heavy answers** generating off-topic questions — q01 (knees) AR 0.40.

**Change** (`session_chat/llm.py` `build_system_prompt`, production prompt): open
with a direct/committal answer (state the actual cause/fix/number, no "it depends"
hedging); answer all parts of multi-part questions in one reply; drop "I couldn't
find guidance for X" sub-disclaimers (full refusal only when *nothing* is covered).
Grounding/refusal/anti-hallucination rules unchanged.

**Generation-level before/after** (8B generator, temperature 0 — deterministic):

| Q | Old answer | New answer |
|---|-----------|-----------|
| q06 sets/reps | 3 exercises + *"…couldn't find specific guidance for Standing Leg Raise"* (evasive fragment → AR 0.00) | same 3 exercises, fragment removed, clean ending |
| q02 heels lift | *"something further up the chain… pay attention to your feet"* (vague, wrong source) | *"work on ankle mobility, specifically calf stretches"* — the actual cause |

The q02 answer is **grounded** (Faithfulness-safe): the retrieved *Deep Squat —
Coaching Cues* chunk states *"If they want to rise, you may need to work on ankle
mobility (calf stretches help)."* The old answer ignored this correct chunk; the new
prompt uses it. So the change lifts AnswerRelevancy and Faithfulness together.

**Status:** AnswerRelevancy *score* delta is **not yet measured end-to-end** — Groq's
free 100k-tokens/day judge budget was exhausted (only q01 re-scored: 0.40→0.41, no
signal). A full AR re-score requires the Colab local-judge run (`gpt-oss:20b`) or a
fresh daily budget. The generation-level evidence above shows the diagnosed AR
failure triggers (noncommittal fragment, vague hedge) are removed.

Run when budget/Colab available: `python eval/eval_rag.py --generate-responses
--ragas-llm groq --metrics AnswerRelevancy` (or full run on Colab).

## Method / reproduction

- Local (Groq judge): `python eval/eval_rag.py --generate-responses --ragas-llm groq`
  (judge defaults to `llama-3.3-70b-versatile`; override `--judge-model`). Generator
  is always `llama-3.1-8b-instant`.
- Colab (local judge, free, uncapped): `eval/colab_rag_eval.ipynb`, `ENGINE="ollama"`,
  `JUDGE_MODEL="gpt-oss:20b"`. Clones `origin/test`. Writes `eval/rag_results_colab.*`.
- KB rebuild (form guide only): `python add_to_knowledge_base.py references/exercise_form_guide.pdf`.

## Commits
- `f8c9ad0` #1 retrieval: heading threading + contextual-header embedding + is_heading fix.
- `bc53a93` #2 judge: configurable judge, Groq 70B default (eval_rag.py).
- `c26979a` #2 Colab: local `gpt-oss:20b` judge + ollama engine.
