# Progress

Quick-glance status for the team. Git history has the details; this is for orientation without digging through commits.

## Pose estimation + classifier
BlazePose (MediaPipe) -> dual-head LSTM classifier (9 exercises + form quality), exported to TFLite for RPi 5 deployment. Stable per `README.md` -- not touched in the work described below. See `README.md` for the training pipeline and `METHODOLOGY.md` for write-up details.

## Post-session RAG chat
Groq-hosted LLM (`llama-3.1-8b-instant`) + local ONNX embeddings answer coaching questions after a session, grounded in a 3-source knowledge base (WHO physical activity guidelines, NSCA conditioning manual, a custom exercise form guide).

**Done:**
- Knowledge base built and embedded (`data/knowledge_base.json`, 1478 chunks across 3 sources)
- Ragas-based eval pipeline (`eval/eval_rag.py`) scoring ContextRecall, Faithfulness, AnswerRelevancy against a 15-query realistic coaching dataset (`eval/rag_ground_truth.json`)
- Targeted knowledge base fixes for queries that scored poorly (sets/reps refusal, side lunge sensation, standing leg raise lean cause, sit-to-stand asymmetry) -- some fully resolved, tracked per-query in eval history
- Source-aware + exercise-aware retrieval re-ranking (`session_chat/retrieval.py`) to stop the broad NSCA manual from crowding out more specific content, and to stop one exercise's content leaking into a different exercise's query. Improved aggregate scores: ContextRecall 0.76->0.80, Faithfulness 0.70->0.81, AnswerRelevancy 0.68->0.76 on the same 15-query set, same Groq judge.

**Known limitation, not a bug to keep chasing:**
ContextRecall can't be scored for a few samples with long ground-truth answers (notably the standing leg raise and sit-to-stand queries) because Groq's free-tier rate limit caps how much the judge model can classify in one call. This only affects *evaluation* -- the live chatbot makes one lightweight generation call per message and has never hit this limit. Document as a free-tier eval-tooling constraint, not a system defect.

**Decided against, for reference (don't re-litigate without new information):**
- Running the eval's judge LLM on Ollama via Colab+ngrok tunnel, to get free higher-volume scoring. Worked for isolated requests but repeatedly hung or crashed under the sustained load a full eval run needs, across multiple GPU tiers and several different fixes. Abandoned for reliability, not for any architectural reason.

**Still open, not actually settled:**
- Switching the judge LLM to OpenAI/Gemini for scoring only. The RPi constraint applies to the *generator* (the chatbot's response must come from what's actually deployed, Groq's `llama-3.1-8b-instant`) -- it does not apply to the judge, which never runs on the RPi and isn't part of what ships; the eval script itself is a Mac/Colab dev tool per `eval/requirements-eval.txt`. So judge != generator is not an architectural problem. The real tradeoffs are practical: a different judge sidesteps Groq's free-tier rate-limit/NaN issue, but means re-running everything for a clean, single-judge baseline (current results aren't comparable across judges), and there's a teammate preference for keeping the same model throughout worth discussing before deciding either way.

**Branch state:**
- `feat/add-rag-metrics` -- eval pipeline, knowledge base content, ground truth dataset. Not yet merged into `test`.
- `feat/source-aware-retrieval` -- retrieval re-ranking fix, branched off `feat/add-rag-metrics`. [PR #15](https://github.com/cemmacabales/pose_est_v2/pull/15) open against `test` (diff currently includes both branches' commits since the base hasn't merged yet).

**Next steps for whoever picks this up:**
- Live chat smoke test for the queries affected by the retrieval fix
- Decide whether to merge `feat/add-rag-metrics` into `test` first to clean up PR #15's diff
- Decide on the judge LLM question above (same-model-as-generator vs. switching judge to OpenAI/Gemini) -- discuss with the team before re-running a new baseline either way
- Optional: try shortening ground-truth/context for the rate-limit-affected samples to get them scoring without a paid API tier
