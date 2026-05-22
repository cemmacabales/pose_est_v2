# Handoff Document — Round 2 Retraining Session

**Date:** 2026-05-22
**Status:** All phases complete. 174 videos collected, model retrained, TFLite exported, RPi-ready.

---

## What Was Accomplished

Completed data collection (Phase 6) and retrained the pose exercise classifier (Phase 7)
with **174 real-world videos from 5 people**, replacing the 26-video initial model with
a diverse 5-person, camera-angle, and incorrect-form dataset.

### Phases Completed

| Phase | Status | Summary |
|-------|--------|---------|
| **Phase 6** | Done | Collected 174 videos from 5 people across all 9 exercises |
| **Phase 7** | Done | Extracted keypoints, built windows (451K), trained model, exported TFLite |
| **Deploy** | Done | `.gitignore` cleaned, `rpi_setup.sh` simplified, pushed and ready for RPi |

---

## Data Collection Results

### Actual vs Planned

| Person | Expected | Actual | Notes |
|--------|----------|--------|-------|
| Person 1 | 30 | 33 | Extra front/lateral variants |
| Person 2 | 30 | 37 | Left/right splits for each incorrect mistake |
| Person 3 | 30 | 47 | Left/right splits — most overshoot |
| Person 4 | 30 | 28 | 2 videos short, descriptive filenames |
| Person 5 | 30 | 30 | Exact naming convention |
| **TOTAL** | **150** | **174** | |

### Per-Exercise Distribution

| Exercise | Correct | Incorrect | Total | People |
|----------|---------|-----------|-------|--------|
| 01 Deep Squat | 12 | 10 | 22 | P1, P2 |
| 02 Hurdle Step | 14 | 12 | 26 | P1, P3 |
| 03 Inline Lunge | 14 | 12 | 26 | P1, P3 |
| 04 Side Lunge | 14 | 12 | 26 | P2, P4 |
| 05 Sit to Stand | 12 | 9 | 21 | P2, P5 |
| 06 Standing Leg Raise | 12 | 12 | 24 | P3, P4 |
| 07 Shoulder Abduction | 5 | 4 | 9 | P4 only |
| 08 Shoulder Extension | 6 | 4 | 10 | P5 only |
| 10 Shoulder Scaption | 6 | 4 | 10 | P5 only |

---

## Model Performance

### Random 80/20 Split (same people, held-out windows)
| Split | Exercise Acc | Quality Acc |
|-------|-------------|-------------|
| Train | 90.0% | 87.9% |
| Val | 88.3% | 86.3% |

### Person-Based Split (unseen people held out)
| Split | Exercise Acc | Quality Acc | Meaning |
|-------|-------------|-------------|---------|
| Train (P1+P2+P3) | 98.6% | 84.4% | Memorized training subjects |
| Val (P4) | 15.5% | 50.9% | Near random (11% is chance) |
| Test (P5) | 10.2% | 66.8% | Near random (11% is chance) |

### ⚠️ Critical Finding: Generalization Gap

The model **cannot recognize exercises from unseen people**. With only 3 training people,
there isn't enough body-type, movement-style, and camera-angle variation for the LSTM
to learn exercise patterns that generalize. This is a **data diversity problem**, not a
model architecture problem.

### Fix Options

| Approach | Effort | Impact |
|----------|--------|--------|
| **20+ training people** | High (data collection) | Most impactful |
| **Joint-angle features** | Medium (code changes) | Body-invariant but camera-angle dependent |
| **3D pose estimation** | High (new model) | Full fix |
| **Per-person normalization** | Low (code) | Tested — didn't help |

---

## The 9 Classes

| Index | Exercise | ID | Training Videos |
|-------|----------|----|-----------------|
| 0 | Deep Squat | 01 | 22 |
| 1 | Hurdle Step | 02 | 26 |
| 2 | Inline Lunge | 03 | 26 |
| 3 | Side Lunge | 04 | 26 |
| 4 | Sit to Stand | 05 | 21 |
| 5 | Standing Leg Raise | 06 | 24 |
| 6 | Shoulder Abduction | 07 | 9 |
| 7 | Shoulder Extension | 08 | 10 |
| 8 | Shoulder Scaption | 10 | 10 |

**Note:** Exercise 09 (Shoulder Rotation) dropped — no video available.

---

## Artifacts

| Artifact | Location | Size | Purpose |
|----------|----------|------|---------|
| `classifier.keras` | `models/` | 504 KB | Final trained model (gitignored) |
| `classifier.tflite` | `models/` | 221 KB | TFLite for RPi5 deployment (tracked) |
| `X_train.npy` | `data/` | 1.3 GB | 451K × 30 × 24 windows (gitignored) |

---

## Repo State

### Tracked (on remote)
```
.gitignore, README.md, requirements.txt, rpi_setup.sh
gui.py, joint_map.py, tts_engine.py, session_logger.py
session_chat/ (5 files)
models/classifier.tflite, models/movenet_thunder_int8.tflite
tests/ (6 files)
HANDOFF.md, RETRAIN_PROGRESS.md
```

### Gitignored (not tracked)
```
data/ — training data, knowledge base, embeddings
eval/ — evaluation scripts, raw/test videos
data_collection/, references/ — checklists, PDFs
models/*.keras — intermediate build artifacts
logs/, .venv/, __pycache__/
build_*.py, extract_*.py, train_*.py, load_*.py — build-only scripts
```

### RPi Deployment Checklist
- [x] `git pull` gets all tracked files
- [x] `bash rpi_setup.sh` installs all deps
- [x] `models/classifier.tflite` is the 174-video, 9-class model
- [x] Knowledge base copied via `scp` (data/knowledge_base.json + data/embedding_model/)
- [x] `GROQ_API_KEY` set in `.env`

---

## Running the Pipeline (Quick Reference)

```bash
source .venv/bin/activate

# Preprocess new videos (if adding more data)
python3 preprocess_train_videos.py

# Extract keypoints (~30 min for 174 videos)
python3 extract_video_keypoints.py --video_dir eval/train_videos_flat

# Build training windows
python3 build_windows.py

# Train model (~30-60 min on M2)
python3 train_classifier.py

# Export TFLite
python3 export_models.py

# Run tests
python3 -m pytest tests/ -v --ignore=tests/test_app.py --ignore=tests/test_llm.py
```

## Tests

```
tests/test_gui_integration.py ..... 7/7
tests/test_retrieval.py ............ 5/5
tests/test_session_logger.py ...... 8/8
tests/test_tts_engine.py .......... 8/8
                                      28/28
```

Note: `test_app.py` and `test_llm.py` require `groq` — install with `pip install groq`.

---

*End of handoff. The model is deployed and runs on RPi5. Generalization to unseen
people is still the open problem.*
