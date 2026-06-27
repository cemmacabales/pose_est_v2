# Methodology — Pose Estimation + Exercise Classifier

This document describes the complete methodology, pipeline, and engineering decisions behind the **Pose Estimation + Exercise Classifier** project. It covers data collection, model architecture, training procedures, real-time inference, and the post-session coaching chat system.

---

## 1. Project Overview

The goal is to build a real-time exercise classification and form-quality assessment system that runs on a **Raspberry Pi 5**. The system:

- Detects human pose from a webcam feed using **MediaPipe BlazePose**.
- Classifies **9 functional movement exercises** in real time.
- Scores **form quality** (correct vs. incorrect) for each exercise.
- Counts **repetitions** per exercise.
- Provides **text-to-speech (TTS) feedback** during the workout.
- Logs the full session and serves a **post-session RAG chat** with a fitness coaching assistant.

**Target hardware:** Raspberry Pi 5 (ARM64, no GPU).  
**Development hardware:** Mac M2 (training, model export, knowledge base building).

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Webcam → OpenCV → BlazePose (33 landmarks)                │
│                     ↓                                       │
│  12 mapped joints (hip-centered) → 16 angle features     │
│                     ↓                                       │
│  30-frame window → Dual-Head LSTM (TFLite)                 │
│                     ↓                                       │
│  Exercise class (9) + Quality class (2)                   │
│                     ↓                                       │
│  Rep Counter + TTS + Session Logger + GUI                  │
│                     ↓                                       │
│  Post-session: Flask → RAG + Groq LLM → Chat UI          │
└─────────────────────────────────────────────────────────────┘
```

### Component Map

| Component | File | Role |
|-----------|------|------|
| Pose Estimation | `pose_estimator.py` | BlazePose wrapper (IMAGE / VIDEO / LIVE_STREAM modes) |
| Joint Mapping | `joint_map.py` | 12 joint indices from 33 BlazePose landmarks |
| Angle Features | `joint_angles.py` | Convert 12 joints → 16 biomechanical angle features |
| Classifier | `models/classifier.tflite` | Dual-head LSTM (exercise + quality) |
| GUI | `gui.py` | Tkinter real-time display, camera switching, QR code |
| Rep Counter | `rep_counter.py` | EMA-threshold-based repetition counting per exercise |
| TTS | `tts_engine.py` | Cross-platform audio feedback (gTTS / espeak / afplay / say) |
| Session Logger | `session_logger.py` | JSON session export with per-exercise metrics |
| Chat Server | `session_chat/app.py` | Flask server for post-session coaching chat |
| RAG Retrieval | `session_chat/retrieval.py` | ONNX embeddings + cosine similarity over PDF knowledge base |
| LLM | `session_chat/llm.py` | Groq API (Llama 3.1-8B) with grounded system prompt |
| Web Demo | `web/index.html` + `app.js` | Browser-based TF.js demo |

---

## 3. Data Collection Methodology

### 3.1 Recording Protocol

Training data was collected from **5 subjects** (Person 1–5). Each person recorded **3 exercises**, with **6 correct videos** and **4 incorrect videos** per exercise, for a planned total of **30 videos per person** (150 videos total). The actual collected dataset was **174 training videos** (some videos were dropped or combined).

**Recording specifications:**
- **Camera:** Smartphone on tripod, stationary.
- **Distance:** 6–8 feet from subject.
- **Framing:** Full body visible (head to toes).
- **Lighting:** Subject faces the light; no backlit windows.
- **Background:** Plain wall.
- **Clothing:** Fitted clothes (no baggy hoodies).
- **Duration:** 15–30 seconds per video.
- **Repetitions:** At least 3–5 reps per video.
- **Format:** MP4 (MOV converted via ffmpeg).

**Angular diversity:** Each correct video was recorded from a different angle or speed to improve generalization:
- Frontal view
- Lateral (side) view
- 45° diagonal view
- Slower / faster tempo
- Farther from camera

### 3.2 Exercise Assignments

Exercises were distributed across 5 people to balance coverage:

| Person | Exercises |
|--------|-----------|
| 1 | Deep Squat (01), Hurdle Step (02), Inline Lunge (03) |
| 2 | Deep Squat (01), Side Lunge (04), Sit to Stand (05) |
| 3 | Hurdle Step (02), Inline Lunge (03), Standing Leg Raise (06) |
| 4 | Side Lunge (04), Standing Leg Raise (06), Shoulder Abduction (07) |
| 5 | Sit to Stand (05), Shoulder Extension (08), Shoulder Scaption (10) |

*Note: Exercise 09 was dropped; Exercise 10 maps to class index 8.*

### 3.3 Form Labeling

Each video was explicitly labeled as **correct** or **incorrect**:

- **Correct:** Smooth, controlled, full range of motion, as taught by a physical therapist.
- **Incorrect:** Exactly **one deliberate mistake** per video. Mistakes were chosen from predefined lists per exercise (e.g., "knees cave inward," "heels lift off ground," "don't go down far enough").

This single-mistake-per-video rule ensures the classifier learns specific form errors rather than conflating multiple failure modes.

### 3.4 Test Videos

**26 additional test videos** from `eval/TEST/` were added later for fine-tuning. These featured:
- Different camera angles (frontal, lateral, 45°)
- Subjects not seen during initial training
- All labeled as **correct** (demonstration/reference videos)

---

## 4. Preprocessing Pipeline

### 4.1 Video Flattening

`preprocess_train_videos.py` flattens the nested folder structure (`eval/train_videos/Person{1..5}/`) into canonical filenames:

```
<exercise_id>_<quality>_P<person>_<take>.mp4
```

Example: `01_correct_P1_1.mp4`

- Handles `.mov` → `.mp4` conversion via ffmpeg (`libx264`, `-crf 23`).
- Skips known stray duplicates (e.g., `01_correct_P1_6.mp4` in Incorrect folder).
- Reports per-exercise, per-person, and per-quality summaries.

### 4.2 Keypoint Extraction

`extract_video_keypoints.py` runs **BlazePose Full model** (`model_complexity=1`) on every frame of every flattened video.

- Frames without a detected person are **skipped** (not zero-padded).
- Outputs: `data/normalized/<stem>.npy` with shape `(frames, 12, 2)`.
- Appends metadata to `data/labels.csv` (`stem`, `exercise_id`, `quality_label`).
- Idempotent: skips already-processed stems.

### 4.3 Test Keypoint Extraction

`extract_test_keypoints.py` maps test filenames to exercise IDs via keyword matching (e.g., "deep squats" → 1, "shoulder scaption" → 10). Test stems are prefixed with `TEST_` and quality is always `correct`.

---

## 5. Feature Extraction

### 5.1 Joint Selection

BlazePose outputs 33 landmarks. The classifier uses **12 joints** (24 coordinate values):

| Body Part | Left | Right |
|-----------|------|-------|
| Shoulder  | 11   | 12    |
| Elbow     | 13   | 14    |
| Wrist     | 15   | 16    |
| Hip       | 23   | 24    |
| Knee      | 25   | 26    |
| Ankle     | 27   | 28    |

### 5.2 Hip-Centered Normalization

All coordinates are normalized by subtracting the **hip midpoint** (`(landmark[23] + landmark[24]) / 2`). This makes the model invariant to absolute position in the frame.

### 5.3 Angle Feature Engineering

`joint_angles.py` converts the 12 hip-centered joints into **16 biomechanical angle features** per frame:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | L elbow angle | Cosine between upper-arm L and forearm L |
| 1 | R elbow angle | Cosine between upper-arm R and forearm R |
| 2 | L knee angle | Cosine between upper-leg L and lower-leg L |
| 3 | R knee angle | Cosine between upper-leg R and lower-leg R |
| 4 | L arm elevation | Cosine between upper-arm L and torso |
| 5 | R arm elevation | Cosine between upper-arm R and torso |
| 6 | L hip flexion | Cosine between upper-leg L and torso |
| 7 | R hip flexion | Cosine between upper-leg R and torso |
| 8 | L arm lateral x | Unit vector x-component of upper-arm L |
| 9 | R arm lateral x | Unit vector x-component of upper-arm R |
| 10 | L leg lateral x | Unit vector x-component of upper-leg L |
| 11 | R leg lateral x | Unit vector x-component of upper-leg R |
| 12 | L arm vertical y | Unit vector y-component of upper-arm L |
| 13 | R arm vertical y | Unit vector y-component of upper-arm R |
| 14 | L leg vertical y | Unit vector y-component of upper-leg L |
| 15 | R leg vertical y | Unit vector y-component of upper-leg R |

**Why cosines?** Cosine similarity is scale-invariant and bounded to `[-1, 1]`, making it ideal for LSTM input without aggressive normalization.

**Vectorization:** `batch_keypoints_to_angles()` processes `(F, 12, 2)` → `(F, 16)` without Python loops.

### 5.4 Motion Detection

`compute_motion(angles)` computes the mean per-feature standard deviation across a window. Near-zero = idle; higher = active motion. Used to suppress classification when the user is standing still.

---

## 6. Window Building & Augmentation

`build_windows.py` constructs training samples from the extracted keypoint sequences.

### 6.1 Window Parameters

- **Window size:** 30 frames (~1 second at 30 FPS)
- **Strides:** 5 and 2 (overlapping windows for temporal density)
- **Minimum frames:** Videos with fewer than 30 frames are discarded

### 6.2 Temporal Augmentation (5× multiplier)

For each raw window, **4 augmented variants** are generated:

1. **Horizontal flip:** Swap left/right joints and negate x-coordinates.
2. **Gaussian noise:** Add `N(0, 0.01)` noise to keypoint coordinates.
3. **Time stretch:** Resample to a random factor between 0.8× and 1.2× using linear interpolation.
4. **Joint masking:** Zero out 1–2 randomly selected joints.

With two strides (5, 2) and 4 augmentations, the total dataset size is **~5× the raw window count**.

### 6.3 Final Dataset

| Dataset | Videos | Raw Windows | Augmented Windows |
|---------|--------|-------------|-------------------|
| Initial training | 174 | ~75,000 | 374,570 |
| After fine-tuning | 200 | ~84,000 | 451,638 |

---

## 7. Model Architecture

### 7.1 Dual-Head LSTM Classifier

```
Input: (batch, 30, 16)
  ↓
LSTM(64, return_sequences=True, unroll=True)
  ↓
LSTM(32, unroll=True)
  ↓
Dense(64, ReLU)
  ↓
Dropout(0.3)
  ├─→ Dense(9, softmax)   → exercise_out
  └─→ Dense(2, softmax)   → quality_out
```

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Initial LR | 0.001 (training) / 0.0001 (fine-tuning) |
| Batch size | 32 |
| Loss | Categorical cross-entropy (both heads) |
| Max epochs | 150 |
| Early stopping | patience=10, restore_best_weights=True |
| Validation split | 80/20 (shuffle seed=42) |

### 7.2 Output Heads

| Head | Classes | Mapping |
|------|---------|---------|
| Exercise | 9 | 0=Deep Squat, 1=Hurdle Step, 2=Inline Lunge, 3=Side Lunge, 4=Sit to Stand, 5=Standing Leg Raise, 6=Shoulder Abduction, 7=Shoulder Extension, 8=Shoulder Scaption |
| Quality | 2 | 0=Incorrect, 1=Correct |

---

## 8. Training Methodology

### 8.1 Session 1: Initial Training (from scratch)

**Dataset:** 174 videos → 374,570 windows (5× augmentation)  
**Hardware:** Mac M2  
**Environment:** Python 3.11 + TensorFlow (`venv-tf`)

**Training progression:**

| Epoch | Val Ex Acc | Val Qu Acc |
|-------|-----------|------------|
| 1 | 86.93% | 76.45% |
| 10 | 95.68% | 88.63% |
| 30 | 96.98% | 91.91% |
| 45 (best) | **97.15%** | **92.93%** |

EarlyStopping triggered at epoch 45. Training exercise accuracy: 97.70%; training quality accuracy: 93.81%.

### 8.2 Session 2: Fine-Tuning

**Dataset:** 200 videos (174 train + 26 test) → ~451,638 windows  
**Strategy:** Load pre-trained `classifier.keras`, lower LR to 0.0001, continue training.

| Metric | Before Fine-Tuning | After Fine-Tuning | Change |
|--------|-------------------|-------------------|--------|
| Val Exercise Accuracy | 97.15% | 96.66% | -0.49% |
| Val Quality Accuracy | 92.93% | 94.37% | +1.44% |
| Train Exercise Accuracy | 97.70% | 97.22% | -0.48% |
| Train Quality Accuracy | 93.81% | 95.48% | +1.67% |

**Analysis:** Adding diverse test videos slightly reduced exercise accuracy (expected with broader distribution) but improved quality accuracy, suggesting better generalization on form assessment.

### 8.3 Model Export

`export_models.py` converts the Keras model to **TFLite**:

- Input: `(1, 30, 16)` float32
- Outputs: `(1, 9)` exercise + `(1, 2)` quality
- File size: ~219 KB
- **No quantization** (required for tfjs-tflite WASM compatibility)
- Verified with dummy inference before deployment

### 8.4 Session 3: Held-out Generalization Audit

The accuracies in §8.1–8.2 are from an **80/20 split taken at the window level**, where overlapping
30-frame windows from the same video and subject can fall on both sides → **data leakage**. To measure
true generalization, a **video-level** held-out evaluation was added (`make_splits.py`,
`build_windows_split.py`, `train_heldout.py`).

- **Split:** 156 train / 44 test videos; augmentation applied to train only; no video in both splits.
- **Test strata:** *Internet-26* (the `eval/TEST/` web videos un-folded from training — subject-disjoint,
  all 9 classes, all `correct`) and *Collected-18* (a stratified holdout incl. incorrect form; subjects
  overlap training, so its quality number is optimistic).
- **Per-video prediction** = mean softmax across each video's windows.

| Metric | Window-split val (§8.1–8.2) | Honest held-out (per video) |
|--------|:---:|:---:|
| Exercise accuracy | ~97% | **65.4%** (Internet-26, subject-disjoint) |
| Quality accuracy | ~93% | **~54%** correct-recall on unseen subjects |

**Analysis:** The model is unchanged — the same recipe still reaches ~97% window-level val; the gap is
measurement honesty, not degradation. The limiting factor is the dataset: only 5 subjects, each exercise
performed by ≤2 of them, and the three shoulder classes (E07/E08/E10) each from a single subject — so a
fully subject-disjoint all-class test is impossible without new data. SitToStand (44%) and the quality
head (false-alarms on unseen subjects) are the concrete weaknesses. Full writeup in
`results_generalization.md`; per-video breakdown in `eval/results/heldout_eval.md`. The audited model is
saved separately as `models/classifier_heldout.keras` (production `classifier.keras` is untouched).

---

## 9. Real-Time Inference Pipeline

### 9.1 Pose Estimation

`gui.py` uses BlazePose in **`LIVE_STREAM` mode** (async):

- `submit_frame(frame, timestamp_ms)` → non-blocking
- Results delivered via callback → `latest_landmarks`
- Two model complexities: **Lite** (0, 25–30 FPS on RPi) and **Full** (1, 10–18 FPS on RPi)

### 9.2 Classification Loop

```
1. Extract 12 joints from latest landmarks
2. Append to 30-frame ring buffer
3. Every 5 frames: compute angles + motion
4. If motion < IDLE_THRESHOLD (0.03) for 2 consecutive windows → IDLE
5. Else: feed (1, 30, 16) to TFLite classifier in a background thread
6. Buffer last 10 predictions; require 5 for stability
7. Majority vote → stable exercise + quality
```

**Background threading:** The TFLite interpreter runs in a dedicated worker thread (`_classifier_worker`) to avoid blocking the GUI main loop.

### 9.3 Idle Detection

- `IDLE_THRESHOLD = 0.03` (mean std of angle features)
- `IDLE_CONFIRM_COUNT = 2` consecutive idle windows before switching to idle state
- Clears prediction buffer and rep counter state on transition to idle

### 9.4 GUI Layout

- **Left panel:** Live video feed with BlazePose skeleton overlay
- **Right sidebar:** Exercise name, quality badge (CORRECT/INCORRECT), rep count, confidence bar, visible keypoints count, FPS
- **Camera selector:** Auto-detects up to 6 cameras; hot-swappable
- **End Session button:** Triggers session logging, TTS announcement, and QR code generation

---

## 10. Repetition Counting

`rep_counter.py` uses a **state-machine + EMA** approach per exercise:

### 10.1 Configuration per Exercise

| Exercise | Features | Direction | Enter | Exit |
|----------|----------|-----------|-------|------|
| Deep Squat, Hurdle Step, Inline Lunge, Side Lunge | [2, 3] (knee angles) | low | 0.5 | 0.8 |
| Sit to Stand, Standing Leg Raise | [6, 7] (hip flexion) | high | -0.2 | -0.5 |
| Shoulder Abduction, Extension, Scaption | [4, 5] (arm elevation) | high | -0.3 | -0.6 |

### 10.2 State Machine

```
neutral → peaked (enter threshold crossed)
peaked → neutral (exit threshold crossed) → COUNT += 1
```

- **EMA smoothing:** `alpha = 0.3` to reduce jitter
- **Direction:** "low" means the value dips below enter and rises above exit; "high" means it rises above enter and dips below exit
- Each exercise maintains its own state; counts persist across session

---

## 11. Text-to-Speech Feedback

`tts_engine.py` provides real-time audio cues with cross-platform fallback:

### 11.1 TTS Strategy

| Platform | Online | Offline Fallback |
|----------|--------|------------------|
| macOS | gTTS → `afplay` | `say` |
| Linux (RPi) | gTTS → `paplay`/`pw-play` | `espeak` → `aplay` |

### 11.2 Cue Logic

- **Session start:** "Session started. Begin your exercise."
- **Exercise detected:** Speak exercise name after 1.5s of stable detection
- **Form error:** "Check your form" after 3s of incorrect quality (5s cooldown)
- **Form restored:** "Good job, form restored" after 3s of incorrect → correct (5s cooldown)
- **Session end:** "Session complete. Great work."

---

## 12. Session Logging

`session_logger.py` records the entire workout as a JSON file:

### 12.1 Segment Structure

A "segment" is a contiguous block of frames classified as the same exercise.

```json
{
  "name": "Deep Squat",
  "segment_start": "14:32:10",
  "duration_seconds": 45,
  "frames_correct": 38,
  "frames_incorrect": 7,
  "form_score_pct": 84,
  "avg_confidence": 0.92,
  "reps": 5
}
```

### 12.2 Session Summary

- Date, start/end time, total duration
- Per-exercise breakdown with form score, confidence, reps
- **Overall form score:** mean of per-exercise form scores
- Saved to `logs/session_YYYYMMDD_HHMMSS.json`

---

## 13. Post-Session RAG Chat

After ending a session, the GUI generates a **QR code** linking to `http://<pi-ip>:5000`. Scanning opens a mobile-friendly chat with a fitness coaching assistant.

### 13.1 RAG Architecture

| Component | Build-time | Runtime |
|-----------|-----------|---------|
| PDF parsing | `pymupdf` + `sentence-transformers` | — |
| Embedding | `torch` + `transformers` | `onnxruntime` + `tokenizers` |
| Knowledge store | `knowledge_base.json` | Loaded into memory |
| LLM | — | Groq API (`llama-3.1-8b-instant`) |

### 13.2 Knowledge Base Building

`build_knowledge_base.py` (run on Mac/Colab, **NOT on RPi**):

1. **Parse PDFs** in `references/` (conditioning manual + behaviour manual)
2. **Extract text + tables** using PyMuPDF in reading order
3. **Chunk** into ~500-character overlapping segments (`overlap=50`)
4. **Embed** with `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
5. **Export ONNX** query encoder + tokenizer for RPi runtime
6. **Save** `data/knowledge_base.json` (~17 MB text + embeddings)

### 13.3 Runtime Retrieval

`session_chat/retrieval.py`:

1. Tokenize user query with `tokenizers` library
2. Embed with ONNX Runtime (mean-pooled, L2-normalized)
3. Compute cosine similarity against all chunk embeddings (pure NumPy)
4. Return top-k=4 chunks with source, page, and score

### 13.4 LLM Prompting

`session_chat/llm.py` builds a **grounded system prompt** containing:

1. **Session data** (exercises, duration, form scores, reps, confidence)
2. **Retrieved knowledge chunks** (with citations)
3. **Strict guardrails:**
   - "You ONLY know what is in SESSION DATA and RETRIEVED KNOWLEDGE"
   - "NEVER use pre-trained knowledge outside these sources"
   - "If the user asks something NOT covered, reply: 'I'm sorry, I can only answer questions about your workout...'"
   - "Always cite sources when using retrieved knowledge"

**Model:** Groq `llama-3.1-8b-instant` (free tier: 1.5M tokens/day).  
**Parameters:** `max_tokens=512`, `temperature=0.7`.

### 13.5 Chat UI

`session_chat/templates/chat.html`:

- Mobile-first responsive design (dark theme)
- Session summary card with color-coded form score (green ≥80, yellow ≥60, red <60)
- Typing indicator animation
- Message history with timestamps
- Auto-scroll to bottom

---

## 14. Web Browser Demo

`web/index.html` + `web/app.js` provides a **browser-based TF.js demo** that runs the same pipeline without Python:

- **MediaPipe Tasks Vision** (CDN) for BlazePose Lite
- **TensorFlow.js** (`tf.loadLayersModel`) for the classifier
- **JavaScript ports** of `joint_angles.py` and `rep_counter.py`
- Same 30-frame window, 5-frame classification interval, idle detection
- Runs on any device with a webcam and modern browser

---

## 15. Testing Strategy

### 15.1 Unit Tests

| Test File | Coverage |
|-----------|----------|
| `test_pose_estimator_live_stream.py` | LIVE_STREAM mode: submit_frame, timestamp deduplication, callback storage |
| `test_joint_angles.py` | Angle computation, batch conversion, motion detection |
| `test_session_logger.py` | Segment logging, session export, JSON schema |
| `test_tts_engine.py` | Audio probing, fallback chains, platform detection |
| `test_app.py` | Flask chat routes, error handling |
| `test_retrieval.py` | ONNX embedding, cosine search, top-k ranking |
| `test_llm.py` | System prompt building, guardrail enforcement |
| `test_rep_counter.py` | State machine transitions, EMA smoothing, per-exercise counts |
| `test_gui_integration.py` | Camera switching, layout updates |
| `web/test_joint_angles.mjs` | JS angle computation parity with Python |
| `web/test_rep_counter.mjs` | JS rep counter parity with Python |

### 15.2 Smoke Tests

- `check_deps.py`: Verifies all runtime dependencies are importable
- `smoke_test_chat.py`: End-to-end chat session test
- `rpi_setup.sh`: Runs full pytest suite after installation

---

## 16. Environment Management

The project uses **two Python virtual environments** to avoid dependency conflicts:

| Environment | Python | Purpose | Key Packages |
|-------------|--------|---------|--------------|
| `.venv` | 3.14.5 (Homebrew) | Extraction, inference, GUI | `mediapipe`, `opencv-python`, `numpy`, `flask`, `groq`, `onnxruntime` |
| `.venv-tf` | 3.11.9 (Homebrew) | Training, model export | `tensorflow`, `numpy` |

**Why two environments?** `mediapipe` and `tensorflow` have conflicting transitive dependencies. The `.venv-tf` environment is only needed for `train_classifier.py`, `fine_tune_classifier.py`, and `export_models.py`.

**Build-time only packages** (not installed on RPi):
- `pymupdf`, `sentence-transformers`, `torch`, `transformers`

---

## 17. Hardware Deployment

### 17.1 Raspberry Pi 5 Setup

`rpi_setup.sh` automates:
1. Python version check (≥3.9)
2. Virtual environment creation
3. Runtime dependency installation (`ai-edge-litert`, `opencv-python`, `mediapipe`, etc.)
4. PulseAudio / Bluetooth speaker support
5. `espeak` TTS installation
6. Model file verification
7. Test execution

### 17.2 Performance Targets

| Configuration | Resolution | FPS |
|--------------|-----------|-----|
| BlazePose Lite | 640×480 | 25–30 |
| BlazePose Full | 640×480 | 10–18 |

### 17.3 Camera Handling

- Requests 640×480 from OpenCV
- If camera driver returns higher resolution (e.g., 1080p), frame is **downscaled before BlazePose processing** to avoid accidental full-HD inference

### 17.4 RPi 5 ARM64 Notes

- MediaPipe now ships an official ARM64 wheel for Raspberry Pi 5 via pip.
- If unavailable, `rpi_setup.sh` falls back to source build (30–40 minutes).
- System deps required: `python3-dev`, `gcc`, `g++`, `libgl1-mesa-dev`, `libgles2-mesa-dev`, `libegl1-mesa-dev`.

---

## 18. File Organization

```
pose_est_v2/
├── data/
│   ├── normalized/          # Per-video keypoints (.npy)
│   ├── labels.csv           # Video metadata
│   ├── X_train.npy         # Training windows
│   ├── y_exercise.npy      # Exercise labels
│   ├── y_quality.npy       # Quality labels
│   ├── knowledge_base.json # Pre-built RAG corpus
│   └── embedding_model/    # ONNX model + tokenizer
├── models/
│   ├── classifier.keras    # Best Keras model
│   ├── classifier.tflite   # RPi deployment model
│   └── pose_landmarker_*.task  # BlazePose models
├── eval/
│   ├── train_videos/       # Raw recorded videos
│   ├── train_videos_flat/  # Flattened canonical videos
│   └── TEST/               # Test/demo videos
├── references/
│   ├── conditioning_manual.pdf
│   └── behaviour_manual.pdf
├── data_collection/
│   └── Person_{1..5}.md    # Recording protocols
├── session_chat/
│   ├── app.py              # Flask server
│   ├── llm.py              # Groq LLM wrapper
│   ├── retrieval.py        # ONNX RAG engine
│   └── templates/chat.html # Mobile chat UI
├── web/
│   ├── index.html          # Browser demo
│   ├── app.js              # TF.js inference pipeline
│   ├── joint_angles.js     # JS angle computation
│   ├── rep_counter.js      # JS rep counter
│   └── models/classifier_tfjs/  # TF.js model export
├── tests/                  # pytest suite
├── logs.md                 # Training metrics & history
├── logs/                   # Session JSON files
├── requirements.txt        # Runtime dependencies
├── rpi_setup.sh            # One-command RPi setup
└── *.py                    # Pipeline scripts
```

---

## 19. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Cosine angles instead of raw keypoints** | Scale-invariant, bounded, no need for heavy normalization |
| **Dual-head LSTM** | Single model handles both classification and quality assessment; shared temporal features |
| **5× augmentation with 2 strides** | Dramatically increases training data without collecting more videos |
| **Hip-centered normalization** | Invariant to subject position in frame |
| **LIVE_STREAM mode** | Non-blocking async inference; much lower latency than per-frame IMAGE mode |
| **TFLite (no quantization)** | Required for tfjs-tflite WASM compatibility; 219 KB is tiny anyway |
| **ONNX embeddings on RPi** | Avoids PyTorch / sentence-transformers on device; ~91 MB model vs. >1 GB PyTorch stack |
| **Groq LLM (cloud)** | Zero on-device weight for LLM; free tier sufficient for personal use |
| **Grounded system prompt** | Prevents hallucination; forces the model to cite sources and stay within its knowledge |
| **Two Python environments** | Resolves mediapipe ↔ tensorflow dependency conflicts |
| **Browser demo (TF.js)** | Demonstrates the pipeline without requiring Python or RPi hardware |

---

## 20. Future Improvements

1. **Temporal quality scoring:** Use the full 30-frame window for quality instead of frame-by-frame majority vote.
2. **Exercise transition detection:** Explicitly model transitions between exercises rather than relying on idle state.
3. **Multi-person support:** BlazePose supports `num_poses > 1`; extend the pipeline to handle multiple subjects.
4. **On-device LLM:** Replace Groq with a small quantized LLM (e.g., Phi-3-mini) for offline operation.
5. **Active learning:** Use low-confidence predictions to flag videos for human annotation and retraining.
6. **Depth camera support:** Integrate Intel RealSense or similar for 3D joint angles and more robust form assessment.

---

*Last updated: 2026-06-07*
