# Handoff Document — Session: RPi 5 Live Deployment (blockers identified, fix plan ready)

**Date:** 2026-05-29
**Status:** Fine-tuning + export done. Code pushed to GitHub. RPi 5 partially working — 2 runtime bugs remain with fixes planned. TTS audio device detection needs rewrite.

---

## Session 4: RPi 5 Live Deployment

### What Worked

1. **Git clone + venv setup** — Success. Using Homebrew Python 3.11.15 (system Python is 3.13.5 which lacks mediapipe wheel).

2. **mediapipe installed** — `pip install mediapipe` worked from inside Python 3.11 venv. Version: **0.10.18**.

3. **All pip deps installed** — `ai-edge-litert`, `opencv-python`, `numpy`, `flask`, etc. all OK.

4. **classifier.tflite loaded** — TFLite inference works via `ai-edge-litert`.

5. **Camera detected** — `/dev/video0` (USB webcam). GUI opens the Tkinter window.

6. **Pose estimation runs** — BlazePose detects 33 landmarks and classifies exercises.

7. **Tests pass** — 47/47 (`pytest tests/ -v`).

8. **Push** — Committed and pushed to GitHub (`e9c3cd7`).

### What's Broken (2 Runtime Errors)

#### Bug 1: `drawing_utils` crash in `pose_estimator.py:90`

```
AttributeError: module 'mediapipe.tasks.python.vision' has no attribute 'drawing_utils'
```

**Root cause:** The RPi-installed mediapipe wheel (0.10.18) is a minimal build that includes the `tasks` API for inference but **omits the `solutions` submodule and `drawing_utils`**. On Mac this module exists, on the community ARM64 wheel it doesn't.

**Fix plan for next session:** Replace `vision.drawing_utils.draw_landmarks()` with a **pure OpenCV fallback** in `pose_estimator.py.draw_landmarks()`:
```python
def draw_landmarks(self, frame, landmarks):
    if landmarks is None: return
    try:
        vision.drawing_utils.draw_landmarks(...)
    except (AttributeError, ImportError):
        self._draw_landmarks_opencv(frame, landmarks)
```

The OpenCV fallback draws 33 circles (`cv2.circle`) for landmarks + lines (`cv2.line`) for the MediaPipe skeleton connections (face, arms, torso, legs). Same visual output, zero dependency on mediapipe's drawing module.

#### Bug 2: TTS ALSA audio error 524

```
aplay: main:850: audio open error: Unknown error 524
[TTS] ALSA device: hw:0,0
```

**Root cause:** `tts_engine.py._probe_alsa()` uses `aplay -l` which returns raw hardware IDs like `hw:0,0`. On the RPi 5, this maps to `vc4-hdmi-0` (HDMI audio) which is not active (no HDMI monitor with speakers connected). Also, Bluetooth speaker is connected but invisible to ALSA hardware probing (`aplay -l` doesn't detect Bluetooth devices).

**Actual audio landscape on RPi 5:**
```
card 0: vc4hdmi0    — HDMI audio (not active, error 524)
card 1: vc4hdmi1    — HDMI audio (not active)
Bluetooth speaker   — connected but not detected (no PulseAudio, no pactl)
```

**Fix plan for next session:**
1. Rewrite `_probe_alsa()` to use `aplay -L` (PCM name listing) instead of `aplay -l` (hardware listing). Return `sysdefault` or `default` — these route through ALSA's software mixer and handle any active audio output.
2. In `_speak_fallback()`: try `aplay -D <device>` first, then `aplay` without `-D`, then `espeak` raw.
3. If all audio fails: print `[TTS] Warning: no audio output available` and skip gracefully (no crash).

### RPi 5 Environment

| Component | Status | Details |
|-----------|--------|---------|
| Python | 3.11.15 (Homebrew) | System Python 3.13.5 also exists but lacks mediapipe wheel |
| mediapipe | 0.10.18 (wheel) | Installed via pip in venv, but minimal build (no drawing_utils) |
| tkinter | Working | Installed via `brew install python-tk@3.11` |
| Camera | `/dev/video0` | USB webcam |
| Audio | HDMI only (`aplay -l`) | Bluetooth speaker connected but needs PulseAudio |
| PulseAudio | NOT installed | `pactl` not found — need `apt-get install pulseaudio pulseaudio-module-bluetooth` |
| classifier.tflite | 219 KB | TFLite inference OK |
| espeak | Installed | TTS fallback available |

### RPi 5 setup commands (what worked)

```bash
# System deps
sudo apt-get install -y espeak python3-venv tk-dev tcl-dev

# Homebrew Python 3.11 + tkinter
brew install python@3.11
brew install python-tk@3.11

# Venv + deps
/home/linuxbrew/.linuxbrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install mediapipe ai-edge-litert opencv-python numpy Pillow \
    gtts "playsound==1.2.2" "qrcode[pil]" flask python-dotenv pytest \
    groq onnxruntime tokenizers

# Test
python gui.py --model lite
# → Tkinter window opens, camera shows, pose detected
# → CRASH on draw_landmarks (Bug 1)
# → TTS error 524 on aplay (Bug 2)
```

### Next Session: Fix 2 Bugs + RPi Audio

1. **Fix `pose_estimator.py.draw_landmarks()`** — try/except + OpenCV fallback
2. **Fix `tts_engine.py._probe_alsa()`** — use `aplay -L` instead of `aplay -l`, return `sysdefault`
3. **Optionally install PulseAudio** for Bluetooth speaker support:
   ```bash
   sudo apt-get install -y pulseaudio pulseaudio-module-bluetooth
   ```
4. **Update `rpi_setup.sh`** — remove mediapipe source build (dead code), add `pip install mediapipe`, add PulseAudio step
5. **Push fixes** to GitHub

---

---

## Session 2: Training + Test Annotation + RPi Planning

### Training Results

Classifier trained with EarlyStopping (patience=10), aborted by epoch 45/150:

| Epoch | Train Loss | Train Ex Acc | Train Qu Acc | Val Loss | Val Ex Acc | Val Qu Acc |
|-------|-----------|-------------|-------------|---------|-----------|-----------|
| 1 | 1.1274 | 79.00% | 71.11% | 0.8204 | 86.93% | 76.45% |
| 5 | 0.5043 | 93.48% | 84.65% | 0.4721 | 94.03% | 85.54% |
| 10 | 0.3748 | 95.49% | 88.46% | 0.3679 | 95.68% | 88.63% |
| 20 | 0.2852 | 96.65% | 91.28% | 0.3379 | 95.88% | 90.27% |
| 30 | 0.2425 | 97.26% | 92.69% | 0.2755 | 96.98% | 91.91% |
| 40 | 0.2162 | 97.59% | 93.56% | 0.2721 | 97.02% | 92.27% |
| 45 | 0.2061 | 97.70% | 93.81% | 0.2607 | 97.15% | 92.93% |

**Final:** 97.15% val exercise, 92.81% val quality accuracy.
Model saved to `models/classifier.keras`.

### Export

`classifier.keras` → `classifier.tflite` (219 KB). Verified with dummy inference.
Input: `(1, 30, 16)`, Outputs: `(1, 9)` exercise + `(1, 2)` quality.

### Training Set Annotation (174 videos)

All 174 training videos annotated with exercise + quality predictions (HUD overlay):

```
eval/results/train_annotated/
  01_DeepSquat/       22 videos
  02_HurdleStep/      26 videos
  03_InlineLunge/     26 videos
  04_SideLunge/       26 videos
  05_SittoStand/      21 videos
  06_StandingLegRaise/24 videos
  07_ShoulderAbduction/ 9 videos
  08_ShoulderExtension/ 10 videos
  10_ShoulderScaption/  10 videos
  summary.csv
```

Results: 100% exercise accuracy, 98.9% quality accuracy (172/174).

### Test Set Annotation (26 unseen videos)

Real-world test videos from `eval/TEST/` with different camera angles (frontal, lateral, 45-degree) and subjects not seen during training. Output:

```
eval/results/test_annotated/   (26 files, 743 MB)
```

### Model Observations

- Exercise classification on training set: near perfect (100%). Likely overfit to training subjects/angles.
- Test videos expose the generalization gap — some exercises get confused on unseen angles.
- Quality classification is the harder task (92.8% val vs 97.2% ex).
- BlazePose detects all 33 landmarks reliably (0 frame drops on test videos).

### RPi 5 Deployment Assessment

- **mediapipe** has NO official Linux ARM64 wheel — this is the primary blocker for RPi.
- Everything else (TFLite, ONNX, OpenCV, Flask) has ARM64 wheels.
- Pipeline is 100% CPU-based; no GPU code anywhere.
- gui.py already requests 640x480 webcam, but BlazePose runs on raw frame before the resize — needs fix.
- Configurable `model_complexity` needed: Lite (0) for RPi speed, Full (1) for desktop accuracy.

---

## Decisions Made (Session 2)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Fine-tuning strategy | Load `classifier.keras`, continue training with LR=0.0001 | Preserves learned weights, adapts to new data |
| Test video quality labels | All assumed "correct" | Demo/reference videos, no incorrect examples |
| RPi target | RPi 5, CPU-only | User's deployment hardware |
| Default model for RPi | BlazePose Lite (`model_complexity=0`) | 20–30 FPS on RPi 5 vs 10–18 FPS with Full |
| Desktop model | BlazePose Full (`model_complexity=1`) | Runs fine on M2 (~25–30 FPS) |
| Model selection | `--model lite\|full` CLI flag | User can choose per deployment |
| Webcam resolution | 640×480, but must verify driver honors `cap.set()` and downscale raw frame before BlazePose | Avoids running detector on accidental 1080p frames |
| Classifier model | `models/classifier.tflite` (219 KB LSTM) | Negligible inference cost (~0.1ms) |
| Training hardware | Mac (M2) only | TensorFlow training not for RPi |

---

## Next Session: Fine-Tune + RPi Setup + Push

### 1. Fine-Tune Model with Test Videos

**Goal:** Improve generalization to unseen camera angles and subjects.

```bash
# Step 1: Extract keypoints from test videos (26 files)
# Step 2: Label by exercise (from filename mapping)
# Step 3: Rebuild windows with expanded dataset
source .venv/bin/activate
python extract_test_keypoints.py          # new script
python build_windows.py                   # now includes test data (~420k windows)

# Step 4: Fine-tune (load weights, not restart)
source .venv-tf/bin/activate
python fine_tune_classifier.py            # new script: loads classifier.keras, LR=0.0001
python export_models.py                   # → classifier.tflite

# Step 5: Verify
source .venv-tf/bin/activate
python eval/annotate_test_videos.py       # re-annotate test set, compare before/after
```

**Filename → exercise ID mapping for test videos:**

| Filename pattern | Exercise ID | Class |
|-----------------|-------------|-------|
| `DEEP SQUATS`, `Squat` | 01 | 0 (Deep Squat) |
| `Hurdle Step` | 02 | 1 (Hurdle Step) |
| `Inline Lunge`, `INLINE LUNGE` | 03 | 2 (Inline Lunge) |
| `Side Lunge`, `SIDE LUNGES` | 04 | 3 (Side Lunge) |
| `Sit to Stand` | 05 | 4 (Sit to Stand) |
| `Standing leg raise`, `Leg Raises`, `STANDING LEG RAISE` | 06 | 5 (Standing Leg Raise) |
| `SHOULDER ABDUCTION` | 07 | 6 (Shoulder Abduction) |
| `SHOULDER EXTENSION` | 08 | 7 (Shoulder Extension) |
| `SHOULDER SCAPTION` | 10 | 8 (Shoulder Scaption) |

### 2. Fix RPi 5 Deployment

**Steps:**

- Fix `gui.py` — verify actual camera resolution, downscale raw frame BEFORE BlazePose if >640x480
- Add `--model lite|full` CLI arg to `gui.py` (default `lite`)
- Fix `rpi_setup.sh` — resolve `mediapipe` ARM64 installation (build from source or community wheel)
- Update `requirements.txt` — clearly split runtime (RPi) vs build-time (Mac training) deps
- Update `README.md` — RPi 5 setup instructions with expected FPS benchmarks
- Push only necessary files to repo (no `.npy`, `.task`, `.mp4`, `.keras`, `.tflite`)

**RPi 5 expected performance:**

| Configuration | Resolution | FPS |
|--------------|-----------|-----|
| Full model (`model_complexity=1`) | 640×480 | 10–18 |
| Lite model (`model_complexity=0`) | 640×480 | 25–30 |

### 3. Repo Cleanup for Push

Stage these files (NOT staged: `data/*.npy`, `models/*.task`, `models/*.keras`, `models/*.tflite`, `eval/results/`):

```
gui.py                          # --model flag + camera fix
pose_estimator.py               # (existing, supports model_complexity)
joint_map.py                    # (existing)
joint_angles.py                 # (existing)
train_classifier.py             # (existing)
fine_tune_classifier.py         # NEW — fine-tuning script
build_windows.py                # (existing)
export_models.py                # (existing)
extract_video_keypoints.py      # (existing)
extract_test_keypoints.py       # NEW — test video keypoint extraction
eval/annotate_video.py          # (existing)
eval/annotate_all_train_videos.py # (existing)
eval/annotate_test_videos.py    # (existing)
rpi_setup.sh                    # UPDATED — mediapipe ARM64 fix
requirements.txt                # UPDATED — runtime vs build-time split
README.md                       # UPDATED — RPi instructions
HANDOFF.md                      # UPDATED (this file)
tests/                          # (existing, all 47 pass)
session_chat/                   # (existing)
tts_engine.py                   # (existing)
session_logger.py               # (existing)
preprocess_train_videos.py      # (existing)
```

### 4. Tests

```bash
source .venv/bin/activate       # or .venv-tf on RPi
pytest tests/ -v                # 47 tests, must pass
python gui.py --model lite      # smoke test
```

---

## Environments (Unchanged)

| Venv | Python | Purpose | Key Package |
|------|--------|---------|-------------|
| `.venv` | 3.14.5 (Homebrew) | Extraction, inference, GUI | `mediapipe` |
| `.venv-tf` | 3.11.9 (Homebrew) | Training, export | `tensorflow` |

If both `mediapipe` and `tensorflow`/`ai-edge-litert` are needed at inference time, use `.venv-tf` (it has both).

---

## Model Architecture (Unchanged)

```
Input: (30, 16)        # 30 time steps × 16 angle features (hip-centered)
  LSTM(64, return_sequences=True)
  LSTM(32)
  Dense(64, ReLU) + Dropout(0.3)
  ├─ exercise_out: Dense(9, softmax)
  └─ quality_out: Dense(2, softmax)
```

Optimizer: Adam(lr=0.001), batch=32, max 150 epochs, EarlyStopping(patience=10).
**Fine-tuning:** Adam(lr=0.0001), same architecture, loads pre-trained weights.

---

## The 9 Exercise Classes (Unchanged)

| Index | Exercise | ID | Training Videos | Test Videos |
|-------|----------|----|----------------|-------------|
| 0 | Deep Squat | 01 | 22 | 2 |
| 1 | Hurdle Step | 02 | 26 | 2 |
| 2 | Inline Lunge | 03 | 26 | 2 |
| 3 | Side Lunge | 04 | 26 | 8 |
| 4 | Sit to Stand | 05 | 21 | 6 |
| 5 | Standing Leg Raise | 06 | 24 | 2 |
| 6 | Shoulder Abduction | 07 | 9 | 1 |
| 7 | Shoulder Extension | 08 | 10 | 1 |
| 8 | Shoulder Scaption | 10 | 10 | 1 |

Note: Class 5 (Standing Leg Raise) includes "Leg Raises" from test set.
"Leg Raises" is standing leg raise (06), not a separate exercise.

---

## Active Artifacts

- `data/normalized/*.npy` — 174 keypoint files (will become 200 after test extraction)
- `data/labels.csv` — 174 entries (will become 200)
- `data/X_train.npy` — (374570, 30, 16) — will be rebuilt with test data
- `models/classifier.keras` — best Keras model
- `models/classifier.tflite` — TFLite for inference (219 KB)
- `models/pose_landmarker_full.task` — BlazePose Full (9.4 MB)
- `eval/results/train_annotated/` — 174 annotated training videos
- `eval/results/test_annotated/` — 26 annotated test videos (743 MB)

---

## Session 1: MoveNet → MediaPipe BlazePose Migration

**Status:** BlazePose migration done. Extraction + windows complete. Training pending.

---

## What Happened This Session

Replaced MoveNet Thunder (TFLite) with MediaPipe BlazePose across the entire codebase. Created a DRY `pose_estimator.py` helper module to eliminate duplicated pose-estimation code across 5 files.

### Code Changes

| File | Action |
|------|--------|
| `pose_estimator.py` | **NEW** — BlazePose wrapper (Task API, auto-downloads model) |
| `joint_map.py` | Replaced VICON_TO_MOVENET with BLAZEPOSE_MAPPED_INDICES, BLAZEPOSE_JOINT_NAMES |
| `joint_angles.py` | Updated docstring (logic unchanged) |
| `extract_video_keypoints.py` | Swap MoveNet TFLite → PoseEstimator |
| `gui.py` | Swap MoveNet TFLite → PoseEstimator, draw_landmarks for 33-landmark skeleton |
| `eval/annotate_video.py` | Same swap |
| `eval/annotate_all_train_videos.py` | Same swap |
| `eval/eval_cross_dataset.py` | Same swap |
| `export_models.py` | Removed MoveNet download/verify, kept classifier→TFLite export |
| `requirements.txt` | Added `mediapipe`, kept `ai-edge-litert` |
| `rpi_setup.sh` | Removed MoveNet check, added `mediapipe` to pip install |
| `tests/test_gui_integration.py` | Replaced TFLite mock with PoseEstimator mock |
| `README.md` | Updated for BlazePose |

### Key Design Decision

Created `pose_estimator.py` instead of inline changes in each file. This wraps MediaPipe's 0.10.x Task API (`mediapipe.tasks.python.vision.PoseLandmarker`) and provides:
- `process_frame(frame)` → list of 33 NormalizedLandmark, or None
- `extract_mapped_joints(landmarks)` → (12, 2) float32 array (static)
- `draw_landmarks(frame, landmarks)` → MediaPipe drawing utils
- `count_visible(landmarks, threshold)` → int (static)

### MediaPipe 0.10.x Note

MediaPipe 0.10.x uses the **new Task API** (NOT the deprecated `mp.solutions.pose` API from 0.9.x). Requires a `.task` model file:
- Downloaded: `models/pose_landmarker_full.task` (~9.4 MB) for `model_complexity=1`
- Lite (`model_complexity=0`) and Heavy (`model_complexity=2`) URLs configured but not downloaded

### Pipeline Progress

```
  preprocess_train_videos.py → DONE (prior session)
  extract_video_keypoints.py → DONE (172/174 videos, ~40 min)
  build_windows.py           → DONE (374,570 windows)
  train_classifier.py        → PENDING (aborted at epoch 8/150)
  export_models.py           → PENDING
  gui.py                     → PENDING (smoke test)
```

### Extraction Results
- 172 new videos processed, 2 skipped (already had .npy)
- Output: `data/normalized/<stem>.npy` — shape `(F, 12, 2)`
- Labels: `data/labels.csv` — 174 entries
- All 9 exercise classes present (IDs 1-8, 10; no 9 — expected)

### Window Building Results
- 374,570 windows from 174 videos
- Exercise distribution: Class 0: 52,565 / Class 1: 62,845 / Class 2: 52,580 / Class 3: 45,990 / Class 4: 43,145 / Class 5: 41,955 / Class 6: 18,995 / Class 7: 24,815 / Class 8: 31,680
- Quality: correct=200,230 / incorrect=174,340
- Output: `data/X_train.npy` (374570, 30, 16), `data/y_exercise.npy`, `data/y_quality.npy`

### Training (Partial — ABORTED)

Aborted at epoch 8/150. Last checkpoint deleted. Results so far were promising:

| Epoch | Train Loss | Train Ex Acc | Train Qu Acc | Val Loss | Val Ex Acc | Val Qu Acc |
|-------|-----------|-------------|-------------|---------|-----------|-----------|
| 1 | 1.1328 | 79.40% | 70.23% | 0.8510 | 86.32% | 76.18% |
| 2 | 0.7572 | 88.86% | 77.88% | 0.6194 | 91.83% | 81.29% |
| 3 | 0.6473 | 91.10% | 80.82% | 0.5876 | 91.92% | 82.24% |
| 4 | 0.5613 | 92.61% | 83.01% | 0.5076 | 93.65% | 83.61% |
| 5 | 0.5112 | 93.49% | 84.36% | 0.5171 | 93.91% | 84.34% |
| 6 | 0.4695 | 94.20% | 85.49% | 0.4607 | 94.68% | 84.47% |
| 7 | 0.4399 | 94.62% | 86.31% | 0.4206 | 94.87% | 86.69% |
| 8 | 0.4178 | 94.97% | 86.92% | 0.4089 | 94.96% | 87.41% |

### Tests
All 47 tests pass (`pytest tests/ -v`).

---

## Next Session: Complete Training + Export + Verify

### 1. Train Classifier
```bash
# Python 3.11 venv (TensorFlow doesn't support 3.14 yet)
source .venv-tf/bin/activate
python -u train_classifier.py
```
- Already have `X_train.npy`, `y_exercise.npy`, `y_quality.npy` ready
- ~1 min per epoch on M2, 150 max epochs with EarlyStopping(patience=10)
- Expect ~30-40 epochs before early stop
- Best model saved at `models/classifier_best.keras`

### 2. Export to TFLite
```bash
source .venv-tf/bin/activate
python export_models.py
```
- Converts `classifier.keras` → `classifier.tflite`
- Verifies with dummy inference

### 3. Verify
```bash
source .venv/bin/activate        # Python 3.14 for mediapipe
pytest tests/ -v
python gui.py                     # smoke test on Mac
```

---

## Environments

| Venv | Python | Purpose | Key Package |
|------|--------|---------|-------------|
| `.venv` | 3.14.5 (Homebrew) | Extraction, inference, GUI | `mediapipe` |
| `.venv-tf` | 3.11.9 (Homebrew) | Training, export | `tensorflow` |

**NOTE:** `.venv`'s `python` symlink was fixed (was pointing to pyenv's 3.11.9, now → homebrew 3.14.5).

### Active Artifacts
- `data/normalized/*.npy` — 174 keypoint files (shape `(F, 12, 2)`)
- `data/labels.csv` — 174 entries
- `data/X_train.npy` — (374570, 30, 16)
- `data/y_exercise.npy` — (374570,)
- `data/y_quality.npy` — (374570,)
- `models/pose_landmarker_full.task` — BlazePose model (9.4 MB)

---

## Model Architecture (Unchanged)

```
Input: (30, 16)        # 30 time steps × 16 angle features
  LSTM(64, return_sequences=True)
  LSTM(32)
  Dense(64, ReLU) + Dropout(0.3)
  ├─ exercise_out: Dense(9, softmax)
  └─ quality_out: Dense(2, softmax)
```

Optimizer: Adam(lr=0.001), batch=32, max 150 epochs, EarlyStopping(patience=10).

---

## The 9 Exercise Classes (Unchanged)

| Index | Exercise | ID | Videos |
|-------|----------|----|--------|
| 0 | Deep Squat | 01 | 22 |
| 1 | Hurdle Step | 02 | 26 |
| 2 | Inline Lunge | 03 | 26 |
| 3 | Side Lunge | 04 | 26 |
| 4 | Sit to Stand | 05 | 21 |
| 5 | Standing Leg Raise | 06 | 24 |
| 6 | Shoulder Abduction | 07 | 9 |
| 7 | Shoulder Extension | 08 | 10 |
| 8 | Shoulder Scaption | 10 | 10 |

---

*End of handoff. Extraction + windows ready. Re-run training next session.*
