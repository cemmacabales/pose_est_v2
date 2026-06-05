# Pose Estimation + Exercise Classifier

Real-time exercise classification running on a Raspberry Pi 5. BlazePose (MediaPipe) extracts 33-landmark pose keypoints from a webcam feed in async LIVE_STREAM mode; a dual-head LSTM classifier identifies 9 functional movement exercises and scores form quality. A post-session RAG chat uses ONNX embeddings + Groq LLM to deliver coaching feedback — no PyTorch on-device.

![Python](https://img.shields.io/badge/python-3.14%20%7C%203.11-blue) ![Platform](https://img.shields.io/badge/platform-RPi%205%20%7C%20macOS-lightgrey) ![MediaPipe](https://img.shields.io/badge/pose-MediaPipe%20BlazePose-green) ![TFLite](https://img.shields.io/badge/model-TFLite%20LSTM-orange)

## Architecture

- **MediaPipe BlazePose** -- pose estimation (33 landmarks, on-device GPU/Coral support)
  - Runs in `LIVE_STREAM` mode: frames submitted via `submit_frame()`, results delivered via callback. Non-blocking; avoids the per-frame latency of `IMAGE` mode. See `pose_estimator.py`.
- **classifier.tflite** -- dual-head LSTM: exercise (9 classes) + quality (correct/incorrect)
  - 174 training videos from 5 people
  - 451,638 windows after augmentation
  - 9 exercises: Deep Squat, Hurdle Step, Inline Lunge, Side Lunge, Sit to Stand,
    Standing Leg Raise, Shoulder Abduction, Shoulder Extension, Shoulder Scaption

## Pipeline Overview

```
preprocess_train_videos.py     # Flatten Person1-5 folders -> canonical filenames
    -> extract_video_keypoints.py # BlazePose keypoints -> data/normalized/*.npy
    -> build_windows.py           # 30-frame windows + 5x augmentation -> X_train.npy
    -> train_classifier.py        # 2-layer LSTM (9 exercise + 2 quality classes)
    -> export_models.py           # .keras -> .tflite for RPi deployment
```

## Quick Start (Mac -- Training & Export)

```bash
# Extraction + window building (Python 3.14, .venv)
source .venv/bin/activate
pip install -r requirements.txt
python preprocess_train_videos.py
python extract_video_keypoints.py --video_dir ./path/to/videos
python build_windows.py

# Training + export (Python 3.11, .venv-tf)
source .venv-tf/bin/activate
pip install tensorflow
python train_classifier.py
python export_models.py

# Fine-tuning (adds 26 test videos)
source .venv/bin/activate
python extract_test_keypoints.py
python build_windows.py

source .venv-tf/bin/activate
python fine_tune_classifier.py
python export_models.py
```

## RPi 5 Deployment (one command)

```bash
git clone <repo>
cd pose_est_v2
bash rpi_setup.sh
```

`rpi_setup.sh` handles: venv creation, dependency install (via `requirements.txt`),
mediapipe ARM64 build from source, espeak setup, model file verification, and test execution.

### RPi 5 Expected FPS

| Configuration | Resolution | Expected FPS |
|--------------|-----------|-------------|
| Lite model (`--model lite`) | 640x480 | 25-30 |
| Full model (`--model full`) | 640x480 | 10-18 |

```bash
# Run with lite model (RPi 5 default)
python gui.py --model lite

# Run with full model (desktop/Mac)
python gui.py --model full
```

### Camera Resolution

The GUI requests 640x480 from the webcam. If the camera driver does not honor
`cap.set()` (common on RPi with some USB cameras), the raw frame is automatically
downscaled to 640x480 BEFORE BlazePose processing to avoid running the pose
detector on accidental 1080p frames.

### mediapipe on RPi 5 (ARM64)

MediaPipe has **no official Linux ARM64 wheel**. `rpi_setup.sh` attempts to
build from source (~30-40 minutes on RPi 5). System dependencies required:
`python3-dev`, `gcc`, `g++`, `libgl1-mesa-dev`, `libgles2-mesa-dev`,
`libegl1-mesa-dev`.

If the build fails, alternatives:
1. Use a community-built ARM64 wheel
2. Run pose estimation on a USB Coral TPU with a different detector
3. Offload to cloud (adds latency)

## Post-Setup (one-time)

```bash
# 1. Copy the pre-built knowledge base from your Mac/Colab to RPi:
scp -r data/knowledge_base.json pi@raspberrypi.local:~/pose_est_v2/data/
scp -r data/embedding_model/ pi@raspberrypi.local:~/pose_est_v2/data/

# 2. Set your Groq API key (free tier -- 1.5M tokens/day)
echo 'GROQ_API_KEY=your_key_here' > .env
#    Get a free key at https://console.groq.com

# 3. Run
python gui.py --model lite
```

## Session Chat / RAG

After ending a workout session, the GUI shows a QR code. Scan it to open a chat
with a fitness coaching assistant that references:
1. **Your current workout session data**
2. **Conditioning manual** -- exercise science & form guidance
3. **Behaviour manual** -- psychology, habit building & motivation

### Building the Knowledge Base (Mac / Colab -- NOT on RPi)

```bash
# Install build-time deps (heavy -- PyTorch + sentence-transformers)
pip install pymupdf sentence-transformers torch transformers

# Build KB + export ONNX model
python build_knowledge_base.py

# Copy the generated artifacts to your RPi
scp -r data/knowledge_base.json data/embedding_model/ pi@raspberrypi.local:~/pose_est_v2/data/
```

### RAG Stack

| Component | Runs On | Weight |
|-----------|---------|--------|
| Session data | RPi 5 (live) | ~1 KB |
| PDF knowledge base | Pre-built, loaded on RPi | ~17 MB |
| ONNX embedding model | Pre-built, loaded on RPi | ~91 MB |
| ONNX Runtime | RPi 5 | ~5-10 MB overhead |
| LLM (Groq) | Cloud (free tier) | 0 MB |
| **Total added footprint on RPi** | | **~115 MB** |

No PyTorch, no `sentence-transformers`, no PDF parsing on the Pi. Everything
heavy happens at build time on your Mac or Colab.

## Joint Mapping

BlazePose provides 33 landmarks. The 12 joints used by the classifier pipeline are:

| Body Part | Left Index | Right Index |
|-----------|-----------|------------|
| Shoulder | 11 | 12 |
| Elbow | 13 | 14 |
| Wrist | 15 | 16 |
| Hip | 23 | 24 |
| Knee | 25 | 26 |
| Ankle | 27 | 28 |

Extracted in the order: `[11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]`
-> L_shoulder, R_shoulder, L_elbow, R_elbow, L_wrist, R_wrist, L_hip, R_hip, L_knee, R_knee, L_ankle, R_ankle.

Coordinates are normalized by hip midpoint. See `joint_map.py` and `pose_estimator.py`.

## Environments

| Venv | Python | Purpose | Key Package |
|------|--------|---------|-------------|
| `.venv` | 3.14.5 (Homebrew) | Extraction, inference, GUI | `mediapipe` |
| `.venv-tf` | 3.11.9 (Homebrew) | Training, export | `tensorflow` |

If both `mediapipe` and `tensorflow`/`ai-edge-litert` are needed at inference time, use `.venv-tf` (it has both).

## Training Log

See [logs.md](logs.md) for detailed training metrics, model architecture, and fine-tuning results.
