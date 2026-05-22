# Pose Estimation + Exercise Classifier

## Pipeline Overview

```
preprocess_train_videos.py     # Flatten Person1-5 folders → canonical filenames
    → extract_video_keypoints.py # MoveNet keypoints → data/normalized/*.npy
    → build_windows.py           # 30-frame windows + 5x augmentation → X_train.npy
    → train_classifier.py        # 2-layer LSTM (9 exercise + 2 quality classes)
    → export_models.py           # .keras → .tflite for RPi deployment
```

## Models

- **movenet_thunder_int8.tflite** — pose estimation (MoveNet Thunder INT8)
- **classifier.tflite** — dual-head LSTM: exercise (9 classes) + quality (correct/incorrect)
  - 174 training videos from 5 people
  - 451,638 windows after augmentation
  - 9 exercises: Deep Squat, Hurdle Step, Inline Lunge, Side Lunge, Sit to Stand,
    Standing Leg Raise, Shoulder Abduction, Shoulder Extension, Shoulder Scaption
  - **Known limitation:** 2D pixel coordinates cause poor generalization (~11%) on unseen people.
    Fix: collect 20+ person dataset or convert to joint-angle features.

## RPi 5 Deployment (one command)

```bash
git clone <repo>
cd pose_est_v2
bash rpi_setup.sh
```

`rpi_setup.sh` handles: venv creation, dependency install (via `requirements.txt`),
espeak setup, model file verification, and test execution.

## Post-Setup (one-time)

```bash
# 1. Copy the pre-built knowledge base from your Mac/Colab to RPi:
scp -r data/knowledge_base.json pi@raspberrypi.local:~/pose_est_v2/data/
scp -r data/embedding_model/ pi@raspberrypi.local:~/pose_est_v2/data/

# 2. Set your Groq API key (free tier — 1.5M tokens/day)
echo 'GROQ_API_KEY=your_key_here' > .env
#    Get a free key at https://console.groq.com

# 3. Run
python gui.py
```

## Session Chat / RAG

After ending a workout session, the GUI shows a QR code. Scan it to open a chat
with a fitness coaching assistant that references:
1. **Your current workout session data**
2. **Conditioning manual** — exercise science & form guidance
3. **Behaviour manual** — psychology, habit building & motivation

### Building the Knowledge Base (Mac / Colab — NOT on RPi)

```bash
# Install build-time deps (heavy — PyTorch + sentence-transformers)
pip install pymupdf sentence-transformers torch transformers

# Build KB + export ONNX model
python build_knowledge_base.py

# Copy the generated artifacts to your RPi
scp -r data/knowledge_base.json data/embedding_model/ pi@raspberrypi.local:~/pose_est_v2/data/
```

### Architecture

| Component | Runs On | Weight |
|-----------|---------|--------|
| Session data | RPi 5 (live) | ~1 KB |
| PDF knowledge base | Pre-built, loaded on RPi | ~17 MB |
| ONNX embedding model | Pre-built, loaded on RPi | ~91 MB |
| ONNX Runtime | RPi 5 | ~5–10 MB overhead |
| LLM (Groq) | Cloud (free tier) | 0 MB |
| **Total added footprint on RPi** | | **~115 MB** |

No PyTorch, no `sentence-transformers`, no PDF parsing on the Pi. Everything
heavy happens at build time on your Mac or Colab.

## Joint Mapping

MoveNet (17 COCO) and Vicon markers share 12 joints. See `joint_map.py` for
the explicit mapping.
