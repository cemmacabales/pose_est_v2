# Pose Estimation + Exercise Classifier

## Desktop (training machine) — run in order:
1. python check_deps.py
2. python load_dataset.py
3. python normalize_joints.py
4. python build_windows.py
5. python train_classifier.py
6. python export_models.py

## RPi 5 (inference + GUI):
git clone <repo> → bash rpi_setup.sh → python gui.py

## Joint mapping:
MoveNet (17 COCO) and UI-PRMD (20 Kinect) share 12 joints.
See joint_map.py for the explicit mapping.

## Models:
- movenet_thunder_int8.tflite — pose estimation (MoveNet Thunder INT8)
- classifier.tflite — dual-head LSTM: exercise (10 classes) + quality (correct/incorrect)

## Session Chat / RAG (NEW)

After ending a workout session, the GUI shows a QR code. Scan it to open a chat with a fitness coaching assistant that references:
1. **Your current workout session data**
2. **Conditioning manual** — exercise science & form guidance
3. **Behaviour manual** — psychology, habit building & motivation

### RPi 5 Setup (one-time)

```bash
# 1. Pull the latest code
git clone <repo>
cd pose_est_v2

# 2. Install runtime dependencies (lightweight — no PyTorch!)
pip install groq onnxruntime tokenizers flask Pillow opencv-python numpy qrcode gtts playsound==1.2.2 python-dotenv

# 3. Copy the pre-built knowledge base from your Mac/Colab
#    (these files are too large for git — ~110 MB total)
scp -r data/knowledge_base.json pi@raspberrypi.local:~/pose_est_v2/data/
scp -r data/embedding_model/ pi@raspberrypi.local:~/pose_est_v2/data/

# 4. Set your Groq API key (free tier — 1.5M tokens/day)
echo 'GROQ_API_KEY=your_key_here' > .env
#    Get a free key at https://console.groq.com

# 5. Run
cd pose_est_v2
python gui.py
```

### Building the Knowledge Base (Mac / Colab — NOT on RPi)

If you update the PDFs in `references/`, rebuild the knowledge base:

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
|---|---|---|
| Session data | RPi 5 (live) | ~1 KB |
| PDF knowledge base | Pre-built, loaded on RPi | ~17 MB |
| ONNX embedding model | Pre-built, loaded on RPi | ~91 MB |
| ONNX Runtime | RPi 5 | ~5–10 MB overhead |
| LLM (Groq) | Cloud (free tier) | 0 MB |
| **Total added footprint on RPi** | | **~115 MB** |

No PyTorch, no `sentence-transformers`, no PDF parsing on the Pi. Everything heavy happens at build time on your Mac or Colab.
