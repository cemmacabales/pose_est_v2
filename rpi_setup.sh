#!/bin/bash
#
# rpi_setup.sh — RPi 5 Pose + Classifier Setup
#
# SETUP VERIFICATION CHECKLIST
# After running: bash rpi_setup.sh
#
# [ ] No pip install errors
# [ ] espeak works: espeak "test" (hear audio)
# [ ] pactl list short sinks (shows audio outputs)
# [ ] python -c "import mediapipe" — no error
# [ ] python -c "import gtts" — no error
# [ ] python -c "import flask" — no error
# [ ] python -c "import qrcode" — no error
# [ ] python -c "import playsound" — no error
# [ ] pytest tests/ -v — all pass
# [ ] python gui.py --model lite launches without errors

set -e

echo "============================================"
echo "  RPi 5 Pose Estimation + Classifier Setup"
echo "============================================"
echo ""

# Check Python >= 3.9
PYTHON_VERSION=$(python3 --version 2>/dev/null | awk '{print $2}')
if [ -z "$PYTHON_VERSION" ]; then
    echo "Error: Python 3 is not installed."
    exit 1
fi

MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$MAJOR" -lt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 9 ]; }; then
    echo "Error: Python >= 3.9 required. Found: $PYTHON_VERSION"
    exit 1
fi

echo "Python version: $PYTHON_VERSION"
echo ""

# Create and activate venv
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip -q

echo ""
echo "--- Installing runtime dependencies ---"
echo ""
pip install ai-edge-litert opencv-python numpy Pillow \
            gtts "playsound==1.2.2" "qrcode[pil]" \
            flask python-dotenv pytest \
            groq onnxruntime tokenizers

echo ""
echo "--- Installing mediapipe (runtime) ---"
echo ""

# mediapipe now ships an ARM64 wheel for RPi 5 via pip.
if python -c "import mediapipe" 2>/dev/null; then
    echo "mediapipe already installed. Skipping."
else
    pip install mediapipe
fi

echo ""
echo "--- Installing PulseAudio for Bluetooth speaker support ---"
echo ""

if ! command -v pactl &>/dev/null; then
    echo "PulseAudio not found. Installing..."
    sudo apt-get install -y -qq pulseaudio pulseaudio-module-bluetooth 2>/dev/null || true
    echo ""
    echo "After reboot, pair your Bluetooth speaker with:"
    echo "  bluetoothctl"
    echo "  scan on"
    echo "  pair <MAC>"
    echo "  connect <MAC>"
    echo "  trust <MAC>"
    echo "Then select it as default sink:"
    echo "  pactl list short sinks"
    echo "  pactl set-default-sink <sink_name>"
    echo ""
else
    echo "PulseAudio already installed."
fi

echo ""
echo "--- Installing espeak (TTS) ---"
sudo apt-get install -y -qq espeak 2>/dev/null || true

echo ""
echo "--- Checking model files ---"

CLASSIFIER_MODEL="./models/classifier.tflite"
POSE_MODEL="./models/pose_landmarker_lite.task"

if [ ! -f "$CLASSIFIER_MODEL" ]; then
    echo "Warning: classifier model not found: $CLASSIFIER_MODEL"
    echo "Copy classifier.tflite from your Mac build:"
    echo "  scp models/classifier.tflite pi@raspberrypi.local:~/pose_est_v2/models/"
fi

if [ ! -f "$POSE_MODEL" ]; then
    echo "Note: BlazePose Lite model will be auto-downloaded on first run."
    echo "  Or pre-download with:"
    echo "  python -c \"from pose_estimator import PoseEstimator; PoseEstimator(model_complexity=0)\""
fi

echo ""
echo "--- Running tests ---"
if [ -f "$CLASSIFIER_MODEL" ]; then
    pytest tests/ -v --tb=short || echo "Some tests failed — review output above."
else
    echo "Skipping tests — classifier.tflite not found."
fi

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Run the GUI with:"
echo "  source .venv/bin/activate"
echo "  python gui.py --model lite"
echo ""
echo "For desktop/Mac use:"
echo "  python gui.py --model full"
echo ""
echo "Build-time deps (PyTorch, sentence-transformers) are NOT installed."
echo "Install them on Mac for knowledge base building only:"
echo "  pip install pymupdf sentence-transformers torch transformers"
echo ""
