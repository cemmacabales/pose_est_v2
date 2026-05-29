#!/bin/bash
#
# rpi_setup.sh — RPi 5 Pose + Classifier Setup
#
# SETUP VERIFICATION CHECKLIST
# After running: bash rpi_setup.sh
#
# [ ] No pip install errors
# [ ] espeak works: espeak "test" (hear audio)
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

# mediapipe has no official Linux ARM64 wheel for RPi 5.
# We attempt to build from source. This takes ~30-40 minutes.
# If a pre-built community wheel is available, it will be auto-detected.
#
# The build requires:
#   - Python development headers
#   - OpenGL ES / EGL development packages
#   - Mesa / libEGL / libGLESv2
#   - Compiler toolchain (gcc, g++, make)

if python -c "import mediapipe" 2>/dev/null; then
    echo "mediapipe already installed. Skipping build."
else
    echo "mediapipe not found. Install system build dependencies..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        python3-dev \
        gcc g++ make \
        libgl1-mesa-dev \
        libgles2-mesa-dev \
        libegl1-mesa-dev \
        pkg-config \
        || true

    echo ""
    echo "Attempting to install mediapipe from source..."
    echo "This may take 30-40 minutes on RPi 5."
    echo ""

    mkdir -p /tmp/mediapipe_build
    cd /tmp/mediapipe_build

    if [ ! -d "mediapipe" ]; then
        git clone --depth 1 --branch v0.10.14 \
            https://github.com/google-ai-edge/mediapipe.git 2>/dev/null || \
        git clone --depth 1 \
            https://github.com/google-ai-edge/mediapipe.git
    fi

    cd mediapipe

    pip install wheel setuptools -q

    if ! pip install . --no-build-isolation 2>/dev/null; then
        echo ""
        echo "============================================"
        echo "  WARNING: mediapipe source build failed."
        echo "============================================"
        echo ""
        echo "This is expected — mediapipe has no official ARM64 wheel."
        echo ""
        echo "ALTERNATIVES:"
        echo "  1. Try a community wheel (search pip for mediapipe ARM64)"
        echo "  2. Use BlazePose via OpenCV's cv2.dnn module instead"
        echo "  3. Offload pose estimation to cloud/edge TPU"
        echo ""
        echo "Pose estimation GUI will NOT work without mediapipe."
        echo "All other components (training, data processing) run on Mac."
        echo ""
    else
        echo "mediapipe installed successfully from source."
    fi

    cd /tmp
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
