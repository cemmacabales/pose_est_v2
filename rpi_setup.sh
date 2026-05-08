# SETUP VERIFICATION CHECKLIST
# After running: bash rpi_setup.sh
#
# [ ] No pip install errors
# [ ] espeak works: espeak "test" (hear audio)
# [ ] python -c "import gtts" — no error
# [ ] python -c "import flask" — no error
# [ ] python -c "from google import genai" — no error
# [ ] python -c "import qrcode" — no error
# [ ] python -c "import playsound" — no error
# [ ] gcloud auth application-default login completed
# [ ] gcloud auth application-default set-quota-project gen-lang-client-0629431240 run
# [ ] VERTEX_PROJECT set in ~/.bashrc
# [ ] VERTEX_LOCATION set in ~/.bashrc
# [ ] pytest tests/ -v — 39 passed
# [ ] python gui.py launches without errors
#!/bin/bash

echo "RPi 5 Pose + Classifier Setup"

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


# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install ai-edge-litert opencv-python numpy Pillow \
            gtts "playsound==1.2.2" "qrcode[pil]" \
            flask google-genai python-dotenv pytest

sudo apt install espeak -y
pip install gtts "playsound==1.2.2" "qrcode[pil]" flask \
            google-genai python-dotenv pytest

echo ""
echo "IMPORTANT: Authenticate with Google Cloud for Vertex AI:"
echo "  1. Install gcloud: https://cloud.google.com/sdk/docs/install"
echo "  2. Run: gcloud auth application-default login"
echo "  3. Run: gcloud auth application-default set-quota-project gen-lang-client-0629431240"
echo ""

if [ -z "$VERTEX_PROJECT" ]; then
  echo "WARNING: VERTEX_PROJECT not set."
  echo "Add to ~/.bashrc: export VERTEX_PROJECT=gen-lang-client-0629431240"
  echo "Add to ~/.bashrc: export VERTEX_LOCATION=us-central1"
  echo ""
fi

# Check model files
MODEL1="./models/movenet_thunder_int8.tflite"
MODEL2="./models/classifier.tflite"

if [ ! -f "$MODEL1" ]; then
    echo "Missing model file: $MODEL1"
    exit 1
fi

if [ ! -f "$MODEL2" ]; then
    echo "Missing model file: $MODEL2"
    exit 1
fi

echo "Running tests..."
pytest tests/ -v --tb=short
if [ $? -ne 0 ]; then
  echo "ERROR: Tests failed. Fix before running gui.py."
  exit 1
fi

echo "Setup complete. Run: python gui.py"
