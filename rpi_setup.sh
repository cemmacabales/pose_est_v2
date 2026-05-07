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

# Install dependencies
pip install tflite-runtime opencv-python numpy Pillow

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

echo "Setup complete. Run: python gui.py"
