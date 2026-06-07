#!/usr/bin/env python3
"""
export_models.py
Convert classifier.keras to TFLite.
"""
import os
import sys

import tensorflow as tf

MODELS_DIR = "./models"
CLASSIFIER_KERAS_PATH = os.path.join(MODELS_DIR, "classifier.keras")
CLASSIFIER_TFLITE_PATH = os.path.join(MODELS_DIR, "classifier.tflite")


def ensure_models_dir() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)


def convert_classifier() -> None:
    print("Loading classifier Keras model ...")
    if not os.path.exists(CLASSIFIER_KERAS_PATH):
        raise FileNotFoundError(f"Classifier not found: {CLASSIFIER_KERAS_PATH}")
    model = tf.keras.models.load_model(CLASSIFIER_KERAS_PATH)
    print("Converting classifier to TFLite ...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # No quantization: keeps FULLY_CONNECTED at op-version <= 9, which is
    # required by the tfjs-tflite alpha.10 WASM runtime.
    tflite_model = converter.convert()
    with open(CLASSIFIER_TFLITE_PATH, "wb") as f:
        f.write(tflite_model)
    size = os.path.getsize(CLASSIFIER_TFLITE_PATH)
    print(f"Saved to {CLASSIFIER_TFLITE_PATH} ({size} bytes)")
    if size < 10_000:
        raise RuntimeError(f"Classifier TFLite file too small: {size} bytes")
    print("Classifier file size OK (>10KB)")


def verify_classifier() -> None:
    print("Loading classifier TFLite model ...")
    interpreter = tf.lite.Interpreter(model_path=CLASSIFIER_TFLITE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Classifier input details:")
    for detail in input_details:
        print(f"  {detail}")
    print("Classifier output details:")
    for detail in output_details:
        print(f"  {detail}")
    dummy_input = tf.zeros((1, 30, 16), dtype=tf.float32)
    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()
    if len(output_details) != 2:
        raise RuntimeError(
            f"Expected 2 outputs, got {len(output_details)}"
        )
    out_0 = interpreter.get_tensor(output_details[0]["index"])
    out_1 = interpreter.get_tensor(output_details[1]["index"])
    print(f"Classifier output 0 shape: {out_0.shape}")
    print(f"Classifier output 1 shape: {out_1.shape}")
    shapes = {out_0.shape, out_1.shape}
    if shapes != {(1, 9), (1, 2)}:
        raise RuntimeError(
            f"Unexpected output shapes: {out_0.shape}, {out_1.shape}, expected (1, 9) and (1, 2)"
        )
    out_exercise = out_0 if out_0.shape == (1, 9) else out_1
    out_quality = out_0 if out_0.shape == (1, 2) else out_1
    print(f"Classifier exercise output shape: {out_exercise.shape}")
    print(f"Classifier quality output shape: {out_quality.shape}")
    print("Classifier dummy inference OK")


def main() -> int:
    ensure_models_dir()
    convert_classifier()
    verify_classifier()
    print("\nAll exports and verifications passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
