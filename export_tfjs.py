#!/usr/bin/env python3
"""
export_tfjs.py
Convert classifier.keras → TF.js layers-model (model.json + weights bin).

Run from the repo root with the .venv-tf environment:
    source .venv-tf/bin/activate && python export_tfjs.py

Requires tf_keras (Keras 2.x) to produce a TF.js-compatible model topology.
Standalone Keras 3.x serializes layers in a format TF.js cannot deserialize
(e.g. InputLayer uses `batch_shape` instead of the `batch_input_shape` that
tf.loadLayersModel expects).
"""
import os
import sys
import types

# tensorflowjs imports tensorflow_hub which uses pkg_resources (removed in
# Python 3.12, and sometimes missing in 3.11 venvs).  Patch it before import.
if "pkg_resources" not in sys.modules:
    pkg = types.ModuleType("pkg_resources")
    from packaging.version import Version
    pkg.parse_version = Version
    sys.modules["pkg_resources"] = pkg

import numpy as np
import keras                         # standalone Keras 3 – loads .keras files
import tf_keras as tk                # legacy Keras 2 – produces TF.js-compatible topology
from tensorflowjs.converters.keras_h5_conversion import save_keras_model

KERAS_PATH  = "./models/classifier.keras"
TFJS_DIR    = "./web/models/classifier_tfjs"


def build_tk_model() -> tk.Model:
    inputs = tk.Input(shape=(30, 16), name="input_layer")
    x = tk.layers.LSTM(64, return_sequences=True, unroll=True, name="lstm")(inputs)
    x = tk.layers.LSTM(32, return_sequences=False, unroll=True, name="lstm_1")(x)
    x = tk.layers.Dense(64, activation="relu", name="dense")(x)
    x = tk.layers.Dropout(0.3, name="dropout")(x)
    out_exercise = tk.layers.Dense(9, activation="softmax", name="exercise")(x)
    out_quality  = tk.layers.Dense(2, activation="softmax", name="quality")(x)
    return tk.Model(inputs=inputs, outputs=[out_exercise, out_quality], name="functional")


def copy_weights(src_model: keras.Model, dst_model: tk.Model) -> None:
    for layer in dst_model.layers:
        try:
            src_layer = src_model.get_layer(layer.name)
            weights = src_layer.get_weights()
            if weights:
                layer.set_weights(weights)
        except ValueError:
            pass


def verify(model: tk.Model) -> None:
    dummy = np.zeros((1, 30, 16), dtype="float32")
    out = model.predict(dummy, verbose=0)
    assert len(out) == 2, f"Expected 2 outputs, got {len(out)}"
    assert out[0].shape == (1, 9), f"exercise shape: {out[0].shape}"
    assert out[1].shape == (1, 2), f"quality shape:  {out[1].shape}"
    assert abs(out[0].sum() - 1.0) < 1e-5, "exercise softmax does not sum to 1"
    assert abs(out[1].sum() - 1.0) < 1e-5, "quality softmax does not sum to 1"
    print("Inference check passed.")


def main() -> int:
    if not os.path.exists(KERAS_PATH):
        print(f"ERROR: {KERAS_PATH} not found", file=sys.stderr)
        return 1

    print("Loading classifier.keras …")
    k3_model = keras.models.load_model(KERAS_PATH)

    print("Building tf_keras model …")
    tk_model = build_tk_model()

    print("Copying weights …")
    copy_weights(k3_model, tk_model)

    print("Verifying inference …")
    verify(tk_model)

    print(f"Saving TF.js model to {TFJS_DIR} …")
    save_keras_model(tk_model, TFJS_DIR)

    size = os.path.getsize(os.path.join(TFJS_DIR, "group1-shard1of1.bin"))
    print(f"Done. Weights bin: {size:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
