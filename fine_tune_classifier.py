#!/usr/bin/env python3
"""
fine_tune_classifier.py — Fine-tune the trained classifier on expanded dataset
(174 training + 26 test videos). Loads classifier.keras and continues training
with a lower learning rate.

Usage:
    source .venv-tf/bin/activate
    python fine_tune_classifier.py

Prerequisite:
    python extract_test_keypoints.py   # extract test keypoints
    python build_windows.py            # rebuild X_train.npy with test data
"""

import os
import time
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.utils import to_categorical


class ProgressCallback(Callback):
    def __init__(self, total_epochs, log_file=None):
        super().__init__()
        self.total_epochs = total_epochs
        self.start_time = time.time()
        self.best_val_loss = float('inf')
        self.epoch_start = None
        self.log_file = log_file

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start
        elapsed = time.time() - self.start_time
        remaining_epochs = self.total_epochs - (epoch + 1)
        eta = (elapsed / (epoch + 1)) * remaining_epochs if epoch > 0 else 0

        val_loss = logs.get('val_loss', 0)
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
        best_marker = '  *** BEST ***' if is_best else ''

        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        eta_mins = int(eta // 60)
        eta_secs = int(eta % 60)

        header = (f"=== Epoch {epoch + 1:3d}/{self.total_epochs} | "
                  f"{epoch_time:5.1f}s | "
                  f"Total: {mins:3d}m{secs:02d}s | "
                  f"ETA: ~{eta_mins}m{eta_secs:02d}s ==={best_marker}")

        train_line = (f"Train | loss: {logs.get('loss', 0):.4f}  "
                      f"ex_acc: {logs.get('exercise_accuracy', 0):.4f}  "
                      f"qu_acc: {logs.get('quality_accuracy', 0):.4f}")

        val_line = (f"Val   | loss: {val_loss:.4f}  "
                    f"ex_acc: {logs.get('val_exercise_accuracy', 0):.4f}  "
                    f"qu_acc: {logs.get('val_quality_accuracy', 0):.4f}")

        out = f"\n{header}\n{train_line}\n{val_line}\n"
        print(out)

        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(out)


def main():
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    X = np.load('./data/X_train.npy')
    y_exercise = np.load('./data/y_exercise.npy')
    y_quality = np.load('./data/y_quality.npy')

    N = X.shape[0]
    print(f"Loaded {N} windows, shape={X.shape}, features={X.shape[2]}")

    if N < 50:
        warnings.warn(f"Warning: Only {N} samples found. (< 50). Continuing anyway.")

    print("Exercise class distribution:")
    for cls in range(9):
        count = int(np.sum(y_exercise == cls))
        print(f"  Class {cls}: {count}")

    correct_count = int(np.sum(y_quality == 1))
    incorrect_count = int(np.sum(y_quality == 0))
    print(f"Quality distribution: correct={correct_count}, incorrect={incorrect_count}")

    # ------------------------------------------------------------------
    # 2. One-hot encode labels
    # ------------------------------------------------------------------
    y_exercise_cat = to_categorical(y_exercise, num_classes=9)
    y_quality_cat = to_categorical(y_quality, num_classes=2)

    # ------------------------------------------------------------------
    # 3. Train/val split (80/20) with shuffle seed=42
    # ------------------------------------------------------------------
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)

    split_idx = int(0.8 * N)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    X_train = X[train_idx]
    X_val = X[val_idx]
    y_exercise_train = y_exercise_cat[train_idx]
    y_exercise_val = y_exercise_cat[val_idx]
    y_quality_train = y_quality_cat[train_idx]
    y_quality_val = y_quality_cat[val_idx]

    # ------------------------------------------------------------------
    # 4. Load pre-trained model
    # ------------------------------------------------------------------
    model_path = './models/classifier.keras'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pre-trained model not found: {model_path}")

    print(f"Loading pre-trained model from {model_path} ...")
    model = tf.keras.models.load_model(model_path)

    print("\nPre-trained model summary:")
    model.summary()

    # ------------------------------------------------------------------
    # 5. Recompile with lower learning rate for fine-tuning
    # ------------------------------------------------------------------
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss={'exercise': 'categorical_crossentropy',
              'quality': 'categorical_crossentropy'},
        metrics={'exercise': 'accuracy', 'quality': 'accuracy'}
    )

    print("\nRecompiled with Adam(lr=0.0001) for fine-tuning.")

    # ------------------------------------------------------------------
    # 6. Callbacks
    # ------------------------------------------------------------------
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
        ModelCheckpoint('./models/classifier_best.keras', save_best_only=True, verbose=0),
        ProgressCallback(total_epochs=150, log_file='./logs/fine_tuning.log')
    ]

    # ------------------------------------------------------------------
    # 7. Fine-tune
    # ------------------------------------------------------------------
    print("\nStarting fine-tuning (150 epochs max, EarlyStopping patience=10, LR=0.0001)...\n")

    history = model.fit(
        X_train,
        {'exercise': y_exercise_train, 'quality': y_quality_train},
        validation_data=(X_val, {'exercise': y_exercise_val, 'quality': y_quality_val}),
        epochs=150,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )

    # ------------------------------------------------------------------
    # 8. Evaluate
    # ------------------------------------------------------------------
    val_metrics = model.evaluate(
        X_val,
        {'exercise': y_exercise_val, 'quality': y_quality_val},
        verbose=0,
        return_dict=True
    )

    print(f"\nFinal val exercise accuracy: {val_metrics.get('exercise_accuracy', 'N/A')}")
    print(f"Final val quality accuracy:  {val_metrics.get('quality_accuracy', 'N/A')}")

    # ------------------------------------------------------------------
    # 9. Save fine-tuned model
    # ------------------------------------------------------------------
    model.save('./models/classifier.keras')
    print("\nFine-tuned model saved to ./models/classifier.keras")


if __name__ == '__main__':
    main()
