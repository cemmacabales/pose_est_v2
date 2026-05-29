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
    X = np.load('./data/X_train.npy')          # (N, 30, 24) → (N, 30, 16) with angles
    y_exercise = np.load('./data/y_exercise.npy')  # (N,)
    y_quality = np.load('./data/y_quality.npy')    # (N,)

    N = X.shape[0]
    if N < 50:
        warnings.warn(f"Warning: Only {N} samples found. (< 50). Continuing anyway.")

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
    X_test = None
    y_exercise_train = y_exercise_cat[train_idx]
    y_exercise_val = y_exercise_cat[val_idx]
    y_exercise_test = None
    y_quality_train = y_quality_cat[train_idx]
    y_quality_val = y_quality_cat[val_idx]
    y_quality_test = None

    # ------------------------------------------------------------------
    # 4. Build model
    # ------------------------------------------------------------------
    inputs = Input(shape=(30, 16))
    x = LSTM(64, return_sequences=True, unroll=True)(inputs)
    x = LSTM(32, unroll=True)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    exercise_out = Dense(9, activation='softmax', name='exercise')(x)
    quality_out = Dense(2, activation='softmax', name='quality')(x)
    model = Model(inputs, [exercise_out, quality_out])

    # ------------------------------------------------------------------
    # 6. Compile
    # ------------------------------------------------------------------
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={'exercise': 'categorical_crossentropy',
              'quality': 'categorical_crossentropy'},
        metrics={'exercise': 'accuracy', 'quality': 'accuracy'}
    )

    # ------------------------------------------------------------------
    # 7. Callbacks
    # ------------------------------------------------------------------
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
        ModelCheckpoint('./models/classifier_best.keras', save_best_only=True, verbose=0),
        ProgressCallback(total_epochs=150, log_file='./logs/training.log')
    ]

    # ------------------------------------------------------------------
    # 8. Train
    # ------------------------------------------------------------------
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
    # 9. Print final val accuracy for both heads
    # ------------------------------------------------------------------
    # Evaluate on validation set to get clean final metrics
    val_metrics = model.evaluate(
        X_val,
        {'exercise': y_exercise_val, 'quality': y_quality_val},
        verbose=0,
        return_dict=True
    )

    print(f"Final val exercise accuracy: {val_metrics.get('exercise_accuracy', 'N/A')}")
    print(f"Final val quality accuracy:  {val_metrics.get('quality_accuracy', 'N/A')}")

    # ------------------------------------------------------------------
    # 10. Evaluate on held-out test set (if available)
    # ------------------------------------------------------------------
    if X_test is not None and X_test.shape[0] > 0:
        test_metrics = model.evaluate(
            X_test,
            {'exercise': y_exercise_test, 'quality': y_quality_test},
            verbose=0,
            return_dict=True
        )
        print(f"Final test exercise accuracy: {test_metrics.get('exercise_accuracy', 'N/A')}")
        print(f"Final test quality accuracy:  {test_metrics.get('quality_accuracy', 'N/A')}")

    # ------------------------------------------------------------------
    # 11. Save final model
    # ------------------------------------------------------------------
    model.save('./models/classifier.keras')
    print("Model saved to ./models/classifier.keras")


if __name__ == '__main__':
    main()
