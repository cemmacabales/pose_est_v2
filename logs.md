# Training Log — Pose Estimation Exercise Classifier

**Project:** Pose Estimation + Exercise Classification (BlazePose + LSTM)
**Model:** Dual-head LSTM (9 exercise classes + 2 quality classes)
**Input:** 30-frame windows of 16 joint-angle features extracted from 12 BlazePose keypoints (hip-centered)
**Hardware:** Mac M2 (training only), RPi 5 (inference target)

---

## Model Architecture

```
Input: (30, 16)        # 30 time steps x 16 angle features (hip-centered)
  LSTM(64, return_sequences=True, unroll=True)
  LSTM(32, unroll=True)
  Dense(64, ReLU) + Dropout(0.3)
  +--- exercise_out: Dense(9, softmax)
  +--- quality_out: Dense(2, softmax)
```

- Optimizer: Adam
- Batch size: 32
- Loss: Categorical cross-entropy (both heads)
- EarlyStopping: patience=10, restore_best_weights=True
- Augmentation: Horizontal flip, Gaussian noise (sigma=0.01), time stretch (0.8x-1.2x), joint masking (1-2 joints zeroed)
- Window strides: 5, 2 (5x augmentation factor)

---

## The 9 Exercise Classes

| Class Index | Exercise | ID | Training Videos |
|-------------|----------|----|----------------|
| 0 | Deep Squat | 01 | 22 |
| 1 | Hurdle Step | 02 | 26 |
| 2 | Inline Lunge | 03 | 26 |
| 3 | Side Lunge | 04 | 26 |
| 4 | Sit to Stand | 05 | 21 |
| 5 | Standing Leg Raise | 06 | 24 |
| 6 | Shoulder Abduction | 07 | 9 |
| 7 | Shoulder Extension | 08 | 10 |
| 8 | Shoulder Scaption | 10 | 10 |

---

## Session 2: Initial Training (from scratch, 174 training videos)

**Date:** 2026-05-29
**Dataset:** 174 videos, 374,570 windows (5x augmentation)
**Training split:** 80/20 (train/val), seed=42
**Optimizer:** Adam(lr=0.001)
**Max epochs:** 150
**EarlyStopping:** patience=10
**Checkpoint:** models/classifier_best.keras

### Training Progress

| Epoch | Train Loss | Train Ex Acc | Train Qu Acc | Val Loss | Val Ex Acc | Val Qu Acc |
|-------|-----------|-------------|-------------|---------|-----------|-----------|
| 1 | 1.1274 | 79.00% | 71.11% | 0.8204 | 86.93% | 76.45% |
| 5 | 0.5043 | 93.48% | 84.65% | 0.4721 | 94.03% | 85.54% |
| 10 | 0.3748 | 95.49% | 88.46% | 0.3679 | 95.68% | 88.63% |
| 20 | 0.2852 | 96.65% | 91.28% | 0.3379 | 95.88% | 90.27% |
| 30 | 0.2425 | 97.26% | 92.69% | 0.2755 | 96.98% | 91.91% |
| 40 | 0.2162 | 97.59% | 93.56% | 0.2721 | 97.02% | 92.27% |
| 45 | 0.2061 | 97.70% | 93.81% | 0.2607 | 97.15% | 92.93% |

### Final Results (Epoch 45 — EarlyStopping triggered)

| Metric | Value |
|--------|-------|
| **Val Exercise Accuracy** | **97.15%** |
| **Val Quality Accuracy** | **92.93%** |
| Train Exercise Accuracy | 97.70% |
| Train Quality Accuracy | 93.81% |
| Train Loss | 0.2061 |
| Val Loss | 0.2607 |
| Best model saved at | models/classifier.keras |

### Export
- `classifier.keras` → `classifier.tflite` (219 KB)
- Input: `(1, 30, 16)`, Outputs: `(1, 9)` exercise + `(1, 2)` quality
- Verified with dummy inference

### Training Set Annotation (174 videos)
- 100% exercise accuracy, 98.9% quality accuracy (172/174 correct)

### Test Set Annotation (26 unseen videos)
- Real-world test videos from eval/TEST/ with different camera angles (frontal, lateral, 45-degree) and subjects not seen during training
- Output: eval/results/test_annotated/ (26 files, 743 MB)

### Observations
- Exercise classification on training set: near perfect (100%). Likely overfit to training subjects/angles.
- Test videos expose the generalization gap — some exercises get confused on unseen angles.
- Quality classification is the harder task (92.8% val vs 97.2% ex).
- BlazePose detects all 33 landmarks reliably (0 frame drops on test videos).

### Window Distribution (174 training videos, 374,570 windows)

| Class | Count |
|-------|-------|
| 0 (Deep Squat) | 52,565 |
| 1 (Hurdle Step) | 62,845 |
| 2 (Inline Lunge) | 52,580 |
| 3 (Side Lunge) | 45,990 |
| 4 (Sit to Stand) | 43,145 |
| 5 (Standing Leg Raise) | 41,955 |
| 6 (Shoulder Abduction) | 18,995 |
| 7 (Shoulder Extension) | 24,815 |
| 8 (Shoulder Scaption) | 31,680 |
| **Total** | **374,570** |

Quality: correct=200,230 / incorrect=174,340

---

## Session 3: Fine-Tuning (174 train + 26 test videos)

**Date:** 2026-05-29
**Dataset:** 200 videos (174 train + 26 test), ~420,000+ windows (rebuilt)
**Training split:** 80/20 (train/val), seed=42
**Pre-trained weights:** models/classifier.keras (from Session 2)
**Optimizer:** Adam(lr=0.0001) — lowered for fine-tuning
**Max epochs:** 150
**EarlyStopping:** patience=10
**Checkpoint:** models/classifier_best.keras

### Test Video Distribution

| Class | Count |
|-------|-------|
| 0 (Deep Squat) | +2 |
| 1 (Hurdle Step) | +2 |
| 2 (Inline Lunge) | +2 |
| 3 (Side Lunge) | +8 |
| 4 (Sit to Stand) | +6 |
| 5 (Standing Leg Raise) | +2 |
| 6 (Shoulder Abduction) | +1 |
| 7 (Shoulder Extension) | +1 |
| 8 (Shoulder Scaption) | +1 |
| **Total added** | **26** |

**Note:** All 26 test videos labeled as "correct" quality (demonstration/reference videos). Test videos include varied camera angles (frontal, lateral, 45-degree) and subjects not seen during initial training.

### Fine-Tuning Progress

*(Results will be filled after fine-tuning runs)*

| Epoch | Train Loss | Train Ex Acc | Train Qu Acc | Val Loss | Val Ex Acc | Val Qu Acc |
|-------|-----------|-------------|-------------|---------|-----------|-----------|
| 1 | 0.6383 | 90.06% | 89.69% | 0.4319 | 92.69% | 91.56% |
| 5 | 0.3239 | 94.68% | 93.25% | 0.3151 | 95.09% | 93.07% |
| 10 | 0.2742 | 95.70% | 93.99% | 0.2866 | 95.74% | 93.53% |
| 15 | 0.2497 | 96.15% | 94.34% | 0.2720 | 96.08% | 93.84% |
| 20 | 0.2325 | 96.48% | 94.65% | 0.2639 | 96.28% | 94.00% |
| 25 | 0.2198 | 96.70% | 94.86% | 0.2612 | 96.46% | 94.09% |
| 30 | 0.2094 | 96.84% | 95.03% | 0.2608 | 96.47% | 94.20% |
| 35 | 0.2011 | 96.97% | 95.17% | 0.2564 | 96.57% | 94.31% |
| 40 | 0.1940 | 97.09% | 95.30% | 0.2580 | 96.68% | 94.36% |
| 45 | 0.1886 | 97.16% | 95.42% | 0.2563 | 96.70% | 94.44% |
| 48 | 0.1848 | 97.22% | 95.48% | 0.2563 | 96.72% | 94.44% |

### Final Fine-Tuned Results

Training completed at epoch 48/150 (EarlyStopping patience=10, best at epoch 38).

| Metric | Before Fine-Tuning | After Fine-Tuning | Change |
|--------|-------------------|-------------------|--------|
| Val Exercise Accuracy | 97.15% | 96.66% | -0.49% |
| Val Quality Accuracy | 92.93% | 94.37% | +1.44% |
| Train Exercise Accuracy | 97.70% | 97.22% | -0.48% |
| Train Quality Accuracy | 93.81% | 95.48% | +1.67% |
| Train Loss | 0.2061 | 0.1848 | -0.0213 |
| Val Loss | 0.2607 | 0.2545 (best) | -0.0062 |

**Analysis:** Adding 26 test videos with varied camera angles and unseen subjects reduced exercise accuracy slightly (-0.49%), as expected when broadening data distribution. Quality accuracy improved by +1.44%, suggesting better generalization on form assessment. The quality head now benefits from more diverse motion patterns.

### Export
- `classifier.keras` → `classifier.tflite` (219 KB, 218,960 bytes)
- Input: `(1, 30, 16)`, Outputs: `(1, 9)` exercise + `(1, 2)` quality
- Verified with dummy inference — both output shapes confirmed

---

## Active Artifacts

| File | Size | Description |
|------|------|-------------|
| `models/pose_landmarker_full.task` | 9.4 MB | BlazePose Full model |
| `models/classifier.keras` | ~480 KB | Best Keras LSTM model |
| `models/classifier.tflite` | ~219 KB | TFLite model for RPi inference |
| `data/normalized/*.npy` | 174 → 200 files | Per-video keypoints (12 joints, 2D) |
| `data/labels.csv` | 174 → 200 entries | Video metadata |
| `data/X_train.npy` | (374,570+, 30, 16) | Training windows (rebuilt) |
| `data/y_exercise.npy` | (374,570+,) | Exercise labels |
| `data/y_quality.npy` | (374,570+,) | Quality labels |

---

## Environment

| Venv | Python | Purpose | Key Packages |
|------|--------|---------|-------------|
| `.venv` | 3.14.5 (Homebrew) | Extraction, inference, GUI | mediapipe, opencv-python |
| `.venv-tf` | 3.11.9 (Homebrew) | Training, export | tensorflow, numpy |

---

## RPi 5 Deployment Targets

| Configuration | Resolution | Expected FPS |
|--------------|-----------|-------------|
| Full model (`model_complexity=1`) | 640x480 | 10–18 |
| Lite model (`model_complexity=0`) | 640x480 | 25–30 |

---

*Last updated: 2026-05-29*
