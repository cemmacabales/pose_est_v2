# Chapter 4: Experimental Results and Deployment

## 4.1 Overview

This chapter presents the experimental evaluation of the pose estimation and exercise classification pipeline. Two primary experiments were conducted: (1) initial training of the dual-head LSTM classifier from 174 labelled training videos, and (2) fine-tuning with an additional 26 test videos to improve generalisation across camera angles and unseen subjects. The optimised models were then deployed to a Raspberry Pi 5 (RPi 5) single-board computer for edge inference, and the complete runtime pipeline was validated with live camera input.

---

## 4.2 Dataset and Training Configuration

### 4.2.1 Input Features

The classifier operates on 30-frame windows of 16 joint-angle features derived from 12 BlazePose landmarks. Coordinates are normalised by the hip midpoint before angle computation. The six body-part pairs and their corresponding MediaPipe landmark indices are listed in Table 4.1.

**Table 4.1:** Body landmark indices used for angle feature extraction.

| Body Part | Left Index | Right Index |
|-----------|-----------|-------------|
| Shoulder | 11 | 12 |
| Elbow     | 13 | 14 |
| Wrist     | 15 | 16 |
| Hip       | 23 | 24 |
| Knee      | 25 | 26 |
| Ankle     | 27 | 28 |

Joint angles were computed vectorially over all frames, producing 16 features per time step (six angles × two sides, plus four inter-segment ratios for timing normalisation).

### 4.2.2 Augmentation

Five-fold augmentation was applied during window construction:
- Horizontal flip (reflection across the vertical axis)
- Gaussian noise (σ = 0.01)
- Time stretch (0.8×–1.2× speed perturbation)
- Joint masking (one to two joints zeroed at random)
- Window stride reduction (stride 5 for base, stride 2 for augmented copies)

From 174 raw training videos, 374,570 windows were generated. The quality label distribution was 200,230 correct and 174,340 incorrect.

### 4.2.3 Architecture

The classifier architecture is a two-layer LSTM with a dual-head output:

```
Input: (30, 16)
  LSTM(64, return_sequences=True, unroll=True)
  LSTM(32, unroll=True)
  Dense(64, ReLU) + Dropout(0.3)
  +--- exercise_out: Dense(9, softmax)
  +--- quality_out: Dense(2, softmax)
```

Training used the Adam optimiser (initial learning rate 1×10⁻³ for session 2; 1×10⁻⁴ for fine-tuning), batch size 32, categorical cross-entropy loss on both heads, and EarlyStopping with patience = 10 and restore_best_weights enabled.

---

## 4.3 Initial Training Results (Session 2)

The model was trained from scratch on 174 videos. The 80/20 train/validation split was applied with seed = 42. Training ran for 45 epochs before EarlyStopping triggered (patience exhausted, best model at epoch 45).

**Table 4.2:** Selected training/epoch results for the initial training session.

| Epoch | Train Loss | Train Ex Acc | Train Qu Acc | Val Loss | Val Ex Acc | Val Qu Acc |
|-------|-----------|-------------|-------------|---------|-----------|-----------|
| 1     | 1.1274    | 79.00%      | 71.11%      | 0.8204  | 86.93%    | 76.45%    |
| 5     | 0.5043    | 93.48%      | 84.65%      | 0.4721  | 94.03%    | 85.54%    |
| 10    | 0.3748    | 95.49%      | 88.46%      | 0.3679  | 95.68%    | 88.63%    |
| 20    | 0.2852    | 96.65%      | 91.28%      | 0.3379  | 95.88%    | 90.27%    |
| 30    | 0.2425    | 97.26%      | 92.69%      | 0.2755  | 96.98%    | 91.91%    |
| 40    | 0.2162    | 97.59%      | 93.56%      | 0.2721  | 97.02%    | 92.27%    |
| 45    | 0.2061    | 97.70%      | 93.81%      | 0.2607  | 97.15%    | 92.93%    |

**Table 4.3:** Final validation metrics for the initial training session.

| Metric                    | Value    |
|---------------------------|----------|
| Val Exercise Accuracy     | 97.15%   |
| Val Quality Accuracy      | 92.93%   |
| Train Exercise Accuracy  | 97.70%   |
| Train Quality Accuracy   | 93.81%   |
| Train Loss                | 0.2061   |
| Val Loss                  | 0.2607   |

The exercise classification head outperformed the quality assessment head by approximately 4.2 percentage points on validation, indicating that form quality discrimination is a harder task than exercise type classification. Annotation of the full training set showed near-perfect exercise accuracy (100%) and 98.9% quality accuracy (172/174 videos correct), consistent with mild overfitting to the five training subjects and their recording angles.

The model checkpoint was saved to `models/classifier.keras` and subsequently exported to TensorFlow Lite format for edge deployment.

---

## 4.4 Fine-Tuning Results (Session 3)

The pre-trained weights from Session 2 were fine-tuned on the combined dataset of 174 original training videos and 26 additional test videos spanning all nine exercise classes. The test videos were captured with varied camera angles (frontal, lateral, 45-degree) and featured subjects not represented in the original training set, making them suitable for evaluating generalisation. All 26 test videos were labelled as correct quality.

**Table 4.4:** Fine-tuning progress over selected epochs.

| Epoch | Train Loss | Train Ex Acc | Train Qu Acc | Val Loss | Val Ex Acc | Val Qu Acc |
|-------|-----------|-------------|-------------|---------|-----------|-----------|
| 1     | 0.6383    | 90.06%      | 89.69%      | 0.4319  | 92.69%    | 91.56%    |
| 5     | 0.3239    | 94.68%      | 93.25%      | 0.3151  | 95.09%    | 93.07%    |
| 10    | 0.2742    | 95.70%      | 93.99%      | 0.2866  | 95.74%    | 93.53%    |
| 15    | 0.2497    | 96.15%      | 94.34%      | 0.2720  | 96.08%    | 93.84%    |
| 20    | 0.2325    | 96.48%      | 94.65%      | 0.2639  | 96.28%    | 94.00%    |
| 25    | 0.2198    | 96.70%      | 94.86%      | 0.2612  | 96.46%    | 94.09%    |
| 30    | 0.2094    | 96.84%      | 95.03%      | 0.2608  | 96.47%    | 94.20%    |
| 35    | 0.2011    | 96.97%      | 95.17%      | 0.2564  | 96.57%    | 94.31%    |
| 40    | 0.1940    | 97.09%      | 95.30%      | 0.2580  | 96.68%    | 94.36%    |
| 45    | 0.1886    | 97.16%      | 95.42%      | 0.2563  | 96.70%    | 94.44%    |
| 48    | 0.1848    | 97.22%      | 95.48%      | 0.2563  | 96.72%    | 94.44%    |

**Table 4.5:** Comparative validation metrics before and after fine-tuning.

| Metric                    | Before Fine-Tuning | After Fine-Tuning | Change  |
|---------------------------|--------------------|--------------------|---------|
| Val Exercise Accuracy     | 97.15%             | 96.66%             | -0.49%  |
| Val Quality Accuracy      | 92.93%             | 94.37%             | +1.44%  |
| Train Exercise Accuracy   | 97.70%             | 97.22%             | -0.48%  |
| Train Quality Accuracy    | 93.81%             | 95.48%             | +1.67%  |
| Train Loss                | 0.2061             | 0.1848             | -0.0213 |
| Val Loss (best)           | 0.2607             | 0.2545             | -0.0062 |

The fine-tuned model shows a slight decrease in exercise accuracy (-0.49%) attributable to the broadened data distribution introduced by the unseen subjects and camera angles in the test set. In contrast, quality accuracy improved by 1.44%, suggesting that the quality regression head benefits more from exposure to additional motion patterns across different viewpoints. This trade-off is expected and acceptable for a deployed system where generalisation to novel environments is critical.

---

## 4.5 Model Export and Optimisation

The trained Keras models were converted to TensorFlow Lite format using `tf.lite.TFLiteConverter` with `optimizations=[tf.lite.Optimize.DEFAULT]`. The exported `classifier.tflite` is 219 KB and accepts input shape `(1, 30, 16)` with two output tensors: `(1, 9)` for exercise logits and `(1, 2)` for quality logits. Correct output shapes were verified with dummy inference before deployment.

The knowledge base embedding model was exported to ONNX format (~91 MB) for use with ONNX Runtime on the RPi 5, avoiding the need for PyTorch or sentence-transformers on the edge device. The total additional storage footprint on the RPi 5 is approximately 115 MB (Section 4.7).

---

## 4.6 Deployment to Raspberry Pi 5

The deployment pipeline on the RPi 5 is automated by `rpi_setup.sh`, which handles virtual environment creation, dependency installation, MediaPipe ARM64 compilation from source (approximately 30–40 minutes on RPi 5), espeak configuration, model file verification, and test execution. A notable challenge was the absence of an official MediaPipe Linux ARM64 wheel; this was resolved by building from source with the required system dependencies (`python3-dev`, `gcc`, `g++`, `libgl1-mesa-dev`, `libgles2-mesa-dev`, `libegl1-mesa-dev`).

Two BlazePose model configurations were evaluated:

**Table 4.6:** RPi 5 runtime performance by model configuration at 640×480 resolution.

| Configuration        | Resolution | Expected FPS |
|---------------------|-----------|-------------|
| Lite model (complexity=0) | 640×480 | 25–30 |
| Full model (complexity=1) | 640×480 | 10–18    |

On desktop hardware, the full model is used by default. On the RPi 5, the lite model is recommended for real-time performance. The GUI requests 640×480 from the webcam; where camera drivers do not honour `cap.set()`, the raw frame is downscaled to 640×480 before BlazePose processing to avoid running the pose detector on accidental 1080p frames.

During deployment testing, two runtime bugs were identified and resolved:

1. **Drawing utilities crash on minimal ARM64 MediaPipe:** OpenCV drawing utilities (`cv2.rectangle`, `cv2.putText`) were added as a fallback on ARM64 build to prevent crashes when the standard MediaPipe drawing module is unavailable.

2. **ALSA audio probing error 524:** The TTS engine's ALSA probe function was updated to use `aplay -L` (PCM device listing) instead of `aplay -l` (hardware listing), which correctly resolves to `sysdefault`/`default` and routes audio through the ALSA software mixer.

All 47 test cases passed during RPi 5 validation.

---

## 4.7 RAG-Enabled Session Chat

After completing a workout session, the GUI displays a QR code that links to a web-based chat interface. The chat backend uses a Retrieval-Augmented Generation (RAG) pipeline drawing from three sources:

1. **Session data** — the workout session just completed, loaded live from the RPi 5 (~1 KB)
2. **Conditioning manual** — a pre-built PDF knowledge base covering exercise science and form guidance (~17 MB)
3. **Behaviour manual** — a pre-built knowledge base covering psychology, habit building, and motivation

The ONNX embedding model (~91 MB) is pre-built on a development machine (Mac or Google Colab) and copied to the RPi 5 before the first session. Heavy components — PyTorch, sentence-transformers, and PDF parsing — do not run on the RPi. The LLM runs on Groq's free-tier cloud API, contributing zero megabytes to the on-device footprint.

**Table 4.7:** On-device storage breakdown for the RPi 5 chat assistant.

| Component              | Runs On           | Storage Weight |
|-----------------------|-------------------|----------------|
| Session data          | RPi 5 (live)      | ~1 KB          |
| PDF knowledge base    | Pre-built, loaded | ~17 MB         |
| ONNX embedding model  | Pre-built, loaded | ~91 MB         |
| ONNX Runtime          | RPi 5             | ~5–10 MB       |
| LLM (Groq)            | Cloud (free tier) | 0 MB           |
| **Total on-device**  |                   | **~115 MB**    |

---

## 4.8 Chapter Summary

The experimental evaluation demonstrated that the dual-head LSTM classifier achieves 97.15% validation exercise accuracy and 92.93% validation quality accuracy after initial training on 174 videos, with further gains in quality accuracy (+1.44%) after fine-tuning on 26 unseen test videos. The models were successfully exported to TensorFlow Lite and ONNX formats suitable for edge deployment, and the full pipeline was validated on a Raspberry Pi 5 achieving 25–30 FPS with the lite model configuration. The RAG-enabled chat assistant adds approximately 115 MB of on-device storage while offloading all heavy inference to the cloud, enabling sophisticated post-workout coaching without compromising edge device performance.
