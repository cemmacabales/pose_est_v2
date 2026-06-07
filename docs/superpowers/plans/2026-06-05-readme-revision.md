# README Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Polish README.md for GitHub by adding an intro paragraph, badge strip, LIVE_STREAM callout, and reordering sections for a CV/ML engineering audience.

**Architecture:** Single file edit to README.md — no new files created. All existing content preserved; new content inserted and sections reordered.

**Tech Stack:** Markdown, shields.io badges

---

### Task 1: Add intro paragraph and badge strip

**Files:**
- Modify: `README.md` (title block, top of file)

- [ ] **Step 1: Replace the title block**

Current content at top of `README.md` (lines 1–2):
```markdown
# Pose Estimation + Exercise Classifier

## Pipeline Overview
```

Replace with:
```markdown
# Pose Estimation + Exercise Classifier

Real-time exercise classification running on a Raspberry Pi 5. BlazePose (MediaPipe) extracts 33-landmark pose keypoints from a webcam feed in async LIVE_STREAM mode; a dual-head LSTM classifier identifies 9 functional movement exercises and scores form quality. A post-session RAG chat uses ONNX embeddings + Groq LLM to deliver coaching feedback — no PyTorch on-device.

![Python](https://img.shields.io/badge/python-3.14%20%7C%203.11-blue) ![Platform](https://img.shields.io/badge/platform-RPi%205%20%7C%20macOS-lightgrey) ![MediaPipe](https://img.shields.io/badge/pose-MediaPipe%20BlazePose-green) ![TFLite](https://img.shields.io/badge/model-TFLite%20LSTM-orange)

```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add intro paragraph and badge strip to README"
```

---

### Task 2: Rename "Models" to "Architecture" and add LIVE_STREAM callout

**Files:**
- Modify: `README.md` (Models section)

- [ ] **Step 1: Replace the Models section header and add LIVE_STREAM note**

Current content (lines 14–20):
```markdown
## Models

- **MediaPipe BlazePose** -- pose estimation (33 landmarks, on-device GPU/Coral support)
- **classifier.tflite** -- dual-head LSTM: exercise (9 classes) + quality (correct/incorrect)
  - 174 training videos from 5 people
  - 451,638 windows after augmentation
  - 9 exercises: Deep Squat, Hurdle Step, Inline Lunge, Side Lunge, Sit to Stand,
    Standing Leg Raise, Shoulder Abduction, Shoulder Extension, Shoulder Scaption
```

Replace with:
```markdown
## Architecture

- **MediaPipe BlazePose** -- pose estimation (33 landmarks, on-device GPU/Coral support)
  - Runs in `LIVE_STREAM` mode: frames submitted via `submit_frame()`, results delivered via callback. Non-blocking; avoids the per-frame latency of `IMAGE` mode. See `pose_estimator.py`.
- **classifier.tflite** -- dual-head LSTM: exercise (9 classes) + quality (correct/incorrect)
  - 174 training videos from 5 people
  - 451,638 windows after augmentation
  - 9 exercises: Deep Squat, Hurdle Step, Inline Lunge, Side Lunge, Sit to Stand,
    Standing Leg Raise, Shoulder Abduction, Shoulder Extension, Shoulder Scaption
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rename Models to Architecture, add LIVE_STREAM callout"
```

---

### Task 3: Reorder sections (Architecture before Pipeline)

**Files:**
- Modify: `README.md` (section order)

- [ ] **Step 1: Move Architecture section above Pipeline Overview**

After Task 2, the file order is: Intro → Badges → Pipeline Overview → Architecture → ...

Reorder so Architecture comes before Pipeline Overview:

```markdown
# Pose Estimation + Exercise Classifier

<intro paragraph>

<badges>

## Architecture
...

## Pipeline Overview
...

## Quick Start (Mac -- Training & Export)
...
```

All remaining sections (RPi Deployment, Post-Setup, Session Chat/RAG, Environments, Joint Mapping, Training Log) stay in their current order after Quick Start.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: move Architecture section above Pipeline Overview"
```

---

### Task 4: Push to remote

**Files:** none

- [ ] **Step 1: Push branch to origin**

```bash
git push origin test
```

Expected output:
```
To https://github.com/cemmacabales/pose_est_v2.git
   <old>..<new>  test -> test
```
