# README Revision Design

**Date:** 2026-06-05
**Scope:** Option A — add intro, badges, section reorder, LIVE_STREAM callout
**Audience:** ML / computer vision engineers
**Goal:** Polish for GitHub showcase

---

## What Changes

### 1. Opening Intro (new)
Add a 3–4 sentence paragraph directly under the title, before any sections. Covers:
- Stack: BlazePose (MediaPipe) + dual-head LSTM classifier
- Inference: async LIVE_STREAM mode, 33-landmark keypoints from webcam
- Output: 9 exercise classes + form quality score
- Deployment: Raspberry Pi 5, post-session RAG chat via ONNX + Groq

### 2. Badge Strip (new)
Single line of shields.io badges under the intro:
- Python `3.14` / `3.11`
- Platform: RPi 5 + macOS
- MediaPipe
- TensorFlow Lite

### 3. Section Reordering
New order:
1. Intro
2. Badges
3. Architecture (renamed from "Models")
4. Pipeline Overview
5. Quick Start (Mac)
6. RPi 5 Deployment
7. Post-Setup
8. Session Chat / RAG
9. Environments
10. Joint Mapping
11. Training Log

Key change: Architecture moves before setup so readers understand the stack before hitting commands.

### 4. LIVE_STREAM Callout (new)
Short note added to Architecture section:
> BlazePose runs in `LIVE_STREAM` mode — frames submitted via `submit_frame()`, results delivered via callback. Non-blocking; avoids the latency of `IMAGE` mode. See `pose_estimator.py`.

## What Stays the Same
All existing commands, tables, FPS benchmarks, joint mapping, environment table, and training log content are preserved verbatim. No technical content is removed or altered.

## Out of Scope
- Screenshots or demo GIFs
- Contribution guide
- License section
- Eval / data collection pipeline docs
