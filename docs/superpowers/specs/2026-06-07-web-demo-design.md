# Web Demo Design — Pose Estimation + Exercise Classifier

**Date:** 2026-06-07  
**Goal:** Deploy the exercise classification model as a browser-based demo on Netlify so colleagues can test it from any device with a webcam.

---

## Overview

A single-page static web app that replicates the RPi GUI pipeline entirely in the browser using WebAssembly. No server required. Netlify serves static files; all inference runs client-side.

The existing Python/Tkinter app on the Raspberry Pi is **not modified** — this is a parallel deployment.

---

## Architecture

```
getUserMedia (webcam)
  └─ <video> element → drawImage() to offscreen <canvas>
       └─ MediaPipe PoseLandmarker (WASM) → 33 landmarks
            └─ extract 12 joints (BLAZEPOSE_MAPPED_INDICES) → hip-center
                 └─ 30-frame ring buffer
                      └─ every 5 frames:
                           ├─ batch_keypoints_to_angles() → (30, 16) tensor
                           ├─ compute_motion() → idle check (threshold 0.03)
                           └─ [not idle] TFLite LSTM → exercise (9-class) + quality (2-class)
                                └─ 10-frame prediction buffer → majority vote
                                     └─ UI update: exercise name, quality badge, confidence bar
            └─ per-frame: keypoints_to_angles() → RepCounter EMA state machine → rep count
  └─ canvas overlay: draw 33-landmark skeleton + connections
```

---

## File Structure

```
web/
  index.html          ← UI shell (dark theme, same palette as RPi app)
  app.js              ← main orchestrator: camera, MediaPipe, TFLite, UI updates
  joint_angles.js     ← direct JS port of joint_angles.py (keypoints_to_angles,
                         batch_keypoints_to_angles, compute_motion)
  rep_counter.js      ← direct JS port of rep_counter.py (EMA + threshold state machine)
  models/
    classifier.tflite ← copied from models/classifier.tflite (214 KB)
netlify.toml          ← publish = "web", no build command
```

---

## Dependencies (CDN — no build step)

| Package | Source | Purpose |
|---------|--------|---------|
| `@mediapipe/tasks-vision` | jsDelivr CDN | BlazePose landmark detection (WASM) |
| `@tensorflow/tfjs` | CDN | TF.js runtime |
| `@tensorflow/tfjs-tflite` | CDN | Load + run `.tflite` model in browser |

MediaPipe's pose model (`pose_landmarker_lite.task`, 5.6 MB) loads from Google's CDN at runtime — no need to bundle it, CORS is open on Google's storage bucket.

`classifier.tflite` (214 KB) is served from Netlify alongside the HTML.

---

## Inference Pipeline Details

### Landmark extraction
- `PoseLandmarker` configured for `VIDEO` running mode (processes canvas frames)
- Indices `[11,12,13,14,15,16,23,24,25,26,27,28]` extracted from 33-landmark result
- Hip-centered: subtract `(landmarks[23].x + landmarks[24].x) / 2` and y equivalent

### Angle computation (`joint_angles.js`)
Ports `keypoints_to_angles()` and `batch_keypoints_to_angles()` from `joint_angles.py`:
- 16 features: elbow angles (cos), knee angles (cos), arm/leg elevation (cos vs torso), lateral and vertical unit vector components
- `compute_motion()`: mean per-feature std across 30-frame window

### Classifier input/output
- Input: `Float32Array` shaped `[1, 30, 16]`
- Output head 1: `[1, 9]` → exercise class (argmax)
- Output head 2: `[1, 2]` → quality class (argmax)
- Prediction buffer: 10 frames, majority vote on exercise index, mean confidence

### Idle detection
- `motion < 0.03` for 2 consecutive windows → idle state
- Clears prediction buffer, resets UI to "Idle" / "WAITING"

### Rep counting (`rep_counter.js`)
Ports `RepCounter` and `_ExerciseState` from `rep_counter.py`:
- EMA smoothing (α = 0.3) on selected angle features
- Per-exercise enter/exit thresholds from `EXERCISE_REP_CONFIG`
- State machine: `neutral → peaked → neutral` = 1 rep
- Driven per-frame from `keypoints_to_angles()` output

---

## UI

**Layout:** Two-column, full-viewport.

| Left column | Right sidebar |
|-------------|---------------|
| `<video>` element (hidden) | EXERCISE — name in large bold white text |
| `<canvas>` overlay (skeleton + video) | QUALITY — green "CORRECT" / red "INCORRECT" badge |
| | REPS — count |
| | CONFIDENCE — colored progress bar (green ≥75%, yellow ≥50%, red <50%) |
| | KEYPOINTS — `N / 33 detected` |

**Color palette** (matches RPi app):
- Background: `#1a1a1a` / `#212121`
- Text: `#FFFFFF` / `#888888` (labels)
- Correct: `#69F0AE` on `#1B5E20`
- Incorrect: `#FF8A80` on `#B71C1C`
- Confidence bar: `#69F0AE` / `#FFD740` / `#FF6E40`

**Camera permission:** browser `getUserMedia` prompts on load; error state shown if denied.

---

## Netlify Deployment

```toml
# netlify.toml
[build]
  publish = "web"
  # no build command — static files only

[[headers]]
  for = "/*"
  [headers.values]
    Cross-Origin-Opener-Policy = "same-origin"
    Cross-Origin-Embedder-Policy = "require-corp"
```

The COOP/COEP headers are required for `SharedArrayBuffer` which MediaPipe's WASM threading uses.

**Deploy flow:**
1. Push to `main` branch on GitHub
2. Netlify auto-deploys `web/` directory
3. Colleagues open the Netlify URL — browser prompts for webcam → real-time demo

---

## Out of Scope

- Post-session RAG chat (requires Groq API key, backend)
- TTS announcements
- Session logging
- Camera selector dropdown (browser will use default or allow OS-level selection)

---

## Open Questions / Risks

| Risk | Mitigation |
|------|------------|
| TFLite LSTM ops unsupported in WASM backend | Test early; fallback: convert to TFJS SavedModel format with `tensorflowjs_converter` |
| MediaPipe WASM requires HTTPS | Netlify provides HTTPS by default |
| COOP/COEP headers break third-party CDN loads | Pin CDN URLs to same-origin if needed; or use `crossOriginIsolated` check |
| Low FPS on mobile | Lite pose model + throttle classifier to every 10 frames on mobile |
