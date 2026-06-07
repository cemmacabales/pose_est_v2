# Idle Detection via Motion Variance Threshold

**Date:** 2026-06-07  
**Status:** Approved

## Problem

The classifier has 9 exercise classes and no "idle" class. It always forces a prediction from those 9 classes — even when the user is standing still or pausing between reps. This causes phantom exercise detections during stillness and micro-pauses.

## Goal

Suppress the classifier output and show "Idle" whenever the user is not actively performing an exercise. This must cover both general stillness (standing around) and brief micro-pauses between repetitions.

## Approach: Motion Variance Threshold

Before pushing a window to the classifier, measure how much the joint angles actually changed across the 30-frame window. If the motion is below a threshold, the person is idle — skip the classifier and update the UI directly.

**No model retraining required. Changes are isolated to `gui.py`.**

## Architecture

```
frame_buffer (30 frames of joint positions)
       ↓
batch_keypoints_to_angles()  →  angles  (30 × 16)
       ↓
motion = mean( std(angles, axis=0) )   # scalar: mean per-feature std deviation
       ↓
motion < IDLE_THRESHOLD?
   YES → increment _idle_count; if _idle_count >= IDLE_CONFIRM_COUNT → show "Idle"
   NO  → reset _idle_count to 0; push to classifier as normal
```

## Constants

| Name | Value | Rationale |
|------|-------|-----------|
| `IDLE_THRESHOLD` | `0.03` | Idle poses score ~0.005–0.02 mean std; active exercises ~0.05–0.20. Tunable. |
| `IDLE_CONFIRM_COUNT` | `2` | Require 2 consecutive idle windows (~10 frames at 5-frame stride) before switching to idle. Prevents flicker. |

Both constants are defined at module level in `gui.py` and easy to tune without touching logic.

## UI Behavior

| State | Exercise label | Quality badge | Confidence | TTS | Logger |
|-------|---------------|---------------|------------|-----|--------|
| Active | Predicted exercise | CORRECT / INCORRECT | Filled bar | Called | `log_frame()` called |
| Idle | `"Idle"` | `"WAITING"` (grey) | Cleared | Not called | Not called |

## State Variables Added to `gui.py`

- `_idle_count: int` — counts consecutive windows where `motion < IDLE_THRESHOLD`. Reset to 0 on any active window. Initialized to 0.

## Buffer Flushing on Idle Entry

When `_idle_count` reaches `IDLE_CONFIRM_COUNT` and the system transitions to idle, both `prediction_buffer` and any pending item in `_classifier_output_queue` are drained. This prevents stale pre-idle predictions from immediately re-appearing the moment motion resumes.

## Files Changed

- `gui.py` — only file modified. No changes to model files, training pipeline, or any other module.

## Tuning Guide

If idle is triggered too aggressively during slow exercises (e.g., slow sit-to-stand), **lower** `IDLE_THRESHOLD` toward `0.02`.  
If idle is not triggered during actual stillness, **raise** `IDLE_THRESHOLD` toward `0.05`.  
If switching between idle/active flickers, **raise** `IDLE_CONFIRM_COUNT` to `3` or `4`.

## Out of Scope

- Retraining the classifier with an idle class
- Changes to `pose_estimator.py`, `joint_angles.py`, `joint_map.py`
- Per-exercise threshold tuning
