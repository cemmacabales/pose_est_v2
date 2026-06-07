# Rep Detection — Design Spec
**Date:** 2026-06-07  
**Branch target:** test

---

## Overview

Add real-time repetition counting to the pose estimation GUI. Each of the 9 FMS-style exercises gets a per-exercise rep total tracked using a finite-state-machine on key joint-angle signals. Rep counts appear in the sidebar, are saved to the session JSON, and are available to the RAG chatbot.

---

## Architecture

### New file: `rep_counter.py`

**`EXERCISE_REP_CONFIG`** — config table mapping each exercise name to:

| Field | Type | Purpose |
|---|---|---|
| `features` | list[int] | Indices into the 16-feature angle vector to average as primary signal |
| `direction` | `"low"` or `"high"` | `"low"` = signal drops at movement peak (squats, lunges); `"high"` = signal rises (shoulder raises, hip flexion) |
| `enter_threshold` | float | Crossing this threshold (in the `direction`) starts the "peaked" state |
| `exit_threshold` | float | Returning past this threshold (opposite direction, hysteresis) completes a rep |

Initial config per exercise:

| Exercise | Features | Direction | Enter | Exit |
|---|---|---|---|---|
| Deep Squat | [2, 3] knee cos | low | 0.5 | 0.8 |
| Hurdle Step | [2, 3] knee cos | low | 0.5 | 0.8 |
| Inline Lunge | [2, 3] knee cos | low | 0.5 | 0.8 |
| Side Lunge | [2, 3] knee cos | low | 0.5 | 0.8 |
| Sit to Stand | [6, 7] hip flexion cos | high | -0.2 | -0.5 |
| Standing Leg Raise | [6, 7] hip flexion cos | high | -0.4 | -0.6 |
| Shoulder Abduction | [4, 5] arm elevation cos | high | -0.3 | -0.6 |
| Shoulder Extension | [4, 5] arm elevation cos | high | -0.3 | -0.6 |
| Shoulder Scaption | [4, 5] arm elevation cos | high | -0.3 | -0.6 |

*Thresholds are initial estimates from cosine geometry; tune after observing real data.*

**`_ExerciseState`** — internal per-exercise tracker:
- `smoothed`: EMA-filtered signal (α = 0.3 — balances responsiveness and noise rejection)
- `state`: `"neutral"` | `"peaked"`
- `count`: int

FSM logic (direction `"low"`):
```
neutral + signal < enter  →  peaked
peaked  + signal > exit   →  neutral, count += 1
```
FSM logic (direction `"high"`):
```
neutral + signal > enter  →  peaked
peaked  + signal < exit   →  neutral, count += 1
```

**`RepCounter`** — public class:
- `update(exercise_name: str, angles: np.ndarray) -> int` — feeds latest frame angles, returns current count for this exercise
- `get_counts() -> dict[str, int]` — returns full per-exercise totals

---

### Changes: `gui.py`

- Import and instantiate `RepCounter` at startup
- In `update()`, after `angles` is computed each processing cycle, call `rep_counter.update(ex_name, angles[-1])`
- **New REPS sidebar section** placed between QUALITY and CONFIDENCE:
  ```
  REPS
  3
  ```
  Updates each processing cycle showing the active exercise's rep count.
- When idle: show `—` for reps.
- Pass `rep_counter.get_counts()` into `_end_session()` and forward to `logger.end_session()`.

---

### Changes: `session_logger.py`

`end_session(rep_counts: dict = None)`:
- Each exercise entry gains `"reps": int` (looked up by exercise name, defaulting to 0).

---

### Changes: `session_chat/llm.py`

Exercise breakdown line in system prompt becomes:
```
- {name}: {dur}s | Form: {form}% | Reps: {reps} | Confidence: {conf}
```
The chatbot can now answer "how many squats did I do?" accurately.

---

## Data flow

```
Frame → PoseEstimator → keypoints → batch_keypoints_to_angles → (30, 16)
                                                                      ↓
                                              angles[-1] (16,) → RepCounter.update()
                                                                      ↓
                                                            GUI sidebar REPS label
                                                            
_end_session() → rep_counter.get_counts() → logger.end_session(rep_counts)
                                                      ↓
                                              session JSON  →  ChatSession system prompt
```

---

## Out of scope

- Per-rep quality tracking (only aggregate quality is tracked)
- Audio announcement of rep counts
- Rep count during idle state
