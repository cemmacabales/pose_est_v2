# Rep Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Count exercise repetitions in real-time, display them in the sidebar, persist them to the session JSON, and surface them in the RAG chatbot.

**Architecture:** A new `RepCounter` class uses a per-exercise FSM (Finite State Machine) on EMA-smoothed angle signals. The GUI reads the latest angles and stable exercise name each processing cycle to drive the counter. Rep counts flow into `SessionLogger.end_session()` and then into the LLM system prompt.

**Tech Stack:** Python, NumPy, tkinter, existing `joint_angles.py` angle features, existing `SessionLogger`, Groq LLM via `session_chat/llm.py`.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `rep_counter.py` | **Create** | `EXERCISE_REP_CONFIG`, `_ExerciseState`, `RepCounter` class |
| `tests/test_rep_counter.py` | **Create** | Unit tests for `RepCounter` |
| `session_logger.py` | **Modify** | `end_session(rep_counts=None)` adds `"reps"` to each exercise entry |
| `tests/test_session_logger.py` | **Modify** | Add tests for `rep_counts` parameter |
| `session_chat/llm.py` | **Modify** | Exercise breakdown line includes `Reps: N` |
| `tests/test_llm.py` | **Modify** | Add test that prompt includes rep count |
| `gui.py` | **Modify** | Import RepCounter, add REPS sidebar section, wire into update loop and `_end_session` |
| `tests/test_gui_integration.py` | **Modify** | Add test for `rep_counts` kwarg passed to `logger.end_session` |

---

## Task 1: Create feature branch

- [ ] **Step 1: Create and switch to feature branch from `test`**

```bash
git checkout test
git checkout -b feat/rep-detection
```

Expected: `Switched to a new branch 'feat/rep-detection'`

---

## Task 2: `rep_counter.py` — core logic (TDD)

**Files:**
- Create: `rep_counter.py`
- Create: `tests/test_rep_counter.py`

### 2a — Write failing tests

- [ ] **Step 1: Create `tests/test_rep_counter.py`**

```python
import numpy as np
import pytest
from rep_counter import RepCounter, EXERCISE_REP_CONFIG


def _angles(overrides: dict) -> np.ndarray:
    a = np.zeros(16, dtype=np.float32)
    for idx, val in overrides.items():
        a[idx] = val
    return a


def _drive(counter, ex, feature_vals, n=25):
    """Push n frames with the given feature values."""
    a = _angles({f: v for f, v in zip(EXERCISE_REP_CONFIG[ex]["features"], feature_vals)})
    for _ in range(n):
        counter.update(ex, a)


def _rep_low(counter, ex):
    """Simulate one full rep for a 'low'-direction exercise."""
    cfg = EXERCISE_REP_CONFIG[ex]
    neutral_val = cfg["exit"] + 0.1
    peak_val = cfg["enter"] - 0.1
    n_feats = len(cfg["features"])
    _drive(counter, ex, [neutral_val] * n_feats, n=10)   # start neutral
    _drive(counter, ex, [peak_val]   * n_feats, n=25)   # descend
    _drive(counter, ex, [neutral_val] * n_feats, n=25)  # return


def _rep_high(counter, ex):
    """Simulate one full rep for a 'high'-direction exercise."""
    cfg = EXERCISE_REP_CONFIG[ex]
    neutral_val = cfg["exit"] - 0.1
    peak_val = cfg["enter"] + 0.1
    n_feats = len(cfg["features"])
    _drive(counter, ex, [neutral_val] * n_feats, n=10)  # start neutral
    _drive(counter, ex, [peak_val]   * n_feats, n=25)  # ascend
    _drive(counter, ex, [neutral_val] * n_feats, n=25) # return


def test_deep_squat_one_rep():
    counter = RepCounter()
    _rep_low(counter, "Deep Squat")
    assert counter.get_counts()["Deep Squat"] == 1


def test_deep_squat_two_reps():
    counter = RepCounter()
    _rep_low(counter, "Deep Squat")
    _rep_low(counter, "Deep Squat")
    assert counter.get_counts()["Deep Squat"] == 2


def test_inline_lunge_one_rep():
    counter = RepCounter()
    _rep_low(counter, "Inline Lunge")
    assert counter.get_counts()["Inline Lunge"] == 1


def test_shoulder_abduction_one_rep():
    counter = RepCounter()
    _rep_high(counter, "Shoulder Abduction")
    assert counter.get_counts()["Shoulder Abduction"] == 1


def test_sit_to_stand_one_rep():
    counter = RepCounter()
    _rep_high(counter, "Sit to Stand")
    assert counter.get_counts()["Sit to Stand"] == 1


def test_staying_at_neutral_counts_zero():
    counter = RepCounter()
    cfg = EXERCISE_REP_CONFIG["Deep Squat"]
    neutral_val = cfg["exit"] + 0.1
    _drive(counter, "Deep Squat", [neutral_val] * len(cfg["features"]), n=50)
    assert counter.get_counts().get("Deep Squat", 0) == 0


def test_partial_rep_no_return_counts_zero():
    counter = RepCounter()
    cfg = EXERCISE_REP_CONFIG["Deep Squat"]
    peak_val = cfg["enter"] - 0.1
    _drive(counter, "Deep Squat", [peak_val] * len(cfg["features"]), n=50)
    assert counter.get_counts().get("Deep Squat", 0) == 0


def test_unknown_exercise_returns_zero():
    counter = RepCounter()
    result = counter.update("Not A Real Exercise", np.zeros(16, dtype=np.float32))
    assert result == 0
    assert counter.get_counts() == {}


def test_get_counts_tracks_multiple_exercises_independently():
    counter = RepCounter()
    _rep_low(counter, "Deep Squat")
    _rep_low(counter, "Deep Squat")
    _rep_low(counter, "Inline Lunge")
    counts = counter.get_counts()
    assert counts["Deep Squat"] == 2
    assert counts["Inline Lunge"] == 1


def test_get_counts_empty_for_new_counter():
    counter = RepCounter()
    assert counter.get_counts() == {}


def test_all_nine_exercises_have_config():
    from joint_map import EXERCISE_NAMES
    for name in EXERCISE_NAMES.values():
        assert name in EXERCISE_REP_CONFIG, f"Missing config for: {name}"


def test_update_returns_current_count():
    counter = RepCounter()
    _rep_low(counter, "Deep Squat")
    result = counter.update("Deep Squat", np.zeros(16, dtype=np.float32))
    assert isinstance(result, int)
    assert result >= 0
```

- [ ] **Step 2: Run tests — verify they fail with ImportError**

```bash
cd /home/ainnovation/pose_est_v2 && python -m pytest tests/test_rep_counter.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'rep_counter'`

### 2b — Implement `rep_counter.py`

- [ ] **Step 3: Create `rep_counter.py`**

```python
import numpy as np

EXERCISE_REP_CONFIG = {
    "Deep Squat":          {"features": [2, 3],  "direction": "low",  "enter": 0.5,  "exit": 0.8},
    "Hurdle Step":         {"features": [2, 3],  "direction": "low",  "enter": 0.5,  "exit": 0.8},
    "Inline Lunge":        {"features": [2, 3],  "direction": "low",  "enter": 0.5,  "exit": 0.8},
    "Side Lunge":          {"features": [2, 3],  "direction": "low",  "enter": 0.5,  "exit": 0.8},
    "Sit to Stand":        {"features": [6, 7],  "direction": "high", "enter": -0.2, "exit": -0.5},
    "Standing Leg Raise":  {"features": [6, 7],  "direction": "high", "enter": -0.4, "exit": -0.6},
    "Shoulder Abduction":  {"features": [4, 5],  "direction": "high", "enter": -0.3, "exit": -0.6},
    "Shoulder Extension":  {"features": [4, 5],  "direction": "high", "enter": -0.3, "exit": -0.6},
    "Shoulder Scaption":   {"features": [4, 5],  "direction": "high", "enter": -0.3, "exit": -0.6},
}

_EMA_ALPHA = 0.3


class _ExerciseState:
    def __init__(self, config: dict):
        self._cfg = config
        self._smoothed: float | None = None
        self._state = "neutral"
        self.count = 0

    def update(self, angles: np.ndarray) -> None:
        raw = float(np.mean(angles[self._cfg["features"]]))
        if self._smoothed is None:
            self._smoothed = raw
        else:
            self._smoothed = _EMA_ALPHA * raw + (1 - _EMA_ALPHA) * self._smoothed

        direction = self._cfg["direction"]
        enter = self._cfg["enter"]
        exit_ = self._cfg["exit"]

        if direction == "low":
            if self._state == "neutral" and self._smoothed < enter:
                self._state = "peaked"
            elif self._state == "peaked" and self._smoothed > exit_:
                self._state = "neutral"
                self.count += 1
        else:
            if self._state == "neutral" and self._smoothed > enter:
                self._state = "peaked"
            elif self._state == "peaked" and self._smoothed < exit_:
                self._state = "neutral"
                self.count += 1


class RepCounter:
    def __init__(self):
        self._states: dict[str, _ExerciseState] = {}

    def _get_state(self, exercise_name: str) -> _ExerciseState | None:
        if exercise_name not in self._states:
            cfg = EXERCISE_REP_CONFIG.get(exercise_name)
            if cfg is None:
                return None
            self._states[exercise_name] = _ExerciseState(cfg)
        return self._states[exercise_name]

    def update(self, exercise_name: str, angles: np.ndarray) -> int:
        state = self._get_state(exercise_name)
        if state is None:
            return 0
        state.update(angles)
        return state.count

    def get_counts(self) -> dict[str, int]:
        return {name: s.count for name, s in self._states.items()}
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
cd /home/ainnovation/pose_est_v2 && python -m pytest tests/test_rep_counter.py -v
```

Expected: all 12 tests pass.

- [ ] **Step 5: Commit**

```bash
git add rep_counter.py tests/test_rep_counter.py
git commit -m "feat: add RepCounter with per-exercise FSM angle threshold detection"
```

---

## Task 3: `session_logger.py` — persist rep counts (TDD)

**Files:**
- Modify: `session_logger.py`
- Modify: `tests/test_session_logger.py`

### 3a — Write failing tests

- [ ] **Step 1: Add these tests to `tests/test_session_logger.py`**

Append after the last existing test:

```python
def test_end_session_with_rep_counts_adds_reps_field(tmp_path):
    logger = SessionLogger(log_dir=str(tmp_path))
    for _ in range(10):
        logger.log_frame(exercise_idx=0, quality=1, confidence=0.9)
    result = logger.end_session(rep_counts={"Deep Squat": 5})
    assert result["exercises"][0]["reps"] == 5


def test_end_session_rep_counts_defaults_to_zero_when_missing(tmp_path):
    logger = SessionLogger(log_dir=str(tmp_path))
    for _ in range(10):
        logger.log_frame(exercise_idx=0, quality=1, confidence=0.9)
    result = logger.end_session(rep_counts={})
    assert result["exercises"][0]["reps"] == 0


def test_end_session_no_rep_counts_arg_defaults_to_zero(tmp_path):
    logger = SessionLogger(log_dir=str(tmp_path))
    for _ in range(10):
        logger.log_frame(exercise_idx=0, quality=1, confidence=0.9)
    result = logger.end_session()
    assert result["exercises"][0]["reps"] == 0


def test_end_session_rep_counts_written_to_json(tmp_path):
    import json
    logger = SessionLogger(log_dir=str(tmp_path))
    for _ in range(10):
        logger.log_frame(exercise_idx=0, quality=1, confidence=0.9)
    result = logger.end_session(rep_counts={"Deep Squat": 3})
    with open(result["log_file"]) as f:
        on_disk = json.load(f)
    assert on_disk["exercises"][0]["reps"] == 3
```

- [ ] **Step 2: Run new tests — verify they fail**

```bash
cd /home/ainnovation/pose_est_v2 && python -m pytest tests/test_session_logger.py::test_end_session_with_rep_counts_adds_reps_field tests/test_session_logger.py::test_end_session_rep_counts_defaults_to_zero_when_missing tests/test_session_logger.py::test_end_session_no_rep_counts_arg_defaults_to_zero tests/test_session_logger.py::test_end_session_rep_counts_written_to_json -v
```

Expected: 4 failures — `end_session() got an unexpected keyword argument 'rep_counts'` or `KeyError: 'reps'`.

### 3b — Update `session_logger.py`

- [ ] **Step 3: Change `end_session` signature and add `reps` to each exercise entry**

In `session_logger.py`, change line 27:

```python
    def end_session(self) -> dict:
```

to:

```python
    def end_session(self, rep_counts: dict | None = None) -> dict:
```

Then, inside the `for i, segment in enumerate(self.segments):` loop, after `avg_confidence = ...` (around line 41), add `reps` to the `exercises.append({...})` dict. The full updated `exercises.append` call becomes:

```python
            exercises.append({
                "name": segment["name"],
                "segment_start": segment["segment_start"].strftime("%H:%M:%S"),
                "duration_seconds": duration_secs,
                "frames_correct": frames_correct,
                "frames_incorrect": frames_incorrect,
                "form_score_pct": form_score_pct,
                "avg_confidence": avg_confidence,
                "reps": (rep_counts or {}).get(segment["name"], 0),
            })
```

- [ ] **Step 4: Run all session logger tests — verify they pass**

```bash
cd /home/ainnovation/pose_est_v2 && python -m pytest tests/test_session_logger.py -v
```

Expected: all tests pass (including original 8 + new 4 = 12 total).

- [ ] **Step 5: Commit**

```bash
git add session_logger.py tests/test_session_logger.py
git commit -m "feat: persist rep counts per exercise in session JSON"
```

---

## Task 4: `session_chat/llm.py` — reps in system prompt (TDD)

**Files:**
- Modify: `session_chat/llm.py`
- Modify: `tests/test_llm.py`

### 4a — Write failing test

- [ ] **Step 1: Add this test to `tests/test_llm.py`**

Add `"reps": 5` to the first exercise in `FAKE_SESSION` at the top of the file, changing:

```python
        {
            "name": "Deep Squat",
            "duration_seconds": 300,
            "frames_correct": 240,
            "frames_incorrect": 60,
            "form_score_pct": 80,
            "avg_confidence": 0.88,
        },
```

to:

```python
        {
            "name": "Deep Squat",
            "duration_seconds": 300,
            "frames_correct": 240,
            "frames_incorrect": 60,
            "form_score_pct": 80,
            "avg_confidence": 0.88,
            "reps": 5,
        },
```

Then append this test at the end of the file:

```python
def test_build_system_prompt_includes_reps():
    prompt = build_system_prompt(FAKE_SESSION)
    assert "Reps: 5" in prompt
```

- [ ] **Step 2: Run the new test — verify it fails**

```bash
cd /home/ainnovation/pose_est_v2 && python -m pytest tests/test_llm.py::test_build_system_prompt_includes_reps -v
```

Expected: FAIL — `"Reps: 5"` not found in prompt.

### 4b — Update `session_chat/llm.py`

- [ ] **Step 3: Update the exercise breakdown line in `build_system_prompt`**

In `session_chat/llm.py`, find this line (around line 68):

```python
        lines.append(f"- {name}: {dur}s | Form: {form}% | Confidence: {conf}")
```

Replace it with:

```python
        reps = ex.get("reps", 0)
        lines.append(f"- {name}: {dur}s | Form: {form}% | Reps: {reps} | Confidence: {conf}")
```

- [ ] **Step 4: Run all LLM tests — verify they pass**

```bash
cd /home/ainnovation/pose_est_v2 && python -m pytest tests/test_llm.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add session_chat/llm.py tests/test_llm.py
git commit -m "feat: include rep counts in chatbot system prompt exercise breakdown"
```

---

## Task 5: `gui.py` — sidebar REPS section and wiring

**Files:**
- Modify: `gui.py`
- Modify: `tests/test_gui_integration.py`

### 5a — Write failing integration test

- [ ] **Step 1: Add test to `tests/test_gui_integration.py`**

Append after the last existing test:

```python
def test_end_session_passes_rep_counts_to_logger(gui_mod):
    gui_mod._end_session()
    call_kwargs = gui_mod.logger.end_session.call_args.kwargs
    assert "rep_counts" in call_kwargs
    assert isinstance(call_kwargs["rep_counts"], dict)
```

- [ ] **Step 2: Run the new test — verify it fails**

```bash
cd /home/ainnovation/pose_est_v2 && python -m pytest tests/test_gui_integration.py::test_end_session_passes_rep_counts_to_logger -v
```

Expected: FAIL — `rep_counts` not in call kwargs.

### 5b — Update `gui.py`

Make the following changes to `gui.py` **in order**:

- [ ] **Step 3: Add import and module-level state**

After the existing import line:
```python
from session_logger import SessionLogger
```

Add:
```python
from rep_counter import RepCounter
```

After the existing module-level `_session_ended = False` line (around line 97), add:

```python
rep_counter = RepCounter()
_latest_angles = None
```

- [ ] **Step 4: Add REPS sidebar section**

Find the existing sidebar layout block for QUALITY (around line 346):

```python
quality_badge.pack(fill=tk.X)

add_spacer(sidebar, 16)

confidence_header = tk.Label(
```

Change to:

```python
quality_badge.pack(fill=tk.X)

add_spacer(sidebar, 16)

reps_header = tk.Label(
    sidebar, text="REPS", font=("Helvetica", 11),
    fg="#888888", bg="#212121", anchor="w"
)
reps_header.pack(fill=tk.X)

reps_label = tk.Label(
    sidebar, text="—", font=("Helvetica", 22, "bold"),
    fg="#FFFFFF", bg="#212121", anchor="w"
)
reps_label.pack(fill=tk.X)

add_spacer(sidebar, 16)

confidence_header = tk.Label(
```

- [ ] **Step 5: Update `update()` — capture latest angles and drive rep counter**

In `update()`, change the `global` declaration from:

```python
    global frame_counter, _idle_count
```

to:

```python
    global frame_counter, _idle_count, _latest_angles
```

Inside the `if len(frame_buffer) == 30 and frame_counter % 5 == 0:` block, after:

```python
        angles = batch_keypoints_to_angles(window)
        motion = compute_motion(angles)
```

Add:

```python
        _latest_angles = angles[-1].copy()
```

In the idle branch (inside `if _idle_count >= IDLE_CONFIRM_COUNT:`), after:

```python
            quality_badge.config(text="WAITING", fg="#888888", bg="#333333")
            draw_confidence_bar(0.0)
            confidence_label.config(text="0%")
```

Add:

```python
            reps_label.config(text="—")
```

In the non-idle branch, inside `if result is not None and len(prediction_buffer) >= 5:`, after the `logger.log_frame(...)` call, add:

```python
            if _latest_angles is not None:
                current_reps = rep_counter.update(exercise_name, _latest_angles)
                reps_label.config(text=str(current_reps))
```

- [ ] **Step 6: Pass rep counts to `logger.end_session` in `_end_session()`**

In `_end_session()`, change:

```python
    session_data = logger.end_session()
```

to:

```python
    session_data = logger.end_session(rep_counts=rep_counter.get_counts())
```

- [ ] **Step 7: Run all tests — verify full suite passes**

```bash
cd /home/ainnovation/pose_est_v2 && python -m pytest tests/ -v
```

Expected: all tests pass. Pay attention to any failures in `test_gui_integration.py` — the fixture mocks tkinter labels, so `reps_label.config()` calls on MagicMock are silent no-ops.

- [ ] **Step 8: Commit**

```bash
git add gui.py tests/test_gui_integration.py
git commit -m "feat: add REPS sidebar section and wire RepCounter into GUI and session end"
```

---

## Task 6: Push PR

- [ ] **Step 1: Verify full test suite one final time**

```bash
cd /home/ainnovation/pose_est_v2 && python -m pytest tests/ -v
```

Expected: all tests pass, 0 failures.

- [ ] **Step 2: Push branch**

```bash
git push -u origin feat/rep-detection
```

- [ ] **Step 3: Open PR targeting `test`**

```bash
gh pr create \
  --base test \
  --title "feat: add rep detection to GUI, session log, and chatbot" \
  --body "$(cat <<'EOF'
## Summary
- Adds `RepCounter` class with per-exercise FSM using EMA-smoothed joint angle signals
- REPS section added to sidebar, showing live rep count for the active exercise
- Rep counts persisted to session JSON (`reps` field on each exercise entry)
- Chatbot system prompt now includes rep counts so users can ask \"how many squats did I do?\"

## How it works
Each exercise is mapped to its primary angle features (e.g., knee cosine for squats, arm elevation cosine for shoulder raises). A state machine transitions neutral → peaked → neutral, counting one rep per round-trip. EMA smoothing (α=0.3) filters noise.

## Test plan
- [ ] `pytest tests/test_rep_counter.py` — 12 unit tests for FSM logic
- [ ] `pytest tests/test_session_logger.py` — 4 new tests for `rep_counts` parameter
- [ ] `pytest tests/test_llm.py` — 1 new test for reps in system prompt
- [ ] `pytest tests/test_gui_integration.py` — 1 new test for kwarg wiring
- [ ] Manual: run `python gui.py`, do 3 squats, verify REPS shows 3

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Return the PR URL.
