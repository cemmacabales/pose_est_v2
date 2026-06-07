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

    def _get_state(self, exercise_name: str) -> "_ExerciseState | None":
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

    def get_counts(self) -> dict:
        return {name: s.count for name, s in self._states.items()}
