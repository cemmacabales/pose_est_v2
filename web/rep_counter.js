// rep_counter.js — port of rep_counter.py

const EMA_ALPHA = 0.3;

const EXERCISE_REP_CONFIG = {
  'Deep Squat':         { features: [2, 3], direction: 'low',  enter: 0.5,  exit: 0.8  },
  'Hurdle Step':        { features: [2, 3], direction: 'low',  enter: 0.5,  exit: 0.8  },
  'Inline Lunge':       { features: [2, 3], direction: 'low',  enter: 0.5,  exit: 0.8  },
  'Side Lunge':         { features: [2, 3], direction: 'low',  enter: 0.5,  exit: 0.8  },
  'Sit to Stand':       { features: [6, 7], direction: 'high', enter: -0.2, exit: -0.5 },
  'Standing Leg Raise': { features: [6, 7], direction: 'high', enter: -0.4, exit: -0.6 },
  'Shoulder Abduction': { features: [4, 5], direction: 'high', enter: -0.3, exit: -0.6 },
  'Shoulder Extension': { features: [4, 5], direction: 'high', enter: -0.3, exit: -0.6 },
  'Shoulder Scaption':  { features: [4, 5], direction: 'high', enter: -0.3, exit: -0.6 },
};

class ExerciseState {
  constructor(cfg) {
    this._cfg = cfg;
    this._smoothed = null;
    this._state = 'neutral';
    this.count = 0;
  }

  update(angles) {
    // angles: Float32Array length 16
    const { features, direction, enter, exit: exit_ } = this._cfg;
    const raw = (angles[features[0]] + angles[features[1]]) / 2;

    this._smoothed = this._smoothed === null
      ? raw
      : EMA_ALPHA * raw + (1 - EMA_ALPHA) * this._smoothed;

    if (direction === 'low') {
      if (this._state === 'neutral' && this._smoothed < enter) {
        this._state = 'peaked';
      } else if (this._state === 'peaked' && this._smoothed > exit_) {
        this._state = 'neutral';
        this.count++;
      }
    } else {
      if (this._state === 'neutral' && this._smoothed > enter) {
        this._state = 'peaked';
      } else if (this._state === 'peaked' && this._smoothed < exit_) {
        this._state = 'neutral';
        this.count++;
      }
    }
  }
}

export class RepCounter {
  constructor() {
    this._states = {};
  }

  update(exerciseName, angles) {
    const cfg = EXERCISE_REP_CONFIG[exerciseName];
    if (!cfg) return 0;
    if (!this._states[exerciseName]) {
      this._states[exerciseName] = new ExerciseState(cfg);
    }
    this._states[exerciseName].update(angles);
    return this._states[exerciseName].count;
  }

  get(exerciseName) {
    return this._states[exerciseName]?.count ?? 0;
  }

  getCounts() {
    const out = {};
    for (const [name, state] of Object.entries(this._states)) {
      out[name] = state.count;
    }
    return out;
  }
}
