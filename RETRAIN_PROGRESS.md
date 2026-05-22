# Retraining Progress Tracker — Round 2

**Status: ALL PHASES COMPLETE.** 174 videos from 5 people → 451K windows → model retrained and deployed.

---

## Phase 0: Backup & Safety
- [x] Backup `data/labels.csv` → `data/labels_backup.csv` (2194 rows)
- [x] Backup `data/normalized/` → `data/normalized_backup/` (2193 files)
- [x] Backup `data/X_train.npy`, `y_exercise.npy`, `y_quality.npy` → `data/backup/`
- [x] Git status documented

## Phase 1: Data Preparation (26 Videos → Keypoints)
- [x] Create `data/retrain_videos/`
- [x] Copy & flatten all `.mp4` from `eval/test videos [POSE_EST] 2/`
- [x] Rename to `XX_correct_N.mp4` format
- [x] Clear old `data/labels.csv` and `data/normalized/*.npy`
- [x] Run `extract_video_keypoints.py --video_dir data/retrain_videos/`
- [x] Verify: 26 `.npy` files, labels.csv has 26 rows, IDs {1-8,10}

## Phase 2: Window Building & Augmentation (First Pass)
- [x] Update `build_windows.py` for 9 classes + augmentation
- [x] Run and verify `X_train.npy`, `y_exercise.npy`, `y_quality.npy`

## Phase 3: Code Refactoring (9 Classes)
- [x] `joint_map.py`: EXERCISE_NAMES 0-based keys {0–8}, imported everywhere
- [x] `train_classifier.py`: num_classes=9, Dense(9)
- [x] `gui.py`: shape==9, remove +1 offset
- [x] `eval/annotate_video.py`: shape==9, remove +1 offset
- [x] `session_logger.py`: import from joint_map, remove +1 offset
- [x] `export_models.py`: verify (1,9) and (1,2)
- [x] `tests/test_gui_integration.py`: mock shape (1,9)
- [x] Run all tests: **38 passed, 0 failed**

## Phase 4: Model Training (26 Videos)
- [x] Run `train_classifier.py`
- [x] Save `classifier.keras` and `classifier_best.keras`

## Phase 5: Export & Validation (26 Videos)
- [x] Delete old `models/classifier.tflite` (10-class)
- [x] Run `export_models.py`
- [x] 100% accuracy on same-session validation (15 videos, same recording)

---

## Phase 6: Expanded Data Collection — DONE (174 Videos)

**Actual results vs planned:**

| Person | Expected | Actual |
|--------|----------|--------|
| Person 1 | 30 | 33 |
| Person 2 | 30 | 37 |
| Person 3 | 30 | 47 |
| Person 4 | 30 | 28 |
| Person 5 | 30 | 30 |
| **TOTAL** | **150** | **174** |

**Per-exercise distribution:**

| Exercise | Correct | Incorrect | Total | People |
|----------|---------|-----------|-------|--------|
| 01 Deep Squat | 12 | 10 | 22 | P1, P2 |
| 02 Hurdle Step | 14 | 12 | 26 | P1, P3 |
| 03 Inline Lunge | 14 | 12 | 26 | P1, P3 |
| 04 Side Lunge | 14 | 12 | 26 | P2, P4 |
| 05 Sit to Stand | 12 | 9 | 21 | P2, P5 |
| 06 Standing Leg Raise | 12 | 12 | 24 | P3, P4 |
| 07 Shoulder Abduction | 5 | 4 | 9 | P4 only |
| 08 Shoulder Extension | 6 | 4 | 10 | P5 only |
| 10 Shoulder Scaption | 6 | 4 | 10 | P5 only |

**Quality split:** 95 correct / 79 incorrect — first time with real incorrect data.

## Phase 7: Retrain with Expanded Data — DONE (174 Videos)

- [x] Preprocess 174 videos via `preprocess_train_videos.py`
- [x] Extract keypoints via `extract_video_keypoints.py`
- [x] Build windows: **451,638 windows** after 5x augmentation
- [x] Retrain model (9 classes, 2-layer LSTM, 114K params)
- [x] Export new TFLite (`classifier.tflite`, 221 KB)
- [x] Validate across person-based splits

### Model Performance

**Random 80/20 split (same people, held-out windows):**
| Split | Exercise Acc | Quality Acc |
|-------|-------------|-------------|
| Train | 90.0% | 87.9% |
| Val | 88.3% | 86.3% |

**Person-based split (unseen people held out):**
| Split | Exercise Acc | Quality Acc |
|-------|-------------|-------------|
| Train (P1+P2+P3) | 98.6% | 84.4% |
| Val (P4) | 15.5% | 50.9% |
| Test (P5) | 10.2% | 66.8% |

**⚠️ Critical finding:** Model memorizes training people (98.6%) but drops to random (~11%) on unseen people. 2D pixel coordinates vary too much across body types, camera angles, and distances. **Fix:** either a 20+ person dataset or converting to joint-angle features.

---

## Deployment

- [x] `.gitignore` cleaned — training/eval files removed from tracking
- [x] `rpi_setup.sh` updated — all RPi deps, no Vertex AI cruft
- [x] `requirements.txt` cleaned — no build-time deps (PyTorch/etc.)
- [x] All deployment tests pass (28/28)
- [x] Pushed to remote

## Exercise ID Mapping

```
01 → index 0 → Deep Squat
02 → index 1 → Hurdle Step
03 → index 2 → Inline Lunge
04 → index 3 → Side Lunge
05 → index 4 → Sit to Stand
06 → index 5 → Standing Leg Raise
07 → index 6 → Shoulder Abduction
08 → index 7 → Shoulder Extension
10 → index 8 → Shoulder Scaption
09 → DROPPED (no video available)
```

## Next Steps (for real generalization)

1. Collect 20+ person dataset (outside the team) with diverse body types/angles
2. OR implement joint-angle features instead of raw (x,y) pixel coordinates
