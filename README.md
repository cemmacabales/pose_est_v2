# Pose Estimation + Exercise Classifier

## Mac (training machine) — run in order:
1. python check_deps.py
2. python load_dataset.py
3. python normalize_joints.py
4. python build_windows.py
5. python train_classifier.py
6. python export_models.py

## RPi 5 (inference + GUI):
git clone <repo> → bash rpi_setup.sh → python gui.py

## Joint mapping:
MoveNet (17 COCO) and UI-PRMD (20 Kinect) share 12 joints.
See joint_map.py for the explicit mapping.

## Models:
- movenet_thunder_int8.tflite — pose estimation (MoveNet Thunder INT8)
- classifier.tflite — dual-head LSTM: exercise (10 classes) + quality (correct/incorrect)
