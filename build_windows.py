import csv
import re
import numpy as np
from pathlib import Path

from joint_angles import batch_keypoints_to_angles

PERSON_RE = re.compile(r"_P(\d+)_")

ANGLE_FEATURES = 16


def main():
    data_dir = Path('./data')
    normalized_dir = data_dir / 'normalized'
    labels_path = data_dir / 'labels.csv'

    windows = []
    exercise_labels = []
    quality_labels = []
    person_labels = []

    with open(labels_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = row['stem']
            exercise_id = int(row['exercise_id'])
            quality_str = row['quality_label']

            person_match = PERSON_RE.search(stem)
            person_id = int(person_match.group(1)) if person_match else 0

            if exercise_id == 0:
                print(f"Warning: skipping {stem} (unknown exercise_id=0)")
                continue

            npy_path = normalized_dir / f"{stem}.npy"
            if not npy_path.exists():
                print(f"Warning: skipping {stem} (file not found: {npy_path})")
                continue

            arr = np.load(npy_path)
            if arr.shape[0] < 30:
                print(f"Warning: skipping {stem} (only {arr.shape[0]} frames, need >= 30)")
                continue

            num_frames = arr.shape[0]
            window_size = 30

            # Map exercise IDs to 0-based class indices
            # E09 dropped, so E10 maps to index 8
            if exercise_id == 10:
                cls = 8
            else:
                cls = exercise_id - 1  # E01->0, E02->1, ..., E08->7

            quality_int = 1 if quality_str == 'correct' else 0

            def _append_angles(kps_window):
                """Convert a (30, 12, 2) keypoint window to angles and append."""
                angles = batch_keypoints_to_angles(kps_window)  # (30, 16)
                windows.append(angles)
                exercise_labels.append(cls)
                quality_labels.append(quality_int)
                person_labels.append(person_id)

            # Generate windows with multiple strides for temporal augmentation
            for stride in [5, 2]:
                for start in range(0, num_frames - window_size + 1, stride):
                    window_kp = arr[start:start + window_size].copy()  # shape (30, 12, 2)

                    # Base window → angles
                    _append_angles(window_kp)

                    # --- Augmentation 1: Horizontal flip (swap L/R ids, negate x) ---
                    flipped = window_kp.copy()
                    for a, b in [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]:
                        flipped[:, [a, b]] = flipped[:, [b, a]]
                    flipped[:, :, 0] = -flipped[:, :, 0]
                    _append_angles(flipped)

                    # --- Augmentation 2: Gaussian noise (applied to keypoints) ---
                    noised = window_kp + np.random.normal(0, 0.01, window_kp.shape)
                    _append_angles(noised.astype(np.float32))

                    # --- Augmentation 3: Time stretch (0.8x – 1.2x) ---
                    stretch_factor = np.random.uniform(0.8, 1.2)
                    num_points = max(2, int(window_size * stretch_factor))
                    kp_flat = window_kp.reshape(window_size, 24)

                    resampled = np.empty((num_points, 24), dtype=np.float32)
                    for feat in range(24):
                        resampled[:, feat] = np.interp(
                            np.linspace(0, window_size - 1, num_points),
                            np.arange(window_size),
                            kp_flat[:, feat],
                        )
                    stretched_flat = np.empty((window_size, 24), dtype=np.float32)
                    for feat in range(24):
                        stretched_flat[:, feat] = np.interp(
                            np.linspace(0, num_points - 1, window_size),
                            np.arange(num_points),
                            resampled[:, feat],
                        )
                    _append_angles(stretched_flat.reshape(window_size, 12, 2))

                    # --- Augmentation 4: Joint masking (zero out 1–2 joints) ---
                    w_2d = window_kp.copy()
                    n_mask = np.random.randint(1, 3)
                    mask_joints = np.random.choice(12, n_mask, replace=False)
                    w_2d[:, mask_joints, :] = 0.0
                    _append_angles(w_2d)

    if len(windows) == 0:
        print("No windows generated.")
        return

    X = np.stack(windows).astype(np.float32)
    y_exercise = np.array(exercise_labels, dtype=np.int32)
    y_quality = np.array(quality_labels, dtype=np.int32)
    y_person = np.array(person_labels, dtype=np.int32)

    np.save(data_dir / 'X_train.npy', X)
    np.save(data_dir / 'y_exercise.npy', y_exercise)
    np.save(data_dir / 'y_quality.npy', y_quality)
    np.save(data_dir / 'y_person.npy', y_person)

    print(f"Total windows: {X.shape[0]}, shape: {X.shape}, features={X.shape[2]}")

    print("Exercise class distribution:")
    for cls in range(9):
        count = int(np.sum(y_exercise == cls))
        print(f"  Class {cls}: {count}")

    correct_count = int(np.sum(y_quality == 1))
    incorrect_count = int(np.sum(y_quality == 0))
    print(f"Quality distribution: correct={correct_count}, incorrect={incorrect_count}")


if __name__ == '__main__':
    main()
