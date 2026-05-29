"""pose_estimator.py — MediaPipe BlazePose wrapper for pose estimation."""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from joint_map import BLAZEPOSE_MAPPED_INDICES

MODELS_DIR = "./models"
POSE_MODEL_MAP = {
    0: "pose_landmarker_lite.task",
    1: "pose_landmarker_full.task",
    2: "pose_landmarker_heavy.task",
}
POSE_MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


def _download_model(url, path):
    import urllib.request
    print(f"Downloading pose model from {url} ...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(url, path)
    size = os.path.getsize(path)
    print(f"Saved to {path} ({size} bytes)")


def _resolve_model_path(model_complexity):
    if model_complexity not in POSE_MODEL_MAP:
        raise ValueError(f"model_complexity must be 0, 1, or 2, got {model_complexity}")
    filename = POSE_MODEL_MAP[model_complexity]
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        _download_model(POSE_MODEL_URLS[model_complexity], path)
    return path


class PoseEstimator:
    def __init__(self, model_complexity=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5, static_image_mode=False):
        model_path = _resolve_model_path(model_complexity)
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def process_frame(self, frame):
        """Run BlazePose on a BGR frame.
        Returns list of 33 NormalizedLandmark objects or None if no person detected."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            return result.pose_landmarks[0]
        return None

    @staticmethod
    def extract_mapped_joints(landmarks, indices=None):
        """Extract 12 mapped joints from BlazePose landmarks, normalized by
        hip midpoint. Returns (len(indices), 2) float32 array or None."""
        if indices is None:
            indices = BLAZEPOSE_MAPPED_INDICES
        if landmarks is None:
            return None
        hip_mid_x = (landmarks[23].x + landmarks[24].x) / 2.0
        hip_mid_y = (landmarks[23].y + landmarks[24].y) / 2.0
        joints = np.zeros((len(indices), 2), dtype=np.float32)
        for i, idx in enumerate(indices):
            joints[i, 0] = landmarks[idx].x - hip_mid_x
            joints[i, 1] = landmarks[idx].y - hip_mid_y
        return joints

    def draw_landmarks(self, frame, landmarks):
        """Draw 33-landmark BlazePose skeleton on BGR frame (mutates in place).
        No-op if landmarks is None."""
        if landmarks is None:
            return
        vision.drawing_utils.draw_landmarks(
            frame,
            landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=vision.drawing_styles.get_default_pose_landmarks_style(),
        )

    @staticmethod
    def count_visible(landmarks, threshold=0.5):
        """Count landmarks with visibility >= threshold."""
        if landmarks is None:
            return 0
        return sum(1 for lm in landmarks if lm.visibility is not None and lm.visibility >= threshold)

    def close(self):
        self.landmarker.close()
