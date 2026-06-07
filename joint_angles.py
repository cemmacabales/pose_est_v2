import numpy as np


def _cos(v1, v2):
    """Cosine of angle between two vectors. Safe for zero-length vectors."""
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-8:
        return 1.0
    return np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)


def _unit(v):
    """Unit vector. Returns zero vector if norm is near zero."""
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v * 0.0
    return v / n


def keypoints_to_angles(joints):
    """
    Convert 12 hip-centered BlazePose keypoints (12, 2) → joint-angle features (16,).

    The 12 joints are ordered by BLAZEPOSE_MAPPED_INDICES:
        [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    →   L_shoulder, R_shoulder, L_elbow, R_elbow, L_wrist, R_wrist,
        L_hip, R_hip, L_knee, R_knee, L_ankle, R_ankle

    Hip midpoint is already at origin (0, 0) — hip-centered input.
    Torso = mid_shoulder − mid_hip = mid_shoulder (since mid_hip ≈ origin).

    Returns 16 features per frame:
      [0]   L elbow angle (cos)
      [1]   R elbow angle (cos)
      [2]   L knee angle (cos)
      [3]   R knee angle (cos)
      [4]   L arm elevation (cos between upper_arm_L and torso)
      [5]   R arm elevation (cos between upper_arm_R and torso)
      [6]   L hip flexion (cos between upper_leg_L and torso)
      [7]   R hip flexion (cos between upper_leg_R and torso)
      [8]   L upper arm lateral (x-comp of unit upper_arm_L)
      [9]   R upper arm lateral (x-comp of unit upper_arm_R)
      [10]  L upper leg lateral (x-comp of unit upper_leg_L)
      [11]  R upper leg lateral (x-comp of unit upper_leg_R)
      [12]  L upper arm vertical (y-comp of unit upper_arm_L)
      [13]  R upper arm vertical (y-comp of unit upper_arm_R)
      [14]  L upper leg vertical (y-comp of unit upper_leg_L)
      [15]  R upper leg vertical (y-comp of unit upper_leg_R)
    """
    joints = np.asarray(joints, dtype=np.float32)

    L_sho = joints[0]   # left shoulder
    R_sho = joints[1]   # right shoulder
    L_elb = joints[2]
    R_elb = joints[3]
    L_wri = joints[4]
    R_wri = joints[5]
    L_hip = joints[6]
    R_hip = joints[7]
    L_kne = joints[8]
    R_kne = joints[9]
    L_ank = joints[10]
    R_ank = joints[11]

    # ── limb vectors ──
    upper_arm_L = L_elb - L_sho
    upper_arm_R = R_elb - R_sho
    forearm_L   = L_wri - L_elb
    forearm_R   = R_wri - R_elb
    upper_leg_L = L_kne - L_hip
    upper_leg_R = R_kne - R_hip
    lower_leg_L = L_ank - L_kne
    lower_leg_R = R_ank - R_kne

    mid_shoulder = (L_sho + R_sho) * 0.5
    torso = mid_shoulder  # hip midpoint = (0, 0) by construction
    torso_unit = _unit(torso)

    # ── unit limb vectors (for directional features) ──
    ua_L_unit = _unit(upper_arm_L)
    ua_R_unit = _unit(upper_arm_R)
    ul_L_unit = _unit(upper_leg_L)
    ul_R_unit = _unit(upper_leg_R)

    return np.array([
        _cos(upper_arm_L, forearm_L),     # 0  L elbow
        _cos(upper_arm_R, forearm_R),     # 1  R elbow
        _cos(upper_leg_L, lower_leg_L),   # 2  L knee
        _cos(upper_leg_R, lower_leg_R),   # 3  R knee
        _cos(upper_arm_L, torso_unit),    # 4  L arm elevation
        _cos(upper_arm_R, torso_unit),    # 5  R arm elevation
        _cos(upper_leg_L, torso_unit),    # 6  L hip flexion
        _cos(upper_leg_R, torso_unit),    # 7  R hip flexion
        ua_L_unit[0],                     # 8  L upper arm x
        ua_R_unit[0],                     # 9  R upper arm x
        ul_L_unit[0],                     # 10 L upper leg x
        ul_R_unit[0],                     # 11 R upper leg x
        ua_L_unit[1],                     # 12 L upper arm y
        ua_R_unit[1],                     # 13 R upper arm y
        ul_L_unit[1],                     # 14 L upper leg y
        ul_R_unit[1],                     # 15 R upper leg y
    ], dtype=np.float32)


def batch_keypoints_to_angles(batch_joints):
    """
    Convert a batch of (F, 12, 2) keypoints → (F, 16) angle features.
    Vectorized over frames — no Python loops.
    """
    kp = np.asarray(batch_joints, dtype=np.float32)       # (F, 12, 2)

    L_sho = kp[:, 0, :]   # (F, 2)
    R_sho = kp[:, 1, :]
    L_elb = kp[:, 2, :]
    R_elb = kp[:, 3, :]
    L_wri = kp[:, 4, :]
    R_wri = kp[:, 5, :]
    L_hip = kp[:, 6, :]
    R_hip = kp[:, 7, :]
    L_kne = kp[:, 8, :]
    R_kne = kp[:, 9, :]
    L_ank = kp[:, 10, :]
    R_ank = kp[:, 11, :]

    # limb vectors (F, 2)
    ua_L = L_elb - L_sho
    ua_R = R_elb - R_sho
    fa_L = L_wri - L_elb
    fa_R = R_wri - R_elb
    ul_L = L_kne - L_hip
    ul_R = R_kne - R_hip
    ll_L = L_ank - L_kne
    ll_R = R_ank - R_kne

    torso = (L_sho + R_sho) * 0.5   # (F, 2) — hip midpoint is origin

    def _norm(v):
        return np.sqrt(np.sum(v * v, axis=1, keepdims=True) + 1e-8)

    def _unit(v):
        return v / _norm(v)

    def _cos(v1, v2):
        n1 = _norm(v1)
        n2 = _norm(v2)
        dot = np.sum(v1 * v2, axis=1)  # (F,)
        return np.clip(dot / (n1.ravel() * n2.ravel()), -1.0, 1.0)

    torso_u = _unit(torso)           # (F, 2)
    ua_L_u = _unit(ua_L)
    ua_R_u = _unit(ua_R)
    ul_L_u = _unit(ul_L)
    ul_R_u = _unit(ul_R)

    F = kp.shape[0]
    out = np.empty((F, 16), dtype=np.float32)

    out[:, 0]  = _cos(ua_L, fa_L)       # L elbow
    out[:, 1]  = _cos(ua_R, fa_R)       # R elbow
    out[:, 2]  = _cos(ul_L, ll_L)       # L knee
    out[:, 3]  = _cos(ul_R, ll_R)       # R knee
    out[:, 4]  = _cos(ua_L, torso_u)    # L arm elevation
    out[:, 5]  = _cos(ua_R, torso_u)    # R arm elevation
    out[:, 6]  = _cos(ul_L, torso_u)    # L hip flexion
    out[:, 7]  = _cos(ul_R, torso_u)    # R hip flexion
    out[:, 8]  = ua_L_u[:, 0]           # L arm x
    out[:, 9]  = ua_R_u[:, 0]           # R arm x
    out[:, 10] = ul_L_u[:, 0]           # L leg x
    out[:, 11] = ul_R_u[:, 0]           # R leg x
    out[:, 12] = ua_L_u[:, 1]           # L arm y
    out[:, 13] = ua_R_u[:, 1]           # R arm y
    out[:, 14] = ul_L_u[:, 1]           # L leg y
    out[:, 15] = ul_R_u[:, 1]           # R leg y

    return out


def compute_motion(angles: np.ndarray) -> float:
    """Mean per-feature std deviation across frames. Near 0 = idle, higher = active."""
    return float(np.std(angles, axis=0).mean())
