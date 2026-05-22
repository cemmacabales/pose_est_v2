# Vicon marker index → MoveNet joint index
# Marker M occupies cols M*3 (x), M*3+1 (y), M*3+2 (z)
VICON_TO_MOVENET = {
    9:  5,   # LSHO  → left_shoulder
    16: 6,   # RSHO  → right_shoulder
    11: 7,   # LELB  → left_elbow
    18: 8,   # RELB  → right_elbow
    13: 9,   # LWRA  → left_wrist
    20: 10,  # RWRA  → right_wrist
    23: 11,  # LASI  → left_hip
    24: 12,  # RASI  → right_hip
    28: 13,  # LKNE  → left_knee
    34: 14,  # RKNE  → right_knee
    30: 15,  # LANK  → left_ankle
    36: 16,  # RANK  → right_ankle
}

LEFT_HIP_MARKER  = 23   # LASI
RIGHT_HIP_MARKER = 24   # RASI

MOVENET_JOINT_NAMES = {
    5:  "left_shoulder",  6:  "right_shoulder",
    7:  "left_elbow",     8:  "right_elbow",
    9:  "left_wrist",     10: "right_wrist",
    11: "left_hip",       12: "right_hip",
    13: "left_knee",      14: "right_knee",
    15: "left_ankle",     16: "right_ankle",
}

EXERCISE_NAMES = {
    0: "Deep Squat",
    1: "Hurdle Step",
    2: "Inline Lunge",
    3: "Side Lunge",
    4: "Sit to Stand",
    5: "Standing Leg Raise",
    6: "Shoulder Abduction",
    7: "Shoulder Extension",
    8: "Shoulder Scaption",
}

COCO_SKELETON_EDGES = [
    (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),
    (6,8),(8,10),(11,12),(5,11),(6,12),(11,13),
    (12,14),(13,15),(14,16)
]
