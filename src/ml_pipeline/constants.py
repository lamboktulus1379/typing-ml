from typing import Dict, Tuple

FINGERS = [
    "left_pinky",
    "left_ring",
    "left_middle",
    "left_index",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
]

FINGER_ERROR_COLUMNS = [f"error_{finger}" for finger in FINGERS]
FINGER_DWELL_COLUMNS = [f"dwell_{finger}" for finger in FINGERS]
FINGER_FLIGHT_COLUMNS = [f"flight_{finger}" for finger in FINGERS]

TRAIN_FEATURE_COLUMNS = [
    "wpm",
    "accuracy",
    *FINGER_ERROR_COLUMNS,
    *FINGER_DWELL_COLUMNS,
    *FINGER_FLIGHT_COLUMNS,
]

TARGET_COLUMN = "weakest_finger"
ALLOWED_WEAKEST_FINGER_LABELS = frozenset(FINGERS)

FEATURE_RANGE_RULES: Dict[str, Tuple[float, float | None]] = {
    "wpm": (0.0, None),
    "accuracy": (0.0, 1.0),
    **{name: (0.0, 1.0) for name in FINGER_ERROR_COLUMNS},
    **{name: (0.0, None) for name in FINGER_DWELL_COLUMNS},
    **{name: (0.0, None) for name in FINGER_FLIGHT_COLUMNS},
}
