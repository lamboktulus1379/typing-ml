from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from services.training_service import TrainingArenaService  # noqa: E402


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


def _build_row(scale: float) -> dict[str, float]:
    row = {
        "wpm": 45.0 + scale,
        "accuracy": 0.90 + (0.002 * scale),
    }
    for index, finger in enumerate(FINGERS):
        row[f"error_{finger}"] = 0.01 + (0.001 * index) + (0.0005 * scale)
        row[f"dwell_{finger}"] = 90.0 + index + (2.0 * scale)
        row[f"flight_{finger}"] = 180.0 + index + (3.0 * scale)
    return row


def _build_training_frame() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for scale in range(6):
        row = _build_row(float(scale))
        row["error_left_pinky"] = 0.25 + (0.01 * scale)
        row["error_right_index"] = 0.02
        row["weakest_finger"] = "left_pinky"
        rows.append(row)

    for scale in range(6, 12):
        row = _build_row(float(scale))
        row["error_left_pinky"] = 0.02
        row["error_right_index"] = 0.25 + (0.01 * (scale - 6))
        row["weakest_finger"] = "right_index"
        rows.append(row)

    return pd.DataFrame(rows)


def test_run_algorithm_arena_returns_leaderboard_and_full_data_model() -> None:
    service = TrainingArenaService.default(random_state=42)
    dataframe = _build_training_frame()

    result = service.run_algorithm_arena(
        dataframe,
        algorithms=("logistic_regression", "random_forest", "xgboost"),
    )

    assert result.total_rows_processed == len(dataframe)
    assert result.winning_algorithm in {"logistic_regression", "random_forest", "xgboost"}
    assert 0.0 <= result.winning_accuracy <= 1.0
    assert 0.0 <= result.winning_f1_score <= 1.0
    assert 0.0 <= result.macro_precision <= 1.0
    assert 0.0 <= result.macro_recall <= 1.0
    assert result.top_predictive_feature in dataframe.columns
    assert result.primary_misclassification
    assert len(result.leaderboard) == 3
    assert {entry.name for entry in result.leaderboard} == {
        "logistic_regression",
        "random_forest",
        "xgboost",
    }
    for entry in result.leaderboard:
        assert 0.0 <= entry.accuracy <= 1.0
        assert 0.0 <= entry.f1_score <= 1.0
        assert entry.execution_time_ms >= 0.0
    assert hasattr(result.retrained_model, "predict")


def test_run_algorithm_arena_deduplicates_exact_rows() -> None:
    service = TrainingArenaService.default(random_state=42)
    dataframe = _build_training_frame()
    duplicated = pd.concat([dataframe, dataframe.iloc[[0]], dataframe.iloc[[7]]], ignore_index=True)

    result = service.run_algorithm_arena(
        duplicated,
        algorithms=("logistic_regression", "random_forest", "xgboost"),
    )

    assert result.total_rows_processed == len(dataframe)


def test_run_algorithm_arena_requires_algorithms() -> None:
    service = TrainingArenaService.default(random_state=42)
    dataframe = _build_training_frame()

    try:
        service.run_algorithm_arena(dataframe, algorithms=())
    except ValueError as ex:
        assert "No candidate algorithms" in str(ex)
    else:
        raise AssertionError("Expected ValueError when no algorithms are provided")