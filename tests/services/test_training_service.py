from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from services.training_service import TrainingArenaService  # noqa: E402
from ml_pipeline.artifacts import ArtifactStore  # noqa: E402


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
        row["user_id"] = "user-a"
        row["error_left_pinky"] = 0.25 + (0.01 * scale)
        row["error_right_index"] = 0.02
        row["weakest_finger"] = "left_pinky"
        rows.append(row)

    for scale in range(6, 12):
        row = _build_row(float(scale))
        row["user_id"] = "user-a"
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


def test_run_algorithm_arena_removes_timing_outliers_before_split() -> None:
    service = TrainingArenaService.default(random_state=42)
    dataframe = _build_training_frame()

    negative_row = dataframe.iloc[[0]].copy(deep=True)
    negative_row["dwell_left_pinky"] = -5.0

    hard_cap_row = dataframe.iloc[[1]].copy(deep=True)
    hard_cap_row["flight_right_index"] = 5005.0

    iqr_row = dataframe.iloc[[2]].copy(deep=True)
    iqr_row["dwell_right_pinky"] = 1200.0

    noisy_dataframe = pd.concat(
        [dataframe, negative_row, hard_cap_row, iqr_row],
        ignore_index=True,
    )

    result = service.run_algorithm_arena(
        noisy_dataframe,
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


def test_run_algorithm_arena_filters_rows_to_requested_user_before_split() -> None:
    service = TrainingArenaService.default(random_state=42)
    user_a_rows = _build_training_frame()
    user_b_rows = user_a_rows.copy(deep=True)
    user_b_rows["user_id"] = "user-b"
    combined = pd.concat([user_a_rows, user_b_rows], ignore_index=True)

    result = service.run_algorithm_arena(
        combined,
        algorithms=("logistic_regression", "random_forest", "xgboost"),
        user_id="user-b",
    )

    assert result.total_rows_processed == len(user_b_rows)
    assert result.winning_algorithm in {"logistic_regression", "random_forest", "xgboost"}


def test_train_personal_model_merges_inline_rows_with_persisted_dataset(tmp_path: Path) -> None:
    service = TrainingArenaService.default(random_state=42)

    persisted_rows = _build_training_frame()
    inline_rows: list[dict[str, float | str]] = []

    for scale in range(12, 16):
        row = _build_row(float(scale))
        row["user_id"] = "user-a"
        row["error_left_pinky"] = 0.28 + (0.01 * (scale - 12))
        row["error_right_index"] = 0.02
        row["weakest_finger"] = "left_pinky"
        inline_rows.append(row)

    for scale in range(16, 20):
        row = _build_row(float(scale))
        row["user_id"] = "user-a"
        row["error_left_pinky"] = 0.02
        row["error_right_index"] = 0.28 + (0.01 * (scale - 16))
        row["weakest_finger"] = "right_index"
        inline_rows.append(row)

    dataset_path = tmp_path / "dataset.csv"
    persisted_rows.to_csv(dataset_path, index=False)

    result = service.train_personal_model(
        user_id="user-a",
        data_path=str(dataset_path),
        dataframe=pd.DataFrame(inline_rows),
        algorithms=("logistic_regression",),
        artifact_store=ArtifactStore(),
        model_output_path_template=str(tmp_path / "model_{user_id}.joblib"),
    )

    assert result.arena_result.total_rows_processed == len(persisted_rows) + len(inline_rows)
