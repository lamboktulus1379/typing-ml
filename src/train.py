import os
import json
import argparse
import datetime
from typing import Any, Callable, Dict, Tuple, cast
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection
import joblib

FINGERS = [
    "left_pinky", "left_ring", "left_middle", "left_index",
    "right_index", "right_middle", "right_ring", "right_pinky"
]

FINGER_ERROR_COLUMNS = [f"error_{f}" for f in FINGERS]
FINGER_DWELL_COLUMNS = [f"dwell_{f}" for f in FINGERS]
FINGER_FLIGHT_COLUMNS = [f"flight_{f}" for f in FINGERS]

TRAIN_FEATURE_COLUMNS = [
    "wpm",
    "accuracy",
    *FINGER_ERROR_COLUMNS,
    *FINGER_DWELL_COLUMNS,
    *FINGER_FLIGHT_COLUMNS
]

FEATURE_RANGE_RULES: Dict[str, Tuple[float, float | None]] = {
    "wpm": (0.0, None),
    "accuracy": (0.0, 1.0),
    **{name: (0.0, 1.0) for name in FINGER_ERROR_COLUMNS},
    **{name: (0.0, None) for name in FINGER_DWELL_COLUMNS},
    **{name: (0.0, None) for name in FINGER_FLIGHT_COLUMNS},
}

ALLOWED_WEAKEST_FINGER_LABELS = set(FINGERS)


def _build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = [c for c in TRAIN_FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    feature_frame = df[TRAIN_FEATURE_COLUMNS].copy()
    if feature_frame.empty:
        raise ValueError("Dataset contains zero rows after feature selection.")

    feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce")

    invalid_numeric_columns = feature_frame.columns[feature_frame.isna().any()].tolist()
    if invalid_numeric_columns:
        raise ValueError(
            "Required feature columns contain null or non-numeric values: "
            f"{invalid_numeric_columns}"
        )

    # Explicitly reject +/-inf values; these can silently pass numeric conversion.
    if not np.isfinite(feature_frame.to_numpy(dtype=float)).all():
        raise ValueError(
            "Required feature columns contain non-finite values (inf or -inf)."
        )

    out_of_range_messages: list[str] = []
    for col_name, (min_value, max_value) in FEATURE_RANGE_RULES.items():
        col_values = feature_frame[col_name]
        if (col_values < min_value).any():
            out_of_range_messages.append(f"{col_name} < {min_value}")
        if max_value is not None and (col_values > max_value).any():
            out_of_range_messages.append(f"{col_name} > {max_value}")

    if out_of_range_messages:
        raise ValueError(
            "Required feature columns contain out-of-range values: "
            f"{out_of_range_messages}"
        )

    return feature_frame


def _build_target_series(df: pd.DataFrame, target: str) -> pd.Series:
    raw_target = df[target]

    if raw_target.isna().any():
        raise ValueError(
            f"Target column '{target}' contains null values; fill or remove them first."
        )

    target_series = raw_target.astype(str).str.strip()
    if (target_series == "").any():
        raise ValueError(
            f"Target column '{target}' contains blank string labels; fix input labels first."
        )

    invalid_labels = sorted(set(target_series.unique()) - ALLOWED_WEAKEST_FINGER_LABELS)
    if invalid_labels:
        raise ValueError(
            f"Target column '{target}' contains unsupported labels: {invalid_labels}. "
            f"Allowed labels: {sorted(ALLOWED_WEAKEST_FINGER_LABELS)}"
        )

    class_counts = target_series.value_counts()
    if class_counts.size < 2:
        raise ValueError(
            f"Target column '{target}' must contain at least 2 classes for training. "
            f"Found classes: {class_counts.to_dict()}"
        )

    sparse_classes = class_counts[class_counts < 2]
    if not sparse_classes.empty:
        raise ValueError(
            "Each class must have at least 2 rows for stratified train/test split. "
            f"Sparse classes: {sparse_classes.to_dict()}"
        )

    return target_series


def _build_model_candidates(random_state: int) -> Dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
    }


def main(
    data_path: str,
    model_out: str,
    report_out: str,
    random_state: int = 42,
    model_type: str = "auto",
) -> None:
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    if df.empty:
        raise ValueError("Training dataset is empty. Provide at least one row.")

    target = "weakest_finger"
    if target not in df.columns:
        raise ValueError(
            f"Dataset is missing required target column '{target}' needed for training."
        )

    x_df = _build_feature_frame(df)
    y = _build_target_series(df, target)

    split_fn = cast(
        Callable[..., Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]],
        cast(Any, sk_model_selection).train_test_split,
    )
    try:
        x_train, x_test, y_train, y_test = split_fn(
            x_df, y, test_size=0.2, random_state=random_state, stratify=y
        )
    except ValueError as ex:
        raise ValueError(
            "Failed to create stratified train/test split. "
            f"Class distribution: {y.value_counts().to_dict()}"
        ) from ex

    candidates = _build_model_candidates(random_state)
    if model_type == "auto":
        selected_name = ""
        selected_score = -1.0
        best_model: Any = None
        candidate_scores: Dict[str, float] = {}

        print("Training model candidates: logistic_regression, random_forest...")
        for name, candidate in candidates.items():
            candidate_any = cast(Any, candidate)
            candidate_any.fit(x_train, y_train)
            score = float(candidate_any.score(x_test, y_test))
            candidate_scores[name] = score
            if score > selected_score:
                selected_score = score
                selected_name = name
                best_model = candidate

        model = cast(Pipeline, best_model)
        print(f"Selected model: {selected_name} (accuracy={selected_score * 100:.2f}%)")
    else:
        if model_type not in candidates:
            raise ValueError(
                f"Unsupported --model-type '{model_type}'. "
                "Use one of: auto, logistic_regression, random_forest"
            )
        selected_name = model_type
        candidate_scores = {}
        model = candidates[model_type]
        print(f"Training {selected_name} model on 26 features...")
        model_any = cast(Any, model)
        model_any.fit(x_train, y_train)

    model_any = cast(Any, model)
    y_pred = model_any.predict(x_test)
    report_fn = cast(Callable[..., Dict[str, Any]], cast(Any, sk_metrics).classification_report)
    report = report_fn(y_test, y_pred, output_dict=True)
    report["selected_model"] = selected_name
    if candidate_scores:
        report["candidate_accuracies"] = candidate_scores

    artifact: Dict[str, Any] = {
        "model": model,
        "model_name": selected_name,
        "feature_names": TRAIN_FEATURE_COLUMNS,
        "target_name": target,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    model_out_dir = os.path.dirname(model_out)
    if model_out_dir:
        os.makedirs(model_out_dir, exist_ok=True)
    cast(Any, joblib).dump(artifact, model_out)

    report_out_dir = os.path.dirname(report_out)
    if report_out_dir:
        os.makedirs(report_out_dir, exist_ok=True)
    with open(report_out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved model to: {model_out}")
    print(f"Selected Model: {selected_name}")
    print(f"Model Accuracy: {report['accuracy'] * 100:.2f}%\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/dataset.csv")
    parser.add_argument("--model-out", default="models/model.joblib")
    parser.add_argument("--report-out", default="reports/training_report.json")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--model-type",
        default="auto",
        choices=["auto", "logistic_regression", "random_forest"],
        help="Training model type. 'auto' trains all candidates and selects the best by holdout accuracy.",
    )

    args = parser.parse_args()
    main(args.data, args.model_out, args.report_out, args.random_state, args.model_type)
