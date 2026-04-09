import os
import argparse
from typing import Any, Callable, Dict, Tuple, cast
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import joblib
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection

FINGERS = [
    "left_pinky", "left_ring", "left_middle", "left_index",
    "right_index", "right_middle", "right_ring", "right_pinky"
]

FINGER_ERROR_COLUMNS = [f"error_{f}" for f in FINGERS]
FINGER_DWELL_COLUMNS = [f"dwell_{f}" for f in FINGERS]
FINGER_FLIGHT_COLUMNS = [f"flight_{f}" for f in FINGERS]

ALLOWED_WEAKEST_FINGER_LABELS = set(FINGERS)

FEATURE_RANGE_RULES: Dict[str, Tuple[float, float | None]] = {
    "wpm": (0.0, None),
    "accuracy": (0.0, 1.0),
    **{name: (0.0, 1.0) for name in FINGER_ERROR_COLUMNS},
    **{name: (0.0, None) for name in FINGER_DWELL_COLUMNS},
    **{name: (0.0, None) for name in FINGER_FLIGHT_COLUMNS},
}

def _load_model_artifact(model_path: str) -> tuple[Any, Dict[str, Any]]:
    """Loads the saved model and its metadata."""
    raw_artifact: Any = cast(Any, joblib).load(model_path)
    if isinstance(raw_artifact, dict):
        artifact = cast(Dict[str, Any], raw_artifact)
        if "model" in artifact:
            return artifact["model"], artifact

    legacy_model = cast(Any, raw_artifact)
    if hasattr(legacy_model, "predict"):
        artifact: Dict[str, Any] = {}
        feature_names_in = getattr(legacy_model, "feature_names_in_", None)
        if feature_names_in is not None and hasattr(feature_names_in, "tolist"):
            artifact["feature_names"] = list(feature_names_in)
        return legacy_model, artifact

    raise ValueError(
        "Model artifact is neither a dict containing 'model' nor a legacy "
        "predictable model object. Re-train with src/train.py to generate "
        "the current dict-based artifact format."
    )


def _validate_feature_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
    if feature_frame.empty:
        raise ValueError("Evaluation feature frame is empty.")

    validated = feature_frame.apply(pd.to_numeric, errors="coerce")

    invalid_numeric_columns = validated.columns[validated.isna().any()].tolist()
    if invalid_numeric_columns:
        raise ValueError(
            "Evaluation features contain null or non-numeric values in columns: "
            f"{invalid_numeric_columns}"
        )

    if not np.isfinite(validated.to_numpy(dtype=float)).all():
        raise ValueError(
            "Evaluation features contain non-finite values (inf or -inf)."
        )

    out_of_range_messages: list[str] = []
    for col_name, (min_value, max_value) in FEATURE_RANGE_RULES.items():
        if col_name not in validated.columns:
            # Model may be legacy or trained on different schema; we only validate known columns present.
            continue
        col_values = validated[col_name]
        if (col_values < min_value).any():
            out_of_range_messages.append(f"{col_name} < {min_value}")
        if max_value is not None and (col_values > max_value).any():
            out_of_range_messages.append(f"{col_name} > {max_value}")

    if out_of_range_messages:
        raise ValueError(
            "Evaluation features contain out-of-range values: "
            f"{out_of_range_messages}"
        )

    return validated


def _build_target_series(df: pd.DataFrame, target: str) -> pd.Series:
    raw_target = df[target]

    if raw_target.isna().any():
        raise ValueError(
            f"Evaluation target column '{target}' contains null values."
        )

    target_series = raw_target.astype(str).str.strip()
    if (target_series == "").any():
        raise ValueError(
            f"Evaluation target column '{target}' contains blank labels."
        )

    invalid_labels = sorted(set(target_series.unique()) - ALLOWED_WEAKEST_FINGER_LABELS)
    if invalid_labels:
        raise ValueError(
            f"Evaluation target column '{target}' contains unsupported labels: {invalid_labels}. "
            f"Allowed labels: {sorted(ALLOWED_WEAKEST_FINGER_LABELS)}"
        )

    class_counts = target_series.value_counts()
    if class_counts.size < 2:
        raise ValueError(
            f"Evaluation target column '{target}' must contain at least 2 classes. "
            f"Found classes: {class_counts.to_dict()}"
        )

    sparse_classes = class_counts[class_counts < 2]
    if not sparse_classes.empty:
        raise ValueError(
            "Each class in evaluation target must have at least 2 rows for stratified split. "
            f"Sparse classes: {sparse_classes.to_dict()}"
        )

    return target_series

def _select_features(df: pd.DataFrame, artifact: Dict[str, Any]) -> pd.DataFrame:
    """Dynamically extracts the exact 26 features the model was trained on."""
    expected_features_raw = artifact.get("feature_names", [])
    expected_features = cast(list[str], expected_features_raw)
    if not expected_features:
        raise ValueError(
            "Artifact does not contain 'feature_names'. This can happen with "
            "legacy model-only artifacts that do not expose schema metadata. "
            "Re-train with src/train.py to generate a dict artifact that includes "
            "feature_names."
        )

    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        raise ValueError(f"Evaluation dataset is missing required features: {missing}")

    selected = df[expected_features].copy()
    return _validate_feature_frame(selected)

def main(data_path: str, model_path: str, fig_dir: str, random_state: int = 42) -> None:
    print(f"Loading evaluation dataset from {data_path}...")
    df = pd.read_csv(data_path)

    if df.empty:
        raise ValueError("Evaluation dataset is empty. Provide at least one row.")

    target = "weakest_finger"
    if target not in df.columns:
        raise ValueError(
            f"Evaluation dataset is missing required target column '{target}'."
        )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    model, artifact = _load_model_artifact(model_path)

    # Automatically extracts the 26 features
    x_df = _select_features(df, artifact)
    y = _build_target_series(df, target)

    split_fn = cast(
        Callable[..., Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]],
        cast(Any, sk_model_selection).train_test_split,
    )

    # We use the provided random_state so we evaluate on a consistent split for that seed.
    try:
        _, x_test, _, y_test = split_fn(
            x_df, y, test_size=0.2, random_state=random_state, stratify=y
        )
    except ValueError as ex:
        raise ValueError(
            "Failed to create stratified evaluation split. "
            f"Class distribution: {y.value_counts().to_dict()}"
        ) from ex

    print("Generating predictions...")
    y_pred = model.predict(x_test)

    # Print report to console
    report_fn = cast(Callable[..., str], cast(Any, sk_metrics).classification_report)
    print("\n=== Classification Report ===")
    print(report_fn(y_test, y_pred))

    # Save confusion matrix plot for the Thesis!
    os.makedirs(fig_dir, exist_ok=True)
    cast(Any, sk_metrics).ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation=45)
    cast(Any, plt).title("Confusion Matrix - Weakest Finger (26 Features)")
    cast(Any, plt).tight_layout()

    plot_path = os.path.join(fig_dir, "confusion_matrix.png")
    cast(Any, plt).savefig(plot_path, dpi=200)
    cast(Any, plt).close()

    print(f"\nSuccess! Saved confusion matrix plot to: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/dataset.csv")
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--fig-dir", default="reports/figures")
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()
    main(args.data, args.model, args.fig_dir, args.random_state)
