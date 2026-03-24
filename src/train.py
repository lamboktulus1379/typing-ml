
import os
import json
import argparse
import datetime
from typing import Any, Callable, Dict, Tuple, cast
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection
import joblib

FINGER_ERROR_COLUMNS = [
    "error_left_pinky",
    "error_left_ring",
    "error_left_middle",
    "error_left_index",
    "error_right_index",
    "error_right_middle",
    "error_right_ring",
    "error_right_pinky",
]

TRAIN_FEATURE_COLUMNS = [
    "wpm",
    "accuracy",
    "error_rate",
    *FINGER_ERROR_COLUMNS,
]


def with_error_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df that always contains an error_rate column.

    Priority:
    1) Keep existing error_rate if present.
    2) Derive from total_errors / total_keypresses if available.
    3) Derive as mean of per-finger error columns.
    """
    out = df.copy()

    if "error_rate" in out.columns:
        out["error_rate"] = pd.to_numeric(out["error_rate"], errors="coerce")
        return out

    if {"total_errors", "total_keypresses"}.issubset(out.columns):
        total_errors = pd.to_numeric(out["total_errors"], errors="coerce")
        total_keypresses = pd.to_numeric(out["total_keypresses"], errors="coerce").clip(lower=1)
        out["error_rate"] = (total_errors / total_keypresses).clip(lower=0, upper=1)
        return out

    if set(FINGER_ERROR_COLUMNS).issubset(out.columns):
        finger_errors = out[FINGER_ERROR_COLUMNS].apply(pd.to_numeric, errors="coerce")
        out["error_rate"] = finger_errors.mean(axis=1).clip(lower=0, upper=1)
        return out

    raise ValueError(
        "Cannot build error_rate feature. Provide 'error_rate' or columns for derivation "
        "(total_errors + total_keypresses, or per-finger error_* columns)."
    )

def _build_feature_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    missing_features = [c for c in TRAIN_FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    features = df[TRAIN_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if features.isnull().any().any():
        null_counts = features.isnull().sum()
        bad_columns = [name for name, count in null_counts.items() if count > 0]
        raise ValueError(f"Found non-numeric/null values in required features: {bad_columns}")

    return features


def main(data_path: str, model_out: str, report_out: str, random_state: int) -> None:
    df = with_error_rate(pd.read_csv(data_path))

    # --- Define target + features ---
    target = "weakest_finger"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    x_df = _build_feature_frame(df, target)
    y = df[target]
    feature_names = list(TRAIN_FEATURE_COLUMNS)

    # --- Train/test split ---
    split_fn = cast(
        Callable[..., Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]],
        cast(Any, sk_model_selection).train_test_split,
    )
    x_train, x_test, y_train, y_test = split_fn(
        x_df, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # --- Model pipeline ---
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model_any = cast(Any, model)
    model_any.fit(x_train, y_train)

    # --- Quick evaluation during training ---
    y_pred = model_any.predict(x_test)
    report_fn = cast(Callable[..., Dict[str, Any]], cast(Any, sk_metrics).classification_report)
    report = report_fn(y_test, y_pred, output_dict=True)

    # --- Save model artifact (model + metadata) ---
    artifact: Dict[str, Any] = {
        "model": model,
        "feature_names": feature_names,
        "target_name": target,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    cast(Any, joblib).dump(artifact, model_out)

    # --- Save report JSON (for later use) ---
    os.makedirs(os.path.dirname(report_out), exist_ok=True)
    with open(report_out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved model to: {model_out}")
    print(f"Saved training report to: {report_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/dataset.csv")
    parser.add_argument("--model-out", default="models/model.joblib")
    parser.add_argument("--report-out", default="reports/train_report.json")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    main(args.data, args.model_out, args.report_out, args.random_state)
