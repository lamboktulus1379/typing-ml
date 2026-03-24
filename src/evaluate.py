
import os
import argparse
from typing import Any, Callable, Dict, List, Optional, Tuple, cast
import pandas as pd
import matplotlib.pyplot as plt

import joblib
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection

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


def with_error_rate(df: pd.DataFrame) -> pd.DataFrame:
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

    return out


def _load_model_artifact(model_path: str) -> Tuple[Any, Optional[Dict[str, Any]]]:
    raw_artifact: Any = cast(Any, joblib).load(model_path)
    if isinstance(raw_artifact, dict) and "model" in raw_artifact:
        artifact = cast(Dict[str, Any], raw_artifact)
        model: Any = artifact["model"]
        return model, artifact
    return cast(Any, raw_artifact), None


def _select_features(
    df: pd.DataFrame,
    target: str,
    artifact: Optional[Dict[str, Any]],
) -> pd.DataFrame:
    if artifact and artifact.get("feature_names"):
        feature_names = cast(List[str], artifact["feature_names"])
        missing = [name for name in feature_names if name not in df.columns]
        if missing:
            raise ValueError(f"Dataset is missing features required by model artifact: {missing}")
        return df[feature_names]

    return df.drop(columns=[target]).select_dtypes(include=["int64", "float64"])


def main(data_path: str, model_path: str, fig_dir: str, random_state: int) -> None:
    df = with_error_rate(pd.read_csv(data_path))

    target = "weakest_finger"
    model, artifact = _load_model_artifact(model_path)

    x_df = _select_features(df, target, artifact)
    y = df[target]

    split_fn = cast(
        Callable[..., Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]],
        cast(Any, sk_model_selection).train_test_split,
    )
    _, x_test, _, y_test = split_fn(
        x_df, y, test_size=0.2, random_state=random_state, stratify=y
    )

    y_pred = model.predict(x_test)

    # Print report to console
    report_fn = cast(Callable[..., str], cast(Any, sk_metrics).classification_report)
    print(report_fn(y_test, y_pred))

    # Save confusion matrix plot
    os.makedirs(fig_dir, exist_ok=True)
    cast(Any, sk_metrics).ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation=45)
    cast(Any, plt).title("Confusion Matrix - Weakest Finger")
    cast(Any, plt).tight_layout()
    cast(Any, plt).savefig(os.path.join(fig_dir, "confusion_matrix.png"), dpi=200)
    cast(Any, plt).close()

    print(f"Saved plots to {fig_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/dataset.csv")
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--fig-dir", default="reports/figures")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    main(args.data, args.model, args.fig_dir, args.random_state)
