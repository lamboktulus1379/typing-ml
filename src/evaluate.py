import os
import argparse
from typing import Any, Callable, Dict, Tuple, cast
import pandas as pd
import matplotlib.pyplot as plt

import joblib
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection

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

    return df[expected_features].copy()

def main(data_path: str, model_path: str, fig_dir: str, random_state: int = 42) -> None:
    print(f"Loading evaluation dataset from {data_path}...")
    df = pd.read_csv(data_path)

    target = "weakest_finger"
    if target not in df.columns:
        raise ValueError(
            f"Evaluation dataset is missing required target column '{target}'."
        )

    model, artifact = _load_model_artifact(model_path)

    # Automatically extracts the 26 features
    x_df = _select_features(df, artifact)
    y = df[target]

    split_fn = cast(
        Callable[..., Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]],
        cast(Any, sk_model_selection).train_test_split,
    )

    # We use the provided random_state so we evaluate on a consistent split for that seed.
    _, x_test, _, y_test = split_fn(
        x_df, y, test_size=0.2, random_state=random_state, stratify=y
    )

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
