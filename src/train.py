import os
import json
import argparse
import datetime
from typing import Any, Callable, Dict, Tuple, cast
import pandas as pd

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

def _build_feature_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    missing_cols = [c for c in TRAIN_FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
    return df[TRAIN_FEATURE_COLUMNS].copy()


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

    target = "weakest_finger"
    x_df = _build_feature_frame(df, target)
    y = df[target]

    split_fn = cast(
        Callable[..., Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]],
        cast(Any, sk_model_selection).train_test_split,
    )
    x_train, x_test, y_train, y_test = split_fn(
        x_df, y, test_size=0.2, random_state=random_state, stratify=y
    )

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

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    cast(Any, joblib).dump(artifact, model_out)

    os.makedirs(os.path.dirname(report_out), exist_ok=True)
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
