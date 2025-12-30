
import os
import json
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from joblib import dump

def main(data_path: str, model_out: str, report_out: str, random_state: int):
    df = pd.read_csv(data_path)

    # --- Define target + features ---
    target = "weakest_finger"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    X = df.drop(columns=[target])
    y = df[target]

    # Keep only numeric features for first baseline (simple)
    X = X.select_dtypes(include=["int64", "float64"])

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # --- Model pipeline ---
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    # --- Quick evaluation during training ---
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    # --- Save model ---
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    dump(model, model_out)

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
