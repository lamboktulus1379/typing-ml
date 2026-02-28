
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


def _load_model_artifact(model_path: str):
    artifact = load(model_path)
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact["model"], artifact
    return artifact, None

def main(data_path: str, model_path: str, fig_dir: str, random_state: int):
    df = pd.read_csv(data_path)

    target = "weakest_finger"
    model, artifact = _load_model_artifact(model_path)

    if artifact and artifact.get("feature_names"):
        feature_names = artifact["feature_names"]
        X = df[feature_names]
    else:
        X = df.drop(columns=[target]).select_dtypes(include=["int64", "float64"])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    y_pred = model.predict(X_test)

    # Print report to console
    print(classification_report(y_test, y_pred))

    # Save confusion matrix plot
    os.makedirs(fig_dir, exist_ok=True)
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation=45)
    plt.title("Confusion Matrix - Weakest Finger")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

    print(f"Saved plots to {fig_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/dataset.csv")
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--fig-dir", default="reports/figures")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    main(args.data, args.model, args.fig_dir, args.random_state)
