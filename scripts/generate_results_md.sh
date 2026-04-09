#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REPORT_JSON="${1:-reports/training_report.json}"
DATASET_CSV="${2:-data/processed/dataset.csv}"
OUT_MD="${3:-reports/results_filled_latest.md}"
MODEL_PATH="${4:-models/model.joblib}"
FIG_PATH="${5:-reports/figures/confusion_matrix.png}"
SEED_VALUE="${6:-42}"

if [[ ! -f "$REPORT_JSON" ]]; then
  echo "Error: report file not found: $REPORT_JSON"
  exit 1
fi

if [[ ! -f "$DATASET_CSV" ]]; then
  echo "Error: dataset file not found: $DATASET_CSV"
  exit 1
fi

python3 - "$REPORT_JSON" "$DATASET_CSV" "$OUT_MD" "$MODEL_PATH" "$FIG_PATH" "$SEED_VALUE" <<'PY'
import json
import csv
import datetime
import sys
from pathlib import Path

report_json = Path(sys.argv[1])
dataset_csv = Path(sys.argv[2])
out_md = Path(sys.argv[3])
model_path = sys.argv[4]
fig_path = sys.argv[5]
seed_value = sys.argv[6]

with report_json.open("r", encoding="utf-8") as f:
    report = json.load(f)

required_top_level = ["selected_model", "accuracy", "macro avg", "weighted avg"]
missing_top_level = [k for k in required_top_level if k not in report]
if missing_top_level:
    raise SystemExit(
        "Training report is missing required keys: "
        f"{missing_top_level}. Re-run training first."
    )

for aggregate_key in ("macro avg", "weighted avg"):
    if not isinstance(report.get(aggregate_key), dict):
        raise SystemExit(
            f"Training report key '{aggregate_key}' must be an object. "
            "Re-run training to regenerate report JSON."
        )

with dataset_csv.open("r", encoding="utf-8", newline="") as f:
    row_count = sum(1 for _ in csv.reader(f)) - 1
if row_count < 0:
    row_count = 0

selected_model = report.get("selected_model", "<unknown>")
accuracy = float(report.get("accuracy", 0.0))
macro = report.get("macro avg", {})
weighted = report.get("weighted avg", {})
candidates = report.get("candidate_accuracies", {})
if candidates is not None and not isinstance(candidates, dict):
    raise SystemExit(
        "Training report key 'candidate_accuracies' must be an object when present."
    )

classes = [
    "left_pinky", "left_ring", "left_middle", "left_index",
    "right_index", "right_middle", "right_ring", "right_pinky",
]

def fmt(x):
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "<n/a>"

def support_val(x):
    try:
        return str(int(round(float(x))))
    except Exception:
        return "<n/a>"

lines = []
lines.append("# Typing-ML Filled Results (Latest Run)")
lines.append("")
lines.append(f"Generated on: {datetime.datetime.now().isoformat(timespec='seconds')}")
lines.append("")
lines.append("## A. Experiment Setup")
lines.append("")
lines.append("| Field | Value |")
lines.append("|---|---|")
lines.append(f"| Dataset path | `{dataset_csv.as_posix()}` |")
lines.append(f"| Number of rows | `{row_count:,}` |")
lines.append(f"| Random seed | `{seed_value}` |")
lines.append("| Train-test split | `80:20 (stratified)` |")
lines.append("| Candidate models | `logistic_regression`, `random_forest` |")
lines.append(f"| Selected model | `{selected_model}` |")
lines.append(f"| Model artifact | `{model_path}` |")
lines.append(f"| Figure path | `{fig_path}` |")
lines.append("")
lines.append("## B. Performance Summary")
lines.append("")
lines.append("| Metric | Value |")
lines.append("|---|---|")
lines.append(f"| Accuracy | `{fmt(accuracy)}` |")
lines.append(f"| Macro Precision | `{fmt(macro.get('precision'))}` |")
lines.append(f"| Macro Recall | `{fmt(macro.get('recall'))}` |")
lines.append(f"| Macro F1-score | `{fmt(macro.get('f1-score'))}` |")
lines.append(f"| Weighted F1-score | `{fmt(weighted.get('f1-score'))}` |")
lines.append("")
lines.append("## C. Per-Class Metrics")
lines.append("")
lines.append("| Class | Precision | Recall | F1-score | Support |")
lines.append("|---|---:|---:|---:|---:|")
for c in classes:
    entry = report.get(c, {})
    lines.append(
        f"| {c} | {fmt(entry.get('precision'))} | {fmt(entry.get('recall'))} | "
        f"{fmt(entry.get('f1-score'))} | {support_val(entry.get('support'))} |"
    )

if isinstance(candidates, dict) and candidates:
    lines.append("")
    lines.append("## D. Candidate Model Accuracy")
    lines.append("")
    lines.append("| Model | Holdout Accuracy |")
    lines.append("|---|---:|")
    for model_name, score in candidates.items():
        lines.append(f"| {model_name} | {fmt(score)} |")

lines.append("")
lines.append("## E. English Narrative (Filled)")
lines.append("")
lines.append(
    f"The selected model for this experiment was `{selected_model}`, chosen based on holdout selection logic."
)
lines.append(f"The model achieved an overall accuracy of `{fmt(accuracy)}` on the test set.")
lines.append(
    "Class-wise metrics can be interpreted from the table above to identify the strongest and most challenging finger labels."
)
lines.append("The confusion matrix figure is available at the path shown in the setup table.")

lines.append("")
lines.append("## F. Narasi Bahasa Indonesia (Terisi)")
lines.append("")
lines.append(
    f"Model terpilih pada eksperimen ini adalah `{selected_model}`, berdasarkan logika pemilihan holdout."
)
lines.append(f"Model menghasilkan akurasi keseluruhan sebesar `{fmt(accuracy)}` pada data uji.")
lines.append(
    "Interpretasi per kelas dapat dilihat pada tabel metrik per kelas untuk menentukan label yang paling kuat dan paling menantang."
)
lines.append("Gambar confusion matrix tersedia pada path yang tercantum di tabel setup.")

lines.append("")
lines.append("## G. Figure Caption")
lines.append("")
lines.append(
    f"Figure X. Confusion Matrix of weakest-finger classification using `{selected_model}` on the test set "
    f"(seed=`{seed_value}`, split=`80:20 stratified`)."
)

out_md.parent.mkdir(parents=True, exist_ok=True)
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote filled results report to: {out_md.as_posix()}")
PY
