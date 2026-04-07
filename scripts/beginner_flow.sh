#!/usr/bin/env bash
set -euo pipefail

# One-command beginner flow for typing-ml:
# 1) install runtime deps
# 2) generate synthetic data
# 3) train model
# 4) evaluate model

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV_NAME="typing-ml"
N_USERS_VALUE=500
SESSIONS_PER_USER_VALUE=20
SEED_VALUE=42
MODEL_TYPE_VALUE="auto"
DATA_PATH_VALUE="data/processed/dataset.csv"
MODEL_PATH_VALUE="models/model.joblib"
REPORT_PATH_VALUE="reports/training_report.json"
FIG_DIR_VALUE="reports/figures"
SKIP_INSTALL=false

usage() {
  cat <<EOF
Usage: bash scripts/beginner_flow.sh [options]

Options:
  --env <name>                  Conda environment name (default: typing-ml)
  --users <n>                   Number of users for synthetic data (default: 500)
  --sessions <n>                Sessions per user (default: 20)
  --seed <n>                    Random seed (default: 42)
  --model-type <name>           auto | logistic_regression | random_forest (default: auto)
  --data-path <path>            Dataset output/input path
  --model-path <path>           Trained model artifact path
  --report-path <path>          Training report output path
  --fig-dir <path>              Evaluation figure output folder
  --skip-install                Skip dependency installation step
  -h, --help                    Show this help

Examples:
  bash scripts/beginner_flow.sh
  bash scripts/beginner_flow.sh --users 200 --sessions 10 --model-type random_forest
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      CONDA_ENV_NAME="$2"
      shift 2
      ;;
    --users)
      N_USERS_VALUE="$2"
      shift 2
      ;;
    --sessions)
      SESSIONS_PER_USER_VALUE="$2"
      shift 2
      ;;
    --seed)
      SEED_VALUE="$2"
      shift 2
      ;;
    --model-type)
      MODEL_TYPE_VALUE="$2"
      shift 2
      ;;
    --data-path)
      DATA_PATH_VALUE="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH_VALUE="$2"
      shift 2
      ;;
    --report-path)
      REPORT_PATH_VALUE="$2"
      shift 2
      ;;
    --fig-dir)
      FIG_DIR_VALUE="$2"
      shift 2
      ;;
    --skip-install)
      SKIP_INSTALL=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda command not found. Install Miniconda/Conda first."
  exit 1
fi

conda_env_exists() {
  conda env list 2>/dev/null | awk 'NR>2 {print $1}' | grep -Fxq "$1"
}

if ! conda_env_exists "$CONDA_ENV_NAME"; then
  echo "Error: Conda environment '$CONDA_ENV_NAME' does not exist."
  echo "Create it first, for example:"
  echo "  conda env create -n \"$CONDA_ENV_NAME\" -f environment.yml"
  exit 1
fi

echo "=== Typing-ML Beginner Flow ==="
echo "Project root: $ROOT_DIR"
echo "Conda env:    $CONDA_ENV_NAME"
echo "Model type:   $MODEL_TYPE_VALUE"
echo "Data path:    $DATA_PATH_VALUE"
echo "Model path:   $MODEL_PATH_VALUE"
echo "Report path:  $REPORT_PATH_VALUE"
echo "Figure dir:   $FIG_DIR_VALUE"

if [[ "$SKIP_INSTALL" == false ]]; then
  echo "[1/4] Installing dependencies..."
  make CONDA_ENV="$CONDA_ENV_NAME" install
else
  echo "[1/4] Skipped dependency installation (--skip-install)."
fi

echo "[2/4] Generating synthetic dataset..."
make \
  CONDA_ENV="$CONDA_ENV_NAME" \
  N_USERS="$N_USERS_VALUE" \
  SESSIONS_PER_USER="$SESSIONS_PER_USER_VALUE" \
  SEED="$SEED_VALUE" \
  DATA_PATH="$DATA_PATH_VALUE" \
  generate-synthetic

echo "[3/4] Training model..."
make \
  CONDA_ENV="$CONDA_ENV_NAME" \
  DATA_PATH="$DATA_PATH_VALUE" \
  MODEL_PATH="$MODEL_PATH_VALUE" \
  REPORT_PATH="$REPORT_PATH_VALUE" \
  SEED="$SEED_VALUE" \
  MODEL_TYPE="$MODEL_TYPE_VALUE" \
  train

echo "[4/4] Evaluating model..."
make \
  CONDA_ENV="$CONDA_ENV_NAME" \
  DATA_PATH="$DATA_PATH_VALUE" \
  MODEL_PATH="$MODEL_PATH_VALUE" \
  FIG_DIR="$FIG_DIR_VALUE" \
  SEED="$SEED_VALUE" \
  evaluate

echo ""
echo "Done. Artifacts generated:"
echo "- Dataset:   $DATA_PATH_VALUE"
echo "- Model:     $MODEL_PATH_VALUE"
echo "- Report:    $REPORT_PATH_VALUE"
echo "- Figures:   $FIG_DIR_VALUE"
