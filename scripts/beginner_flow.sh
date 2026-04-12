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
ALGORITHM_VALUE="logistic_regression"
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
  --algorithm <name>            logistic_regression | random_forest | xgboost
                                 (default: logistic_regression)
  --model-type <name>           Deprecated alias for --algorithm
  --data-path <path>            Dataset output/input path
  --model-path <path>           Trained model artifact path
  --report-path <path>          Training report output path
  --fig-dir <path>              Evaluation figure output folder
  --skip-install                Skip dependency installation step
  -h, --help                    Show this help

Examples:
  bash scripts/beginner_flow.sh
  bash scripts/beginner_flow.sh --users 200 --sessions 10 --algorithm random_forest
EOF
}

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_integer() {
  [[ "$1" =~ ^-?[0-9]+$ ]]
}

validate_inputs() {
  if ! is_positive_int "$N_USERS_VALUE"; then
    echo "Error: --users must be a positive integer, got: $N_USERS_VALUE"
    exit 1
  fi

  if ! is_positive_int "$SESSIONS_PER_USER_VALUE"; then
    echo "Error: --sessions must be a positive integer, got: $SESSIONS_PER_USER_VALUE"
    exit 1
  fi

  if ! is_integer "$SEED_VALUE"; then
    echo "Error: --seed must be an integer, got: $SEED_VALUE"
    exit 1
  fi

  case "$ALGORITHM_VALUE" in
    logistic_regression|random_forest|xgboost) ;;
    *)
      echo "Error: --algorithm must be one of: logistic_regression, random_forest, xgboost"
      echo "       Received: $ALGORITHM_VALUE"
      exit 1
      ;;
  esac

  for value_and_name in \
    "$DATA_PATH_VALUE:--data-path" \
    "$MODEL_PATH_VALUE:--model-path" \
    "$REPORT_PATH_VALUE:--report-path" \
    "$FIG_DIR_VALUE:--fig-dir"; do
    value="${value_and_name%%:*}"
    name="${value_and_name##*:}"
    if [[ -z "${value// }" ]]; then
      echo "Error: $name must be a non-empty path"
      exit 1
    fi
  done
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
      ALGORITHM_VALUE="$2"
      shift 2
      ;;
    --algorithm)
      ALGORITHM_VALUE="$2"
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

validate_inputs

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
echo "Algorithm:    $ALGORITHM_VALUE"
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
  ALGORITHM="$ALGORITHM_VALUE" \
  train

echo "[4/4] Evaluating model..."
make \
  CONDA_ENV="$CONDA_ENV_NAME" \
  DATA_PATH="$DATA_PATH_VALUE" \
  MODEL_PATH="$MODEL_PATH_VALUE" \
  FIG_DIR="$FIG_DIR_VALUE" \
  SEED="$SEED_VALUE" \
  evaluate

EXPECTED_FIGURE="$FIG_DIR_VALUE/confusion_matrix.png"
missing_artifacts=()
[[ -f "$DATA_PATH_VALUE" ]] || missing_artifacts+=("$DATA_PATH_VALUE")
[[ -f "$MODEL_PATH_VALUE" ]] || missing_artifacts+=("$MODEL_PATH_VALUE")
[[ -f "$REPORT_PATH_VALUE" ]] || missing_artifacts+=("$REPORT_PATH_VALUE")
[[ -f "$EXPECTED_FIGURE" ]] || missing_artifacts+=("$EXPECTED_FIGURE")

if (( ${#missing_artifacts[@]} > 0 )); then
  echo ""
  echo "Error: flow completed but expected artifacts were not found:"
  for artifact in "${missing_artifacts[@]}"; do
    echo "- $artifact"
  done
  exit 1
fi

echo ""
echo "Done. Artifacts generated:"
echo "- Dataset:   $DATA_PATH_VALUE"
echo "- Model:     $MODEL_PATH_VALUE"
echo "- Report:    $REPORT_PATH_VALUE"
echo "- Figures:   $EXPECTED_FIGURE"
