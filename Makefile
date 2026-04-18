CONDA_ENV ?= typing-ml
PYTHON ?= conda run -n $(CONDA_ENV) python
APP_MODULE ?= src.api:app
HOST ?= 127.0.0.1
PORT ?= 8000
DATA_PATH ?= data/processed/dataset.csv
MODEL_PATH ?= models/model.joblib
REPORT_PATH ?= reports/training_report.json
FIG_DIR ?= reports/figures
N_USERS ?= 500
SESSIONS_PER_USER ?= 20
SEED ?= 42
ALGORITHM ?= logistic_regression
RESULTS_MD ?= reports/results_filled_latest.md
COMPARE_SUMMARY_PATH ?= reports/algorithm_comparison.json
COMPARE_MARKDOWN_PATH ?= reports/algorithm_comparison.md
COMPARE_LATENCY_RUNS ?= 50

.PHONY: help install dev run generate-synthetic train evaluate compare-algorithms ml-pipeline e2e refresh-results e2e-report test-api test

help:
	@echo "Available targets:"
	@echo "  make CONDA_ENV=typing-ml <target> - Run target in a Conda environment"
	@echo "  make install  - Install FastAPI runtime dependencies"
	@echo "  make dev      - Run FastAPI with auto-reload (local dev)"
	@echo "  make run      - Run FastAPI without reload"
	@echo "  make generate-synthetic - Generate synthetic dataset"
	@echo "  make train     - Train model from dataset"
	@echo "    - ALGORITHM=logistic_regression|random_forest|xgboost (default: logistic_regression)"
	@echo "  make evaluate  - Evaluate model and save confusion matrix"
	@echo "  make compare-algorithms - Train and compare logistic_regression, random_forest, xgboost"
	@echo "  make ml-pipeline - Generate data, train, and evaluate"
	@echo "  make e2e       - Full end-to-end flow (standard thesis pipeline)"
	@echo "  make refresh-results - Build reports/results_filled_latest.md from latest JSON"
	@echo "  make e2e-report - Run e2e then refresh thesis-ready results markdown"
	@echo "  make test-api  - Run Python API end-to-end tests"
	@echo "  make test      - Alias for test-api"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install fastapi "uvicorn[standard]" pandas scikit-learn joblib matplotlib pytest httpx \
		opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http \
		opentelemetry-instrumentation-fastapi opentelemetry-instrumentation-requests

dev:
	$(PYTHON) -m uvicorn $(APP_MODULE) --reload --host $(HOST) --port $(PORT)

run:
	$(PYTHON) -m uvicorn $(APP_MODULE) --host $(HOST) --port $(PORT)

generate-synthetic:
	$(PYTHON) src/generate_synthetic_data.py --n-users $(N_USERS) --sessions-per-user $(SESSIONS_PER_USER) --seed $(SEED) --output $(DATA_PATH)

train:
	$(PYTHON) src/train.py --data $(DATA_PATH) --model-out $(MODEL_PATH) --report-out $(REPORT_PATH) --random-state $(SEED) --algorithm $(ALGORITHM)

evaluate:
	$(PYTHON) src/evaluate.py --data $(DATA_PATH) --model $(MODEL_PATH) --fig-dir $(FIG_DIR) --random-state $(SEED)

compare-algorithms:
	$(PYTHON) src/compare_algorithms.py --data $(DATA_PATH) --model-dir models --report-dir reports --summary-out $(COMPARE_SUMMARY_PATH) --markdown-out $(COMPARE_MARKDOWN_PATH) --random-state $(SEED) --latency-runs $(COMPARE_LATENCY_RUNS)

ml-pipeline: generate-synthetic train evaluate

e2e: ml-pipeline

refresh-results:
	bash scripts/generate_results_md.sh $(REPORT_PATH) $(DATA_PATH) $(RESULTS_MD) $(MODEL_PATH) $(FIG_DIR)/confusion_matrix.png $(SEED)

e2e-report: e2e refresh-results

test-api:
	$(PYTHON) -m pytest tests/api/test_api_e2e.py -q

test: test-api
