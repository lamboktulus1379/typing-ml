CONDA_ENV ?= typing-ml
PYTHON ?= conda run -n $(CONDA_ENV) python
APP_MODULE ?= src.api:app
HOST ?= 127.0.0.1
PORT ?= 8000

.PHONY: help install dev run worker check-worker-env

help:
	@echo "Available targets:"
	@echo "  make CONDA_ENV=typing-ml <target> - Run target in a Conda environment"
	@echo "  make install  - Install runtime dependencies from requirements.txt"
	@echo "  make dev      - Run FastAPI with auto-reload (local dev)"
	@echo "  make run      - Run FastAPI without reload"
	@echo "  make worker   - Run the Kafka/Redis background worker"
	@echo "  make check-worker-env - Verify worker dependencies in the Conda env"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

dev:
	$(PYTHON) -m uvicorn $(APP_MODULE) --reload --host $(HOST) --port $(PORT)

run:
	$(PYTHON) -m uvicorn $(APP_MODULE) --host $(HOST) --port $(PORT)

worker:
	$(PYTHON) -m src.worker

check-worker-env:
	$(PYTHON) -c "import confluent_kafka, redis, fastapi; print('worker env ok')"
