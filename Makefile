CONDA_ENV ?= typing-ml
PYTHON ?= conda run -n $(CONDA_ENV) python
APP_MODULE ?= src.api:app
HOST ?= 127.0.0.1
PORT ?= 8000

.PHONY: help install dev run

help:
	@echo "Available targets:"
	@echo "  make CONDA_ENV=typing-ml <target> - Run target in a Conda environment"
	@echo "  make install  - Install FastAPI runtime dependencies"
	@echo "  make dev      - Run FastAPI with auto-reload (local dev)"
	@echo "  make run      - Run FastAPI without reload"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install fastapi "uvicorn[standard]" pandas scikit-learn joblib

dev:
	$(PYTHON) -m uvicorn $(APP_MODULE) --reload --host $(HOST) --port $(PORT)

run:
	$(PYTHON) -m uvicorn $(APP_MODULE) --host $(HOST) --port $(PORT)
