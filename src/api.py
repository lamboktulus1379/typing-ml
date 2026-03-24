"""FastAPI inference service for typing-ml.

This module exposes a small REST API to load a trained sklearn model and run
predictions.

Key ideas for beginners:
- The API loads the model once at startup (fast per-request).
- Input validation is done with Pydantic models (schema + types).
- We keep a strict feature schema (feature_names) to match training.

Run:
    uvicorn src.api:app --reload --port 8000

Then open:
    http://127.0.0.1:8000/docs
"""

import os
from typing import Any, Dict, List, Optional, cast

import pandas as pd
from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel, Field


def load_model_artifact(model_path: str) -> tuple[Any, Dict[str, Any]]:
    """Load a model artifact from disk.

    New format (recommended): a dict containing:
    - model: sklearn estimator or Pipeline
    - feature_names: list[str] used in training
    - target_name: name of the prediction target

    Old format (legacy): the sklearn Pipeline itself.
    """
    raw_artifact: Any = cast(Any, joblib).load(model_path)
    if isinstance(raw_artifact, dict) and "model" in raw_artifact:
        artifact = cast(Dict[str, Any], raw_artifact)
        model: Any = artifact["model"]
        return model, artifact

    # Backward compatibility: old joblib contained only the sklearn pipeline
    model: Any = cast(Any, raw_artifact)
    feature_names = getattr(model, "feature_names_in_", None)
    meta: Dict[str, Any] = {
        "model": "<legacy_pipeline>",
        "feature_names": list(feature_names) if feature_names is not None else None,
        "target_name": "weakest_finger",
    }
    return model, meta


class PredictRequest(BaseModel):
    """Request payload for single-row prediction."""

    features: Dict[str, float] = Field(
        ..., description="Numeric feature map, e.g. {'wpm': 55.2, 'accuracy': 0.93, ...}"
    )


class PredictBatchRequest(BaseModel):
    """Request payload for batch prediction."""

    rows: List[Dict[str, float]] = Field(
        ..., description="List of feature maps, each like PredictRequest.features"
    )


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="typing-ml",
        version="0.1.0",
        description=(
            "REST API for testing the weakest-finger classifier. "
            "See /metadata for required features and /docs for interactive Swagger UI."
        ),
    )

    model_path = os.getenv("TYPING_ML_MODEL_PATH", "models/model.joblib")
    model, artifact = load_model_artifact(model_path)

    feature_names: Optional[List[str]] = artifact.get("feature_names")

    def build_frame(rows: List[Dict[str, float]]) -> pd.DataFrame:
        """Convert incoming JSON feature maps into a pandas DataFrame.

        If the artifact provides feature_names, we require all of them and keep
        the exact column order from training.
        """
        if feature_names:
            missing = sorted({k for k in feature_names if any(k not in r for r in rows)})
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Missing required features",
                        "missing": missing,
                    },
                )
            # Build in exact training column order
            return pd.DataFrame(rows, columns=feature_names)

        # No saved schema: best-effort ordering by sorted keys
        all_keys = sorted({k for r in rows for k in r.keys()})
        return pd.DataFrame(rows, columns=all_keys)

    @app.get(
        "/health",
        summary="Health check",
        description="Returns OK if the server is running.",
    )
    def health() -> Dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        return {"status": "ok"}

    @app.get(
        "/metadata",
        summary="Model metadata",
        description="Returns required feature names, classes, and artifact info.",
    )
    def metadata() -> Dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        classes = getattr(getattr(model, "named_steps", {}).get("clf", model), "classes_", None)
        return {
            "model_path": model_path,
            "target_name": artifact.get("target_name", "weakest_finger"),
            "feature_names": feature_names,
            "classes": list(classes) if classes is not None else None,
            "created_at": artifact.get("created_at"),
        }

    @app.post(
        "/predict",
        summary="Predict weakest finger (single row)",
        description=(
            "Send one feature map and get one predicted label. "
            "If the model supports predict_proba, class probabilities are returned too."
        ),
    )
    def predict(req: PredictRequest) -> Dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        X = build_frame([req.features])

        pred = model.predict(X)[0]
        result: Dict[str, Any] = {"prediction": pred}

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            classes = getattr(model, "classes_", None)
            if classes is None:
                # For Pipeline, classes live on the classifier step
                clf = getattr(model, "named_steps", {}).get("clf")
                classes = getattr(clf, "classes_", None)
            if classes is not None:
                result["probabilities"] = {str(c): float(p) for c, p in zip(classes, probs)}

        return result

    @app.post(
        "/predict_batch",
        summary="Predict weakest finger (batch)",
        description="Send multiple rows and get predictions (and optional probabilities).",
    )
    def predict_batch(req: PredictBatchRequest) -> Dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        if not req.rows:
            raise HTTPException(status_code=400, detail={"error": "rows must be non-empty"})

        X = build_frame(req.rows)
        preds = model.predict(X)

        out: Dict[str, Any] = {"predictions": [p for p in preds]}

        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X)
            clf = getattr(model, "named_steps", {}).get("clf", model)
            classes = getattr(clf, "classes_", None)
            if classes is not None:
                out["probabilities"] = [
                    {str(c): float(p) for c, p in zip(classes, row)} for row in probas
                ]

        return out

    return app


app = create_app()
