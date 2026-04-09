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
import logging
import json
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, cast

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
import joblib
from pydantic import BaseModel


MAX_TRACE_PAYLOAD_CHARS = int(os.getenv("TYPING_ML_TRACE_PAYLOAD_CHARS", "4096"))
logger = logging.getLogger("typing-ml")


def truncate_payload(raw: bytes | str, max_chars: int = MAX_TRACE_PAYLOAD_CHARS) -> str:
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...(truncated)"


def compact_json_if_possible(payload: str) -> str:
    try:
        return json.dumps(json.loads(payload), separators=(",", ":"))
    except json.JSONDecodeError:
        return payload


def configure_optional_otel(app: FastAPI) -> None:
    """Enable OTLP tracing when OpenTelemetry packages are available.

    Aspire injects standard OTEL_* environment variables when WithOtlpExporter()
    is configured on this resource.
    """
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not otlp_endpoint:
        logger.info("OpenTelemetry endpoint is not configured; skipping exporter setup.")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
    except Exception as ex:  # pragma: no cover - optional dependency path
        logger.info("OpenTelemetry packages are not installed, payload tracing will stay log-only: %s", ex)
        return

    service_name = os.getenv("OTEL_SERVICE_NAME", "typing-ml-api")
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
    RequestsInstrumentor().instrument()
    logger.info("OpenTelemetry tracing enabled for typing-ml FastAPI service")


def start_model_internal_span(span_name: str, attributes: Optional[Dict[str, Any]] = None):
    """Start an internal model span when OpenTelemetry is available."""
    try:
        from opentelemetry import trace

        tracer = trace.get_tracer("typing-ml.model")
        return tracer.start_as_current_span(span_name, attributes=attributes or {})
    except Exception:
        return nullcontext(None)


def start_model_inference_span(span_name: str, row_count: int, feature_count: int):
    """Start an internal inference span when OpenTelemetry is available."""
    return start_model_internal_span(
        span_name,
        attributes={
            "typing.ml.model.rows": row_count,
            "typing.ml.model.feature_count": feature_count,
        },
    )


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


class TypingSessionData(BaseModel):
    wpm: float
    accuracy: float
    error_left_pinky: float
    error_left_ring: float
    error_left_middle: float
    error_left_index: float
    error_right_index: float
    error_right_middle: float
    error_right_ring: float
    error_right_pinky: float
    dwell_left_pinky: float
    dwell_left_ring: float
    dwell_left_middle: float
    dwell_left_index: float
    dwell_right_index: float
    dwell_right_middle: float
    dwell_right_ring: float
    dwell_right_pinky: float
    flight_left_pinky: float
    flight_left_ring: float
    flight_left_middle: float
    flight_left_index: float
    flight_right_index: float
    flight_right_middle: float
    flight_right_ring: float
    flight_right_pinky: float

class PredictRequest(BaseModel):
    row: TypingSessionData

class PredictBatchRequest(BaseModel):
    rows: List[TypingSessionData]


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

    if not logging.getLogger().handlers:
        logging.basicConfig(level=os.getenv("TYPING_ML_LOG_LEVEL", "INFO"))

    configure_optional_otel(app)

    @app.middleware("http")
    async def trace_payload_middleware(request: Request, call_next):
        raw_request_body = await request.body()

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": raw_request_body, "more_body": False}

        request_with_body = Request(request.scope, receive)
        response = await call_next(request_with_body)

        response_chunks = []
        async for chunk in response.body_iterator:
            response_chunks.append(chunk)
        raw_response_body = b"".join(response_chunks)

        request_payload = compact_json_if_possible(truncate_payload(raw_request_body))
        response_payload = compact_json_if_possible(truncate_payload(raw_response_body))

        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            if span is not None and span.is_recording():
                span.set_attribute("typing.ml.http.method", request.method)
                span.set_attribute("typing.ml.http.path", request.url.path)
                span.set_attribute("typing.ml.http.request.payload", request_payload)
                span.set_attribute("typing.ml.http.response.payload", response_payload)
                span.set_attribute("typing.ml.http.status_code", response.status_code)
        except Exception:
            # Optional OpenTelemetry path; logs still contain payload data.
            pass

        logger.info(
            "typing-ml request method=%s path=%s status=%s request=%s response=%s",
            request.method,
            request.url.path,
            response.status_code,
            request_payload,
            response_payload,
        )

        return Response(
            content=raw_response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
            background=response.background,
        )

    model_path = os.getenv("TYPING_ML_MODEL_PATH", "models/model.joblib")
    with start_model_internal_span("typing-ml.model.load", attributes={"typing.ml.model.path": model_path}) as span:
        model, artifact = load_model_artifact(model_path)
        if span is not None and span.is_recording():
            span.set_attribute("typing.ml.model.class", model.__class__.__name__)

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
        with start_model_internal_span("typing-ml.model.metadata") as span:
            classes = getattr(getattr(model, "named_steps", {}).get("clf", model), "classes_", None)
            classes_list = list(classes) if classes is not None else None

            if span is not None and span.is_recording():
                span.set_attribute("typing.ml.model.class", model.__class__.__name__)
                span.set_attribute("typing.ml.model.feature_count", len(feature_names) if feature_names else 0)
                span.set_attribute("typing.ml.model.class_count", len(classes_list) if classes_list else 0)

            return {
                "model_path": model_path,
                "target_name": artifact.get("target_name", "weakest_finger"),
                "feature_names": feature_names,
                "classes": classes_list,
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
        X = build_frame([req.row.model_dump()])

        with start_model_inference_span("typing-ml.model.predict", row_count=1, feature_count=len(X.columns)) as span:
            pred = model.predict(X)[0]
            result: Dict[str, Any] = {"prediction": pred}

            has_predict_proba = hasattr(model, "predict_proba")
            if has_predict_proba:
                probs = model.predict_proba(X)[0]
                classes = getattr(model, "classes_", None)
                if classes is None:
                    # For Pipeline, classes live on the classifier step
                    clf = getattr(model, "named_steps", {}).get("clf")
                    classes = getattr(clf, "classes_", None)
                if classes is not None:
                    result["probabilities"] = {str(c): float(p) for c, p in zip(classes, probs)}

            if span is not None and span.is_recording():
                span.set_attribute("typing.ml.model.class", model.__class__.__name__)
                span.set_attribute("typing.ml.model.has_predict_proba", has_predict_proba)
                span.set_attribute("typing.ml.model.prediction", str(pred))

            return result

    @app.post(
        "/predict_batch",
        summary="Predict weakest finger (batch)",
        description="Send multiple rows and get predictions (and optional probabilities).",
    )
    def predict_batch(req: PredictBatchRequest) -> Dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        if not req.rows:
            raise HTTPException(status_code=400, detail={"error": "rows must be non-empty"})

        X = build_frame([row.model_dump() for row in req.rows])

        with start_model_inference_span(
            "typing-ml.model.predict_batch",
            row_count=len(req.rows),
            feature_count=len(X.columns),
        ) as span:
            preds = model.predict(X)

            out: Dict[str, Any] = {"predictions": [p for p in preds]}

            has_predict_proba = hasattr(model, "predict_proba")
            if has_predict_proba:
                probas = model.predict_proba(X)
                clf = getattr(model, "named_steps", {}).get("clf", model)
                classes = getattr(clf, "classes_", None)
                if classes is not None:
                    out["probabilities"] = [
                        {str(c): float(p) for c, p in zip(classes, row)} for row in probas
                    ]

            if span is not None and span.is_recording():
                span.set_attribute("typing.ml.model.class", model.__class__.__name__)
                span.set_attribute("typing.ml.model.has_predict_proba", has_predict_proba)
                span.set_attribute("typing.ml.model.prediction_count", len(preds))

            return out

    return app


app = create_app()
