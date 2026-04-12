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
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

try:
    from src.ml_pipeline.artifacts import ArtifactStore
except ModuleNotFoundError:
    from ml_pipeline.artifacts import ArtifactStore


MAX_TRACE_PAYLOAD_CHARS = int(os.getenv("TYPING_ML_TRACE_PAYLOAD_CHARS", "4096"))
logger = logging.getLogger("typing-ml")


def truncate_payload(raw: bytes | str, max_chars: int = MAX_TRACE_PAYLOAD_CHARS) -> str:
    """Convert payload to text and cap it to avoid oversized trace attributes."""

    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...(truncated)"


def compact_json_if_possible(payload: str) -> str:
    """Return compact JSON when payload is valid JSON, otherwise unchanged text."""

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


class InferenceService:
    """Application service for schema-aware inference and metadata access."""

    def __init__(self, model: Any, artifact: Dict[str, Any], model_path: str) -> None:
        self.model = model
        self.artifact = artifact
        self.model_path = model_path
        self.feature_names: Optional[List[str]] = artifact.get("feature_names")

    def build_frame(self, rows: List[Dict[str, float]]) -> pd.DataFrame:
        """Build a prediction frame and enforce training feature schema when available."""

        if self.feature_names:
            missing = sorted({k for k in self.feature_names if any(k not in r for r in rows)})
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Missing required features",
                        "missing": missing,
                    },
                )
            return pd.DataFrame(rows, columns=self.feature_names)

        all_keys = sorted({k for r in rows for k in r.keys()})
        return pd.DataFrame(rows, columns=all_keys)

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata used by clients to shape prediction requests."""

        classes = getattr(getattr(self.model, "named_steps", {}).get("clf", self.model), "classes_", None)
        classes_list = list(classes) if classes is not None else None
        return {
            "model_path": self.model_path,
            "target_name": self.artifact.get("target_name", "weakest_finger"),
            "feature_names": self.feature_names,
            "classes": classes_list,
            "created_at": self.artifact.get("created_at"),
        }

    def predict_one(self, row: Dict[str, float]) -> Dict[str, Any]:
        """Predict one row and optionally include class probabilities."""

        dataframe = self.build_frame([row])
        prediction = self.model.predict(dataframe)[0]
        result: Dict[str, Any] = {"prediction": prediction}

        has_predict_proba = hasattr(self.model, "predict_proba")
        if has_predict_proba:
            probs = self.model.predict_proba(dataframe)[0]
            classes = getattr(self.model, "classes_", None)
            if classes is None:
                clf = getattr(self.model, "named_steps", {}).get("clf")
                classes = getattr(clf, "classes_", None)
            if classes is not None:
                result["probabilities"] = {str(c): float(p) for c, p in zip(classes, probs)}

        return result

    def predict_many(self, rows: List[Dict[str, float]]) -> Dict[str, Any]:
        """Predict many rows and optionally include class probabilities per row."""

        if not rows:
            raise HTTPException(status_code=400, detail={"error": "rows must be non-empty"})

        dataframe = self.build_frame(rows)
        predictions = self.model.predict(dataframe)
        out: Dict[str, Any] = {"predictions": [p for p in predictions]}

        has_predict_proba = hasattr(self.model, "predict_proba")
        if has_predict_proba:
            probas = self.model.predict_proba(dataframe)
            clf = getattr(self.model, "named_steps", {}).get("clf", self.model)
            classes = getattr(clf, "classes_", None)
            if classes is not None:
                out["probabilities"] = [
                    {str(c): float(p) for c, p in zip(classes, row_probs)}
                    for row_probs in probas
                ]

        return out


class TypingSessionData(BaseModel):
    """Input feature schema for one typing session."""

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
    """Single-row prediction request payload."""

    row: TypingSessionData

class PredictBatchRequest(BaseModel):
    """Batch prediction request payload."""

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
        """Capture request/response payload snippets for logs and optional tracing."""

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
    artifact_store = ArtifactStore()
    with start_model_internal_span("typing-ml.model.load", attributes={"typing.ml.model.path": model_path}) as span:
        model, artifact = artifact_store.load_model_artifact(model_path)
        if span is not None and span.is_recording():
            span.set_attribute("typing.ml.model.class", model.__class__.__name__)

    inference_service = InferenceService(model, artifact, model_path)

    @app.get(
        "/health",
        summary="Health check",
        description="Returns OK if the server is running.",
    )
    def health() -> Dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        """Lightweight liveness endpoint."""

        return {"status": "ok"}

    @app.get(
        "/metadata",
        summary="Model metadata",
        description="Returns required feature names, classes, and artifact info.",
    )
    def metadata() -> Dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Expose model schema and class metadata for API clients."""

        with start_model_internal_span("typing-ml.model.metadata") as span:
            metadata_payload = inference_service.get_metadata()
            classes_list = metadata_payload.get("classes")
            feature_names = metadata_payload.get("feature_names")

            if span is not None and span.is_recording():
                span.set_attribute("typing.ml.model.class", model.__class__.__name__)
                span.set_attribute("typing.ml.model.feature_count", len(feature_names) if feature_names else 0)
                span.set_attribute("typing.ml.model.class_count", len(classes_list) if classes_list else 0)

            return metadata_payload

    @app.post(
        "/predict",
        summary="Predict weakest finger (single row)",
        description=(
            "Send one feature map and get one predicted label. "
            "If the model supports predict_proba, class probabilities are returned too."
        ),
    )
    def predict(req: PredictRequest) -> Dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Predict weakest finger for a single input row."""

        row_payload = req.row.model_dump()
        feature_count = len(row_payload)

        with start_model_inference_span("typing-ml.model.predict", row_count=1, feature_count=feature_count) as span:
            result = inference_service.predict_one(row_payload)
            has_predict_proba = "probabilities" in result
            pred = result.get("prediction")

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
        """Predict weakest finger labels for multiple input rows."""

        rows_payload = [row.model_dump() for row in req.rows]
        feature_count = len(rows_payload[0]) if rows_payload else 0

        with start_model_inference_span(
            "typing-ml.model.predict_batch",
            row_count=len(rows_payload),
            feature_count=feature_count,
        ) as span:
            out = inference_service.predict_many(rows_payload)
            preds = out.get("predictions", [])
            has_predict_proba = "probabilities" in out

            if span is not None and span.is_recording():
                span.set_attribute("typing.ml.model.class", model.__class__.__name__)
                span.set_attribute("typing.ml.model.has_predict_proba", has_predict_proba)
                span.set_attribute("typing.ml.model.prediction_count", len(preds))

            return out

    return app


app = create_app()
