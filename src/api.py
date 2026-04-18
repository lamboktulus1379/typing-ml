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
from datetime import datetime, timezone
from contextlib import nullcontext
from dataclasses import dataclass
from threading import Lock, RLock
from typing import Any, Dict, List, Optional, Sequence, cast

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

try:
    from src.ml_pipeline.artifacts import ArtifactStore, ModelArtifact
    from src.ml_pipeline.constants import (
        ALLOWED_WEAKEST_FINGER_LABELS,
        FEATURE_RANGE_RULES,
        TARGET_COLUMN,
        TRAIN_FEATURE_COLUMNS,
    )
    from src.ml_pipeline.model_factory import Algorithm, ModelPipelineFactory
    from src.ml_pipeline.validation import FeatureFrameValidator, TargetSeriesValidator
    from src.services.training_service import TrainingArenaService
except ModuleNotFoundError:
    from ml_pipeline.artifacts import ArtifactStore, ModelArtifact
    from ml_pipeline.constants import (
        ALLOWED_WEAKEST_FINGER_LABELS,
        FEATURE_RANGE_RULES,
        TARGET_COLUMN,
        TRAIN_FEATURE_COLUMNS,
    )
    from ml_pipeline.model_factory import Algorithm, ModelPipelineFactory
    from ml_pipeline.validation import FeatureFrameValidator, TargetSeriesValidator
    from services.training_service import TrainingArenaService


MAX_TRACE_PAYLOAD_CHARS = int(os.getenv("TYPING_ML_TRACE_PAYLOAD_CHARS", "4096"))
logger = logging.getLogger("typing-ml")

DEFAULT_MODEL_PATH = "models/model.joblib"
DEFAULT_PRODUCTION_MODEL_PATH = "models/model_production.joblib"
DEFAULT_TRAIN_REPORTS_DIR = "reports/retrain_runs"
DEFAULT_RETRAIN_RANDOM_STATE = int(os.getenv("TYPING_ML_RETRAIN_RANDOM_STATE", "42"))
DEFAULT_RETRAIN_ALGORITHMS: tuple[str, ...] = (
    Algorithm.LOGISTIC_REGRESSION.value,
    Algorithm.RANDOM_FOREST.value,
    Algorithm.XGBOOST.value,
)
DEFAULT_MODEL_PATH_BY_ALGORITHM = {
    "logistic_regression": "models/model_compare_logistic_regression.joblib",
    "random_forest": "models/model_compare_random_forest.joblib",
    "xgboost": "models/model_compare_xgboost.joblib",
}


@dataclass(frozen=True)
class RuntimeInferenceState:
    """Thread-safe snapshot holder for active inference model and service."""

    inference_service: "InferenceService"
    model: Any


def resolve_retrain_algorithms_from_env() -> tuple[str, ...]:
    """Resolve retraining candidate algorithms from environment configuration."""

    raw_algorithms = os.getenv("TYPING_ML_RETRAIN_ALGORITHMS")
    if not raw_algorithms:
        return DEFAULT_RETRAIN_ALGORITHMS

    requested = tuple(
        value.strip().lower()
        for value in raw_algorithms.split(",")
        if value.strip()
    )
    if not requested:
        raise RuntimeError(
            "TYPING_ML_RETRAIN_ALGORITHMS is set but contains no valid algorithm values."
        )

    allowed = set(Algorithm.choices())
    invalid = sorted(set(requested) - allowed)
    if invalid:
        raise RuntimeError(
            "Unsupported TYPING_ML_RETRAIN_ALGORITHMS values: "
            f"{invalid}. Supported values: {sorted(allowed)}"
        )

    return requested


def resolve_model_selection_from_env() -> tuple[str, Optional[str]]:
    """Resolve model path and selected algorithm from environment variables.

    Selection precedence:
    1) TYPING_ML_MODEL_PATH
    2) TYPING_ML_MODEL_ALGORITHM + optional TYPING_ML_MODEL_PATH_<ALGORITHM>
    3) default model path
    """

    explicit_model_path = os.getenv("TYPING_ML_MODEL_PATH")
    selected_algorithm_raw = os.getenv("TYPING_ML_MODEL_ALGORITHM")
    selected_algorithm = (
        selected_algorithm_raw.strip().lower() if selected_algorithm_raw else None
    )

    if explicit_model_path:
        return explicit_model_path, selected_algorithm

    if selected_algorithm:
        if selected_algorithm not in DEFAULT_MODEL_PATH_BY_ALGORITHM:
            raise RuntimeError(
                "Unsupported TYPING_ML_MODEL_ALGORITHM='"
                f"{selected_algorithm}'. Supported values: "
                f"{sorted(DEFAULT_MODEL_PATH_BY_ALGORITHM.keys())}"
            )

        algorithm_specific_path = os.getenv(
            f"TYPING_ML_MODEL_PATH_{selected_algorithm.upper()}"
        )
        if algorithm_specific_path:
            return algorithm_specific_path, selected_algorithm

        return DEFAULT_MODEL_PATH_BY_ALGORITHM[selected_algorithm], selected_algorithm

    return DEFAULT_MODEL_PATH, None


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

    def __init__(
        self,
        model: Any,
        artifact: Dict[str, Any],
        model_path: str,
        model_algorithm: Optional[str] = None,
    ) -> None:
        self.model = model
        self.artifact = artifact
        self.model_path = model_path
        self.model_algorithm = model_algorithm or artifact.get("model_name")
        self.feature_names: Optional[List[str]] = artifact.get("feature_names")
        self.label_classes: Optional[List[str]] = artifact.get("label_classes")

    def _decode_label(self, value: Any) -> Any:
        """Decode model output label when artifact stores encoded class mapping."""

        if not self.label_classes or isinstance(value, str):
            return value

        try:
            index = int(value)
        except (TypeError, ValueError):
            return value

        if index < 0 or index >= len(self.label_classes):
            return value

        return self.label_classes[index]

    def _resolve_output_classes(self) -> Optional[List[Any]]:
        """Resolve classes used by predict_proba output and decode if needed."""

        classes = getattr(self.model, "classes_", None)
        if classes is None:
            clf = getattr(self.model, "named_steps", {}).get("clf", self.model)
            classes = getattr(clf, "classes_", None)
        if classes is None:
            return None

        return [self._decode_label(class_value) for class_value in classes]

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

        classes = self._resolve_output_classes()
        classes_list = [str(class_value) for class_value in classes] if classes is not None else None
        return {
            "model_path": self.model_path,
            "model_algorithm": self.model_algorithm,
            "target_name": self.artifact.get("target_name", "weakest_finger"),
            "feature_names": self.feature_names,
            "classes": classes_list,
            "created_at": self.artifact.get("created_at"),
        }

    def predict_one(self, row: Dict[str, float]) -> Dict[str, Any]:
        """Predict one row and optionally include class probabilities."""

        dataframe = self.build_frame([row])
        prediction = self._decode_label(self.model.predict(dataframe)[0])
        result: Dict[str, Any] = {"prediction": prediction}

        has_predict_proba = hasattr(self.model, "predict_proba")
        if has_predict_proba:
            probs = self.model.predict_proba(dataframe)[0]
            classes = self._resolve_output_classes()
            if classes is not None:
                result["probabilities"] = {str(c): float(p) for c, p in zip(classes, probs)}

        return result

    def predict_many(self, rows: List[Dict[str, float]]) -> Dict[str, Any]:
        """Predict many rows and optionally include class probabilities per row."""

        if not rows:
            raise HTTPException(status_code=400, detail={"error": "rows must be non-empty"})

        dataframe = self.build_frame(rows)
        predictions = self.model.predict(dataframe)
        out: Dict[str, Any] = {"predictions": [self._decode_label(prediction) for prediction in predictions]}

        has_predict_proba = hasattr(self.model, "predict_proba")
        if has_predict_proba:
            probas = self.model.predict_proba(dataframe)
            classes = self._resolve_output_classes()
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


class TypingSessionTrainingData(TypingSessionData):
    """Retraining row payload schema including supervised target label."""

    weakest_finger: str


class RetrainRequest(BaseModel):
    """Retraining payload contract with dry-run safety mode."""

    rows: List[TypingSessionTrainingData]
    is_dry_run: bool = Field(
        default=True,
        description=(
            "If true, trains the model and returns metrics but does NOT save "
            "the model to disk."
        ),
    )


class TrainMetricsResponse(BaseModel):
    """Winner metrics preserved for backwards-compatible clients."""

    f1_score: float
    accuracy: float


class TrainLeaderboardEntryResponse(BaseModel):
    """Public leaderboard entry for one evaluated algorithm."""

    name: str
    accuracy: float
    f1_score: float
    execution_time_ms: float


class TrainResponse(BaseModel):
    """Typed retraining response for the Algorithm Arena flow."""

    status: str
    winning_algorithm: str
    winning_f1_score: float
    total_rows_processed: int
    leaderboard: List[TrainLeaderboardEntryResponse]
    algorithm: str
    accuracy: float
    f1_score: float
    rows_processed: int
    trained_rows: int
    metrics: TrainMetricsResponse
    candidates: Dict[str, TrainLeaderboardEntryResponse]
    model_path: Optional[str] = None
    report_path: Optional[str] = None


def build_train_report_path(status: str) -> str:
    """Create a timestamped JSON report path for one retraining run."""

    reports_dir = os.getenv("TYPING_ML_TRAIN_REPORTS_DIR", DEFAULT_TRAIN_REPORTS_DIR)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return os.path.join(reports_dir, f"train_{status}_{timestamp}.json")


def persist_train_report(
    artifact_store: ArtifactStore,
    response: TrainResponse,
    *,
    is_dry_run: bool,
    random_state: int,
) -> str:
    """Persist a JSON report artifact for thesis audit and reproducibility."""

    report_path = build_train_report_path(response.status)
    report_payload = {
        "status": response.status,
        "is_dry_run": is_dry_run,
        "random_state": random_state,
        "target_column": TARGET_COLUMN,
        "winning_algorithm": response.winning_algorithm,
        "winning_f1_score": response.winning_f1_score,
        "total_rows_processed": response.total_rows_processed,
        "leaderboard": [entry.model_dump() for entry in response.leaderboard],
        "metrics": response.metrics.model_dump(),
        "algorithm": response.algorithm,
        "accuracy": response.accuracy,
        "f1_score": response.f1_score,
        "rows_processed": response.rows_processed,
        "trained_rows": response.trained_rows,
        "model_path": response.model_path,
        "report_created_at": datetime.now(timezone.utc).isoformat(),
    }
    artifact_store.save_report(report_payload, report_path)
    return report_path


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

    model_path, selected_algorithm = resolve_model_selection_from_env()
    artifact_store = ArtifactStore()
    with start_model_internal_span("typing-ml.model.load", attributes={"typing.ml.model.path": model_path}) as span:
        model, artifact = artifact_store.load_model_artifact(model_path)
        if span is not None and span.is_recording():
            span.set_attribute("typing.ml.model.class", model.__class__.__name__)
            if selected_algorithm:
                span.set_attribute("typing.ml.model.algorithm", selected_algorithm)

    inference_service = InferenceService(model, artifact, model_path, selected_algorithm)
    retrain_algorithms = resolve_retrain_algorithms_from_env()
    production_model_path = os.getenv(
        "TYPING_ML_PRODUCTION_MODEL_PATH",
        DEFAULT_PRODUCTION_MODEL_PATH,
    )

    runtime_lock = RLock()
    training_lock = Lock()
    runtime_state = RuntimeInferenceState(inference_service=inference_service, model=model)
    feature_validator = FeatureFrameValidator(FEATURE_RANGE_RULES)
    target_validator = TargetSeriesValidator(ALLOWED_WEAKEST_FINGER_LABELS)
    training_service = TrainingArenaService(
        model_factory=ModelPipelineFactory(random_state=DEFAULT_RETRAIN_RANDOM_STATE),
        feature_validator=feature_validator,
        target_validator=target_validator,
        random_state=DEFAULT_RETRAIN_RANDOM_STATE,
    )

    def get_runtime_snapshot() -> RuntimeInferenceState:
        with runtime_lock:
            return runtime_state

    def replace_runtime_state(new_state: RuntimeInferenceState) -> None:
        nonlocal runtime_state
        with runtime_lock:
            runtime_state = new_state

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

        runtime_snapshot = get_runtime_snapshot()
        current_service = runtime_snapshot.inference_service
        current_model = runtime_snapshot.model

        with start_model_internal_span("typing-ml.model.metadata") as span:
            metadata_payload = current_service.get_metadata()
            classes_list = metadata_payload.get("classes")
            feature_names = metadata_payload.get("feature_names")

            if span is not None and span.is_recording():
                span.set_attribute("typing.ml.model.class", current_model.__class__.__name__)
                span.set_attribute("typing.ml.model.feature_count", len(feature_names) if feature_names else 0)
                span.set_attribute("typing.ml.model.class_count", len(classes_list) if classes_list else 0)

            return metadata_payload

    @app.post(
        "/train",
        summary="Retrain model (supports dry run and production mode)",
        description=(
            "Train logistic_regression, random_forest, and xgboost on in-memory payload rows, "
            "select the winner by macro F1-score on a deterministic 80/20 split, retrain the winner "
            "on 100% of the dataset, and optionally persist the production artifact."
        ),
        response_model=TrainResponse,
    )
    def train(
        payload: RetrainRequest | List[TypingSessionTrainingData],
    ) -> TrainResponse:  # pyright: ignore[reportUnusedFunction]
        """Run one retraining cycle and optionally replace runtime inference model."""

        if isinstance(payload, list):
            rows = payload
            is_dry_run = True
        else:
            rows = payload.rows
            is_dry_run = payload.is_dry_run

        if not rows:
            raise HTTPException(status_code=400, detail={"error": "rows must be non-empty"})

        if not training_lock.acquire(blocking=False):
            raise HTTPException(
                status_code=409,
                detail={"error": "Retraining is already in progress"},
            )

        try:
            normalized_rows: List[Dict[str, Any]] = []
            for row in rows:
                row_payload = row.model_dump()
                row_payload[TARGET_COLUMN] = str(row_payload[TARGET_COLUMN]).strip().lower()

                # Accept legacy percentage-style accuracy and normalize to ratio.
                accuracy_value = float(row_payload["accuracy"])
                if 1.0 < accuracy_value <= 100.0:
                    row_payload["accuracy"] = accuracy_value / 100.0

                normalized_rows.append(row_payload)

            dataframe = pd.DataFrame(normalized_rows)

            try:
                arena_result = training_service.run_algorithm_arena(
                    dataframe,
                    algorithms=retrain_algorithms,
                )
            except ValueError as ex:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Invalid retraining payload",
                        "detail": str(ex),
                    },
                ) from ex
            except RuntimeError as ex:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Retraining failed",
                        "detail": str(ex),
                    },
                )

            leaderboard = [
                TrainLeaderboardEntryResponse(
                    name=entry.name,
                    accuracy=entry.accuracy,
                    f1_score=entry.f1_score,
                    execution_time_ms=entry.execution_time_ms,
                )
                for entry in arena_result.leaderboard
            ]
            candidates_payload = {entry.name: entry for entry in leaderboard}
            metrics = TrainMetricsResponse(
                f1_score=arena_result.winning_f1_score,
                accuracy=arena_result.winning_accuracy,
            )

            if is_dry_run:
                logger.info(
                    "typing-ml retrain dry-run completed rows=%s winner=%s accuracy=%.4f macro_f1=%.4f",
                    arena_result.total_rows_processed,
                    arena_result.winning_algorithm,
                    arena_result.winning_accuracy,
                    arena_result.winning_f1_score,
                )

                response = TrainResponse(
                    status="success_dry_run",
                    winning_algorithm=arena_result.winning_algorithm,
                    winning_f1_score=arena_result.winning_f1_score,
                    total_rows_processed=arena_result.total_rows_processed,
                    leaderboard=leaderboard,
                    algorithm=arena_result.winning_algorithm,
                    accuracy=arena_result.winning_accuracy,
                    f1_score=arena_result.winning_f1_score,
                    rows_processed=arena_result.total_rows_processed,
                    trained_rows=arena_result.total_rows_processed,
                    metrics=metrics,
                    candidates=candidates_payload,
                )
                report_path = persist_train_report(
                    artifact_store,
                    response,
                    is_dry_run=True,
                    random_state=DEFAULT_RETRAIN_RANDOM_STATE,
                )
                return response.model_copy(update={"report_path": report_path})

            winner_artifact = ModelArtifact.from_training(
                model=arena_result.retrained_model,
                model_name=arena_result.winning_algorithm,
                feature_names=TRAIN_FEATURE_COLUMNS,
                target_name=TARGET_COLUMN,
                label_classes=arena_result.retrained_label_classes,
            )
            artifact_store.save_model_artifact(winner_artifact, production_model_path)

            new_service = InferenceService(
                arena_result.retrained_model,
                winner_artifact.to_dict(),
                production_model_path,
                arena_result.winning_algorithm,
            )
            replace_runtime_state(
                RuntimeInferenceState(inference_service=new_service, model=arena_result.retrained_model)
            )

            logger.info(
                "typing-ml retrain production completed rows=%s winner=%s accuracy=%.4f macro_f1=%.4f path=%s",
                arena_result.total_rows_processed,
                arena_result.winning_algorithm,
                arena_result.winning_accuracy,
                arena_result.winning_f1_score,
                production_model_path,
            )

            response = TrainResponse(
                status="success_production",
                winning_algorithm=arena_result.winning_algorithm,
                winning_f1_score=arena_result.winning_f1_score,
                total_rows_processed=arena_result.total_rows_processed,
                leaderboard=leaderboard,
                algorithm=arena_result.winning_algorithm,
                accuracy=arena_result.winning_accuracy,
                f1_score=arena_result.winning_f1_score,
                rows_processed=arena_result.total_rows_processed,
                trained_rows=arena_result.total_rows_processed,
                metrics=metrics,
                candidates=candidates_payload,
                model_path=production_model_path,
            )
            report_path = persist_train_report(
                artifact_store,
                response,
                is_dry_run=False,
                random_state=DEFAULT_RETRAIN_RANDOM_STATE,
            )
            return response.model_copy(update={"report_path": report_path})
        finally:
            training_lock.release()

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

        runtime_snapshot = get_runtime_snapshot()
        current_service = runtime_snapshot.inference_service
        current_model = runtime_snapshot.model
        row_payload = req.row.model_dump()
        feature_count = len(row_payload)

        with start_model_inference_span("typing-ml.model.predict", row_count=1, feature_count=feature_count) as span:
            result = current_service.predict_one(row_payload)
            has_predict_proba = "probabilities" in result
            pred = result.get("prediction")

            if span is not None and span.is_recording():
                span.set_attribute("typing.ml.model.class", current_model.__class__.__name__)
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

        runtime_snapshot = get_runtime_snapshot()
        current_service = runtime_snapshot.inference_service
        current_model = runtime_snapshot.model
        rows_payload = [row.model_dump() for row in req.rows]
        feature_count = len(rows_payload[0]) if rows_payload else 0

        with start_model_inference_span(
            "typing-ml.model.predict_batch",
            row_count=len(rows_payload),
            feature_count=feature_count,
        ) as span:
            out = current_service.predict_many(rows_payload)
            preds = out.get("predictions", [])
            has_predict_proba = "probabilities" in out

            if span is not None and span.is_recording():
                span.set_attribute("typing.ml.model.class", current_model.__class__.__name__)
                span.set_attribute("typing.ml.model.has_predict_proba", has_predict_proba)
                span.set_attribute("typing.ml.model.prediction_count", len(preds))

            return out

    return app


app = create_app()
