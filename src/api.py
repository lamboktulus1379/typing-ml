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
import re
from datetime import datetime, timezone
from contextlib import nullcontext
from dataclasses import dataclass
from threading import Lock, RLock
from typing import Any, Dict, List, Optional, Sequence, cast

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

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
DEFAULT_PRODUCTION_MODEL_DIR = "models/production"
DEFAULT_ACTIVE_MODEL_METADATA_PATH = "models/active_production_model.json"
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


def resolve_active_model_metadata_path() -> str:
    """Return the metadata pointer path for the active promoted model."""

    return os.getenv(
        "TYPING_ML_ACTIVE_MODEL_METADATA_PATH",
        DEFAULT_ACTIVE_MODEL_METADATA_PATH,
    )


def _sanitize_user_id_for_path(user_id: str) -> str:
    """Normalize a user id into a filesystem-safe path segment."""

    normalized_user_id = user_id.strip().lower()
    sanitized = re.sub(r"[^a-z0-9_-]+", "_", normalized_user_id)
    return sanitized.strip("_") or "anonymous"


def build_production_model_artifact_path(algorithm: str, user_id: Optional[str] = None) -> str:
    """Create a unique immutable artifact path for one promoted production model."""

    models_dir = os.getenv("TYPING_ML_PRODUCTION_MODEL_DIR", DEFAULT_PRODUCTION_MODEL_DIR)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    normalized_algorithm = algorithm.strip().lower()
    if user_id and user_id.strip():
        sanitized_user_id = _sanitize_user_id_for_path(user_id)
        return os.path.join(models_dir, f"typing-prod-{timestamp}-{sanitized_user_id}-{normalized_algorithm}.joblib")

    return os.path.join(models_dir, f"typing-prod-{timestamp}-{normalized_algorithm}.joblib")


def load_active_model_pointer(artifact_store: ArtifactStore) -> Optional[Dict[str, Any]]:
    """Load the active promoted model pointer when it exists and targets a valid artifact."""

    metadata_path = resolve_active_model_metadata_path()
    if not os.path.exists(metadata_path):
        return None

    try:
        payload = artifact_store.load_json(metadata_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as ex:
        logger.warning("Failed to read active production model metadata from %s: %s", metadata_path, ex)
        return None

    model_path = payload.get("model_path")
    if not isinstance(model_path, str) or not model_path:
        logger.warning("Active production model metadata is missing a usable model_path: %s", metadata_path)
        return None

    if not os.path.exists(model_path):
        logger.warning("Active production model metadata points to a missing artifact: %s", model_path)
        return None

    payload["active_model_metadata_path"] = metadata_path
    return payload


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


def resolve_model_selection_from_env() -> tuple[bool, str, Optional[str]]:
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
        return True, explicit_model_path, selected_algorithm

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
            return True, algorithm_specific_path, selected_algorithm

        return True, DEFAULT_MODEL_PATH_BY_ALGORITHM[selected_algorithm], selected_algorithm

    return False, DEFAULT_MODEL_PATH, None


def resolve_runtime_model_selection(
    artifact_store: ArtifactStore,
) -> tuple[str, Optional[str], Optional[Dict[str, Any]]]:
    """Resolve runtime model path with explicit env overrides first, then active pointer metadata."""

    has_explicit_selection, model_path, selected_algorithm = resolve_model_selection_from_env()
    if has_explicit_selection:
        return model_path, selected_algorithm, None

    active_pointer = load_active_model_pointer(artifact_store)
    if active_pointer is not None:
        return active_pointer["model_path"], cast(Optional[str], active_pointer.get("model_algorithm")), active_pointer

    legacy_production_model_path = os.getenv(
        "TYPING_ML_PRODUCTION_MODEL_PATH",
        DEFAULT_PRODUCTION_MODEL_PATH,
    )
    if os.path.exists(legacy_production_model_path):
        return legacy_production_model_path, None, None

    return model_path, selected_algorithm, None


def resolve_global_fallback_model_path() -> str:
    """Return the global baseline model path used for cold-start fallback."""

    return os.getenv(
        "TYPING_ML_GLOBAL_FALLBACK_MODEL_PATH",
        os.getenv("TYPING_ML_PRODUCTION_MODEL_PATH", DEFAULT_PRODUCTION_MODEL_PATH),
    )


def resolve_personalized_model_path(user_id: str) -> str:
    """Resolve the preferred personalized model path for one user.

    Preference order:
    1. Explicit template override via environment variable
    2. Latest immutable production artifact matching the sanitized user id
    3. Legacy fixed per-user production artifact path
    """

    sanitized_user_id = _sanitize_user_id_for_path(user_id)
    explicit_template = os.getenv("TYPING_ML_PERSONALIZED_MODEL_PATH_TEMPLATE")
    if explicit_template:
        return explicit_template.format(user_id=sanitized_user_id)

    production_dir = os.getenv("TYPING_ML_PRODUCTION_MODEL_DIR", DEFAULT_PRODUCTION_MODEL_DIR)
    if os.path.isdir(production_dir):
        matched_artifacts = sorted(
            os.path.join(production_dir, file_name)
            for file_name in os.listdir(production_dir)
            if file_name.endswith(".joblib")
            and file_name.startswith("typing-prod-")
            and f"-{sanitized_user_id}-" in file_name
        )
        if matched_artifacts:
            return matched_artifacts[-1]

    return os.path.join("models", f"model_production_{sanitized_user_id}.joblib")


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
        active_model_metadata_path: Optional[str] = None,
    ) -> None:
        self.model = model
        self.artifact = artifact
        self.model_path = model_path
        self.model_algorithm = model_algorithm or artifact.get("model_name")
        self.active_model_metadata_path = active_model_metadata_path
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
            "active_model_metadata_path": self.active_model_metadata_path,
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

    model_config = ConfigDict(populate_by_name=True)

    user_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("user_id", "userId"),
    )
    row: TypingSessionData

class PredictBatchRequest(BaseModel):
    """Batch prediction request payload."""

    rows: List[TypingSessionData]


class TypingSessionTrainingData(TypingSessionData):
    """Retraining row payload schema including supervised target label."""

    model_config = ConfigDict(populate_by_name=True)

    weakest_finger: str
    user_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("user_id", "userId"),
    )


class RetrainRequest(BaseModel):
    """Retraining payload contract with dry-run safety mode."""

    model_config = ConfigDict(populate_by_name=True)

    user_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("user_id", "userId"),
    )
    rows: List[TypingSessionTrainingData]
    is_dry_run: bool = Field(
        default=True,
        description=(
            "If true, trains the model and returns metrics but does NOT save "
            "the model to disk."
        ),
    )


class TrainEvaluationResponse(BaseModel):
    """Per-algorithm evaluation payload for the .NET orchestrator."""

    algorithmName: str
    accuracy: float
    f1Score: float
    executionTimeMs: int


class TrainOrchestratorResponse(BaseModel):
    """Exact retraining response contract expected by the .NET orchestrator."""

    winningAlgorithmName: str
    macroPrecision: float
    macroRecall: float
    topPredictiveFeature: str
    primaryMisclassification: str
    evaluations: List[TrainEvaluationResponse]


def build_train_report_path(status: str) -> str:
    """Create a timestamped JSON report path for one retraining run."""

    reports_dir = os.getenv("TYPING_ML_TRAIN_REPORTS_DIR", DEFAULT_TRAIN_REPORTS_DIR)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return os.path.join(reports_dir, f"train_{status}_{timestamp}.json")


def persist_train_report(
    artifact_store: ArtifactStore,
    response: TrainOrchestratorResponse,
    *,
    status: str,
    is_dry_run: bool,
    random_state: int,
    total_rows_processed: int,
    model_path: Optional[str] = None,
    active_model_metadata_path: Optional[str] = None,
) -> str:
    """Persist a JSON report artifact for thesis audit and reproducibility."""

    report_path = build_train_report_path(status)
    report_payload = {
        "status": status,
        "is_dry_run": is_dry_run,
        "random_state": random_state,
        "target_column": TARGET_COLUMN,
        "winning_algorithm_name": response.winningAlgorithmName,
        "macro_precision": response.macroPrecision,
        "macro_recall": response.macroRecall,
        "top_predictive_feature": response.topPredictiveFeature,
        "primary_misclassification": response.primaryMisclassification,
        "total_rows_processed": total_rows_processed,
        "evaluations": [entry.model_dump() for entry in response.evaluations],
        "model_path": model_path,
        "active_model_metadata_path": active_model_metadata_path,
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

    artifact_store = ArtifactStore()
    model_path, selected_algorithm, active_model_pointer = resolve_runtime_model_selection(artifact_store)
    with start_model_internal_span("typing-ml.model.load", attributes={"typing.ml.model.path": model_path}) as span:
        model, artifact = artifact_store.load_model_artifact(model_path)
        if span is not None and span.is_recording():
            span.set_attribute("typing.ml.model.class", model.__class__.__name__)
            if selected_algorithm:
                span.set_attribute("typing.ml.model.algorithm", selected_algorithm)

    inference_service = InferenceService(
        model,
        artifact,
        model_path,
        selected_algorithm,
        cast(Optional[str], active_model_pointer.get("active_model_metadata_path")) if active_model_pointer else None,
    )
    retrain_algorithms = resolve_retrain_algorithms_from_env()
    active_model_metadata_path = resolve_active_model_metadata_path()

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

    def load_inference_service_for_model_path(model_path: str) -> tuple[InferenceService, Any]:
        """Load one inference service instance from the requested artifact path."""

        model, artifact = artifact_store.load_model_artifact(model_path)
        service = InferenceService(
            model,
            artifact,
            model_path,
            cast(Optional[str], artifact.get("model_name")),
        )
        return service, model

    def resolve_predict_runtime(user_id: Optional[str]) -> tuple[InferenceService, Any, bool]:
        """Resolve personalized inference first, then fall back to the global baseline."""

        runtime_snapshot = get_runtime_snapshot()
        if not user_id or not user_id.strip():
            return runtime_snapshot.inference_service, runtime_snapshot.model, False

        normalized_user_id = user_id.strip()
        personalized_model_path = resolve_personalized_model_path(normalized_user_id)

        try:
            personalized_service, personalized_model = load_inference_service_for_model_path(personalized_model_path)
            return personalized_service, personalized_model, False
        except FileNotFoundError:
            global_model_path = resolve_global_fallback_model_path()
            logger.info(
                "Personalized model not found for user_id=%s at path=%s. Falling back to global model path=%s",
                normalized_user_id,
                personalized_model_path,
                global_model_path,
            )
            global_service, global_model = load_inference_service_for_model_path(global_model_path)
            return global_service, global_model, True

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
        response_model=TrainOrchestratorResponse,
    )
    def train(
        payload: RetrainRequest | List[TypingSessionTrainingData],
    ) -> TrainOrchestratorResponse:  # pyright: ignore[reportUnusedFunction]
        """Run one retraining cycle and optionally replace runtime inference model."""

        if isinstance(payload, list):
            rows = payload
            is_dry_run = True
            requested_user_id: Optional[str] = None
        else:
            rows = payload.rows
            is_dry_run = payload.is_dry_run
            requested_user_id = payload.user_id.strip()

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

                if requested_user_id is not None and not row_payload.get("user_id"):
                    row_payload["user_id"] = requested_user_id

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
                    user_id=requested_user_id,
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

            evaluations = [
                TrainEvaluationResponse(
                    algorithmName=entry.name,
                    accuracy=entry.accuracy,
                    f1Score=entry.f1_score,
                    executionTimeMs=int(round(entry.execution_time_ms)),
                )
                for entry in arena_result.leaderboard
            ]
            response = TrainOrchestratorResponse(
                winningAlgorithmName=arena_result.winning_algorithm,
                macroPrecision=arena_result.macro_precision,
                macroRecall=arena_result.macro_recall,
                topPredictiveFeature=arena_result.top_predictive_feature,
                primaryMisclassification=arena_result.primary_misclassification,
                evaluations=evaluations,
            )

            if is_dry_run:
                logger.info(
                    "typing-ml retrain dry-run completed rows=%s winner=%s accuracy=%.4f macro_f1=%.4f",
                    arena_result.total_rows_processed,
                    arena_result.winning_algorithm,
                    arena_result.winning_accuracy,
                    arena_result.winning_f1_score,
                )
                persist_train_report(
                    artifact_store,
                    response,
                    status="success_dry_run",
                    is_dry_run=True,
                    random_state=DEFAULT_RETRAIN_RANDOM_STATE,
                    total_rows_processed=arena_result.total_rows_processed,
                )
                return response

            winner_artifact = ModelArtifact.from_training(
                model=arena_result.retrained_model,
                model_name=arena_result.winning_algorithm,
                feature_names=TRAIN_FEATURE_COLUMNS,
                target_name=TARGET_COLUMN,
                label_classes=arena_result.retrained_label_classes,
            )
            production_model_path = build_production_model_artifact_path(
                arena_result.winning_algorithm,
                requested_user_id,
            )
            artifact_store.save_model_artifact(winner_artifact, production_model_path)
            artifact_store.save_json(
                {
                    "model_path": production_model_path,
                    "model_algorithm": arena_result.winning_algorithm,
                    "promoted_at": datetime.now(timezone.utc).isoformat(),
                    "created_at": winner_artifact.created_at,
                },
                active_model_metadata_path,
            )

            new_service = InferenceService(
                arena_result.retrained_model,
                winner_artifact.to_dict(),
                production_model_path,
                arena_result.winning_algorithm,
                active_model_metadata_path,
            )
            replace_runtime_state(
                RuntimeInferenceState(inference_service=new_service, model=arena_result.retrained_model)
            )

            logger.info(
                "typing-ml retrain production completed rows=%s winner=%s accuracy=%.4f macro_f1=%.4f path=%s active_metadata=%s",
                arena_result.total_rows_processed,
                arena_result.winning_algorithm,
                arena_result.winning_accuracy,
                arena_result.winning_f1_score,
                production_model_path,
                active_model_metadata_path,
            )

            persist_train_report(
                artifact_store,
                response,
                status="success_production",
                is_dry_run=False,
                random_state=DEFAULT_RETRAIN_RANDOM_STATE,
                total_rows_processed=arena_result.total_rows_processed,
                model_path=production_model_path,
                active_model_metadata_path=active_model_metadata_path,
            )
            return response
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

        requested_user_id = req.user_id.strip() if req.user_id else None
        current_service, current_model, is_fallback_used = resolve_predict_runtime(requested_user_id)
        row_payload = req.row.model_dump()
        feature_count = len(row_payload)

        with start_model_inference_span("typing-ml.model.predict", row_count=1, feature_count=feature_count) as span:
            result = current_service.predict_one(row_payload)
            result["is_fallback_used"] = is_fallback_used
            has_predict_proba = "probabilities" in result
            pred = result.get("prediction")

            if span is not None and span.is_recording():
                span.set_attribute("typing.ml.model.class", current_model.__class__.__name__)
                span.set_attribute("typing.ml.model.has_predict_proba", has_predict_proba)
                span.set_attribute("typing.ml.model.prediction", str(pred))
                span.set_attribute("typing.ml.model.is_fallback_used", is_fallback_used)
                if requested_user_id:
                    span.set_attribute("typing.ml.model.user_id", requested_user_id)

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
