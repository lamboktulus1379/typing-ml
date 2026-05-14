from __future__ import annotations

import json
import logging
import os
import re
import signal
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd
import pyodbc
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection
from confluent_kafka import Consumer, KafkaError, Producer
from confluent_kafka.admin import AdminClient, NewTopic
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError
from redis import Redis

try:
    from src.ml_pipeline.artifacts import ArtifactStore, ModelArtifact
    from src.ml_pipeline.cleaning import TimingOutlierAnalysis, analyze_preprocessing_outliers
    from src.ml_pipeline.constants import (
        ALLOWED_WEAKEST_FINGER_LABELS,
        FEATURE_RANGE_RULES,
        TARGET_COLUMN,
        TRAIN_FEATURE_COLUMNS,
    )
    from src.ml_pipeline.model_factory import Algorithm, ModelPipelineFactory
    from src.ml_pipeline.validation import FeatureFrameValidator, TargetSeriesValidator
except ModuleNotFoundError:
    from ml_pipeline.artifacts import ArtifactStore, ModelArtifact
    from ml_pipeline.cleaning import TimingOutlierAnalysis, analyze_preprocessing_outliers
    from ml_pipeline.constants import (
        ALLOWED_WEAKEST_FINGER_LABELS,
        FEATURE_RANGE_RULES,
        TARGET_COLUMN,
        TRAIN_FEATURE_COLUMNS,
    )
    from ml_pipeline.model_factory import Algorithm, ModelPipelineFactory
    from ml_pipeline.validation import FeatureFrameValidator, TargetSeriesValidator


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("typing_ml.worker")

GENERATION_REQUEST_TOPIC = "data-generation-requests"
GENERATION_STREAM_TOPIC = "telemetry-data-stream"
PREPROCESSING_REQUEST_TOPIC = "data-preprocessing-requests"
PREPROCESSING_RESULT_TOPIC = "data-preprocessing-results"
TRAINING_REQUEST_TOPIC = "data-training-requests"
TRAINING_RESULT_TOPIC = "data-training-results"
REDIS_KEY_PREFIX = "dataops:task:"
DEFAULT_CHUNK_SIZE = 5_000
DEFAULT_KAFKA_GROUP_ID = "typing-ml-dataops-worker"
DEFAULT_SQL_ODBC_DRIVER = "ODBC Driver 18 for SQL Server"
DEFAULT_MODEL_REGISTRY_DIR = "models/registry"
DEFAULT_RANDOM_STATE = 42
DEFAULT_OUTLIER_SAMPLE_SIZE = 10
OUTLIER_TIME_MS = 5_000


def configure_optional_otel() -> None:
    """Enable OTLP tracing for the worker when OpenTelemetry packages are available."""
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not otlp_endpoint:
        logger.info("OpenTelemetry endpoint is not configured; skipping exporter setup.")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except Exception as ex:  # pragma: no cover - optional dependency path
        logger.info("OpenTelemetry packages are not installed, worker tracing will stay log-only: %s", ex)
        return

    service_name = os.getenv("OTEL_SERVICE_NAME", "typing-ml-worker")
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    logger.info("OpenTelemetry tracing enabled for typing-ml worker")


def start_worker_span(span_name: str, attributes: dict[str, Any] | None = None):
    """Start a worker span when OpenTelemetry is available."""
    try:
        from opentelemetry import trace

        tracer = trace.get_tracer("typing-ml.worker")
        return tracer.start_as_current_span(span_name, attributes=attributes or {})
    except Exception:
        return nullcontext(None)


def update_span_attributes(**attributes: Any) -> None:
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if not span.is_recording():
            return

        for key, value in attributes.items():
            if value is None:
                continue
            if isinstance(value, UUID):
                span.set_attribute(key, str(value))
            elif isinstance(value, (str, bool, int, float)):
                span.set_attribute(key, value)
            else:
                span.set_attribute(key, str(value))
    except Exception:
        return


def record_span_exception(exc: Exception) -> None:
    try:
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        span = trace.get_current_span()
        if not span.is_recording():
            return

        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, str(exc)))
    except Exception:
        return

FINGERS = (
    "LeftPinky",
    "LeftRing",
    "LeftMiddle",
    "LeftIndex",
    "RightIndex",
    "RightMiddle",
    "RightRing",
    "RightPinky",
)
ERROR_COLUMNS = tuple(f"Error{finger}" for finger in FINGERS)
DWELL_COLUMNS = tuple(f"Dwell{finger}" for finger in FINGERS)
FLIGHT_COLUMNS = tuple(f"Flight{finger}" for finger in FINGERS)
TIME_COLUMNS = DWELL_COLUMNS + FLIGHT_COLUMNS
ORDERED_COLUMNS = ("WPM", "Accuracy", *ERROR_COLUMNS, *DWELL_COLUMNS, *FLIGHT_COLUMNS)


class DataGenerationRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    task_id: UUID = Field(alias="taskId")
    sessions: PositiveInt
    inject_outliers: bool = Field(alias="injectOutliers")


class TelemetryChunkEnvelope(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    task_id: UUID = Field(alias="taskId")
    is_last_chunk: bool = Field(alias="isLastChunk")
    data: list[dict[str, Any]]


class DataPreprocessingRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    task_id: UUID = Field(alias="taskId")


class IqrSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    q1: float
    q3: float
    iqr: float
    lower_bound: float = Field(alias="lowerBound")
    upper_bound: float = Field(alias="upperBound")


class OutlierSample(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_feature_id: UUID = Field(alias="sessionFeatureId")
    dwell_time_avg: float = Field(alias="dwellTimeAvg")
    flight_time_avg: float = Field(alias="flightTimeAvg")


class DataPreprocessingSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    total_rows_scanned: int = Field(alias="totalRowsScanned")
    total_rows_after_preprocessing: int = Field(alias="totalRowsAfterPreprocessing")
    total_outliers_detected: int = Field(alias="totalOutliersDetected")
    dwell_time_avg: IqrSummary = Field(alias="dwellTimeAvg")
    flight_time_avg: IqrSummary = Field(alias="flightTimeAvg")
    sample_outliers: list[OutlierSample] = Field(alias="sampleOutliers")


class DataPreprocessingResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    task_id: UUID = Field(alias="taskId")
    is_last_chunk: bool = Field(default=True, alias="isLastChunk")
    clear_existing_outliers: bool = Field(default=True, alias="clearExistingOutliers")
    outlier_ids: list[UUID] = Field(default_factory=list, alias="outlierIds")
    summary: DataPreprocessingSummary


class DataTrainingRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    task_id: UUID = Field(alias="taskId")
    data_type: str = Field(alias="dataType", pattern="^(real|synthetic)$")
    user_email: str | None = Field(default=None, alias="userEmail")


class DataTrainingMetricsResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    task_id: UUID = Field(alias="taskId")
    algorithm: str
    accuracy: float
    f1_score: float = Field(alias="f1Score")
    total_training_samples_used: int = Field(alias="totalTrainingSamplesUsed")
    saved_to_model_registry: bool = Field(alias="savedToModelRegistry")
    model_registry_path: str = Field(alias="modelRegistryPath")
    classification_report: dict[str, Any] = Field(alias="classificationReport")


@dataclass(frozen=True)
class WorkerConfig:
    kafka_bootstrap_servers: str
    redis_url: str
    sql_connection_string: str
    generation_request_topic: str = GENERATION_REQUEST_TOPIC
    generation_stream_topic: str = GENERATION_STREAM_TOPIC
    preprocessing_request_topic: str = PREPROCESSING_REQUEST_TOPIC
    preprocessing_result_topic: str = PREPROCESSING_RESULT_TOPIC
    training_request_topic: str = TRAINING_REQUEST_TOPIC
    training_result_topic: str = TRAINING_RESULT_TOPIC
    consumer_group_id: str = DEFAULT_KAFKA_GROUP_ID
    chunk_size: int = DEFAULT_CHUNK_SIZE
    model_registry_dir: str = DEFAULT_MODEL_REGISTRY_DIR
    random_state: int = DEFAULT_RANDOM_STATE


class DataOpsWorker:
    def __init__(self, config: WorkerConfig) -> None:
        self._config = config
        self._redis = Redis.from_url(config.redis_url, decode_responses=True)
        self._consumer = Consumer(
            {
                "bootstrap.servers": config.kafka_bootstrap_servers,
                "group.id": config.consumer_group_id,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
            }
        )
        self._producer: Producer | None = None
        self._running = True
        self._artifact_store = ArtifactStore()
        self._feature_validator = FeatureFrameValidator(FEATURE_RANGE_RULES)
        self._target_validator = TargetSeriesValidator(ALLOWED_WEAKEST_FINGER_LABELS)
        self._model_factory = ModelPipelineFactory(random_state=config.random_state)

    def run_forever(self) -> None:
        subscribed_topics = [
            self._config.generation_request_topic,
            self._config.preprocessing_request_topic,
            self._config.training_request_topic,
        ]
        self._ensure_topics_exist(
            [
                self._config.generation_request_topic,
                self._config.generation_stream_topic,
                self._config.preprocessing_request_topic,
                self._config.preprocessing_result_topic,
                self._config.training_request_topic,
                self._config.training_result_topic,
            ]
        )
        logger.info(
            "Starting DataOps worker Kafka=%s Redis=%s SqlConfigured=%s Topics=%s ChunkSize=%s",
            self._config.kafka_bootstrap_servers,
            self._config.redis_url,
            bool(self._config.sql_connection_string),
            subscribed_topics,
            self._config.chunk_size,
        )
        with start_worker_span(
            "worker.consumer.start",
            {
                "messaging.system": "kafka",
                "messaging.kafka.group": self._config.consumer_group_id,
                "typing.ml.chunk_size": self._config.chunk_size,
                "typing.ml.sql_configured": bool(self._config.sql_connection_string),
            },
        ):
            update_span_attributes(
                **{
                    "typing.ml.topic.generation": self._config.generation_request_topic,
                    "typing.ml.topic.preprocessing": self._config.preprocessing_request_topic,
                    "typing.ml.topic.training": self._config.training_request_topic,
                }
            )
        self._consumer.subscribe(subscribed_topics)

        while self._running:
            message = self._consumer.poll(1.0)
            if message is None:
                continue

            if message.error():
                if message.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error("Kafka consumer error: %s", message.error())
                continue

            payload_text = message.value()
            if payload_text is None:
                self._consumer.commit(message=message, asynchronous=False)
                continue

            request: BaseModel | None = None
            with start_worker_span(
                "worker.kafka.message",
                {
                    "messaging.system": "kafka",
                    "messaging.destination": message.topic(),
                    "messaging.kafka.group": self._config.consumer_group_id,
                },
            ):
                try:
                    if message.topic() == self._config.generation_request_topic:
                        request = DataGenerationRequest.model_validate_json(payload_text)
                        update_span_attributes(task_id=request.task_id, topic=message.topic(), sessions=request.sessions)
                        self._process_generation_request(request)
                    elif message.topic() == self._config.preprocessing_request_topic:
                        request = DataPreprocessingRequest.model_validate_json(payload_text)
                        update_span_attributes(task_id=request.task_id, topic=message.topic())
                        self._process_preprocessing_request(request)
                    elif message.topic() == self._config.training_request_topic:
                        request = DataTrainingRequest.model_validate_json(payload_text)
                        update_span_attributes(
                            task_id=request.task_id,
                            topic=message.topic(),
                            data_type=request.data_type,
                            user_email=request.user_email,
                        )
                        self._process_training_request(request)
                    else:
                        logger.warning("Skipping message from unsupported topic=%s", message.topic())
                        update_span_attributes(topic=message.topic(), unsupported_topic=True)

                    self._consumer.commit(message=message, asynchronous=False)
                except ValidationError as exc:
                    record_span_exception(exc)
                    logger.error("Invalid request payload skipped for topic=%s: %s", message.topic(), exc)
                    self._consumer.commit(message=message, asynchronous=False)
                except Exception as exc:  # noqa: BLE001
                    record_span_exception(exc)
                    logger.exception("Worker failed while handling topic=%s: %s", message.topic(), exc)
                    if request is not None and hasattr(request, "task_id"):
                        self._set_failed_state(getattr(request, "task_id"), str(exc))
                    self._consumer.commit(message=message, asynchronous=False)

    def _ensure_topics_exist(self, topic_names: list[str]) -> None:
        admin_client = AdminClient({"bootstrap.servers": self._config.kafka_bootstrap_servers})
        metadata = admin_client.list_topics(timeout=10)
        existing_topics = set(metadata.topics.keys())
        missing_topics = [topic for topic in topic_names if topic not in existing_topics]

        if not missing_topics:
            logger.info("Verified Kafka topics exist: %s", sorted(set(topic_names)))
            return

        logger.info("Creating missing Kafka topics: %s", missing_topics)
        creation_results = admin_client.create_topics(
            [NewTopic(topic, num_partitions=1, replication_factor=1) for topic in missing_topics]
        )

        for topic_name, future in creation_results.items():
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                if "TOPIC_ALREADY_EXISTS" in str(exc):
                    continue
                raise RuntimeError(f"Failed to create Kafka topic {topic_name}: {exc}") from exc

        logger.info("Kafka topics ready: %s", sorted(set(topic_names)))

    def stop(self) -> None:
        self._running = False
        try:
            self._consumer.close()
        finally:
            if self._producer is not None:
                self._producer.flush(5)
            self._redis.close()

    def _process_generation_request(self, request: DataGenerationRequest) -> None:
        with start_worker_span(
            "worker.process.generation",
            {
                "task.id": str(request.task_id),
                "typing.ml.sessions": request.sessions,
                "typing.ml.inject_outliers": request.inject_outliers,
            },
        ):
            logger.info(
                "Processing generation taskId=%s sessions=%s injectOutliers=%s",
                request.task_id,
                request.sessions,
                request.inject_outliers,
            )
            self._merge_task_state(
                request.task_id,
                status="Processing",
                progressPercentage=10,
                totalSessions=request.sessions,
                processedSessions=0,
                errorMessage=None,
            )

            rng = np.random.default_rng()
            processed_sessions = 0

            for chunk_start in range(0, request.sessions, self._config.chunk_size):
                chunk_size = min(self._config.chunk_size, request.sessions - chunk_start)
                rows = generate_chunk(
                    rng=rng,
                    absolute_start_index=chunk_start,
                    chunk_size=chunk_size,
                    inject_outliers=request.inject_outliers,
                )

                processed_sessions += chunk_size
                is_last_chunk = processed_sessions >= request.sessions
                envelope = TelemetryChunkEnvelope(
                    taskId=request.task_id,
                    isLastChunk=is_last_chunk,
                    data=rows,
                )
                self._publish_json(self._config.generation_stream_topic, envelope)

                progress_percentage = 90 if is_last_chunk else min(
                    89,
                    10 + int((processed_sessions / request.sessions) * 80),
                )
                self._merge_task_state(
                    request.task_id,
                    status="Processing",
                    progressPercentage=progress_percentage,
                    totalSessions=request.sessions,
                    processedSessions=processed_sessions,
                    errorMessage=None,
                )
                update_span_attributes(processed_sessions=processed_sessions, progress_percentage=progress_percentage)

    def _process_preprocessing_request(self, request: DataPreprocessingRequest) -> None:
        with start_worker_span("worker.process.preprocessing", {"task.id": str(request.task_id)}):
            logger.info("Processing preprocessing taskId=%s", request.task_id)
            self._merge_task_state(
                request.task_id,
                status="Preprocessing",
                progressPercentage=15,
                errorMessage=None,
            )

            dataframe = load_preprocessing_dataframe(self._config.sql_connection_string)
            update_span_attributes(rows_scanned=len(dataframe.index))
            if dataframe.empty:
                raise ValueError("Preprocessing dataset is empty. No SessionFeature rows were found.")

            analysis = analyze_preprocessing_outliers(
                dataframe,
                timing_columns=("dwell_time_avg", "flight_time_avg"),
            )
            summary = build_preprocessing_summary(dataframe, analysis)
            outlier_ids = [UUID(value) for value in analysis.outlier_dataframe["session_feature_id"].astype(str).tolist()]
            update_span_attributes(outlier_count=len(outlier_ids), rows_after_preprocessing=summary.total_rows_after_preprocessing)
            result = DataPreprocessingResult(
                taskId=request.task_id,
                outlierIds=outlier_ids,
                summary=summary,
            )
            self._publish_json(self._config.preprocessing_result_topic, result)
            self._merge_task_state(
                request.task_id,
                status="PreprocessingPublished",
                progressPercentage=90,
                errorMessage=None,
            )

    def _process_training_request(self, request: DataTrainingRequest) -> None:
        with start_worker_span(
            "worker.process.training",
            {
                "task.id": str(request.task_id),
                "typing.ml.data_type": request.data_type,
                "enduser.id": request.user_email or "",
            },
        ):
            logger.info("Processing training taskId=%s", request.task_id)
            self._merge_task_state(
                request.task_id,
                status="Training",
                progressPercentage=15,
                errorMessage=None,
            )

            dataframe = load_training_dataframe(self._config.sql_connection_string)
            update_span_attributes(rows_scanned=len(dataframe.index))
            if dataframe.empty:
                raise ValueError("Training dataset is empty after filtering IsOutlier = 0.")

            metrics_result = train_random_forest_from_dataframe(
                dataframe,
                task_id=request.task_id,
                model_registry_dir=self._config.model_registry_dir,
                data_type=request.data_type,
                user_email=request.user_email,
                artifact_store=self._artifact_store,
                feature_validator=self._feature_validator,
                target_validator=self._target_validator,
                model_factory=self._model_factory,
                random_state=self._config.random_state,
            )
            update_span_attributes(
                algorithm=metrics_result.algorithm,
                accuracy=metrics_result.accuracy,
                f1_score=metrics_result.f1_score,
                total_training_samples_used=metrics_result.total_training_samples_used,
            )
            self._publish_json(self._config.training_result_topic, metrics_result)
            self._merge_task_state(
                request.task_id,
                status="TrainingPublished",
                progressPercentage=90,
                errorMessage=None,
            )

    def _publish_json(self, topic: str, payload: BaseModel) -> None:
        with start_worker_span(
            "worker.kafka.publish",
            {
                "messaging.system": "kafka",
                "messaging.destination": topic,
                "payload.type": payload.__class__.__name__,
            },
        ):
            producer = self._get_producer()
            encoded_payload = json.dumps(payload.model_dump(mode="json", by_alias=True), separators=(",", ":"))
            producer.produce(topic, value=encoded_payload)
            producer.flush(5)

    def _get_producer(self) -> Producer:
        if self._producer is None:
            self._producer = Producer({"bootstrap.servers": self._config.kafka_bootstrap_servers})
        return self._producer

    def _set_failed_state(self, task_id: UUID, error_message: str) -> None:
        self._merge_task_state(
            task_id,
            status="Failed",
            progressPercentage=0,
            errorMessage=error_message,
        )

    def _merge_task_state(self, task_id: UUID, **updates: Any) -> None:
        redis_key = f"{REDIS_KEY_PREFIX}{task_id}"
        current_payload = self._redis.get(redis_key)
        state: dict[str, Any] = (
            json.loads(current_payload)
            if current_payload
            else {
                "taskId": str(task_id),
                "status": "Starting",
                "progressPercentage": 0,
                "totalSessions": 0,
                "processedSessions": 0,
                "errorMessage": None,
            }
        )
        state.update(updates)
        state["taskId"] = str(task_id)
        self._redis.set(redis_key, json.dumps(state, separators=(",", ":")))


def resolve_config() -> WorkerConfig:
    kafka_bootstrap_servers = os.environ.get("ConnectionStrings__kafka")
    if not kafka_bootstrap_servers:
        raise RuntimeError("Missing ConnectionStrings__kafka environment variable.")

    redis_raw = os.environ.get("ConnectionStrings__redis") or os.environ.get("ConnectionStrings__cache")
    if not redis_raw:
        raise RuntimeError(
            "Missing Redis connection string. Set ConnectionStrings__redis or the legacy ConnectionStrings__cache."
        )

    sql_connection_string = os.environ.get("ConnectionStrings__typing") or os.environ.get("ConnectionStrings__sql")
    if not sql_connection_string:
        raise RuntimeError("Missing SQL connection string. Set ConnectionStrings__typing or ConnectionStrings__sql.")

    chunk_size = int(os.environ.get("DATAOPS_CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE)))
    if chunk_size <= 0:
        raise RuntimeError("DATAOPS_CHUNK_SIZE must be greater than zero.")

    return WorkerConfig(
        kafka_bootstrap_servers=kafka_bootstrap_servers,
        redis_url=normalize_redis_url(redis_raw),
        sql_connection_string=sql_connection_string,
        generation_request_topic=os.environ.get("DATAOPS_REQUEST_TOPIC")
        or os.environ.get("Kafka__Topics__DataGenerationRequests")
        or GENERATION_REQUEST_TOPIC,
        generation_stream_topic=os.environ.get("DATAOPS_STREAM_TOPIC")
        or os.environ.get("Kafka__Topics__TelemetryDataStream")
        or GENERATION_STREAM_TOPIC,
        preprocessing_request_topic=os.environ.get("DATAOPS_PREPROCESSING_REQUEST_TOPIC")
        or os.environ.get("Kafka__Topics__DataPreprocessingRequests")
        or PREPROCESSING_REQUEST_TOPIC,
        preprocessing_result_topic=os.environ.get("DATAOPS_PREPROCESSING_RESULT_TOPIC")
        or os.environ.get("Kafka__Topics__DataPreprocessingResults")
        or PREPROCESSING_RESULT_TOPIC,
        training_request_topic=os.environ.get("DATAOPS_TRAINING_REQUEST_TOPIC")
        or os.environ.get("Kafka__Topics__DataTrainingRequests")
        or TRAINING_REQUEST_TOPIC,
        training_result_topic=os.environ.get("DATAOPS_TRAINING_RESULT_TOPIC")
        or os.environ.get("Kafka__Topics__DataTrainingResults")
        or TRAINING_RESULT_TOPIC,
        consumer_group_id=os.environ.get("DATAOPS_CONSUMER_GROUP_ID", DEFAULT_KAFKA_GROUP_ID),
        chunk_size=chunk_size,
        model_registry_dir=os.environ.get("TYPING_ML_MODEL_REGISTRY_DIR", DEFAULT_MODEL_REGISTRY_DIR),
        random_state=int(os.environ.get("TYPING_ML_RANDOM_STATE", str(DEFAULT_RANDOM_STATE))),
    )


def normalize_redis_url(raw_value: str) -> str:
    if "://" in raw_value:
        return raw_value

    head, *tail = [segment.strip() for segment in raw_value.split(",") if segment.strip()]
    password = ""
    database = "0"

    for item in tail:
        if item.startswith("password="):
            password = item.split("=", 1)[1]
        elif item.startswith("db="):
            database = item.split("=", 1)[1]

    auth_segment = f":{password}@" if password else ""
    return f"redis://{auth_segment}{head}/{database}"


def build_odbc_connection_string(raw_connection_string: str) -> str:
    normalized = raw_connection_string.strip().strip(";")
    if not normalized:
        raise ValueError("SQL connection string is empty.")

    if "driver=" in normalized.lower():
        return normalized

    driver = os.environ.get("TYPING_ML_SQL_ODBC_DRIVER", DEFAULT_SQL_ODBC_DRIVER)
    segments = [segment.strip() for segment in normalized.split(";") if segment.strip()]
    rewritten_segments = [f"DRIVER={{{driver}}}"]

    for segment in segments:
        key, separator, value = segment.partition("=")
        if not separator:
            rewritten_segments.append(segment)
            continue

        normalized_key = key.strip().lower()
        normalized_value = value.strip()
        if normalized_key == "user id":
            rewritten_segments.append(f"UID={normalized_value}")
        elif normalized_key == "password":
            rewritten_segments.append(f"PWD={normalized_value}")
        elif normalized_key == "initial catalog":
            rewritten_segments.append(f"Database={normalized_value}")
        elif normalized_key == "trustservercertificate":
            rewritten_segments.append(f"TrustServerCertificate={normalize_sql_boolean(normalized_value)}")
        elif normalized_key == "encrypt":
            rewritten_segments.append(f"Encrypt={normalize_sql_boolean(normalized_value)}")
        else:
            rewritten_segments.append(f"{key.strip()}={normalized_value}")

    return ";".join(rewritten_segments)


def normalize_sql_boolean(raw_value: str) -> str:
    if raw_value.lower() in {"true", "yes", "1"}:
        return "yes"
    if raw_value.lower() in {"false", "no", "0"}:
        return "no"
    return raw_value


def load_preprocessing_dataframe(sql_connection_string: str) -> pd.DataFrame:
    return load_sql_dataframe(
        sql_connection_string,
        """
        SELECT
            CAST(sf.Id AS nvarchar(36)) AS session_feature_id,
            CAST((
                sf.DwellLeftPinky + sf.DwellLeftRing + sf.DwellLeftMiddle + sf.DwellLeftIndex +
                sf.DwellRightIndex + sf.DwellRightMiddle + sf.DwellRightRing + sf.DwellRightPinky
            ) / 8.0 AS float) AS dwell_time_avg,
            CAST((
                sf.FlightLeftPinky + sf.FlightLeftRing + sf.FlightLeftMiddle + sf.FlightLeftIndex +
                sf.FlightRightIndex + sf.FlightRightMiddle + sf.FlightRightRing + sf.FlightRightPinky
            ) / 8.0 AS float) AS flight_time_avg,
            CAST(ISNULL(sf.IsOutlier, 0) AS bit) AS is_outlier
        FROM dbo.SessionFeature AS sf;
        """,
    )


def load_training_dataframe(sql_connection_string: str) -> pd.DataFrame:
    return load_sql_dataframe(
        sql_connection_string,
        """
        SELECT
            CAST(ts.Wpm AS float) AS wpm,
            CAST(CASE WHEN ts.Accuracy > 1 THEN ts.Accuracy / 100.0 ELSE ts.Accuracy END AS float) AS accuracy,
            CAST(sf.ErrorLeftPinky AS float) AS error_left_pinky,
            CAST(sf.ErrorLeftRing AS float) AS error_left_ring,
            CAST(sf.ErrorLeftMiddle AS float) AS error_left_middle,
            CAST(sf.ErrorLeftIndex AS float) AS error_left_index,
            CAST(sf.ErrorRightIndex AS float) AS error_right_index,
            CAST(sf.ErrorRightMiddle AS float) AS error_right_middle,
            CAST(sf.ErrorRightRing AS float) AS error_right_ring,
            CAST(sf.ErrorRightPinky AS float) AS error_right_pinky,
            CAST(sf.DwellLeftPinky AS float) AS dwell_left_pinky,
            CAST(sf.DwellLeftRing AS float) AS dwell_left_ring,
            CAST(sf.DwellLeftMiddle AS float) AS dwell_left_middle,
            CAST(sf.DwellLeftIndex AS float) AS dwell_left_index,
            CAST(sf.DwellRightIndex AS float) AS dwell_right_index,
            CAST(sf.DwellRightMiddle AS float) AS dwell_right_middle,
            CAST(sf.DwellRightRing AS float) AS dwell_right_ring,
            CAST(sf.DwellRightPinky AS float) AS dwell_right_pinky,
            CAST(sf.FlightLeftPinky AS float) AS flight_left_pinky,
            CAST(sf.FlightLeftRing AS float) AS flight_left_ring,
            CAST(sf.FlightLeftMiddle AS float) AS flight_left_middle,
            CAST(sf.FlightLeftIndex AS float) AS flight_left_index,
            CAST(sf.FlightRightIndex AS float) AS flight_right_index,
            CAST(sf.FlightRightMiddle AS float) AS flight_right_middle,
            CAST(sf.FlightRightRing AS float) AS flight_right_ring,
            CAST(sf.FlightRightPinky AS float) AS flight_right_pinky,
            LOWER(LTRIM(RTRIM(COALESCE(ts.WeakestFinger,
                (
                    SELECT TOP (1) v.FingerName
                    FROM (VALUES
                        ('left_pinky', sf.ErrorLeftPinky),
                        ('left_ring', sf.ErrorLeftRing),
                        ('left_middle', sf.ErrorLeftMiddle),
                        ('left_index', sf.ErrorLeftIndex),
                        ('right_index', sf.ErrorRightIndex),
                        ('right_middle', sf.ErrorRightMiddle),
                        ('right_ring', sf.ErrorRightRing),
                        ('right_pinky', sf.ErrorRightPinky)
                    ) AS v(FingerName, ErrorRate)
                    ORDER BY v.ErrorRate DESC, v.FingerName ASC
                )
            )))) AS weakest_finger
        FROM dbo.SessionFeature AS sf
        INNER JOIN dbo.TypingSession AS ts ON ts.Id = sf.TypingSessionId
        WHERE ISNULL(sf.IsOutlier, 0) = 0;
        """,
    )


def load_sql_dataframe(sql_connection_string: str, query: str) -> pd.DataFrame:
    with start_worker_span(
        "worker.sql.load_dataframe",
        {
            "db.system": "mssql",
            "db.operation": "SELECT",
        },
    ):
        odbc_connection_string = build_odbc_connection_string(sql_connection_string)
        update_span_attributes(db_query_preview=" ".join(query.split())[:160])
        with pyodbc.connect(odbc_connection_string) as connection:
            dataframe = pd.read_sql_query(query, connection)
            update_span_attributes(db_row_count=len(dataframe.index))
            return dataframe


def build_preprocessing_summary(
    source_dataframe: pd.DataFrame,
    analysis: TimingOutlierAnalysis,
) -> DataPreprocessingSummary:
    dwell_bounds = analysis.bounds_by_column.get("dwell_time_avg")
    flight_bounds = analysis.bounds_by_column.get("flight_time_avg")
    if dwell_bounds is None or flight_bounds is None:
        raise ValueError("Preprocessing analysis did not produce IQR bounds for dwell_time_avg and flight_time_avg.")

    sample_dataframe = analysis.outlier_dataframe.head(DEFAULT_OUTLIER_SAMPLE_SIZE)
    sample_outliers = [
        OutlierSample(
            sessionFeatureId=UUID(str(row["session_feature_id"])),
            dwellTimeAvg=float(row["dwell_time_avg"]),
            flightTimeAvg=float(row["flight_time_avg"]),
        )
        for _, row in sample_dataframe.iterrows()
    ]
    return DataPreprocessingSummary(
        totalRowsScanned=int(len(source_dataframe.index)),
        totalRowsAfterPreprocessing=int(len(analysis.cleaned_dataframe.index)),
        totalOutliersDetected=int(len(analysis.outlier_dataframe.index)),
        dwellTimeAvg=IqrSummary(
            q1=dwell_bounds.q1,
            q3=dwell_bounds.q3,
            iqr=dwell_bounds.iqr,
            lowerBound=dwell_bounds.lower_bound,
            upperBound=dwell_bounds.upper_bound,
        ),
        flightTimeAvg=IqrSummary(
            q1=flight_bounds.q1,
            q3=flight_bounds.q3,
            iqr=flight_bounds.iqr,
            lowerBound=flight_bounds.lower_bound,
            upperBound=flight_bounds.upper_bound,
        ),
        sampleOutliers=sample_outliers,
    )


def train_random_forest_from_dataframe(
    dataframe: pd.DataFrame,
    *,
    task_id: UUID,
    model_registry_dir: str,
    data_type: str,
    user_email: str | None,
    artifact_store: ArtifactStore,
    feature_validator: FeatureFrameValidator,
    target_validator: TargetSeriesValidator,
    model_factory: ModelPipelineFactory,
    random_state: int,
) -> DataTrainingMetricsResult:
    feature_frame = feature_validator.validate(
        dataframe,
        required_columns=TRAIN_FEATURE_COLUMNS,
        context="Clean training dataset",
    )
    target_series = target_validator.validate(
        dataframe[TARGET_COLUMN],
        target_name=TARGET_COLUMN,
        context="Clean training dataset",
    )

    try:
        x_train, x_test, y_train, y_test = sk_model_selection.train_test_split(
            feature_frame,
            target_series,
            test_size=0.2,
            random_state=random_state,
            stratify=target_series,
        )
    except ValueError as ex:
        raise ValueError(
            "Failed to create stratified train/test split for cleaned training dataset. "
            f"Class distribution: {target_series.value_counts().to_dict()}"
        ) from ex

    model = model_factory.create(Algorithm.RANDOM_FOREST.value)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    classification_report = sk_metrics.classification_report(y_test, predictions, output_dict=True)
    accuracy = float(sk_metrics.accuracy_score(y_test, predictions))
    f1_score = float(sk_metrics.f1_score(y_test, predictions, average="macro"))

    normalized_data_type = data_type.strip().lower()
    if normalized_data_type not in {"real", "synthetic"}:
        raise ValueError("dataType must be either 'real' or 'synthetic'.")

    lineage_folder = "production" if normalized_data_type == "real" else "experiments"
    model_directory = Path(model_registry_dir) / lineage_folder
    model_directory.mkdir(parents=True, exist_ok=True)

    safe_user_email = sanitize_filename_segment(user_email or "unknown_user")
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_file_name = f"{Algorithm.RANDOM_FOREST.value}_{normalized_data_type}_{safe_user_email}_{timestamp_utc}.pkl"
    model_registry_path = (model_directory / model_file_name).resolve()

    artifact = ModelArtifact.from_training(
        model=model,
        model_name=Algorithm.RANDOM_FOREST.value,
        feature_names=TRAIN_FEATURE_COLUMNS,
        target_name=TARGET_COLUMN,
    )
    artifact_store.save_model_artifact(artifact, str(model_registry_path))

    return DataTrainingMetricsResult(
        taskId=task_id,
        algorithm=Algorithm.RANDOM_FOREST.value,
        accuracy=accuracy,
        f1Score=f1_score,
        totalTrainingSamplesUsed=int(len(feature_frame.index)),
        savedToModelRegistry=True,
        modelRegistryPath=str(model_registry_path),
        classificationReport=classification_report,
    )


def sanitize_filename_segment(value: str) -> str:
    normalized = value.strip().replace(" ", "_")
    normalized = re.sub(r"[^A-Za-z0-9@._-]+", "_", normalized)
    return normalized or "unknown"


def generate_chunk(
    *,
    rng: np.random.Generator,
    absolute_start_index: int,
    chunk_size: int,
    inject_outliers: bool,
) -> list[dict[str, Any]]:
    row_numbers = np.arange(absolute_start_index + 1, absolute_start_index + chunk_size + 1)

    telemetry: dict[str, np.ndarray] = {
        "WPM": clip_rounded(rng.normal(loc=48, scale=9, size=chunk_size), minimum=12, maximum=180),
        "Accuracy": np.round(rng.uniform(0.86, 0.995, size=chunk_size), 4),
    }

    for column in ERROR_COLUMNS:
        telemetry[column] = np.round(rng.uniform(0.0, 0.06, size=chunk_size), 4)

    for column in DWELL_COLUMNS:
        telemetry[column] = clip_rounded(rng.normal(loc=105, scale=18, size=chunk_size), minimum=25, maximum=900)

    for column in FLIGHT_COLUMNS:
        telemetry[column] = clip_rounded(rng.normal(loc=185, scale=28, size=chunk_size), minimum=30, maximum=1200)

    if inject_outliers:
        outlier_mask = row_numbers % 10 == 0
        if np.any(outlier_mask):
            for column in TIME_COLUMNS:
                telemetry[column][outlier_mask] = OUTLIER_TIME_MS

    return materialize_rows(telemetry, chunk_size)


def clip_rounded(values: np.ndarray, *, minimum: int, maximum: int) -> np.ndarray:
    return np.clip(np.rint(values).astype(int), minimum, maximum)


def materialize_rows(telemetry: dict[str, np.ndarray], chunk_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index in range(chunk_size):
        rows.append({column: coerce_scalar(telemetry[column][row_index]) for column in ORDERED_COLUMNS})
    return rows


def coerce_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def install_signal_handlers(worker: DataOpsWorker) -> None:
    def _handle_signal(signum: int, _frame: Any) -> None:
        logger.info("Received signal %s, shutting down worker.", signum)
        worker.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


def main() -> None:
    configure_optional_otel()
    config = resolve_config()
    worker = DataOpsWorker(config)
    install_signal_handlers(worker)
    worker.run_forever()


if __name__ == "__main__":
    main()
