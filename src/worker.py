from __future__ import annotations

import json
import logging
import os
import signal
import sys
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import numpy as np
from confluent_kafka import Consumer, KafkaError, Producer
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError
from redis import Redis


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("typing_ml.worker")

REQUEST_TOPIC = "data-generation-requests"
STREAM_TOPIC = "telemetry-data-stream"
REDIS_KEY_PREFIX = "dataops:task:"
DEFAULT_CHUNK_SIZE = 5_000
DEFAULT_KAFKA_GROUP_ID = "typing-ml-dataops-worker"
OUTLIER_TIME_MS = 5_000

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


@dataclass(frozen=True)
class WorkerConfig:
    kafka_bootstrap_servers: str
    redis_url: str
    request_topic: str = REQUEST_TOPIC
    stream_topic: str = STREAM_TOPIC
    consumer_group_id: str = DEFAULT_KAFKA_GROUP_ID
    chunk_size: int = DEFAULT_CHUNK_SIZE


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

    def run_forever(self) -> None:
        logger.info(
            "Starting DataOps worker Kafka=%s Redis=%s RequestTopic=%s StreamTopic=%s ChunkSize=%s",
            self._config.kafka_bootstrap_servers,
            self._config.redis_url,
            self._config.request_topic,
            self._config.stream_topic,
            self._config.chunk_size,
        )
        self._consumer.subscribe([self._config.request_topic])

        while self._running:
            message = self._consumer.poll(1.0)
            if message is None:
                continue

            if message.error():
                if message.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error("Kafka consumer error: %s", message.error())
                continue

            request: DataGenerationRequest | None = None
            try:
                request = DataGenerationRequest.model_validate_json(message.value())
                self._process_request(request)
                self._consumer.commit(message=message, asynchronous=False)
            except ValidationError as exc:
                logger.error("Invalid request payload skipped: %s", exc)
                self._consumer.commit(message=message, asynchronous=False)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Worker failed while handling request: %s", exc)
                if request is not None:
                    self._set_failed_state(request.task_id, request.sessions, str(exc))
                self._consumer.commit(message=message, asynchronous=False)

    def stop(self) -> None:
        self._running = False
        try:
            self._consumer.close()
        finally:
            if self._producer is not None:
                self._producer.flush(5)
            self._redis.close()

    def _process_request(self, request: DataGenerationRequest) -> None:
        logger.info(
            "Processing taskId=%s sessions=%s injectOutliers=%s",
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

        # Generate one chunk at a time so large requests never materialize the full
        # dataset in memory. Each iteration only holds a single 5,000-row payload.
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
            self._publish_chunk(envelope)

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

    def _publish_chunk(self, envelope: TelemetryChunkEnvelope) -> None:
        payload = json.dumps(envelope.model_dump(mode="json", by_alias=True), separators=(",", ":"))
        producer = self._get_producer()
        producer.produce(self._config.stream_topic, value=payload)
        producer.flush(5)

    def _get_producer(self) -> Producer:
        if self._producer is None:
            self._producer = Producer({"bootstrap.servers": self._config.kafka_bootstrap_servers})
        return self._producer

    def _set_failed_state(self, task_id: UUID, total_sessions: int, error_message: str) -> None:
        self._merge_task_state(
            task_id,
            status="Failed",
            progressPercentage=0,
            totalSessions=total_sessions,
            processedSessions=0,
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
        raise RuntimeError("Missing Redis connection string. Set ConnectionStrings__redis or the legacy ConnectionStrings__cache.")

    chunk_size = int(os.environ.get("DATAOPS_CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE)))
    if chunk_size <= 0:
        raise RuntimeError("DATAOPS_CHUNK_SIZE must be greater than zero.")

    return WorkerConfig(
        kafka_bootstrap_servers=kafka_bootstrap_servers,
        redis_url=normalize_redis_url(redis_raw),
        request_topic=os.environ.get("DATAOPS_REQUEST_TOPIC")
        or os.environ.get("Kafka__Topics__DataGenerationRequests")
        or REQUEST_TOPIC,
        stream_topic=os.environ.get("DATAOPS_STREAM_TOPIC")
        or os.environ.get("Kafka__Topics__TelemetryDataStream")
        or STREAM_TOPIC,
        consumer_group_id=os.environ.get("DATAOPS_CONSUMER_GROUP_ID", DEFAULT_KAFKA_GROUP_ID),
        chunk_size=chunk_size,
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
            # Mark every 10th logical row with extreme timing values to stress-test
            # downstream validation, anomaly handling, and SQL ingestion paths.
            for column in TIME_COLUMNS:
                telemetry[column][outlier_mask] = OUTLIER_TIME_MS

    return materialize_rows(telemetry, chunk_size)


def clip_rounded(values: np.ndarray, *, minimum: int, maximum: int) -> np.ndarray:
    return np.clip(np.rint(values).astype(int), minimum, maximum)


def materialize_rows(telemetry: dict[str, np.ndarray], chunk_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index in range(chunk_size):
        rows.append(
            {
                column: coerce_scalar(telemetry[column][row_index])
                for column in ORDERED_COLUMNS
            }
        )
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
    config = resolve_config()
    worker = DataOpsWorker(config)
    install_signal_handlers(worker)
    worker.run_forever()


if __name__ == "__main__":
    main()
