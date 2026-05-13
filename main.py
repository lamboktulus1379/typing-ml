from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager, suppress
from typing import Any, Iterable
from uuid import UUID

import numpy as np
from confluent_kafka import Consumer, KafkaError, Producer
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from redis.asyncio import Redis


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("typing-ml.dataops")

REQUEST_TOPIC = "data-generation-requests"
STREAM_TOPIC = "telemetry-data-stream"
REDIS_KEY_PREFIX = "dataops:task:"
DEFAULT_CHUNK_SIZE = 5_000
DEFAULT_KAFKA_GROUP_ID = "typing-ml-dataops"
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


class AppConfig(BaseModel):
    kafka_bootstrap_servers: str = Field(min_length=1)
    redis_url: str = Field(min_length=1)
    request_topic: str = REQUEST_TOPIC
    stream_topic: str = STREAM_TOPIC
    consumer_group_id: str = DEFAULT_KAFKA_GROUP_ID
    chunk_size: PositiveInt = DEFAULT_CHUNK_SIZE


class DataGenerationRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    task_id: UUID = Field(alias="taskId")
    sessions: PositiveInt
    inject_outliers: bool = Field(alias="injectOutliers")


class TelemetryRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    WPM: int
    Accuracy: float
    ErrorLeftPinky: float
    ErrorLeftRing: float
    ErrorLeftMiddle: float
    ErrorLeftIndex: float
    ErrorRightIndex: float
    ErrorRightMiddle: float
    ErrorRightRing: float
    ErrorRightPinky: float
    DwellLeftPinky: int
    DwellLeftRing: int
    DwellLeftMiddle: int
    DwellLeftIndex: int
    DwellRightIndex: int
    DwellRightMiddle: int
    DwellRightRing: int
    DwellRightPinky: int
    FlightLeftPinky: int
    FlightLeftRing: int
    FlightLeftMiddle: int
    FlightLeftIndex: int
    FlightRightIndex: int
    FlightRightMiddle: int
    FlightRightRing: int
    FlightRightPinky: int


class TelemetryChunkMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    task_id: UUID = Field(alias="taskId")
    is_last_chunk: bool = Field(alias="isLastChunk")
    data: list[TelemetryRow]


def resolve_config() -> AppConfig:
    kafka_bootstrap_servers = (
        os.getenv("ConnectionStrings__kafka")
        or os.getenv("KAFKA_BROKER")
        or os.getenv("KAFKA_SERVER")
    )
    if not kafka_bootstrap_servers:
        raise RuntimeError(
            "Kafka bootstrap servers are not configured. Set ConnectionStrings__kafka or KAFKA_BROKER."
        )

    redis_raw = (
        os.getenv("ConnectionStrings__redis")
        or os.getenv("ConnectionStrings__cache")
        or os.getenv("REDIS_URL")
    )
    if not redis_raw:
        raise RuntimeError(
            "Redis connection is not configured. Set ConnectionStrings__redis, ConnectionStrings__cache, or REDIS_URL."
        )

    return AppConfig(
        kafka_bootstrap_servers=kafka_bootstrap_servers,
        redis_url=normalize_redis_url(redis_raw),
        request_topic=os.getenv("DATAOPS_REQUEST_TOPIC", REQUEST_TOPIC),
        stream_topic=os.getenv("DATAOPS_STREAM_TOPIC", STREAM_TOPIC),
        consumer_group_id=os.getenv("DATAOPS_CONSUMER_GROUP_ID", DEFAULT_KAFKA_GROUP_ID),
        chunk_size=int(os.getenv("DATAOPS_CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE))),
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


class DataOpsPipelineWorker:
    def __init__(self, config: AppConfig) -> None:
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
        self._producer = Producer({"bootstrap.servers": config.kafka_bootstrap_servers})
        self._stop_event = asyncio.Event()
        self._consumer_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        logger.info(
            "Starting DataOps worker with Kafka=%s Redis=%s RequestTopic=%s StreamTopic=%s ChunkSize=%s",
            self._config.kafka_bootstrap_servers,
            self._config.redis_url,
            self._config.request_topic,
            self._config.stream_topic,
            self._config.chunk_size,
        )
        self._consumer.subscribe([self._config.request_topic])
        self._consumer_task = asyncio.create_task(self._consume_loop())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._consumer_task is not None:
            self._consumer_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._consumer_task

        await asyncio.to_thread(self._consumer.close)
        await asyncio.to_thread(self._producer.flush, 5)
        await self._redis.close()
        await self._redis.connection_pool.disconnect()

    async def _consume_loop(self) -> None:
        while not self._stop_event.is_set():
            message = await asyncio.to_thread(self._consumer.poll, 1.0)
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
                await self._process_request(request)
                await asyncio.to_thread(self._consumer.commit, message=message, asynchronous=False)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to process generation request: %s", exc)
                if request is not None:
                    await self._set_failed_state(request.task_id, str(exc), request.sessions)
                await asyncio.to_thread(self._consumer.commit, message=message, asynchronous=False)

    async def _process_request(self, request: DataGenerationRequest) -> None:
        logger.info(
            "Received generation request taskId=%s sessions=%s injectOutliers=%s",
            request.task_id,
            request.sessions,
            request.inject_outliers,
        )
        await self._merge_task_state(
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
            rows = generate_telemetry_chunk(
                rng=rng,
                absolute_start_index=chunk_start,
                chunk_size=chunk_size,
                inject_outliers=request.inject_outliers,
            )
            processed_sessions += chunk_size
            is_last_chunk = processed_sessions >= request.sessions

            chunk_message = TelemetryChunkMessage(
                taskId=request.task_id,
                isLastChunk=is_last_chunk,
                data=[TelemetryRow(**row) for row in rows],
            )

            await self._publish_chunk(chunk_message)

            progress = 90 if is_last_chunk else min(
                89,
                10 + int((processed_sessions / request.sessions) * 80),
            )
            await self._merge_task_state(
                request.task_id,
                status="Processing",
                progressPercentage=progress,
                totalSessions=request.sessions,
                processedSessions=processed_sessions,
                errorMessage=None,
            )

        logger.info("Finished streaming taskId=%s to topic=%s", request.task_id, self._config.stream_topic)

    async def _publish_chunk(self, chunk_message: TelemetryChunkMessage) -> None:
        payload = json.dumps(chunk_message.model_dump(mode="json"), separators=(",", ":"))

        def _send() -> None:
            self._producer.produce(self._config.stream_topic, value=payload)
            self._producer.flush(5)

        await asyncio.to_thread(_send)

    async def _set_failed_state(self, task_id: UUID, error_message: str, total_sessions: int) -> None:
        await self._merge_task_state(
            task_id,
            status="Failed",
            progressPercentage=0,
            totalSessions=total_sessions,
            processedSessions=0,
            errorMessage=error_message,
        )

    async def _merge_task_state(self, task_id: UUID, **updates: Any) -> None:
        redis_key = f"{REDIS_KEY_PREFIX}{task_id}"
        current_payload = await self._redis.get(redis_key)
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
        await self._redis.set(redis_key, json.dumps(state, separators=(",", ":")))


def generate_telemetry_chunk(
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
    ordered_columns = ("WPM", "Accuracy", *ERROR_COLUMNS, *DWELL_COLUMNS, *FLIGHT_COLUMNS)
    rows: list[dict[str, Any]] = []

    for row_index in range(chunk_size):
        rows.append(
            {
                column: coerce_scalar(telemetry[column][row_index])
                for column in ordered_columns
            }
        )

    return rows


def coerce_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


@asynccontextmanager
async def lifespan(app: FastAPI) -> Iterable[None]:
    config = resolve_config()
    worker = DataOpsPipelineWorker(config)
    app.state.worker = worker
    await worker.start()

    try:
        yield
    finally:
        await worker.stop()


app = FastAPI(
    title="typing-ml DataOps Worker",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "typing-ml-dataops-worker", "status": "running"}
