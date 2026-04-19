# Requirements: typing-ml FastAPI Retraining Worker

## Goal

Provide a training worker endpoint that accepts in-memory dataset rows, trains candidate models, selects the best model by F1-score, writes immutable production artifacts, updates an active-model pointer, and hot-reloads runtime inference without server restart.

## In Scope

- New endpoint: `POST /train`.
- Pydantic validation for retraining payload.
- Candidate algorithm training for:
  - logistic_regression
  - random_forest
  - xgboost
- Deterministic winner selection by F1-score.
- Save winner artifact to a timestamped immutable path under `models/production/`.
- Maintain an active-model pointer file at `models/active_production_model.json`.
- Hot-reload in-process model used by `/predict` and `/predict_batch`.

## Out of Scope

- Orchestration scheduling logic (handled by Typing API).
- SQL persistence of training history (handled by Typing API).
- Frontend wizard behavior.

## Endpoint Contract

### Request

- Method: `POST`
- Route: `/train`
- Payload: array of typing-session feature rows.

Minimum row shape:

```json
[
  {
    "wpm": 68,
    "accuracy": 100,
    "dwell_left_pinky": 100.5,
    "flight_left_pinky": 223.7,
    "error_left_pinky": 0,
    "weakest_finger": "left_pinky"
  }
]
```

Operational requirement:
- Full feature schema used by current model factory should be validated before training starts.

### Response

```json
{
  "algorithm": "xgboost",
  "accuracy": 0.94,
  "f1_score": 0.91,
  "rows_processed": 150
}
```

Optional extension (recommended for UI evaluation step):
- Include per-algorithm comparison metrics in a `candidates` object for transparency.

## Training Pipeline Requirements

- ML-RQ-01 Schema validation:
  - Validate required numeric features and target `weakest_finger` via Pydantic and pipeline validators.

- ML-RQ-02 Dataframe preparation:
  - Convert request payload to Pandas DataFrame.
  - Split into feature frame and target series.

- ML-RQ-03 Standardization:
  - Use existing standardized pipeline (`StandardScaler + classifier`) from `ModelPipelineFactory`.

- ML-RQ-04 Candidate training:
  - Train 3 candidate models in one retrain call.
  - Run candidates concurrently when possible.

- ML-RQ-05 Evaluation:
  - Compute accuracy and F1-score per candidate on the same split policy.

- ML-RQ-06 Winner selection:
  - Primary criterion: highest F1-score.
  - Deterministic tie-break requirement:
    - Higher accuracy wins ties.
    - If still tied, lexical order of algorithm key for deterministic reproducibility.

- ML-RQ-07 Artifact output:
  - Save winner as dict-based artifact to a timestamped immutable path such as `models/production/typing-prod-<timestamp>-<algorithm>.joblib`.
  - Artifact names must never overwrite a previous production retrain artifact.

- ML-RQ-08 Active pointer metadata:
  - Persist `models/active_production_model.json` containing the active artifact path, algorithm, and promotion timestamp.
  - Runtime startup should prefer the active-model pointer when no explicit model path override is configured.

- ML-RQ-09 Hot reload:
  - Replace active in-memory model/artifact atomically after successful save.
  - `/predict`, `/predict_batch`, and `/metadata` must reflect new model immediately after successful train response.

## Runtime Safety Requirements

- ML-NQ-01 Concurrency safety:
  - Inference requests must not observe partially-updated model state during hot reload.

- ML-NQ-02 Failure safety:
  - If any unrecoverable training failure occurs, keep current active model unchanged.

- ML-NQ-03 Observability:
  - Log request row count, candidate metrics, selected winner, artifact path, and active pointer path.

- ML-NQ-04 Backward compatibility:
  - Existing endpoints (`/health`, `/metadata`, `/predict`, `/predict_batch`) remain operational.

## Acceptance Criteria

- ML-AC-01 `POST /train` validates payload and returns winner metrics.
- ML-AC-02 Winner artifact is written to a timestamped immutable production path and no previous artifact is overwritten.
- ML-AC-03 Active-model pointer metadata is updated to the newly promoted artifact.
- ML-AC-04 Inference endpoints use reloaded model without process restart.
- ML-AC-05 Failed retraining does not replace active model.
- ML-AC-06 Tie behavior is deterministic and test-covered.

## Implementation Task Breakdown

- ML-T01 Add Pydantic request model for training rows and payload list.
- ML-T02 Implement trainer orchestration using existing ml_pipeline services.
- ML-T03 Add concurrent candidate execution and metric aggregation.
- ML-T04 Add deterministic winner selector.
- ML-T05 Persist winning artifact to an immutable timestamped production path.
- ML-T06 Write and read active-model pointer metadata.
- ML-T07 Implement thread-safe model hot-reload in FastAPI app state.
- ML-T08 Add `/train` endpoint wiring and response contract.
- ML-T09 Add tests for happy path, invalid payload, pointer resolution, and hot-reload behavior.
