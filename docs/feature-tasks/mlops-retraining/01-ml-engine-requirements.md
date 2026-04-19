# Requirements: typing-ml FastAPI Retraining Worker

## Goal

Provide a training worker endpoint that accepts in-memory dataset rows, trains candidate models, selects the best model by F1-score, writes immutable production artifacts, updates an active-model pointer, hot-reloads runtime inference without server restart, and supports per-user personalized training requests without changing the winner/evaluation response contract used by the .NET orchestrator. Also support inference cold-start handling so `/predict` can fall back to a global baseline model when a personalized model does not yet exist for the requested user.

## In Scope

- New endpoint: `POST /train`.
- Pydantic validation for retraining payload.
- Candidate algorithm training for:
  - logistic_regression
  - random_forest
  - xgboost
- Deterministic winner selection by F1-score.
- Save winner artifact to a timestamped immutable path under `models/production/`.
- Personalized retraining request support via top-level `user_id`.
- Strict per-user dataset filtering before the 80/20 train/test split.
- Per-user production artifact naming so one user's promoted artifact does not overwrite another's file.
- Maintain an active-model pointer file at `models/active_production_model.json`.
- Hot-reload in-process model used by `/predict` and `/predict_batch`.
- Cold-start fallback in `POST /predict` for user-specific inference.
- Predict response metadata flag describing whether global fallback was used.

## Out of Scope

- Orchestration scheduling logic (handled by Typing API).
- SQL persistence of training history (handled by Typing API).
- Frontend wizard behavior.

## Endpoint Contract

### Predict Inference Request

- Method: `POST`
- Route: `/predict`
- Payload: object with one `row` and optional `user_id`.

Example shape:

```json
{
  "user_id": "9f55a4ee-7be6-4c54-a5c6-bf173ea2ad74",
  "row": {
    "wpm": 68,
    "accuracy": 1.0,
    "dwell_left_pinky": 100.5,
    "flight_left_pinky": 223.7,
    "error_left_pinky": 0.0
  }
}
```

Predict operational requirements:
- If `user_id` is present, inference must first attempt to load that user's personalized production artifact.
- If the personalized artifact is missing, inference must catch `FileNotFoundError` and fall back to the configured global baseline model.
- Predict responses must include `is_fallback_used` so upstream callers can distinguish personalized inference from global fallback inference.

### Request

- Method: `POST`
- Route: `/train`
- Payload: object with `user_id`, `is_dry_run`, and array of typing-session feature rows.

Minimum row shape:

```json
{
  "user_id": "9f55a4ee-7be6-4c54-a5c6-bf173ea2ad74",
  "is_dry_run": true,
  "rows": [
    {
      "wpm": 68,
      "accuracy": 100,
      "dwell_left_pinky": 100.5,
      "flight_left_pinky": 223.7,
      "error_left_pinky": 0,
      "weakest_finger": "left_pinky"
    }
  ]
}
```

Operational requirement:
- Full feature schema used by current model factory should be validated before training starts.
- The retraining request must include a non-empty `user_id` value for personalized retrain mode.
- Before deduplication and before the 80/20 split, the worker must reduce the dataset to rows matching the requested `user_id` only.
- If the filtered dataset is empty, the worker must reject the request.

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
  - Normalize top-level `user_id` and apply a strict filter so the DataFrame contains only rows for that user.
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
  - Save winner as dict-based artifact to a timestamped immutable path such as `models/production/typing-prod-<timestamp>-<user_id>-<algorithm>.joblib`.
  - Artifact names must never overwrite a previous production retrain artifact.

- ML-RQ-08 Active pointer metadata:
  - Persist `models/active_production_model.json` containing the active artifact path, algorithm, and promotion timestamp.
  - Runtime startup should prefer the active-model pointer when no explicit model path override is configured.

- ML-RQ-09 Hot reload:
  - Replace active in-memory model/artifact atomically after successful save.
  - `/predict`, `/predict_batch`, and `/metadata` must reflect new model immediately after successful train response.

- ML-RQ-10 Predict personalized model selection:
  - `POST /predict` must accept optional `user_id`/`userId`.
  - When `user_id` is provided, inference must attempt to resolve and load the latest personalized model artifact for that user without modifying existing feature validation or prediction logic.

- ML-RQ-11 Predict cold-start fallback:
  - Personalized inference loading must wrap the user-specific artifact load in a `try-except FileNotFoundError` block.
  - On `FileNotFoundError`, inference must load the configured global baseline model instead of failing the request.

- ML-RQ-12 Predict response metadata:
  - `POST /predict` responses must include `is_fallback_used: bool`.
  - `is_fallback_used` is `false` when a personalized model is used and `true` when the global baseline model is used.

## Runtime Safety Requirements

- ML-NQ-01 Concurrency safety:
  - Inference requests must not observe partially-updated model state during hot reload.

- ML-NQ-02 Failure safety:
  - If any unrecoverable training failure occurs, keep current active model unchanged.

- ML-NQ-03 Observability:
  - Log request row count, candidate metrics, selected winner, artifact path, and active pointer path.

- ML-NQ-04 Backward compatibility:
  - Existing endpoints (`/health`, `/metadata`, `/predict`, `/predict_batch`) remain operational.
  - The `/train` response schema used by the .NET orchestrator must remain unchanged.

## Acceptance Criteria

- ML-AC-01 `POST /train` validates payload and returns winner metrics.
- ML-AC-02 Winner artifact is written to a timestamped immutable production path and no previous artifact is overwritten.
- ML-AC-03 Active-model pointer metadata is updated to the newly promoted artifact.
- ML-AC-04 Inference endpoints use reloaded model without process restart.
- ML-AC-05 Failed retraining does not replace active model.
- ML-AC-06 Tie behavior is deterministic and test-covered.
- ML-AC-07 Personalized retraining requests filter the dataset to the requested `user_id` before the 80/20 split.
- ML-AC-08 Production artifact filenames include the requested `user_id` while keeping the response schema unchanged.
- ML-AC-09 `POST /predict` uses a personalized model when a matching user artifact exists.
- ML-AC-10 `POST /predict` falls back to the global baseline model when the personalized artifact is missing.
- ML-AC-11 `POST /predict` includes `is_fallback_used` in the response payload.

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
- ML-T10 Add request-level `user_id` support and pre-split dataset filtering in the FastAPI retraining path.
- ML-T11 Add per-user production artifact naming and tests proving the response contract is unchanged.
- ML-T12 Add `user_id` support to the predict request contract and route wiring.
- ML-T13 Add inference service helpers for personalized model resolution plus global fallback on `FileNotFoundError`.
- ML-T14 Add predict endpoint tests covering personalized inference, cold-start fallback, and `is_fallback_used`.
