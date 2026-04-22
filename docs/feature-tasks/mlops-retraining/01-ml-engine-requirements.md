# Requirements: typing-ml FastAPI Retraining Worker

## Goal

Provide training worker endpoints that load the persisted dataset, train candidate models, select the best model by F1-score, save a fixed global or personalized production artifact, hot-reload runtime inference without server restart, and support per-user personalized training requests without changing the core winner/evaluation logic. Also support inference cold-start handling so `/predict` can fall back to a global baseline model when a personalized model does not yet exist for the requested user.

## AutoML Algorithm Selection via K-Fold Cross Validation

To improve academic rigor and reduce the risk of a lucky or unlucky single split, the AutoML selection stage uses 5-fold cross validation instead of a single 80/20 holdout split. In a standard holdout evaluation, the measured performance can change materially depending on which rows happen to fall into the training partition and which rows happen to fall into the test partition. That makes the winner sensitive to partition noise rather than model quality.

Under 5-fold cross validation, the cleaned personalized dataset is partitioned into five folds. Each candidate algorithm is trained five times. On each iteration, four folds are used for fitting and the remaining fold is used for validation. This process is repeated until every fold has served once as the validation fold. The three candidate algorithms evaluated in this AutoML stage are XGBoost, Random Forest, and Logistic Regression.

For each algorithm, the validation outcome from the five folds is summarized as a Mean Macro F1-Score. Macro F1 is used because it weights each class equally and is therefore more appropriate than raw accuracy when class balance may vary across users or fatigue states. The Winning Algorithm is the model with the highest Mean Macro F1-Score across the five folds. After that selection step, the winner is refit on 100% of the cleaned personalized dataset so the persisted artifact is trained on all available evidence before it is saved.

## In Scope

- New endpoint: `POST /train/global`.
- New endpoint: `POST /train/personal`.
- Pydantic validation for retraining payload.
- Dataset loading from the processed training CSV.
- Candidate algorithm training for:
  - logistic_regression
  - random_forest
  - xgboost
- Deterministic winner selection by F1-score.
- Save the global winner artifact exactly as `models/model_production_global.joblib`.
- Save the personalized winner artifact exactly as `models/model_production_{user_id}.joblib`.
- Personalized retraining request support via top-level `user_id`.
- Strict per-user dataset filtering before the 5-fold cross-validation stage.
- Personalized requests must reject insufficient data when filtered rows are fewer than 20.
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

### Global Training Request

- Method: `POST`
- Route: `/train/global`
- Payload: empty body.

Operational requirement:
- The worker must load the full processed dataset from disk.
- Global training must use all available rows without user filtering.
- The winning promoted artifact must be written exactly to `models/model_production_global.joblib`.

### Personalized Training Request

- Method: `POST`
- Route: `/train/personal`
- Payload: object with `user_id` and optional `rows`.

Minimum row shape:

```json
{
  "user_id": "9f55a4ee-7be6-4c54-a5c6-bf173ea2ad74",
  "rows": []
}
```

Operational requirement:
- Full feature schema used by current model factory should be validated before training starts.
- The retraining request must include a non-empty `user_id` value for personalized retrain mode.
- If `rows` is omitted or empty, the worker must load the persisted processed dataset from disk.
- If `rows` is present, the worker must merge those supplied rows with the persisted processed dataset when that dataset is available.
- If `rows` is present and the persisted processed dataset is unavailable, the worker may train from the supplied rows only.
- Before deduplication and before 5-fold cross validation, the worker must reduce the effective dataset to rows matching the requested `user_id` only.
- If the filtered dataset contains fewer than 20 rows, the worker must reject the request with HTTP 400 and message `Insufficient data`.
- The winning promoted artifact must be written exactly to `models/model_production_{user_id}.joblib`.

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
  - Load the persisted training dataset from disk into a Pandas DataFrame.
  - Permit the personalized route to receive pre-filterable in-memory rows from the Typing API admin workflow.
  - For personalized training, normalize top-level `user_id` and apply a strict filter so the DataFrame contains only rows for that user.
  - Before feature extraction and before 5-fold cross-validation scoring, run a timing-telemetry cleaning step for dwell/flight features.
  - The cleaning step must drop rows with impossible negative timing values.
  - The cleaning step must drop rows above logical hard caps:
    - dwell timing columns: `> 2000 ms`
    - flight timing columns: `> 3000 ms`
  - The cleaning step must then apply an upper-bound IQR filter to dwell/flight timing columns using `Q3 + 1.5 * IQR`.
  - The worker must log or print DataFrame shape before and after cleaning for auditability.
  - Split into feature frame and target series.

- ML-RQ-03 Standardization:
  - Use existing standardized pipeline (`StandardScaler + classifier`) from `ModelPipelineFactory`.

- ML-RQ-04 Candidate training:
  - Train 3 candidate models in one retrain call.
  - Evaluate candidates with 5-fold cross validation.
  - Run candidates concurrently when possible.

- ML-RQ-05 Evaluation:
  - Compute accuracy and F1-score per candidate across the same 5 folds.
  - Use Mean Macro F1-Score as the primary AutoML selection metric.

- ML-RQ-06 Winner selection:
  - Primary criterion: highest Mean Macro F1-Score.
  - Deterministic tie-break requirement:
    - Higher mean accuracy wins ties.
    - If still tied, lexical order of algorithm key for deterministic reproducibility.

- ML-RQ-07 Artifact output:
  - Save the global winner as dict-based artifact exactly to `models/model_production_global.joblib`.
  - Save the personalized winner as dict-based artifact exactly to `models/model_production_{user_id}.joblib`.
  - The saved artifact must preserve the same feature names, target metadata, and label class metadata already used by inference.

- ML-RQ-08 Runtime selection semantics:
  - `POST /predict` must first attempt to load `models/model_production_{user_id}.joblib` when `user_id` is supplied.
  - If that file does not exist, the API must log a warning and fall back to `models/model_production_global.joblib`.
  - Runtime startup should continue to support the existing default model selection logic for non-personalized flows.

- ML-RQ-09 Hot reload:
  - Replace active in-memory model/artifact atomically after successful save.
  - `/predict`, `/predict_batch`, and `/metadata` must reflect new model immediately after successful train response.

- ML-RQ-10 Predict personalized model selection:
  - `POST /predict` must accept `user_id`/`userId` plus one keystroke feature row.
  - When `user_id` is provided, inference must attempt to resolve and load `models/model_production_{user_id}.joblib` for that user without modifying existing feature validation or prediction logic.

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

- ML-AC-01 `POST /train/global` loads the dataset from disk, trains candidate models, and returns the winner metrics.
- ML-AC-02 Global training writes the promoted artifact exactly to `models/model_production_global.joblib`.
- ML-AC-03 `POST /train/personal` filters the dataset to the requested `user_id` before 5-fold cross validation.
- ML-AC-04 Personalized training writes the promoted artifact exactly to `models/model_production_{user_id}.joblib`.
- ML-AC-05 Personalized training rejects requests with fewer than 20 filtered rows using HTTP 400 `Insufficient data`.
- ML-AC-05a Personalized training accepts inline admin-supplied rows and uses them even when the persisted processed dataset does not already contain that user.
- ML-AC-06 Inference endpoints use the updated global or personalized model without process restart.
- ML-AC-07 Tie behavior remains deterministic and test-covered.
- ML-AC-08 `POST /predict` uses a personalized model when a matching user artifact exists.
- ML-AC-09 `POST /predict` falls back to the global baseline model when the personalized artifact is missing.
- ML-AC-10 `POST /predict` includes `is_fallback_used` in the response payload.

## Implementation Task Breakdown

- ML-T01 Add dataset-backed training request models for `/train/global` and `/train/personal`.
- ML-T02 Implement dataset loading helpers in `src/services/training_service.py`.
- ML-T02a Implement reusable dwell/flight outlier cleaning before cross-validation scoring.
- ML-T03a Replace single-split winner selection with 5-fold cross validation scored by macro F1.
- ML-T03 Reuse existing algorithm arena evaluation and deterministic winner selection for both global and personalized training.
- ML-T04 Persist the global winner to `models/model_production_global.joblib`.
- ML-T05 Persist the personalized winner to `models/model_production_{user_id}.joblib`.
- ML-T06 Keep thread-safe model hot-reload in FastAPI app state.
- ML-T07 Add `/train/global` and `/train/personal` endpoint wiring and response contracts.
- ML-T07a Extend `/train/personal` so it can normalize optional inline rows from the Typing API admin workflow.
- ML-T08 Update `/predict` personalized model resolution to the fixed personalized and global production paths.
- ML-T09 Add tests for global training, personalized training, insufficient-data rejection, personalized predict, cold-start fallback, and `is_fallback_used`.
