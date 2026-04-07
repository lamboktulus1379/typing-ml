# API Documentation (FastAPI)

This API loads the trained model from `models/model.joblib` and exposes REST endpoints so you can test predictions from Postman/curl.

## Start the server

From the repo root:

```bash
conda activate typing-ml
uvicorn src.api:app --reload --port 8000
```

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## Model file

By default the API reads:
- `models/model.joblib`

You can override it with an environment variable:

```bash
export TYPING_ML_MODEL_PATH="models/model.joblib"
uvicorn src.api:app --reload
```

## Endpoints

### GET `/health`

Purpose: quick check that the server is running.

Example:

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

Response:

```json
{"status": "ok"}
```

### GET `/metadata`

Purpose: see the model’s expected input features and output classes.

Example:

```bash
curl -s http://127.0.0.1:8000/metadata | python -m json.tool
```

Response fields:
- `feature_names`: list of required feature keys (in training order)
- `classes`: possible labels for `weakest_finger`
- `created_at`: when the artifact was created (UTC)

### POST `/predict`

Purpose: predict `weakest_finger` for one row.

Request JSON shape:

```json
{
  "row": {
    "wpm": 55,
    "accuracy": 0.93,
    "error_left_pinky": 0.02,
    "error_left_ring": 0.01,
    "error_left_middle": 0.01,
    "error_left_index": 0.01,
    "error_right_index": 0.02,
    "error_right_middle": 0.01,
    "error_right_ring": 0.01,
    "error_right_pinky": 0.01,
    "dwell_left_pinky": 115,
    "dwell_left_ring": 98,
    "dwell_left_middle": 91,
    "dwell_left_index": 86,
    "dwell_right_index": 88,
    "dwell_right_middle": 93,
    "dwell_right_ring": 99,
    "dwell_right_pinky": 121,
    "flight_left_pinky": 220,
    "flight_left_ring": 198,
    "flight_left_middle": 180,
    "flight_left_index": 171,
    "flight_right_index": 174,
    "flight_right_middle": 182,
    "flight_right_ring": 203,
    "flight_right_pinky": 231
  }
}
```

Example:

```bash
curl -s http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"row": {"wpm": 55, "accuracy": 0.93, "error_left_pinky": 0.02, "error_left_ring": 0.01, "error_left_middle": 0.01, "error_left_index": 0.01, "error_right_index": 0.02, "error_right_middle": 0.01, "error_right_ring": 0.01, "error_right_pinky": 0.01, "dwell_left_pinky": 115, "dwell_left_ring": 98, "dwell_left_middle": 91, "dwell_left_index": 86, "dwell_right_index": 88, "dwell_right_middle": 93, "dwell_right_ring": 99, "dwell_right_pinky": 121, "flight_left_pinky": 220, "flight_left_ring": 198, "flight_left_middle": 180, "flight_left_index": 171, "flight_right_index": 174, "flight_right_middle": 182, "flight_right_ring": 203, "flight_right_pinky": 231}}' \
  | python -m json.tool
```

Response:
- `prediction`: predicted label
- `probabilities` (optional): probability per class (if the model supports `predict_proba`)

### POST `/predict_batch`

Purpose: predict for many rows at once.

Request JSON shape:

```json
{
  "rows": [
    {"wpm": 55, "accuracy": 0.93, "error_left_pinky": 0.02, "error_left_ring": 0.01, "error_left_middle": 0.01, "error_left_index": 0.01, "error_right_index": 0.02, "error_right_middle": 0.01, "error_right_ring": 0.01, "error_right_pinky": 0.01, "dwell_left_pinky": 115, "dwell_left_ring": 98, "dwell_left_middle": 91, "dwell_left_index": 86, "dwell_right_index": 88, "dwell_right_middle": 93, "dwell_right_ring": 99, "dwell_right_pinky": 121, "flight_left_pinky": 220, "flight_left_ring": 198, "flight_left_middle": 180, "flight_left_index": 171, "flight_right_index": 174, "flight_right_middle": 182, "flight_right_ring": 203, "flight_right_pinky": 231},
    {"wpm": 42, "accuracy": 0.91, "error_left_pinky": 0.01, "error_left_ring": 0.01, "error_left_middle": 0.02, "error_left_index": 0.01, "error_right_index": 0.03, "error_right_middle": 0.01, "error_right_ring": 0.01, "error_right_pinky": 0.01, "dwell_left_pinky": 123, "dwell_left_ring": 100, "dwell_left_middle": 96, "dwell_left_index": 90, "dwell_right_index": 93, "dwell_right_middle": 98, "dwell_right_ring": 105, "dwell_right_pinky": 130, "flight_left_pinky": 240, "flight_left_ring": 214, "flight_left_middle": 201, "flight_left_index": 189, "flight_right_index": 193, "flight_right_middle": 205, "flight_right_ring": 220, "flight_right_pinky": 246}
  ]
}
```

## Integration impact for .NET backend / frontend

If your .NET backend or frontend calls this API, update request contracts to match:

- `/predict`: use `row` (not `features`).
- `/predict_batch`: use `rows` array of full records.
- Include dwell and flight features; these are part of the standard model input.
- `error_rate` is not part of the current request schema.

Suggested migration checklist:

1. Update DTOs/ViewModels in .NET for `row` and `rows` payloads.
2. Update FE request builders/forms to send the full feature set.
3. Confirm required keys with `GET /metadata` before deployment.
4. Add contract tests for `/predict` and `/predict_batch` payload validation.

## Common errors

### 400 Missing required features

If the model artifact contains `feature_names`, the API requires all of them.

Example response:

```json
{
  "detail": {
    "error": "Missing required features",
    "missing": ["accuracy", "wpm"]
  }
}
```

## Learning notes (why it’s built this way)

- We save `feature_names` during training so the API can build input in the exact same column order as training.
- The FastAPI `/docs` page is the easiest way to learn: it shows schema, lets you try requests, and generates curl.

## Automated API E2E tests

This repository includes automated end-to-end tests for:

- `GET /health`
- `GET /metadata`
- `POST /predict`
- `POST /predict_batch`
- payload compatibility checks (reject old `features` shape)

Run tests:

```bash
make CONDA_ENV=typing-ml test-api
```

Test file:

- `tests/api/test_api_e2e.py`
