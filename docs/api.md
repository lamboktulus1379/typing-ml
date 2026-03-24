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
  "features": {
    "wpm": 55,
    "accuracy": 0.93,
    "error_rate": 0.07,
    "error_left_pinky": 0.02,
    "error_left_ring": 0.01,
    "error_left_middle": 0.01,
    "error_left_index": 0.01,
    "error_right_index": 0.02,
    "error_right_middle": 0.01,
    "error_right_ring": 0.01,
    "error_right_pinky": 0.01
  }
}
```

Example:

```bash
curl -s http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": {"wpm": 55, "accuracy": 0.93, "error_rate": 0.07, "error_left_pinky": 0.02, "error_left_ring": 0.01, "error_left_middle": 0.01, "error_left_index": 0.01, "error_right_index": 0.02, "error_right_middle": 0.01, "error_right_ring": 0.01, "error_right_pinky": 0.01}}' \
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
    {"wpm": 55, "accuracy": 0.93, "error_rate": 0.07, "error_left_pinky": 0.02, "error_left_ring": 0.01, "error_left_middle": 0.01, "error_left_index": 0.01, "error_right_index": 0.02, "error_right_middle": 0.01, "error_right_ring": 0.01, "error_right_pinky": 0.01},
    {"wpm": 42, "accuracy": 0.91, "error_rate": 0.09, "error_left_pinky": 0.01, "error_left_ring": 0.01, "error_left_middle": 0.02, "error_left_index": 0.01, "error_right_index": 0.03, "error_right_middle": 0.01, "error_right_ring": 0.01, "error_right_pinky": 0.01}
  ]
}
```

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
