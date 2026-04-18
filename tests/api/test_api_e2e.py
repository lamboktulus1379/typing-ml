import importlib
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FINGERS = [
    "left_pinky",
    "left_ring",
    "left_middle",
    "left_index",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
]

FEATURE_NAMES = [
    "wpm",
    "accuracy",
    *[f"error_{f}" for f in FINGERS],
    *[f"dwell_{f}" for f in FINGERS],
    *[f"flight_{f}" for f in FINGERS],
]


def _build_row(scale: float) -> dict[str, float]:
    row: dict[str, float] = {
        "wpm": 45.0 + scale,
        "accuracy": 0.9 + 0.01 * scale,
    }
    for i, finger in enumerate(FINGERS):
        row[f"error_{finger}"] = 0.01 + (i * 0.001) + (0.001 * scale)
        row[f"dwell_{finger}"] = 90.0 + i + (2.0 * scale)
        row[f"flight_{finger}"] = 180.0 + i + (3.0 * scale)
    return row


class EncodedDummyModel:
    """Minimal model stub that emits encoded class indices."""

    def __init__(self) -> None:
        self.classes_ = [0, 1]

    def predict(self, frame: pd.DataFrame) -> list[int]:
        return [1 for _ in range(len(frame))]

    def predict_proba(self, frame: pd.DataFrame) -> list[list[float]]:
        return [[0.1, 0.9] for _ in range(len(frame))]


class AlgorithmSwitchDummyModel:
    """Model stub to verify algorithm-based model switching from environment."""

    def __init__(self) -> None:
        self.classes_ = [0, 1]

    def predict(self, frame: pd.DataFrame) -> list[int]:
        return [0 for _ in range(len(frame))]

    def predict_proba(self, frame: pd.DataFrame) -> list[list[float]]:
        return [[0.8, 0.2] for _ in range(len(frame))]


@pytest.fixture
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    model_path = tmp_path / "model.joblib"

    # Minimal training data just to create a valid artifact with predict_proba.
    rows = [_build_row(0.0), _build_row(1.0), _build_row(2.0), _build_row(3.0)]
    labels = ["left_index", "right_index", "left_index", "right_index"]
    x_train = pd.DataFrame(rows, columns=FEATURE_NAMES)

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )
    clf.fit(x_train, labels)

    artifact = {
        "model": clf,
        "model_name": "logistic_regression",
        "feature_names": FEATURE_NAMES,
        "target_name": "weakest_finger",
        "created_at": "2026-03-28T00:00:00+00:00",
    }
    joblib.dump(artifact, model_path)

    monkeypatch.setenv("TYPING_ML_MODEL_PATH", str(model_path))

    # Ensure import side effects pick up patched environment variable.
    if "src.api" in sys.modules:
        del sys.modules["src.api"]
    api_module = importlib.import_module("src.api")

    app = api_module.create_app()
    return TestClient(app)


@pytest.fixture
def api_client_encoded_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    model_path = tmp_path / "model_encoded.joblib"

    artifact = {
        "model": EncodedDummyModel(),
        "model_name": "xgboost",
        "feature_names": FEATURE_NAMES,
        "target_name": "weakest_finger",
        "label_classes": ["left_index", "right_index"],
        "created_at": "2026-03-28T00:00:00+00:00",
    }
    joblib.dump(artifact, model_path)

    monkeypatch.setenv("TYPING_ML_MODEL_PATH", str(model_path))

    if "src.api" in sys.modules:
        del sys.modules["src.api"]
    api_module = importlib.import_module("src.api")

    app = api_module.create_app()
    return TestClient(app)


@pytest.fixture
def api_client_algorithm_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    model_path = tmp_path / "model_random_forest.joblib"

    artifact = {
        "model": AlgorithmSwitchDummyModel(),
        "model_name": "random_forest",
        "feature_names": FEATURE_NAMES,
        "target_name": "weakest_finger",
        "label_classes": ["rf_class_a", "rf_class_b"],
        "created_at": "2026-03-28T00:00:00+00:00",
    }
    joblib.dump(artifact, model_path)

    monkeypatch.delenv("TYPING_ML_MODEL_PATH", raising=False)
    monkeypatch.setenv("TYPING_ML_MODEL_ALGORITHM", "random_forest")
    monkeypatch.setenv("TYPING_ML_MODEL_PATH_RANDOM_FOREST", str(model_path))

    if "src.api" in sys.modules:
        del sys.modules["src.api"]
    api_module = importlib.import_module("src.api")

    app = api_module.create_app()
    return TestClient(app)


def test_health(api_client: TestClient) -> None:
    res = api_client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_metadata(api_client: TestClient) -> None:
    res = api_client.get("/metadata")
    assert res.status_code == 200
    body = res.json()
    assert body["model_algorithm"] == "logistic_regression"
    assert body["target_name"] == "weakest_finger"
    assert isinstance(body["feature_names"], list)
    assert len(body["feature_names"]) == 26
    assert set(body["classes"]) == {"left_index", "right_index"}


def test_predict_single(api_client: TestClient) -> None:
    payload = {"row": _build_row(0.5)}
    res = api_client.post("/predict", json=payload)

    assert res.status_code == 200
    body = res.json()
    assert "prediction" in body
    assert body["prediction"] in {"left_index", "right_index"}
    assert "probabilities" in body
    assert set(body["probabilities"].keys()) == {"left_index", "right_index"}


def test_predict_batch(api_client: TestClient) -> None:
    payload = {"rows": [_build_row(0.5), _build_row(1.5)]}
    res = api_client.post("/predict_batch", json=payload)

    assert res.status_code == 200
    body = res.json()
    assert "predictions" in body
    assert len(body["predictions"]) == 2
    assert "probabilities" in body
    assert len(body["probabilities"]) == 2


def test_predict_rejects_old_features_shape(api_client: TestClient) -> None:
    old_payload = {"features": _build_row(0.5)}
    res = api_client.post("/predict", json=old_payload)

    # Pydantic should reject because "row" is required.
    assert res.status_code == 422
    body = res.json()
    assert "detail" in body


def test_predict_batch_rejects_empty_rows(api_client: TestClient) -> None:
    res = api_client.post("/predict_batch", json={"rows": []})
    assert res.status_code == 400
    assert res.json()["detail"]["error"] == "rows must be non-empty"


def test_metadata_decodes_label_classes_for_encoded_model(
    api_client_encoded_labels: TestClient,
) -> None:
    res = api_client_encoded_labels.get("/metadata")
    assert res.status_code == 200
    assert res.json()["classes"] == ["left_index", "right_index"]


def test_predict_decodes_label_classes_for_encoded_model(
    api_client_encoded_labels: TestClient,
) -> None:
    payload = {"row": _build_row(0.5)}
    res = api_client_encoded_labels.post("/predict", json=payload)

    assert res.status_code == 200
    body = res.json()
    assert body["prediction"] == "right_index"
    assert set(body["probabilities"].keys()) == {"left_index", "right_index"}


def test_metadata_uses_algorithm_switch_model_path(
    api_client_algorithm_switch: TestClient,
) -> None:
    res = api_client_algorithm_switch.get("/metadata")
    assert res.status_code == 200
    body = res.json()
    assert body["model_algorithm"] == "random_forest"
    assert body["model_path"].endswith("model_random_forest.joblib")
    assert body["classes"] == ["rf_class_a", "rf_class_b"]


def test_predict_uses_algorithm_switch_model(
    api_client_algorithm_switch: TestClient,
) -> None:
    payload = {"row": _build_row(0.25)}
    res = api_client_algorithm_switch.post("/predict", json=payload)

    assert res.status_code == 200
    body = res.json()
    assert body["prediction"] == "rf_class_a"
    assert set(body["probabilities"].keys()) == {"rf_class_a", "rf_class_b"}


def test_create_app_rejects_unsupported_algorithm_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TYPING_ML_MODEL_PATH", raising=False)
    monkeypatch.setenv("TYPING_ML_MODEL_ALGORITHM", "svm")

    if "src.api" in sys.modules:
        del sys.modules["src.api"]

    with pytest.raises(RuntimeError, match="Unsupported TYPING_ML_MODEL_ALGORITHM"):
        importlib.import_module("src.api")
