import importlib
import importlib.util
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def _build_retraining_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []

    for scale in range(6):
        row = _build_row(float(scale))
        row["error_left_pinky"] = 0.20 + (0.01 * scale)
        row["error_right_pinky"] = 0.01
        row["weakest_finger"] = "left_pinky"
        rows.append(row)

    for scale in range(6, 12):
        row = _build_row(float(scale))
        row["error_left_pinky"] = 0.01
        row["error_right_pinky"] = 0.20 + (0.01 * (scale - 6))
        row["weakest_finger"] = "right_pinky"
        rows.append(row)

    return rows


def _build_dataset_training_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []

    for scale in range(12):
        row = _build_row(float(scale) / 10.0)
        row["accuracy"] = 0.91 + (0.002 * scale)
        row["error_left_pinky"] = 0.25 + (0.005 * scale)
        row["error_right_pinky"] = 0.01
        row["weakest_finger"] = "left_pinky"
        row["user_id"] = "user-test-1"
        rows.append(row)

    for scale in range(12, 24):
        row = _build_row(float(scale) / 10.0)
        row["accuracy"] = 0.91 + (0.002 * (scale - 12))
        row["error_left_pinky"] = 0.01
        row["error_right_pinky"] = 0.25 + (0.005 * (scale - 12))
        row["weakest_finger"] = "right_pinky"
        row["user_id"] = "user-test-1"
        rows.append(row)

    for scale in range(4):
        row = _build_row((float(scale) + 0.5) / 10.0)
        row["accuracy"] = 0.9 + (0.002 * scale)
        row["error_left_pinky"] = 0.22 + (0.003 * scale)
        row["error_right_pinky"] = 0.01
        row["weakest_finger"] = "left_pinky"
        row["user_id"] = "missing-user"
        rows.append(row)

    for scale in range(4, 8):
        row = _build_row((float(scale) + 0.5) / 10.0)
        row["accuracy"] = 0.9 + (0.002 * (scale - 4))
        row["error_left_pinky"] = 0.01
        row["error_right_pinky"] = 0.22 + (0.003 * (scale - 4))
        row["weakest_finger"] = "right_pinky"
        row["user_id"] = "missing-user"
        rows.append(row)

    return rows


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


class PersonalizedPredictDummyModel:
    """Model stub that identifies a personalized inference artifact."""

    def __init__(self) -> None:
        self.classes_ = ["personalized_left", "personalized_right"]

    def predict(self, frame: pd.DataFrame) -> list[str]:
        return ["personalized_right" for _ in range(len(frame))]

    def predict_proba(self, frame: pd.DataFrame) -> list[list[float]]:
        return [[0.05, 0.95] for _ in range(len(frame))]


class GlobalFallbackPredictDummyModel:
    """Model stub that identifies the global cold-start fallback artifact."""

    def __init__(self) -> None:
        self.classes_ = ["global_left", "global_right"]

    def predict(self, frame: pd.DataFrame) -> list[str]:
        return ["global_left" for _ in range(len(frame))]

    def predict_proba(self, frame: pd.DataFrame) -> list[list[float]]:
        return [[0.9, 0.1] for _ in range(len(frame))]


@pytest.fixture
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    model_path = tmp_path / "model.joblib"
    production_model_dir = tmp_path / "production"
    active_model_metadata_path = tmp_path / "active_production_model.json"
    train_reports_dir = tmp_path / "reports"
    training_dataset_path = tmp_path / "dataset.csv"

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
    pd.DataFrame(_build_dataset_training_rows()).to_csv(training_dataset_path, index=False)

    monkeypatch.setenv("TYPING_ML_MODEL_PATH", str(model_path))
    monkeypatch.setenv("TYPING_ML_PRODUCTION_MODEL_DIR", str(production_model_dir))
    monkeypatch.setenv("TYPING_ML_ACTIVE_MODEL_METADATA_PATH", str(active_model_metadata_path))
    monkeypatch.setenv("TYPING_ML_TRAIN_REPORTS_DIR", str(train_reports_dir))
    monkeypatch.setenv("TYPING_ML_TRAINING_DATASET_PATH", str(training_dataset_path))
    monkeypatch.setenv("TYPING_ML_GLOBAL_FALLBACK_MODEL_PATH", str(tmp_path / "model_production_global.joblib"))
    monkeypatch.setenv("TYPING_ML_PERSONALIZED_MODEL_PATH_TEMPLATE", str(tmp_path / "model_production_{user_id}.joblib"))

    # Ensure import side effects pick up patched environment variable.
    if "src.api" in sys.modules:
        del sys.modules["src.api"]
    api_module = importlib.import_module("src.api")

    app = api_module.create_app()
    app.extra["active_model_metadata_path"] = str(active_model_metadata_path)
    app.extra["train_reports_dir"] = str(train_reports_dir)
    app.extra["training_dataset_path"] = str(training_dataset_path)
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


@pytest.fixture
def api_client_predict_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    global_model_path = tmp_path / "model_production_global.joblib"
    personalized_model_path = tmp_path / "model_production_user-test-1.joblib"

    global_artifact = {
        "model": GlobalFallbackPredictDummyModel(),
        "model_name": "global_baseline",
        "feature_names": FEATURE_NAMES,
        "target_name": "weakest_finger",
        "created_at": "2026-04-19T00:00:00+00:00",
    }
    personalized_artifact = {
        "model": PersonalizedPredictDummyModel(),
        "model_name": "logistic_regression",
        "feature_names": FEATURE_NAMES,
        "target_name": "weakest_finger",
        "created_at": "2026-04-19T00:10:00+00:00",
    }

    joblib.dump(global_artifact, global_model_path)
    joblib.dump(personalized_artifact, personalized_model_path)

    monkeypatch.setenv("TYPING_ML_MODEL_PATH", str(global_model_path))
    monkeypatch.setenv("TYPING_ML_GLOBAL_FALLBACK_MODEL_PATH", str(global_model_path))
    monkeypatch.setenv("TYPING_ML_PERSONALIZED_MODEL_PATH_TEMPLATE", str(tmp_path / "model_production_{user_id}.joblib"))

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


def test_predict_uses_personalized_model_when_available(
    api_client_predict_fallback: TestClient,
) -> None:
    payload = {"userId": "user-test-1", "row": _build_row(0.5)}
    res = api_client_predict_fallback.post("/predict", json=payload)

    assert res.status_code == 200
    body = res.json()
    assert body["prediction"] == "personalized_right"
    assert set(body["probabilities"].keys()) == {"personalized_left", "personalized_right"}
    assert body["is_fallback_used"] is False


def test_predict_falls_back_to_global_model_when_personalized_model_is_missing(
    api_client_predict_fallback: TestClient,
) -> None:
    payload = {"userId": "missing-user", "row": _build_row(0.5)}
    res = api_client_predict_fallback.post("/predict", json=payload)

    assert res.status_code == 200
    body = res.json()
    assert body["prediction"] == "global_left"
    assert set(body["probabilities"].keys()) == {"global_left", "global_right"}
    assert body["is_fallback_used"] is True


def test_predict_without_user_id_uses_global_model(api_client_predict_fallback: TestClient) -> None:
    payload = {"row": _build_row(0.5)}
    res = api_client_predict_fallback.post("/predict", json=payload)

    assert res.status_code == 200
    body = res.json()
    assert body["prediction"] == "global_left"
    assert body["is_fallback_used"] is True


def test_train_global_saves_fixed_global_model_path(api_client: TestClient) -> None:
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("xgboost is not installed in this environment")

    response = api_client.post("/train/global", json={})

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "winningAlgorithmName",
        "macroPrecision",
        "macroRecall",
        "topPredictiveFeature",
        "primaryMisclassification",
        "evaluations",
    }
    global_model_path = Path(api_client.app.extra["training_dataset_path"]).parent / "model_production_global.joblib"
    assert global_model_path.exists()


def test_train_personal_saves_fixed_personal_model_path(api_client: TestClient) -> None:
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("xgboost is not installed in this environment")

    response = api_client.post("/train/personal", json={"userId": "user-test-1"})

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "winningAlgorithmName",
        "macroPrecision",
        "macroRecall",
        "topPredictiveFeature",
        "primaryMisclassification",
        "evaluations",
    }
    personal_model_path = Path(api_client.app.extra["training_dataset_path"]).parent / "model_production_user-test-1.joblib"
    assert personal_model_path.exists()


def test_train_personal_accepts_inline_rows_for_user_missing_from_dataset(api_client: TestClient) -> None:
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("xgboost is not installed in this environment")

    inline_rows: list[dict[str, float | str]] = []
    for scale in range(12):
        row = _build_row(10.0 + float(scale))
        row["error_left_pinky"] = 0.28 + (0.004 * scale)
        row["error_right_pinky"] = 0.01
        row["weakest_finger"] = "left_pinky"
        inline_rows.append(row)

    for scale in range(12, 24):
        row = _build_row(10.0 + float(scale))
        row["error_left_pinky"] = 0.01
        row["error_right_pinky"] = 0.28 + (0.004 * (scale - 12))
        row["weakest_finger"] = "right_pinky"
        inline_rows.append(row)

    response = api_client.post(
        "/train/personal",
        json={"userId": "inline-user", "rows": inline_rows},
    )

    assert response.status_code == 200
    personal_model_path = Path(api_client.app.extra["training_dataset_path"]).parent / "model_production_inline-user.joblib"
    assert personal_model_path.exists()


def test_train_personal_rejects_insufficient_rows(api_client: TestClient) -> None:
    response = api_client.post("/train/personal", json={"userId": "missing-user"})

    assert response.status_code == 400
    assert response.json()["detail"] == "Insufficient data"


def test_train_hot_reloads_model_and_updates_metadata(api_client: TestClient) -> None:
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("xgboost is not installed in this environment")

    retrain_payload = _build_retraining_rows()
    retrain_response = api_client.post(
        "/train",
        json={"userId": "user-test-1", "is_dry_run": False, "rows": retrain_payload},
    )

    assert retrain_response.status_code == 200
    body = retrain_response.json()
    assert set(body.keys()) == {
        "winningAlgorithmName",
        "macroPrecision",
        "macroRecall",
        "topPredictiveFeature",
        "primaryMisclassification",
        "evaluations",
    }
    assert body["winningAlgorithmName"] in {"logistic_regression", "random_forest", "xgboost"}
    assert 0.0 <= body["macroPrecision"] <= 1.0
    assert 0.0 <= body["macroRecall"] <= 1.0
    assert body["topPredictiveFeature"] in FEATURE_NAMES
    assert body["primaryMisclassification"]
    assert len(body["evaluations"]) == 3
    assert {entry["algorithmName"] for entry in body["evaluations"]} == {
        "logistic_regression",
        "random_forest",
        "xgboost",
    }
    for entry in body["evaluations"]:
        assert set(entry.keys()) == {"algorithmName", "accuracy", "f1Score", "executionTimeMs"}
        assert 0.0 <= entry["accuracy"] <= 1.0
        assert 0.0 <= entry["f1Score"] <= 1.0
        assert isinstance(entry["executionTimeMs"], int)
        assert entry["executionTimeMs"] >= 0

    pointer_path = Path(api_client.app.extra["active_model_metadata_path"])
    assert pointer_path.exists()
    pointer_payload = __import__("json").loads(pointer_path.read_text(encoding="utf-8"))
    assert pointer_payload["model_algorithm"] == body["winningAlgorithmName"]
    assert pointer_payload["model_path"].endswith(".joblib")
    assert "typing-prod-" in pointer_payload["model_path"]
    assert "user-test-1" in pointer_payload["model_path"]
    assert Path(pointer_payload["model_path"]).exists()

    report_dir = Path(api_client.app.extra["train_reports_dir"])
    report_files = sorted(report_dir.glob("train_success_production_*.json"))
    assert report_files
    report_payload = __import__("json").loads(report_files[-1].read_text(encoding="utf-8"))
    assert report_payload["winning_algorithm_name"] == body["winningAlgorithmName"]
    assert report_payload["total_rows_processed"] == len(retrain_payload)
    assert len(report_payload["evaluations"]) == 3

    metadata_response = api_client.get("/metadata")
    assert metadata_response.status_code == 200
    metadata_body = metadata_response.json()
    assert metadata_body["model_algorithm"] == body["winningAlgorithmName"]
    assert metadata_body["model_path"] == pointer_payload["model_path"]
    assert metadata_body["active_model_metadata_path"] == str(pointer_path)

    predict_response = api_client.post("/predict", json={"row": _build_row(2.0)})
    assert predict_response.status_code == 200
    assert "prediction" in predict_response.json()


def test_train_dry_run_reports_deduplicated_rows_processed(api_client: TestClient) -> None:
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("xgboost is not installed in this environment")

    unique_rows = _build_retraining_rows()
    duplicated_rows = unique_rows + [dict(unique_rows[0]), dict(unique_rows[6])]

    response = api_client.post(
        "/train",
        json={"userId": "user-test-1", "is_dry_run": True, "rows": duplicated_rows},
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "winningAlgorithmName",
        "macroPrecision",
        "macroRecall",
        "topPredictiveFeature",
        "primaryMisclassification",
        "evaluations",
    }
    assert len(body["evaluations"]) == 3
    assert {entry["algorithmName"] for entry in body["evaluations"]} == {
        "logistic_regression",
        "random_forest",
        "xgboost",
    }

    report_dir = Path(api_client.app.extra["train_reports_dir"])
    report_files = sorted(report_dir.glob("train_success_dry_run_*.json"))
    assert report_files
    report_payload = __import__("json").loads(report_files[-1].read_text(encoding="utf-8"))
    assert report_payload["total_rows_processed"] == len(unique_rows)
    assert report_payload["total_rows_processed"] < len(duplicated_rows)
    assert report_payload["winning_algorithm_name"] == body["winningAlgorithmName"]
    assert len(report_payload["evaluations"]) == 3


def test_train_report_contains_audit_payload(api_client: TestClient) -> None:
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("xgboost is not installed in this environment")

    retrain_payload = _build_retraining_rows()
    response = api_client.post(
        "/train",
        json={"userId": "user-test-1", "is_dry_run": True, "rows": retrain_payload},
    )

    assert response.status_code == 200
    body = response.json()
    report_dir = Path(api_client.app.extra["train_reports_dir"])
    report_files = sorted(report_dir.glob("train_success_dry_run_*.json"))
    assert report_files
    report_payload = __import__("json").loads(report_files[-1].read_text(encoding="utf-8"))

    assert report_payload["status"] == "success_dry_run"
    assert report_payload["is_dry_run"] is True
    assert report_payload["winning_algorithm_name"] == body["winningAlgorithmName"]
    assert len(report_payload["evaluations"]) == 3
    assert 0.0 <= report_payload["macro_precision"] <= 1.0
    assert 0.0 <= report_payload["macro_recall"] <= 1.0


def test_train_rejects_empty_rows(api_client: TestClient) -> None:
    response = api_client.post("/train", json={"userId": "user-test-1", "rows": []})
    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "rows must be non-empty"


def test_train_filters_rows_to_requested_user(api_client: TestClient) -> None:
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("xgboost is not installed in this environment")

    retrain_payload = _build_retraining_rows()
    mixed_rows = []
    target_indexes = {0, 1, 2, 6, 7, 8}
    for index, row in enumerate(retrain_payload):
        mixed_row = dict(row)
        mixed_row["userId"] = "user-target" if index in target_indexes else "user-other"
        mixed_rows.append(mixed_row)

    response = api_client.post(
        "/train",
        json={"userId": "user-target", "is_dry_run": True, "rows": mixed_rows},
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "winningAlgorithmName",
        "macroPrecision",
        "macroRecall",
        "topPredictiveFeature",
        "primaryMisclassification",
        "evaluations",
    }

    report_dir = Path(api_client.app.extra["train_reports_dir"])
    report_files = sorted(report_dir.glob("train_success_dry_run_*.json"))
    assert report_files
    report_payload = __import__("json").loads(report_files[-1].read_text(encoding="utf-8"))
    assert report_payload["total_rows_processed"] == len(target_indexes)


def test_create_app_rejects_unsupported_algorithm_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TYPING_ML_MODEL_PATH", raising=False)
    monkeypatch.setenv("TYPING_ML_MODEL_ALGORITHM", "svm")

    if "src.api" in sys.modules:
        del sys.modules["src.api"]

    with pytest.raises(RuntimeError, match="Unsupported TYPING_ML_MODEL_ALGORITHM"):
        importlib.import_module("src.api")


def test_metadata_uses_active_model_pointer_when_no_explicit_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production_model_path = tmp_path / "production" / "typing-prod-20260419T120000_000000Z-random_forest.joblib"
    active_model_metadata_path = tmp_path / "active_production_model.json"

    production_model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": AlgorithmSwitchDummyModel(),
        "model_name": "random_forest",
        "feature_names": FEATURE_NAMES,
        "target_name": "weakest_finger",
        "label_classes": ["rf_class_a", "rf_class_b"],
        "created_at": "2026-04-19T12:00:00+00:00",
    }
    joblib.dump(artifact, production_model_path)
    active_model_metadata_path.write_text(
        __import__("json").dumps(
            {
                "model_path": str(production_model_path),
                "model_algorithm": "random_forest",
                "promoted_at": "2026-04-19T12:01:00+00:00",
                "created_at": "2026-04-19T12:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("TYPING_ML_MODEL_PATH", raising=False)
    monkeypatch.delenv("TYPING_ML_MODEL_ALGORITHM", raising=False)
    monkeypatch.delenv("TYPING_ML_MODEL_PATH_RANDOM_FOREST", raising=False)
    monkeypatch.setenv("TYPING_ML_ACTIVE_MODEL_METADATA_PATH", str(active_model_metadata_path))

    if "src.api" in sys.modules:
        del sys.modules["src.api"]
    api_module = importlib.import_module("src.api")

    app = api_module.create_app()
    client = TestClient(app)

    res = client.get("/metadata")
    assert res.status_code == 200
    body = res.json()
    assert body["model_path"] == str(production_model_path)
    assert body["model_algorithm"] == "random_forest"
    assert body["active_model_metadata_path"] == str(active_model_metadata_path)
