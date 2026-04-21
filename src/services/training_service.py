"""FastAPI retraining service for thesis-oriented Algorithm Arena evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

try:
    from src.ml_pipeline.artifacts import ArtifactStore, ModelArtifact
    from src.ml_pipeline.constants import (
        ALLOWED_WEAKEST_FINGER_LABELS,
        FEATURE_RANGE_RULES,
        TARGET_COLUMN,
        TRAIN_FEATURE_COLUMNS,
    )
    from src.ml_pipeline.interfaces import (
        FeatureValidatorProtocol,
        ModelFactoryProtocol,
        TargetValidatorProtocol,
    )
    from src.ml_pipeline.model_factory import Algorithm, ModelPipelineFactory
    from src.ml_pipeline.validation import FeatureFrameValidator, TargetSeriesValidator
except ModuleNotFoundError:
    from ml_pipeline.artifacts import ArtifactStore, ModelArtifact
    from ml_pipeline.constants import (
        ALLOWED_WEAKEST_FINGER_LABELS,
        FEATURE_RANGE_RULES,
        TARGET_COLUMN,
        TRAIN_FEATURE_COLUMNS,
    )
    from ml_pipeline.interfaces import (
        FeatureValidatorProtocol,
        ModelFactoryProtocol,
        TargetValidatorProtocol,
    )
    from ml_pipeline.model_factory import Algorithm, ModelPipelineFactory
    from ml_pipeline.validation import FeatureFrameValidator, TargetSeriesValidator


DEFAULT_TRAINING_DATASET_PATH = "data/processed/dataset.csv"
DEFAULT_GLOBAL_MODEL_PATH = "models/model_production_global.joblib"
DEFAULT_PERSONAL_MODEL_TEMPLATE = "models/model_production_{user_id}.joblib"
DEFAULT_MIN_PERSONAL_TRAINING_ROWS = 20


@dataclass(frozen=True)
class AlgorithmLeaderboardEntry:
    """Public metrics for one evaluated algorithm candidate."""

    name: str
    accuracy: float
    f1_score: float
    execution_time_ms: float
    model: Any
    label_classes: list[str] | None = None


@dataclass(frozen=True)
class AlgorithmArenaResult:
    """Full result of one Algorithm Arena retraining run."""

    winning_algorithm: str
    winning_accuracy: float
    winning_f1_score: float
    macro_precision: float
    macro_recall: float
    top_predictive_feature: str
    primary_misclassification: str
    total_rows_processed: int
    xai_global: dict[str, Any]
    leaderboard: tuple[AlgorithmLeaderboardEntry, ...]
    retrained_model: Any
    retrained_label_classes: list[str] | None = None


@dataclass(frozen=True)
class PersistedTrainingResult:
    """Persisted global or personalized training output."""

    arena_result: AlgorithmArenaResult
    artifact: ModelArtifact
    model_output_path: str


@dataclass
class TrainingArenaService:
    """Validates payloads, evaluates candidate algorithms, and retrains the winner."""

    model_factory: ModelFactoryProtocol
    feature_validator: FeatureValidatorProtocol
    target_validator: TargetValidatorProtocol
    random_state: int = 42

    @classmethod
    def default(cls, *, random_state: int = 42) -> "TrainingArenaService":
        """Build a default service graph using built-in pipeline components."""

        return cls(
            model_factory=ModelPipelineFactory(random_state=random_state),
            feature_validator=FeatureFrameValidator(FEATURE_RANGE_RULES),
            target_validator=TargetSeriesValidator(ALLOWED_WEAKEST_FINGER_LABELS),
            random_state=random_state,
        )

    def load_training_dataset(self, data_path: str = DEFAULT_TRAINING_DATASET_PATH) -> pd.DataFrame:
        """Load the persisted processed dataset used for global or personalized training."""

        dataframe = pd.read_csv(data_path)
        if dataframe.empty:
            raise ValueError("Training dataset is empty. Provide at least one row.")

        if TARGET_COLUMN not in dataframe.columns:
            raise ValueError(
                f"Training dataset is missing required target column '{TARGET_COLUMN}'."
            )

        return dataframe

    def train_global_model(
        self,
        *,
        data_path: str = DEFAULT_TRAINING_DATASET_PATH,
        algorithms: Sequence[str],
        artifact_store: ArtifactStoreProtocol,
        model_output_path: str = DEFAULT_GLOBAL_MODEL_PATH,
    ) -> PersistedTrainingResult:
        """Load the full dataset, train the global model, and persist the winner."""

        dataframe = self.load_training_dataset(data_path)
        arena_result = self.run_algorithm_arena(dataframe, algorithms=algorithms)
        return self._persist_training_result(arena_result, artifact_store=artifact_store, model_output_path=model_output_path)

    def train_personal_model(
        self,
        *,
        user_id: str,
        data_path: str = DEFAULT_TRAINING_DATASET_PATH,
        dataframe: pd.DataFrame | None = None,
        algorithms: Sequence[str],
        artifact_store: ArtifactStoreProtocol,
        model_output_path_template: str = DEFAULT_PERSONAL_MODEL_TEMPLATE,
        minimum_rows: int = DEFAULT_MIN_PERSONAL_TRAINING_ROWS,
    ) -> PersistedTrainingResult:
        """Load the dataset, filter to one user, enforce minimum rows, and persist the winner."""

        effective_dataframe = dataframe.copy() if dataframe is not None else self.load_training_dataset(data_path)
        filtered_dataframe = self._filter_dataframe_for_user(effective_dataframe, user_id=user_id)
        if len(filtered_dataframe) < minimum_rows:
            raise ValueError("Insufficient data")

        sanitized_user_id = self._sanitize_user_id_for_path(user_id)
        arena_result = self.run_algorithm_arena(filtered_dataframe, algorithms=algorithms)
        model_output_path = model_output_path_template.format(user_id=sanitized_user_id)
        return self._persist_training_result(arena_result, artifact_store=artifact_store, model_output_path=model_output_path)

    def run_algorithm_arena(
        self,
        dataframe: pd.DataFrame,
        *,
        algorithms: Sequence[str],
        user_id: str | None = None,
    ) -> AlgorithmArenaResult:
        """Run split evaluation, pick a winner, and retrain it on the full dataset."""

        if dataframe.empty:
            raise ValueError("Retraining payload is empty. Provide at least one row.")

        filtered_dataframe = self._filter_dataframe_for_user(dataframe, user_id=user_id)
        if filtered_dataframe.empty:
            raise ValueError("Retraining payload is empty after user_id filtering.")

        if TARGET_COLUMN not in dataframe.columns:
            raise ValueError(
                f"Retraining payload is missing required target column '{TARGET_COLUMN}'."
            )

        normalized_dataframe = filtered_dataframe.drop_duplicates().reset_index(drop=True)
        if normalized_dataframe.empty:
            raise ValueError("Retraining payload contains zero unique rows after deduplication.")

        feature_frame = self.feature_validator.validate(
            normalized_dataframe,
            required_columns=TRAIN_FEATURE_COLUMNS,
            context="Retraining payload",
        )
        target_series = self.target_validator.validate(
            normalized_dataframe[TARGET_COLUMN],
            target_name=TARGET_COLUMN,
            context="Retraining payload",
        )

        if not algorithms:
            raise ValueError("No candidate algorithms were configured for retraining.")

        try:
            x_train, x_test, y_train, y_test = sk_model_selection.train_test_split(
                feature_frame,
                target_series,
                test_size=0.2,
                random_state=self.random_state,
                stratify=target_series,
            )
        except ValueError as ex:
            raise ValueError(
                "Failed to create stratified 80/20 train/test split. "
                f"Class distribution: {target_series.value_counts().to_dict()}"
            ) from ex

        leaderboard = tuple(
            self._evaluate_algorithm(
                algorithm,
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                full_target_series=target_series,
            )
            for algorithm in algorithms
        )
        winner = self._choose_winner(leaderboard)
        winner_predictions = self._predict_with_optional_decoder(winner.model, x_test, winner.label_classes)
        macro_precision = float(
            sk_metrics.precision_score(y_test, winner_predictions, average="macro", zero_division=0)
        )
        macro_recall = float(
            sk_metrics.recall_score(y_test, winner_predictions, average="macro", zero_division=0)
        )
        top_predictive_feature = self._extract_top_predictive_feature(winner.model, list(x_train.columns))
        primary_misclassification = self._build_primary_misclassification(y_test, winner_predictions)
        xai_global = self._build_xai_global(
            winner=winner,
            feature_names=list(x_train.columns),
            y_true=y_test,
            predictions=winner_predictions,
        )

        retrained_model, retrained_label_classes, _ = self._fit_model(
            winner.name,
            x_train=feature_frame,
            y_train=target_series,
            full_target_series=target_series,
        )

        return AlgorithmArenaResult(
            winning_algorithm=winner.name,
            winning_accuracy=winner.accuracy,
            winning_f1_score=winner.f1_score,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            top_predictive_feature=top_predictive_feature,
            primary_misclassification=primary_misclassification,
            total_rows_processed=int(len(normalized_dataframe)),
            xai_global=xai_global,
            leaderboard=leaderboard,
            retrained_model=retrained_model,
            retrained_label_classes=retrained_label_classes,
        )

    @staticmethod
    def _filter_dataframe_for_user(dataframe: pd.DataFrame, *, user_id: str | None) -> pd.DataFrame:
        """Reduce the payload to one requested user before train/test splitting."""

        if user_id is None:
            return dataframe.copy()

        normalized_user_id = user_id.strip().casefold()
        if not normalized_user_id:
            raise ValueError("Retraining payload user_id must be non-empty.")

        if "user_id" not in dataframe.columns:
            raise ValueError("Retraining payload is missing required user_id column for personalized training.")

        filtered_dataframe = dataframe[
            dataframe["user_id"].astype(str).str.strip().str.casefold() == normalized_user_id
        ].copy()

        if filtered_dataframe.empty:
            raise ValueError(f"No retraining rows found for user_id '{user_id}'.")

        return filtered_dataframe.reset_index(drop=True)

    def _evaluate_algorithm(
        self,
        algorithm: str,
        *,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        full_target_series: pd.Series,
    ) -> AlgorithmLeaderboardEntry:
        model, label_classes, execution_time_ms = self._fit_model(
            algorithm,
            x_train=x_train,
            y_train=y_train,
            full_target_series=full_target_series,
        )
        predictions = self._predict_with_optional_decoder(model, x_test, label_classes)

        accuracy = float(sk_metrics.accuracy_score(y_test, predictions))
        f1_score = float(sk_metrics.f1_score(y_test, predictions, average="macro"))

        return AlgorithmLeaderboardEntry(
            name=algorithm,
            accuracy=accuracy,
            f1_score=f1_score,
            execution_time_ms=execution_time_ms,
            model=model,
            label_classes=label_classes,
        )

    @staticmethod
    def _sanitize_user_id_for_path(user_id: str) -> str:
        normalized = user_id.strip().lower()
        sanitized = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in normalized)
        sanitized = sanitized.strip("_")
        return sanitized or "anonymous"

    @staticmethod
    def _persist_training_result(
        arena_result: AlgorithmArenaResult,
        *,
        artifact_store: ArtifactStoreProtocol,
        model_output_path: str,
    ) -> PersistedTrainingResult:
        artifact = ModelArtifact.from_training(
            model=arena_result.retrained_model,
            model_name=arena_result.winning_algorithm,
            feature_names=TRAIN_FEATURE_COLUMNS,
            target_name=TARGET_COLUMN,
            label_classes=arena_result.retrained_label_classes,
        )
        artifact_store.save_model_artifact(artifact, model_output_path)
        return PersistedTrainingResult(
            arena_result=arena_result,
            artifact=artifact,
            model_output_path=str(Path(model_output_path)),
        )

    def _fit_model(
        self,
        algorithm: str,
        *,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        full_target_series: pd.Series,
    ) -> tuple[Any, list[str] | None, float]:
        model = self.model_factory.create(algorithm)

        label_encoder: LabelEncoder | None = None
        y_train_for_fit: Any = y_train
        if algorithm == Algorithm.XGBOOST.value:
            label_encoder = LabelEncoder()
            label_encoder.fit(full_target_series)
            y_train_for_fit = label_encoder.transform(y_train)

        started_at = perf_counter()
        model.fit(x_train, y_train_for_fit)
        execution_time_ms = (perf_counter() - started_at) * 1000.0

        return model, (label_encoder.classes_.tolist() if label_encoder is not None else None), execution_time_ms

    @staticmethod
    def _predict_with_optional_decoder(
        model: Any,
        x_test: pd.DataFrame,
        label_classes: list[str] | None,
    ) -> Any:
        predictions = model.predict(x_test)
        if label_classes is None:
            return predictions

        label_encoder = LabelEncoder()
        label_encoder.classes_ = pd.Index(label_classes).to_numpy(dtype=object)
        encoded_predictions = pd.Series(predictions).astype(int).to_numpy()
        return label_encoder.inverse_transform(encoded_predictions)

    @staticmethod
    def _choose_winner(
        leaderboard: Sequence[AlgorithmLeaderboardEntry],
    ) -> AlgorithmLeaderboardEntry:
        if not leaderboard:
            raise ValueError("No candidate models were produced during retraining.")

        ranked = sorted(
            leaderboard,
            key=lambda entry: (-entry.f1_score, -entry.accuracy, entry.name),
        )
        return ranked[0]

    @staticmethod
    def _extract_top_predictive_feature(model: Any, feature_names: list[str]) -> str:
        estimator = getattr(model, "named_steps", {}).get("clf", model)

        importance_scores = getattr(estimator, "feature_importances_", None)
        if importance_scores is None:
            coefficients = getattr(estimator, "coef_", None)
            if coefficients is not None:
                importance_scores = np.abs(np.asarray(coefficients, dtype=float))
                if importance_scores.ndim > 1:
                    importance_scores = importance_scores.mean(axis=0)

        if importance_scores is None:
            return feature_names[0] if feature_names else "unknown_feature"

        flattened_scores = np.asarray(importance_scores, dtype=float).reshape(-1)
        if flattened_scores.size == 0:
            return feature_names[0] if feature_names else "unknown_feature"

        feature_index = int(np.argmax(flattened_scores))
        if 0 <= feature_index < len(feature_names):
            return feature_names[feature_index]

        return feature_names[0] if feature_names else "unknown_feature"

    @staticmethod
    def _build_primary_misclassification(y_true: pd.Series, predictions: Any) -> str:
        y_true_values = [str(value) for value in y_true.tolist()]
        prediction_values = [str(value) for value in pd.Series(predictions).tolist()]
        labels = sorted(set(y_true_values) | set(prediction_values))

        if not labels:
            return "No primary misclassification detected on holdout data"

        confusion = sk_metrics.confusion_matrix(y_true_values, prediction_values, labels=labels)
        off_diagonal = confusion.copy()
        np.fill_diagonal(off_diagonal, 0)

        largest_error = int(off_diagonal.max()) if off_diagonal.size else 0
        if largest_error <= 0:
            return "No primary misclassification detected on holdout data"

        true_index, predicted_index = np.argwhere(off_diagonal == largest_error)[0]
        return (
            f"True class {labels[int(true_index)]} was frequently misclassified as "
            f"{labels[int(predicted_index)]}"
        )

    def _build_xai_global(
        self,
        *,
        winner: AlgorithmLeaderboardEntry,
        feature_names: list[str],
        y_true: pd.Series,
        predictions: Any,
    ) -> dict[str, Any]:
        return {
            "confusion_matrix": self._build_confusion_matrix_payload(y_true, predictions),
            "feature_importances": self._build_feature_importances_payload(
                winner.name,
                winner.model,
                feature_names,
            ),
        }

    @staticmethod
    def _build_confusion_matrix_payload(y_true: pd.Series, predictions: Any) -> dict[str, Any]:
        normalized_binary = TrainingArenaService._normalize_binary_fatigue_labels(y_true, predictions)
        if normalized_binary is not None:
            y_true_binary, prediction_binary = normalized_binary
            matrix = confusion_matrix(y_true_binary, prediction_binary, labels=[0, 1])
            return {
                "labels": [0, 1],
                "class_names": {"0": "Normal", "1": "Fatigued"},
                "matrix": matrix.astype(int).tolist(),
            }

        y_true_values = [str(value) for value in y_true.tolist()]
        prediction_values = [str(value) for value in pd.Series(predictions).tolist()]
        labels = sorted(set(y_true_values) | set(prediction_values))
        matrix = confusion_matrix(y_true_values, prediction_values, labels=labels)
        return {
            "labels": labels,
            "matrix": matrix.astype(int).tolist(),
        }

    @staticmethod
    def _build_feature_importances_payload(
        algorithm: str,
        model: Any,
        feature_names: list[str],
    ) -> list[dict[str, Any]]:
        if algorithm not in {Algorithm.XGBOOST.value, Algorithm.RANDOM_FOREST.value}:
            return []

        estimator = getattr(model, "named_steps", {}).get("clf", model)
        importance_scores = getattr(estimator, "feature_importances_", None)
        if importance_scores is None:
            return []

        flattened_scores = np.asarray(importance_scores, dtype=float).reshape(-1)
        if flattened_scores.size == 0 or flattened_scores.size != len(feature_names):
            return []

        importance_frame = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": flattened_scores,
            }
        )
        top_importances = (
            importance_frame.sort_values("importance", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        return [
            {
                "feature": str(row["feature"]),
                "importance": float(row["importance"]),
            }
            for _, row in top_importances.iterrows()
        ]

    @staticmethod
    def _normalize_binary_fatigue_labels(
        y_true: pd.Series,
        predictions: Any,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        def normalize_value(value: Any) -> int | None:
            if isinstance(value, (np.integer, int, bool)):
                numeric_value = int(value)
                return numeric_value if numeric_value in {0, 1} else None

            if isinstance(value, (np.floating, float)):
                numeric_value = float(value)
                if numeric_value in {0.0, 1.0}:
                    return int(numeric_value)
                return None

            normalized_text = str(value).strip().casefold()
            mapping = {
                "0": 0,
                "normal": 0,
                "false": 0,
                "1": 1,
                "fatigued": 1,
                "fatigue": 1,
                "true": 1,
            }
            return mapping.get(normalized_text)

        y_true_normalized = [normalize_value(value) for value in y_true.tolist()]
        prediction_normalized = [normalize_value(value) for value in pd.Series(predictions).tolist()]
        if any(value is None for value in y_true_normalized + prediction_normalized):
            return None

        return (
            np.asarray(y_true_normalized, dtype=int),
            np.asarray(prediction_normalized, dtype=int),
        )