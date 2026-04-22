# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

"""Training orchestration layer for weakest-finger classification."""

from dataclasses import dataclass
from typing import Any, Dict, cast

import pandas as pd
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection
from sklearn.preprocessing import LabelEncoder

from .artifacts import ArtifactStore, ModelArtifact
from .cleaning import clean_timing_outliers
from .constants import (
    ALLOWED_WEAKEST_FINGER_LABELS,
    FEATURE_RANGE_RULES,
    TARGET_COLUMN,
    TRAIN_FEATURE_COLUMNS,
)
from .model_factory import Algorithm, ModelPipelineFactory
from .interfaces import (
    ArtifactStoreProtocol,
    FeatureValidatorProtocol,
    ModelFactoryProtocol,
    TargetValidatorProtocol,
)
from .validation import FeatureFrameValidator, TargetSeriesValidator


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable input configuration for a training run."""

    data_path: str
    model_out: str
    report_out: str
    algorithm: str
    random_state: int = 42


@dataclass
class TrainingService:
    """Coordinates validation, train/test split, fitting, and artifact persistence."""

    model_factory: ModelFactoryProtocol
    artifact_store: ArtifactStoreProtocol
    feature_validator: FeatureValidatorProtocol
    target_validator: TargetValidatorProtocol

    @classmethod
    def default(cls, *, random_state: int = 42) -> "TrainingService":
        """Build a default service graph using built-in implementations."""

        return cls(
            model_factory=ModelPipelineFactory(random_state=random_state),
            artifact_store=ArtifactStore(),
            feature_validator=FeatureFrameValidator(FEATURE_RANGE_RULES),
            target_validator=TargetSeriesValidator(ALLOWED_WEAKEST_FINGER_LABELS),
        )

    def train(self, config: TrainingConfig) -> Dict[str, Any]:
        """Run end-to-end training and return a classification report payload."""

        print(f"Loading data from {config.data_path}...")
        dataframe = pd.read_csv(config.data_path)
        if dataframe.empty:
            raise ValueError("Training dataset is empty. Provide at least one row.")

        if TARGET_COLUMN not in dataframe.columns:
            raise ValueError(
                f"Dataset is missing required target column '{TARGET_COLUMN}' needed for training."
            )

        cleaned_dataframe = clean_timing_outliers(
            dataframe,
            log_prefix="Training dataset timing cleaning",
        )
        if cleaned_dataframe.empty:
            raise ValueError("Training dataset contains zero rows after timing outlier cleaning.")

        feature_frame = self.feature_validator.validate(
            cleaned_dataframe,
            required_columns=TRAIN_FEATURE_COLUMNS,
            context="Training dataset",
        )
        target_series = self.target_validator.validate(
            cleaned_dataframe[TARGET_COLUMN],
            target_name=TARGET_COLUMN,
            context="Training dataset",
        )

        try:
            train_test_split_fn = cast(Any, sk_model_selection.train_test_split)
            split_result = train_test_split_fn(
                feature_frame,
                target_series,
                test_size=0.2,
                random_state=config.random_state,
                stratify=target_series,
            )
            x_train, x_test, y_train, y_test = cast(
                tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
                tuple(split_result),
            )
        except ValueError as ex:
            raise ValueError(
                "Failed to create stratified train/test split. "
                f"Class distribution: {target_series.value_counts().to_dict()}"
            ) from ex

        print(f"Training algorithm: {config.algorithm}")
        model = cast(Any, self.model_factory.create(config.algorithm))

        label_encoder: LabelEncoder | None = None
        y_train_for_fit: Any = y_train
        if config.algorithm == Algorithm.XGBOOST.value:
            label_encoder = LabelEncoder()
            label_encoder.fit(target_series)
            y_train_for_fit = label_encoder.transform(y_train)

        model.fit(x_train, y_train_for_fit)

        predictions = model.predict(x_test)
        if label_encoder is not None:
            encoded_predictions = pd.Series(predictions).astype(int).to_numpy()
            predictions = label_encoder.inverse_transform(encoded_predictions)

        classification_report_fn = cast(Any, sk_metrics.classification_report)
        report = cast(
            Dict[str, Any],
            classification_report_fn(y_test, predictions, output_dict=True),
        )
        report["selected_model"] = config.algorithm

        artifact = ModelArtifact.from_training(
            model=model,
            model_name=config.algorithm,
            feature_names=TRAIN_FEATURE_COLUMNS,
            target_name=TARGET_COLUMN,
            label_classes=(label_encoder.classes_.tolist() if label_encoder is not None else None),
        )
        self.artifact_store.save_model_artifact(artifact, config.model_out)
        self.artifact_store.save_report(report, config.report_out)

        print(f"Saved model to: {config.model_out}")
        print(f"Selected Model: {config.algorithm}")
        accuracy = float(report.get("accuracy", 0.0))
        print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

        return report
