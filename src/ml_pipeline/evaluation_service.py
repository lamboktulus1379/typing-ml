# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

"""Evaluation orchestration layer for trained weakest-finger models."""

import os
from dataclasses import dataclass
from typing import Any, Dict, cast

import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection

from .artifacts import ArtifactStore
from .constants import ALLOWED_WEAKEST_FINGER_LABELS, FEATURE_RANGE_RULES, TARGET_COLUMN
from .interfaces import ArtifactStoreProtocol, FeatureValidatorProtocol, TargetValidatorProtocol
from .validation import FeatureFrameValidator, TargetSeriesValidator


@dataclass(frozen=True)
class EvaluationConfig:
    """Immutable input configuration for an evaluation run."""

    data_path: str
    model_path: str
    fig_dir: str
    random_state: int = 42


@dataclass
class EvaluationService:
    """Coordinates artifact loading, validation, metrics, and confusion matrix export."""

    artifact_store: ArtifactStoreProtocol
    feature_validator: FeatureValidatorProtocol
    target_validator: TargetValidatorProtocol

    @classmethod
    def default(cls) -> "EvaluationService":
        """Build a default service graph using built-in implementations."""

        return cls(
            artifact_store=ArtifactStore(),
            feature_validator=FeatureFrameValidator(FEATURE_RANGE_RULES),
            target_validator=TargetSeriesValidator(ALLOWED_WEAKEST_FINGER_LABELS),
        )

    @staticmethod
    def _decode_predictions(predictions: Any, label_classes: Any) -> Any:
        """Decode integer class predictions into label strings when metadata is present."""

        if not label_classes:
            return predictions

        classes = list(label_classes)
        decoded: list[str] = []
        for value in predictions:
            if isinstance(value, str):
                decoded.append(value)
                continue

            index = int(value)
            if index < 0 or index >= len(classes):
                raise ValueError(
                    "Model prediction index is outside the saved label class range. "
                    f"index={index}, classes={classes}"
                )
            decoded.append(classes[index])

        return decoded

    def evaluate(self, config: EvaluationConfig) -> Dict[str, Any]:
        """Run end-to-end evaluation and save confusion matrix artifacts."""

        print(f"Loading evaluation dataset from {config.data_path}...")
        dataframe = pd.read_csv(config.data_path)
        if dataframe.empty:
            raise ValueError("Evaluation dataset is empty. Provide at least one row.")

        if TARGET_COLUMN not in dataframe.columns:
            raise ValueError(
                f"Evaluation dataset is missing required target column '{TARGET_COLUMN}'."
            )

        if not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model artifact not found: {config.model_path}")

        model, artifact = self.artifact_store.load_model_artifact(config.model_path)
        expected_features_raw = artifact.get("feature_names", [])
        expected_features = list(expected_features_raw)
        if not expected_features:
            raise ValueError(
                "Artifact does not contain 'feature_names'. This can happen with "
                "legacy model-only artifacts that do not expose schema metadata. "
                "Re-train with src/train.py to generate a dict artifact that includes "
                "feature_names."
            )

        feature_frame = self.feature_validator.validate(
            dataframe,
            required_columns=expected_features,
            context="Evaluation dataset",
        )
        target_series = self.target_validator.validate(
            dataframe[TARGET_COLUMN],
            target_name=TARGET_COLUMN,
            context="Evaluation dataset",
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
            _, x_test, _, y_test = cast(
                tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
                tuple(split_result),
            )
        except ValueError as ex:
            raise ValueError(
                "Failed to create stratified evaluation split. "
                f"Class distribution: {target_series.value_counts().to_dict()}"
            ) from ex

        print("Generating predictions...")
        raw_predictions = model.predict(x_test)
        predictions = self._decode_predictions(raw_predictions, artifact.get("label_classes"))

        classification_report_fn = cast(Any, sk_metrics.classification_report)
        report_text = cast(str, classification_report_fn(y_test, predictions))
        print("\n=== Classification Report ===")
        print(report_text)

        os.makedirs(config.fig_dir, exist_ok=True)
        confusion_display_from_predictions = cast(Any, sk_metrics.ConfusionMatrixDisplay.from_predictions)
        confusion_display_from_predictions(
            y_test,
            predictions,
            xticks_rotation=45,
        )
        plt.title("Confusion Matrix - Weakest Finger (26 Features)")
        plt.tight_layout()

        plot_path = os.path.join(config.fig_dir, "confusion_matrix.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()

        print(f"\nSuccess! Saved confusion matrix plot to: {plot_path}")

        return {
            "plot_path": plot_path,
            "sample_count": len(y_test),
        }
