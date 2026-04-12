"""Shared OOP building blocks for typing-ml pipelines."""

from .constants import (
    ALLOWED_WEAKEST_FINGER_LABELS,
    FEATURE_RANGE_RULES,
    FINGERS,
    TARGET_COLUMN,
    TRAIN_FEATURE_COLUMNS,
)
from .evaluation_service import EvaluationConfig, EvaluationService
from .interfaces import (
    ArtifactStoreProtocol,
    FeatureValidatorProtocol,
    ModelFactoryProtocol,
    TargetValidatorProtocol,
)
from .model_factory import Algorithm, ModelPipelineFactory
from .training_service import TrainingConfig, TrainingService
from .validation import FeatureFrameValidator, TargetSeriesValidator

__all__ = [
    "Algorithm",
    "ALLOWED_WEAKEST_FINGER_LABELS",
    "ArtifactStoreProtocol",
    "EvaluationConfig",
    "EvaluationService",
    "FeatureValidatorProtocol",
    "FeatureFrameValidator",
    "FEATURE_RANGE_RULES",
    "FINGERS",
    "ModelFactoryProtocol",
    "ModelPipelineFactory",
    "TARGET_COLUMN",
    "TargetValidatorProtocol",
    "TargetSeriesValidator",
    "TRAIN_FEATURE_COLUMNS",
    "TrainingConfig",
    "TrainingService",
]
