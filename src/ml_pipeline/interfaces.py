"""Interface contracts for dependency-inverted ML pipeline services.

These Protocols allow services to depend on behavior, not concrete classes.
"""

from typing import Any, Dict, Iterable, Protocol, runtime_checkable

import pandas as pd
from sklearn.pipeline import Pipeline

from .artifacts import ModelArtifact


@runtime_checkable
class ModelFactoryProtocol(Protocol):
    """Builds a training pipeline for a requested algorithm."""

    def create(self, algorithm: str) -> Pipeline:
        ...


@runtime_checkable
class ArtifactStoreProtocol(Protocol):
    """Persists and retrieves model artifacts and training reports."""

    def save_model_artifact(self, artifact: ModelArtifact, output_path: str) -> None:
        ...

    def save_report(self, report: Dict[str, Any], report_path: str) -> None:
        ...

    def load_model_artifact(self, model_path: str) -> tuple[Any, Dict[str, Any]]:
        ...


@runtime_checkable
class FeatureValidatorProtocol(Protocol):
    """Validates and normalizes feature frames before model operations."""

    def validate(
        self,
        source_df: pd.DataFrame,
        *,
        required_columns: Iterable[str],
        context: str,
    ) -> pd.DataFrame:
        ...


@runtime_checkable
class TargetValidatorProtocol(Protocol):
    """Validates target labels and class distribution constraints."""

    def validate(self, source_series: pd.Series, *, target_name: str, context: str) -> pd.Series:
        ...
