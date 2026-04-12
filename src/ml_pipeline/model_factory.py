"""Factory for constructing training-ready sklearn pipelines by algorithm."""

import importlib
from dataclasses import dataclass
from enum import Enum
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Algorithm(str, Enum):
    """Supported training algorithms exposed by the CLI."""

    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"

    @classmethod
    def choices(cls) -> list[str]:
        """Return valid CLI values for argparse choices."""

        return [member.value for member in cls]


@dataclass(frozen=True)
class ModelPipelineFactory:
    """Builds a scaler+classifier pipeline based on selected algorithm."""

    random_state: int = 42

    def create(self, algorithm: str | Algorithm) -> Pipeline:
        """Create a standardized sklearn Pipeline for training/inference."""

        normalized_algorithm = self._parse_algorithm(algorithm)
        classifier = self._build_classifier(normalized_algorithm)

        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", classifier),
            ]
        )

    def _parse_algorithm(self, algorithm: str | Algorithm) -> Algorithm:
        """Normalize string/enum input into an Algorithm value."""

        if isinstance(algorithm, Algorithm):
            return algorithm

        try:
            return Algorithm(algorithm)
        except ValueError as ex:
            raise ValueError(
                f"Unsupported --algorithm '{algorithm}'. "
                f"Use one of: {Algorithm.choices()}"
            ) from ex

    def _build_classifier(self, algorithm: Algorithm) -> Any:
        """Instantiate a concrete classifier for the selected algorithm."""

        if algorithm == Algorithm.LOGISTIC_REGRESSION:
            return LogisticRegression(max_iter=2000)

        if algorithm == Algorithm.RANDOM_FOREST:
            return RandomForestClassifier(n_estimators=100, random_state=self.random_state)

        if algorithm == Algorithm.XGBOOST:
            try:
                xgb = importlib.import_module("xgboost")
            except ModuleNotFoundError as ex:
                raise RuntimeError(
                    "xgboost is required for --algorithm xgboost. "
                    "Install dependencies from environment.yml first."
                ) from ex
            return xgb.XGBClassifier(random_state=self.random_state)

        raise ValueError(f"Unhandled algorithm: {algorithm.value}")
