"""Artifact persistence helpers for model and report files."""

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, cast

import joblib


@dataclass(frozen=True)
class ModelArtifact:
    """Structured model artifact payload saved to disk via joblib."""

    model: Any
    model_name: str
    feature_names: list[str]
    target_name: str
    created_at: str

    @classmethod
    def from_training(
        cls,
        *,
        model: Any,
        model_name: str,
        feature_names: list[str],
        target_name: str,
    ) -> "ModelArtifact":
        """Create a model artifact with a UTC creation timestamp."""

        return cls(
            model=model,
            model_name=model_name,
            feature_names=feature_names,
            target_name=target_name,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize artifact to a joblib-friendly dictionary."""

        return {
            "model": self.model,
            "model_name": self.model_name,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ArtifactStore:
    """Persistence boundary for model artifacts and report payloads."""

    def save_model_artifact(self, artifact: ModelArtifact, output_path: str) -> None:
        """Write model artifact to disk and create parent folders if needed."""

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        joblib.dump(artifact.to_dict(), output_path)

    def save_report(self, report: Dict[str, Any], report_path: str) -> None:
        """Write JSON report to disk with stable indentation."""

        report_dir = os.path.dirname(report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)

        with open(report_path, "w", encoding="utf-8") as file_handle:
            import json

            json.dump(report, file_handle, indent=2)

    def load_model_artifact(self, model_path: str) -> tuple[Any, Dict[str, Any]]:
        """Load dict-based artifacts and gracefully handle legacy model-only files."""

        raw_artifact = joblib.load(model_path)
        if isinstance(raw_artifact, dict):
            artifact = cast(Dict[str, Any], raw_artifact)
            if "model" in artifact:
                return artifact["model"], artifact

        legacy_model = cast(Any, raw_artifact)
        if hasattr(legacy_model, "predict"):
            artifact: Dict[str, Any] = {}
            feature_names_in = getattr(legacy_model, "feature_names_in_", None)
            if feature_names_in is not None and hasattr(feature_names_in, "tolist"):
                artifact["feature_names"] = list(feature_names_in)
            return legacy_model, artifact

        raise ValueError(
            "Model artifact is neither a dict containing 'model' nor a legacy "
            "predictable model object. Re-train with src/train.py to generate "
            "the current dict-based artifact format."
        )
