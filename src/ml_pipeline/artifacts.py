"""Artifact persistence helpers for model and report files."""

import json
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
    label_classes: list[str] | None = None

    @classmethod
    def from_training(
        cls,
        *,
        model: Any,
        model_name: str,
        feature_names: list[str],
        target_name: str,
        label_classes: list[str] | None = None,
    ) -> "ModelArtifact":
        """Create a model artifact with a UTC creation timestamp."""

        return cls(
            model=model,
            model_name=model_name,
            feature_names=feature_names,
            target_name=target_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            label_classes=label_classes,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize artifact to a joblib-friendly dictionary."""

        artifact = {
            "model": self.model,
            "model_name": self.model_name,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "created_at": self.created_at,
        }
        if self.label_classes:
            artifact["label_classes"] = self.label_classes

        return artifact


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

        self.save_json(report, report_path)

    def save_json(self, payload: Dict[str, Any], output_path: str) -> None:
        """Write a JSON payload to disk with stable indentation."""

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2)

    def load_json(self, input_path: str) -> Dict[str, Any]:
        """Load a JSON payload from disk."""

        with open(input_path, "r", encoding="utf-8") as file_handle:
            return cast(Dict[str, Any], json.load(file_handle))

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
