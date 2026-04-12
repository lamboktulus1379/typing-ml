from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureFrameValidator:
    range_rules: Mapping[str, Tuple[float, float | None]]

    def validate(
        self,
        source_df: pd.DataFrame,
        *,
        required_columns: Iterable[str],
        context: str,
    ) -> pd.DataFrame:
        required = list(required_columns)
        missing_cols = [column for column in required if column not in source_df.columns]
        if missing_cols:
            raise ValueError(f"{context} is missing required columns: {missing_cols}")

        feature_frame = source_df[required].copy()
        if feature_frame.empty:
            raise ValueError(f"{context} contains zero rows after feature selection.")

        feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce")

        invalid_numeric_columns = feature_frame.columns[feature_frame.isna().any()].tolist()
        if invalid_numeric_columns:
            raise ValueError(
                f"{context} contains null or non-numeric values in required columns: "
                f"{invalid_numeric_columns}"
            )

        if not np.isfinite(feature_frame.to_numpy(dtype=float)).all():
            raise ValueError(
                f"{context} contains non-finite values (inf or -inf) in required features."
            )

        out_of_range_messages: list[str] = []
        for col_name, (min_value, max_value) in self.range_rules.items():
            if col_name not in feature_frame.columns:
                continue
            col_values = feature_frame[col_name]
            if (col_values < min_value).any():
                out_of_range_messages.append(f"{col_name} < {min_value}")
            if max_value is not None and (col_values > max_value).any():
                out_of_range_messages.append(f"{col_name} > {max_value}")

        if out_of_range_messages:
            raise ValueError(
                f"{context} contains out-of-range values in required features: "
                f"{out_of_range_messages}"
            )

        return feature_frame


@dataclass(frozen=True)
class TargetSeriesValidator:
    allowed_labels: frozenset[str]
    min_classes: int = 2
    min_samples_per_class: int = 2

    def validate(self, source_series: pd.Series, *, target_name: str, context: str) -> pd.Series:
        if source_series.isna().any():
            raise ValueError(
                f"{context} target column '{target_name}' contains null values; fill or remove them first."
            )

        target_series = source_series.astype(str).str.strip()
        if (target_series == "").any():
            raise ValueError(
                f"{context} target column '{target_name}' contains blank labels; fix input labels first."
            )

        invalid_labels = sorted(set(target_series.unique()) - self.allowed_labels)
        if invalid_labels:
            raise ValueError(
                f"{context} target column '{target_name}' contains unsupported labels: {invalid_labels}. "
                f"Allowed labels: {sorted(self.allowed_labels)}"
            )

        class_counts = target_series.value_counts()
        if class_counts.size < self.min_classes:
            raise ValueError(
                f"{context} target column '{target_name}' must contain at least "
                f"{self.min_classes} classes. Found classes: {class_counts.to_dict()}"
            )

        sparse_classes = class_counts[class_counts < self.min_samples_per_class]
        if not sparse_classes.empty:
            raise ValueError(
                f"{context} target column '{target_name}' has classes with fewer than "
                f"{self.min_samples_per_class} rows: {sparse_classes.to_dict()}"
            )

        return target_series
