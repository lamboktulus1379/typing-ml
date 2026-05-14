from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable, Mapping

import pandas as pd


logger = logging.getLogger(__name__)

DEFAULT_DWELL_HARD_CAP_MS = 2000.0
DEFAULT_FLIGHT_HARD_CAP_MS = 3000.0
DEFAULT_IQR_MULTIPLIER = 1.5


@dataclass(frozen=True)
class IqrBounds:
    q1: float
    q3: float
    iqr: float
    lower_bound: float
    upper_bound: float


@dataclass(frozen=True)
class TimingOutlierAnalysis:
    cleaned_dataframe: pd.DataFrame
    outlier_dataframe: pd.DataFrame
    bounds_by_column: dict[str, IqrBounds]


def clean_timing_outliers(
    dataframe: pd.DataFrame,
    *,
    dwell_hard_cap_ms: float = DEFAULT_DWELL_HARD_CAP_MS,
    flight_hard_cap_ms: float = DEFAULT_FLIGHT_HARD_CAP_MS,
    iqr_multiplier: float = DEFAULT_IQR_MULTIPLIER,
    log_prefix: str = "Timing cleaning",
) -> pd.DataFrame:
    """Remove impossible and noisy timing rows before feature extraction and splitting.

    This function is the stable cleaner used by the existing API and training flows.
    The cleaner supports both generic telemetry columns (`dwell_time`, `flight_time`)
    and this repository's per-finger timing schema (`dwell_*`, `flight_*`).
    """

    cleaned = dataframe.copy()
    before_shape = cleaned.shape

    dwell_columns = _select_timing_columns(cleaned.columns, explicit_names=("dwell_time",), prefix="dwell_")
    flight_columns = _select_timing_columns(cleaned.columns, explicit_names=("flight_time",), prefix="flight_")
    timing_columns = [*dwell_columns, *flight_columns]

    if not timing_columns:
        logger.info("%s skipped: no dwell/flight timing columns found. shape=%s", log_prefix, before_shape)
        return cleaned

    negative_mask = pd.Series(False, index=cleaned.index)
    for column in timing_columns:
        numeric_column = pd.to_numeric(cleaned[column], errors="coerce")
        negative_mask |= numeric_column < 0
    cleaned = cleaned.loc[~negative_mask].copy()

    hard_cap_mask = pd.Series(False, index=cleaned.index)
    for column in dwell_columns:
        numeric_column = pd.to_numeric(cleaned[column], errors="coerce")
        hard_cap_mask |= numeric_column > dwell_hard_cap_ms
    for column in flight_columns:
        numeric_column = pd.to_numeric(cleaned[column], errors="coerce")
        hard_cap_mask |= numeric_column > flight_hard_cap_ms
    cleaned = cleaned.loc[~hard_cap_mask].copy()

    iqr_mask = pd.Series(False, index=cleaned.index)
    for column in timing_columns:
        numeric_column = pd.to_numeric(cleaned[column], errors="coerce")
        non_null_column = numeric_column.dropna()
        if non_null_column.empty:
            continue

        q1 = float(non_null_column.quantile(0.25))
        q3 = float(non_null_column.quantile(0.75))
        iqr = q3 - q1
        upper_bound = q3 + (iqr_multiplier * iqr)
        iqr_mask |= numeric_column > upper_bound

    cleaned = cleaned.loc[~iqr_mask].copy()

    after_shape = cleaned.shape
    logger.info(
        "%s completed: shape %s -> %s (removed=%s rows)",
        log_prefix,
        before_shape,
        after_shape,
        before_shape[0] - after_shape[0],
    )
    print(f"{log_prefix}: shape {before_shape} -> {after_shape}")
    return cleaned.reset_index(drop=True)


def analyze_preprocessing_outliers(
    dataframe: pd.DataFrame,
    *,
    timing_columns: Iterable[str],
    iqr_multiplier: float = DEFAULT_IQR_MULTIPLIER,
    hard_cap_rules: Mapping[str, float] | None = None,
) -> TimingOutlierAnalysis:
    """Build worker-facing outlier analysis for preprocessing and EDA summaries."""

    analyzed = dataframe.copy()
    ordered_columns = [column for column in timing_columns if column in analyzed.columns]
    if not ordered_columns:
        return TimingOutlierAnalysis(
            cleaned_dataframe=analyzed.copy(),
            outlier_dataframe=analyzed.iloc[0:0].copy(),
            bounds_by_column={},
        )

    negative_mask = pd.Series(False, index=analyzed.index)
    for column in ordered_columns:
        numeric_column = pd.to_numeric(analyzed[column], errors="coerce")
        negative_mask |= numeric_column < 0
        analyzed[column] = numeric_column

    hard_cap_mask = pd.Series(False, index=analyzed.index)
    for column, max_value in (hard_cap_rules or {}).items():
        if column not in analyzed.columns:
            continue
        numeric_column = pd.to_numeric(analyzed[column], errors="coerce")
        hard_cap_mask |= numeric_column > max_value

    candidate_dataframe = analyzed.loc[~negative_mask & ~hard_cap_mask].copy()

    iqr_mask = pd.Series(False, index=candidate_dataframe.index)
    bounds_by_column: dict[str, IqrBounds] = {}

    for column in ordered_columns:
        if column not in candidate_dataframe.columns:
            continue

        numeric_column = pd.to_numeric(candidate_dataframe[column], errors="coerce")
        non_null_column = numeric_column.dropna()
        if non_null_column.empty:
            continue

        q1 = float(non_null_column.quantile(0.25))
        q3 = float(non_null_column.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - (iqr_multiplier * iqr)
        upper_bound = q3 + (iqr_multiplier * iqr)
        bounds_by_column[column] = IqrBounds(
            q1=q1,
            q3=q3,
            iqr=iqr,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        iqr_mask |= (numeric_column < lower_bound) | (numeric_column > upper_bound)

    outlier_dataframe = candidate_dataframe.loc[iqr_mask].copy()
    cleaned_dataframe = candidate_dataframe.loc[~iqr_mask].copy()
    return TimingOutlierAnalysis(
        cleaned_dataframe=cleaned_dataframe,
        outlier_dataframe=outlier_dataframe,
        bounds_by_column=bounds_by_column,
    )


def _select_timing_columns(
    available_columns: Iterable[str],
    *,
    explicit_names: tuple[str, ...],
    prefix: str,
) -> list[str]:
    ordered_columns = list(available_columns)
    return [
        column
        for column in ordered_columns
        if column in explicit_names or column.startswith(prefix)
    ]
