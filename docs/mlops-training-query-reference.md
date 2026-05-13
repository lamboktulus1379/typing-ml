# MLOps Training Query Reference

## Purpose

This document defines the SQL reference used by the offline thesis notebook `notebooks/visualize_outliers.ipynb`.

The goal is to keep the notebook aligned with the production-style MLOps retraining source instead of relying on ad-hoc CSV exports or a simplified standalone table. The notebook remains read-only and exists only for analysis and Chapter 4 visualization.

## Why This Query Exists

The production retraining flow in the Typing backend does not train from a `TypingTelemetry` flat table. It builds training rows from:

- `SessionFeature`
- `TypingSession`

The backend then:

1. keeps only rows where `SessionFeature.IsUsedForTraining = 0`,
2. joins each feature row to its parent typing session,
3. derives the supervised label `weakest_finger`, and
4. maps the per-finger telemetry columns into the payload expected by `typing-ml`.

The notebook query mirrors the same source-of-truth tables and the same label-resolution rule, but projects only the fields needed for thesis visualization:

- `Sesi`
- `FlightTime`
- `DwellTime`

## Production Alignment

The SQL below is intentionally aligned with these backend behaviors:

- Pending retraining rows are selected from `SessionFeature`.
- A row is eligible only when `IsUsedForTraining = 0`.
- Each row is joined to `TypingSession` through `TypingSessionId`.
- `weakest_finger` prefers `TypingSession.WeakestFinger` when it is valid.
- If `TypingSession.WeakestFinger` is empty or invalid, the label is inferred from the maximum error metric across the eight finger-specific error columns.
- Rows are ordered by `SessionFeature.CreatedAt`, which matches the retraining flow ordering.

## Reference SQL

```sql
WITH PendingTrainingRows AS (
    SELECT
        sf.Id AS SessionFeatureId,
        ts.Id AS TypingSessionId,
        sf.CreatedAt AS FeatureCapturedAt,
        ts.CreatedAt AS SessionCapturedAt,
        CAST(ts.Wpm AS float) AS Wpm,
        CAST(CASE WHEN ts.Accuracy <= 1 THEN ts.Accuracy * 100.0 ELSE ts.Accuracy END AS float) AS Accuracy,
        CAST(sf.DwellLeftPinky AS float) AS dwell_left_pinky,
        CAST(sf.DwellLeftRing AS float) AS dwell_left_ring,
        CAST(sf.DwellLeftMiddle AS float) AS dwell_left_middle,
        CAST(sf.DwellLeftIndex AS float) AS dwell_left_index,
        CAST(sf.DwellRightIndex AS float) AS dwell_right_index,
        CAST(sf.DwellRightMiddle AS float) AS dwell_right_middle,
        CAST(sf.DwellRightRing AS float) AS dwell_right_ring,
        CAST(sf.DwellRightPinky AS float) AS dwell_right_pinky,
        CAST(sf.FlightLeftPinky AS float) AS flight_left_pinky,
        CAST(sf.FlightLeftRing AS float) AS flight_left_ring,
        CAST(sf.FlightLeftMiddle AS float) AS flight_left_middle,
        CAST(sf.FlightLeftIndex AS float) AS flight_left_index,
        CAST(sf.FlightRightIndex AS float) AS flight_right_index,
        CAST(sf.FlightRightMiddle AS float) AS flight_right_middle,
        CAST(sf.FlightRightRing AS float) AS flight_right_ring,
        CAST(sf.FlightRightPinky AS float) AS flight_right_pinky,
        CASE
            WHEN LOWER(LTRIM(RTRIM(ISNULL(ts.WeakestFinger, '')))) IN (
                'left_pinky', 'left_ring', 'left_middle', 'left_index',
                'right_index', 'right_middle', 'right_ring', 'right_pinky'
            )
                THEN LOWER(LTRIM(RTRIM(ts.WeakestFinger)))
            ELSE inferred.finger_name
        END AS weakest_finger
    FROM SessionFeature sf
    INNER JOIN TypingSession ts
        ON ts.Id = sf.TypingSessionId
    OUTER APPLY (
        SELECT TOP (1) errors.finger_name
        FROM (VALUES
            ('left_pinky', CAST(sf.ErrorLeftPinky AS float)),
            ('left_ring', CAST(sf.ErrorLeftRing AS float)),
            ('left_middle', CAST(sf.ErrorLeftMiddle AS float)),
            ('left_index', CAST(sf.ErrorLeftIndex AS float)),
            ('right_index', CAST(sf.ErrorRightIndex AS float)),
            ('right_middle', CAST(sf.ErrorRightMiddle AS float)),
            ('right_ring', CAST(sf.ErrorRightRing AS float)),
            ('right_pinky', CAST(sf.ErrorRightPinky AS float))
        ) AS errors(finger_name, error_value)
        ORDER BY errors.error_value DESC, errors.finger_name ASC
    ) AS inferred
    WHERE sf.IsUsedForTraining = 0
)
SELECT
    ROW_NUMBER() OVER (ORDER BY FeatureCapturedAt ASC) AS Sesi,
    TypingSessionId,
    SessionFeatureId,
    SessionCapturedAt,
    FeatureCapturedAt,
    weakest_finger,
    CAST((
        flight_left_pinky + flight_left_ring + flight_left_middle + flight_left_index +
        flight_right_index + flight_right_middle + flight_right_ring + flight_right_pinky
    ) / 8.0 AS float) AS FlightTime,
    CAST((
        dwell_left_pinky + dwell_left_ring + dwell_left_middle + dwell_left_index +
        dwell_right_index + dwell_right_middle + dwell_right_ring + dwell_right_pinky
    ) / 8.0 AS float) AS DwellTime
FROM PendingTrainingRows
WHERE weakest_finger IS NOT NULL
ORDER BY Sesi;
```

## Thesis-Specific Projection

The notebook does not need the full retraining payload for the boxplot figure. It only keeps:

- `Sesi`: sequential row number for the chapter figure
- `FlightTime`: average of the eight per-finger flight metrics
- `DwellTime`: average of the eight per-finger dwell metrics

This keeps the visualization concise while preserving alignment with the real retraining source.

## Engineering Notes

- Treat this query as a documented reference query, not a duplicated business rule owned by the notebook.
- If the Typing backend changes the training-row extraction logic in `MlRetrainingService` or the `SessionFeature` repository, update this document and the notebook together in the same change set.
- Keep the notebook read-only against SQL Server.
- Prefer placeholders or environment-driven secrets in notebooks instead of hardcoding credentials.

## Ownership

- Production query semantics: Typing backend MLOps retraining flow
- Offline analysis consumer: `typing-ml/notebooks/visualize_outliers.ipynb`
- Documentation owner: thesis / analytics maintenance flow
