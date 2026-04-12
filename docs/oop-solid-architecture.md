# OOP + SOLID Architecture Guide

This project now uses a service-oriented architecture in `src/ml_pipeline` to keep each module focused and easy to test.

## Design Goals

- Single Responsibility: each class handles one concern.
- Open/Closed: add algorithms or validators by extending dedicated modules.
- Liskov Substitution: services depend on contracts and can accept compatible implementations.
- Interface Segregation: contracts are small and purpose-specific.
- Dependency Inversion: use protocols in services instead of hard-coding concrete implementations.

## Package Layout

- `src/ml_pipeline/constants.py`
  - Shared domain constants: features, target name, label set, value ranges.

- `src/ml_pipeline/validation.py`
  - `FeatureFrameValidator` for feature schema/range checks.
  - `TargetSeriesValidator` for target-label checks and split safety checks.

- `src/ml_pipeline/model_factory.py`
  - `ModelPipelineFactory` creates `Pipeline(StandardScaler + classifier)`.
  - Supports `logistic_regression`, `random_forest`, and `xgboost`.

- `src/ml_pipeline/artifacts.py`
  - `ArtifactStore` persists and loads model artifacts/reports.
  - Handles both dict-based and legacy model-only artifacts.

- `src/ml_pipeline/training_service.py`
  - `TrainingService` orchestrates train split, fit, reporting, and artifact persistence.

- `src/ml_pipeline/evaluation_service.py`
  - `EvaluationService` orchestrates evaluation split, metrics output, and confusion matrix export.

- `src/ml_pipeline/interfaces.py`
  - Protocol contracts for factories, validators, and artifact store.
  - Core mechanism for dependency inversion in services.

## Interface Contracts

Services depend on these contracts:

- `ModelFactoryProtocol`
- `ArtifactStoreProtocol`
- `FeatureValidatorProtocol`
- `TargetValidatorProtocol`

Concrete classes (`ModelPipelineFactory`, `ArtifactStore`, validators) satisfy these contracts and are injected via `TrainingService.default()` and `EvaluationService.default()`.

## Entry Point Pattern

The CLI files are intentionally thin:

- `src/train.py`
- `src/evaluate.py`

They only parse args, build config objects, and delegate to services. This keeps orchestration logic centralized and easier to test.

## API Integration

`src/api.py` reuses `ArtifactStore` and wraps prediction behavior in `InferenceService`.
This aligns online inference with the same artifact conventions used by training/evaluation.

## How to Extend

### Add a new algorithm

1. Add enum value in `Algorithm` (`model_factory.py`).
2. Implement classifier branch in `_build_classifier`.
3. Ensure dependency is in `environment.yml` if needed.
4. Use it with `python src/train.py --algorithm <new_name>`.

### Replace artifact persistence backend

1. Implement `ArtifactStoreProtocol` in a new class.
2. Inject your implementation into `TrainingService` and `EvaluationService`.

### Add custom validation rules

1. Implement `FeatureValidatorProtocol` and/or `TargetValidatorProtocol`.
2. Inject into services for environment-specific validation behavior.

## Recommended Workflow

1. Generate or prepare dataset.
2. Train with explicit algorithm selection.
3. Evaluate and inspect confusion matrix.
4. Serve model via FastAPI for integration testing.

This structure keeps each responsibility isolated while preserving end-to-end reproducibility.
