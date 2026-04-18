# Algorithm Comparison

- Data path: data/processed/dataset.csv
- Random state: 42
- Algorithms evaluated: 3

## Quality Metrics

| Algorithm | Status | Accuracy | Macro F1 | Weighted F1 | Support |
|---|---|---:|---:|---:|---:|
| logistic_regression | ok | 1.0000 | 1.0000 | 1.0000 | 2000 |
| random_forest | ok | 1.0000 | 1.0000 | 1.0000 | 2000 |
| xgboost | ok | 1.0000 | 1.0000 | 1.0000 | 2000 |

## Efficiency Tie-Breakers

| Algorithm | Model Size (bytes) | Predict ms/call | Test Rows |
|---|---:|---:|---:|
| logistic_regression | 5245 | 0.5580 | 2000 |
| random_forest | 3152096 | 9.8171 | 2000 |
| xgboost | 700711 | 2.0493 | 2000 |

## Recommendation

- Best by quality metrics: logistic_regression (accuracy=1.0000, macro_f1=1.0000)
- Quality tie count: 3. Best practical choice by speed and size: logistic_regression.
