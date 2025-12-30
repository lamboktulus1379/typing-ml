
# Results Log

## Experiment 001 — Baseline weakest-finger classifier (LogReg)
**Date:** 2025-12-29  
**Dataset:** data/processed/dataset.csv  
**Split:** 80/20 stratified, random_state=42  
**Features:** numeric session stats (wpm, accuracy, finger error rates)  
**Model:** StandardScaler + LogisticRegression(max_iter=1000)

### Metrics (test set)
- Accuracy: ...
- Macro F1: ...
- Notes: Which fingers are confused? (see confusion matrix)

### Figures
- confusion_matrix.png

### Observations
- Model often confuses right_index vs right_middle when error rates are close.
- Users with high overall accuracy show weaker signal.

### Next steps
- Add rolling history features (last 3 sessions)
- Compare with RandomForest
- Try class_weight="balanced" if label imbalance exists
