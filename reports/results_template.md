# Typing-ML Thesis Results Template

Use this template after each experiment run (for example after `make e2e`).

## A. Experiment Setup

| Field | Value |
|---|---|
| Dataset path | `data/processed/dataset.csv` |
| Number of rows | `<fill>` |
| Random seed | `42` |
| Train-test split | `80:20 (stratified)` |
| Candidate models | `logistic_regression`, `random_forest` |
| Selected model | `<fill from reports/training_report.json>` |

## B. Performance Summary

| Metric | Value |
|---|---|
| Accuracy | `<fill>` |
| Macro Precision | `<fill>` |
| Macro Recall | `<fill>` |
| Macro F1-score | `<fill>` |
| Weighted F1-score | `<fill>` |

## C. Per-Class Metrics

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| left_pinky | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| left_ring | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| left_middle | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| left_index | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| right_index | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| right_middle | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| right_ring | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| right_pinky | `<fill>` | `<fill>` | `<fill>` | `<fill>` |

## D. English Narrative Starters

1. The selected model for this experiment was `<model_name>`, chosen based on the highest holdout accuracy.
2. The model achieved an overall accuracy of `<value>` on the test set.
3. Class-wise analysis shows that `<best_class>` has the highest recall, while `<hard_class>` remains the most challenging class.
4. The confusion matrix indicates that misclassification mainly occurs between `<class_a>` and `<class_b>`.
5. These results suggest that dwell-time and flight-time features contribute meaningful signal for weakest-finger classification.

## E. Bahasa Indonesia Narrative Starters

1. Model terpilih pada eksperimen ini adalah `<model_name>`, berdasarkan akurasi holdout tertinggi.
2. Model menghasilkan akurasi keseluruhan sebesar `<value>` pada data uji.
3. Analisis per kelas menunjukkan bahwa `<kelas_terbaik>` memiliki recall tertinggi, sedangkan `<kelas_sulit>` masih menjadi kelas yang paling menantang.
4. Confusion matrix menunjukkan bahwa kesalahan klasifikasi paling sering terjadi antara `<kelas_a>` dan `<kelas_b>`.
5. Hasil ini mengindikasikan bahwa fitur dwell time dan flight time memberikan sinyal yang bermakna untuk klasifikasi jari terlemah.

## F. Figure Caption Template

Figure X. Confusion Matrix of weakest-finger classification using `<model_name>` on the test set (seed=`42`, split=`80:20 stratified`).
