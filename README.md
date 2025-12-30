# Typing-ML (Thesis Project)

Beginner-friendly ML workflow for my thesis.

**Phase 1:** Predict `weakest_finger` from typing session summary stats (WPM, accuracy, per-finger error rates).  
**Phase 2:** Use predictions to recommend typing drills that target weak fingers.

---

## Repository Structure

- `notebooks/01_eda.ipynb` — explore/clean data and export processed dataset
- `src/train.py` — reproducible training pipeline
- `src/evaluate.py` — evaluation + plots
- `reports/results.md` — experiment notes (thesis-friendly)

Suggested folders:
- `data/raw/` — raw data (ignored by git)
- `data/processed/` — cleaned data (ignored by git)
- `models/` — saved models (ignored by git)

---

## Setup (Conda)

```bash
conda create -n typing-ml python=3.11 -y
conda activate typing-ml
conda install -y numpy pandas scikit-learn matplotlib seaborn jupyterlab joblib -c conda-forge
```

(Optional) export environment:

```bash
conda env export > environment.yml
```

---

## Data

Expected file:

- `data/raw/sessions.csv`

Minimum columns:
- `user_id`, `session_id`, `timestamp`
- `wpm`, `accuracy`
- `error_left_pinky`, `error_left_ring`, `error_left_middle`, `error_left_index`
- `error_right_index`, `error_right_middle`, `error_right_ring`, `error_right_pinky`
- `weakest_finger` (target label)

> Note: `data/raw/` and `data/processed/` are usually ignored by git to avoid committing large/private datasets.

---

## Generate Synthetic Data (optional)

If you don’t have real data yet, you can generate a small synthetic dataset to test the pipeline.

```bash
mkdir -p data/raw
python src/generate_synthetic_data.py
```

This will create:
- `data/raw/sessions.csv`

---

## Run EDA

Start JupyterLab:

```bash
conda activate typing-ml
jupyter lab
```

Open:
- `notebooks/01_eda.ipynb`

Run all cells to generate:
- `data/processed/dataset.csv`

---

## Train

```bash
python src/train.py --data data/processed/dataset.csv
```

---

## Evaluate

```bash
python src/evaluate.py --data data/processed/dataset.csv --model models/model.joblib
```

Outputs:
- metrics in terminal
- confusion matrix plot in `reports/figures/` (if enabled)

---

## Notes

- If you want plots visible on GitHub, commit `notebooks/01_eda.ipynb` **with outputs**.
- Notebooks appear as “modified” after running because outputs are stored inside `.ipynb`.

---

## Generate Synthetic Data (optional)

```bash
mkdir -p data/raw
python src/generate_synthetic_data.py

