# HDD Failure Prediction — CPU-friendly incremental pipeline

Overview
--------
This repository contains a small, readable pipeline to:
1. Clean raw Backblaze-style SMART CSVs (`data/raw/`).
2. Validate, lazily generate a small set of engineered features, and save per-file processed CSVs to `data/processed/`.
3. Incrementally train a LightGBM model over processed files, saving checkpoints after each file.

Design decisions
----------------
- **One processed file per input** (good for debugging and incremental updates).
- **Lazy feature engineering**: only a selected set of SMART features are engineered (rolling mean/std, delta) to limit feature explosion.
- **CPU only** — no CUDA or GPU dependencies, so repo is clone-friendly.
- **Incremental training**: use LightGBM `init_model` approach; small number of boosting rounds per file and checkpoint after each file.

Quick start
-----------
1. Create your venv and install requirements:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   pip install -r requirements.txt
