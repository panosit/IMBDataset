# Reproducibility guide

## Setup
```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/verify_dataset.py
python scripts/run_experiment.py --config configs/baseline.json
python -m pytest
```

## Artifact map
- `data/raw/imdb_dataset.csv`: immutable source data; verify checksum before analysis.
- `configs/baseline.json`: versioned experiment specification.
- `outputs/baseline/manifest.json`: Git commit, platform, packages, data digest, and configuration.
- `outputs/baseline/train_split.csv`, `test_split.csv`: row IDs for the fixed split.
- `outputs/baseline/results.json`: nested-CV and held-out metrics, confidence intervals, calibration, timing, and selected model.
- `outputs/baseline/*.joblib`: fitted models; generated and intentionally ignored.

Never edit a result artifact manually. Rerun the configuration and retain the manifest alongside every thesis table or figure.
