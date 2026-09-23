# Reproducible IMDB Sentiment Research

This repository is a **doctoral-research framework** for leakage-safe sentiment-classification experiments. It is not a completed empirical thesis: numerical findings must be generated from the versioned protocol and reported with their manifests, uncertainty intervals, and limitations.

## Research contribution

The project addresses comparative performance **and** reliability: discriminative performance, calibration, robustness, error taxonomy, interpretability, and external validity. The preregistered questions and ethical constraints are in [`docs/RESEARCH_PROTOCOL.md`](docs/RESEARCH_PROTOCOL.md); the formal manuscript draft is in [`docs/thesis/`](docs/thesis/).

## Layout

```text
src/imdb_sentiment/  reusable data, metrics, and experiment code
tests/               automated unit and regression tests
configs/             versioned experiment specifications
scripts/             reproducible command-line entry points
data/raw/            source data (immutable after checksum verification)
data/processed/      generated data only (ignored)
outputs/             generated run artifacts only (ignored)
docs/                protocol, reproducibility guide, and thesis draft
```

## Protocol safeguards

- Validates schema and removes exact duplicate rows before splitting.
- Uses a deterministic, stratified 80/20 hold-out partition and saves split row IDs.
- Fits TF–IDF **inside** every CV fold through an sklearn `Pipeline`.
- Uses nested cross-validation for selection and reports the final held-out evaluation separately.
- Emits ROC-AUC, PR-AUC, F1, balanced accuracy, MCC, Brier score, calibration bins, 95% stratified-bootstrap intervals, timing, environment, dataset SHA-256, and Git commit.
- Treats unexecuted configurations and document placeholders as non-results.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/verify_dataset.py
python scripts/run_experiment.py --config configs/baseline.json
python -m pytest
```

The default experiment can take substantial time because it performs nested cross-validation over three model families. Start with a reduced copy of `configs/baseline.json` for smoke testing; never use a smoke-test result in the thesis.

## Thesis readiness checklist

Before submission, complete the pending empirical work: lock hypotheses, execute all planned seeds/configurations, conduct paired statistical comparisons, run robustness and external-validation experiments, perform two-annotator error analysis, verify every literature citation, and replace all manuscript placeholders with generated tables/figures. See [`docs/RESEARCH_PROTOCOL.md`](docs/RESEARCH_PROTOCOL.md) and [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).
