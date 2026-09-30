"""Loading helpers for artifacts emitted by the baseline experiment runner."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


class BaselineArtifactError(RuntimeError):
    """Raised when a completed baseline run cannot be loaded safely."""


def load_selected_baseline(output_dir: str | Path) -> tuple[str, Any, dict]:
    """Load the selected fitted pipeline and its validated results summary.

    The experiment runner writes one fitted sklearn pipeline per candidate and a
    ``results.json`` file identifying the candidate selected from nested CV.
    Loading through this function prevents downstream tools from relying on the
    obsolete root-level model and vectorizer files.
    """
    output_dir = Path(output_dir)
    results_path = output_dir / "results.json"
    if not results_path.exists():
        raise BaselineArtifactError(
            f"Baseline results not found at {results_path}. "
            "Run 'python scripts/run_experiment.py --config configs/baseline.json' first."
        )
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
        selected_name = results["selected_model"]
        if selected_name not in results["models"]:
            raise KeyError("selected model is not present in results['models']")
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise BaselineArtifactError(f"Invalid baseline results file: {results_path}") from error

    model_path = output_dir / f"{selected_name}.joblib"
    if not model_path.exists():
        raise BaselineArtifactError(f"Selected model artifact not found at {model_path}.")
    return selected_name, joblib.load(model_path), results
