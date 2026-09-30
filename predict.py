"""Predict sentiment using the selected pipeline from ``outputs/baseline``.

Fill in ``review.json`` and run ``python predict.py`` after executing the
baseline experiment. The selected artifact is a complete sklearn pipeline, so
this script intentionally does not load a separate TF-IDF vectorizer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from imdb_sentiment.artifacts import BaselineArtifactError, load_selected_baseline
from imdb_sentiment.data import clean_text

BASELINE_OUTPUT_DIR = BASE_DIR / "outputs/baseline"
INPUT_JSON = BASE_DIR / "review.json"


def load_review(path: str | Path) -> str:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("review"), str) or not data["review"].strip():
        raise ValueError(f"'{path}' must contain a non-empty string 'review' field.")
    return data["review"]


def main() -> None:
    try:
        model_name, model, _ = load_selected_baseline(BASELINE_OUTPUT_DIR)
        review_text = load_review(INPUT_JSON)
    except (BaselineArtifactError, FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        print(f"Error: {error}")
        sys.exit(1)

    cleaned = clean_text(review_text)
    prediction = int(model.predict([cleaned])[0])
    probability = float(model.predict_proba([cleaned])[0][1])
    label = "POSITIVE" if prediction else "NEGATIVE"
    confidence = probability if prediction else 1 - probability
    print(f"\nInput: {INPUT_JSON}")
    print(f"Model selected by nested CV: {model_name}")
    print(f'Review: "{review_text}"')
    print(f"\nSentiment: {label} (confidence: {confidence:.1%})")


if __name__ == "__main__":
    main()
