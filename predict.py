"""
IMDB Sentiment Predictor
==========================
Fill in review.json with a movie review, then run:

    python predict.py

It loads the trained model + TF-IDF vectorizer produced by main.py
and prints whether the review is Positive or Negative, with probability.
"""

import json
import re
import sys
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_sentiment_model.joblib"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.joblib"
INPUT_JSON = BASE_DIR / "review.json"

HTML_TAG_RE = re.compile(r"<.*?>")


def clean_text(text: str) -> str:
    text = HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def load_review(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "review" not in data or not data["review"].strip():
        raise ValueError(f"'{path}' must contain a non-empty 'review' field.")

    return data["review"]


def main():
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    except FileNotFoundError:
        print("Model artifacts not found. Run 'python main.py' first to train and save the model.")
        sys.exit(1)

    try:
        review_text = load_review(INPUT_JSON)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    cleaned = clean_text(review_text)
    X = vectorizer.transform([cleaned])

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    print(f"\nInput: {INPUT_JSON}")
    print(f'Review: "{review_text}"')
    print("\n=== Result ===")
    if prediction == 1:
        print(f"Sentiment: POSITIVE (confidence: {probability:.1%})")
    else:
        print(f"Sentiment: NEGATIVE (confidence: {1 - probability:.1%})")


if __name__ == "__main__":
    main()
