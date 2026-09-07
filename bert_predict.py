"""
IMDB Sentiment Predictor - DistilBERT
========================================
Fill in review.json with a movie review, then run:

    python bert_predict.py

It loads the fine-tuned DistilBERT model + tokenizer produced by
bert_finetune.py and prints whether the review is Positive or Negative,
with confidence.
"""

import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "bert_sentiment_model"
INPUT_JSON = BASE_DIR / "review.json"
MAX_LENGTH = 256

HTML_TAG_RE = re.compile(r"<.*?>")


def clean_text(text: str) -> str:
    return HTML_TAG_RE.sub(" ", text)


def load_review(path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "review" not in data or not data["review"].strip():
        raise ValueError(f"'{path}' must contain a non-empty 'review' field.")

    return data["review"]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not MODEL_DIR.exists():
        print(f"Model not found at {MODEL_DIR}. Run 'python bert_finetune.py' first to train and save it.")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    try:
        review_text = load_review(INPUT_JSON)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    cleaned = clean_text(review_text)
    inputs = tokenizer(
        cleaned, truncation=True, padding="max_length", max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]

    prediction = torch.argmax(probs).item()
    probability = probs[1].item()

    print(f"\nInput: {INPUT_JSON}")
    print(f'Review: "{review_text}"')
    print("\n=== Result ===")
    if prediction == 1:
        print(f"Sentiment: POSITIVE (confidence: {probability:.1%})")
    else:
        print(f"Sentiment: NEGATIVE (confidence: {1 - probability:.1%})")


if __name__ == "__main__":
    main()
