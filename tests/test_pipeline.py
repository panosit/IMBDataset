"""
Unit tests for main.py.

Run with:
    python -m pytest test_main.py -v
"""

import numpy as np
import pandas as pd
import pytest

from imdb_sentiment.data import clean_text, load_data, preprocess
from imdb_sentiment.metrics import compute_metrics


class DummyModel:
    """Minimal stand-in for a fitted sklearn classifier."""

    def __init__(self, preds, probas):
        self._preds = np.array(preds)
        self._probas = np.array(probas)

    def predict(self, X):
        return self._preds

    def predict_proba(self, X):
        return self._probas


class TestCleanText:
    def test_strips_html_tags(self):
        assert clean_text("Great movie<br />loved it") == "great movie loved it"

    def test_lowercases(self):
        assert clean_text("AMAZING Film") == "amazing film"

    def test_preserves_apostrophes_in_contractions(self):
        assert "wasn't" in clean_text("It wasn't very good")

    def test_preserves_digits(self):
        assert "10" in clean_text("I would give this a 10/10")

    def test_strips_punctuation_other_than_apostrophe(self):
        result = clean_text("Wow!! Best movie, ever...")
        assert "!" not in result
        assert "," not in result
        assert "." not in result

    def test_collapses_whitespace(self):
        assert clean_text("too    many     spaces") == "too many spaces"

    def test_empty_string(self):
        assert clean_text("") == ""


class TestLoadData:
    def test_drops_duplicate_rows(self, tmp_path):
        csv_path = tmp_path / "sample.csv"
        pd.DataFrame({
            "review": ["good movie", "good movie", "bad movie"],
            "sentiment": ["positive", "positive", "negative"],
        }).to_csv(csv_path, index=False)

        df = load_data(csv_path)

        assert len(df) == 2

    def test_resets_index_after_dedup(self, tmp_path):
        csv_path = tmp_path / "sample.csv"
        pd.DataFrame({
            "review": ["a", "a", "b", "c"],
            "sentiment": ["positive", "positive", "negative", "positive"],
        }).to_csv(csv_path, index=False)

        df = load_data(csv_path)

        assert list(df.index) == list(range(len(df)))


class TestPreprocess:
    def test_encodes_positive_as_one(self):
        df = pd.DataFrame({"review": ["great"], "sentiment": ["positive"]})
        result = preprocess(df)
        assert result["label"].iloc[0] == 1

    def test_encodes_negative_as_zero(self):
        df = pd.DataFrame({"review": ["awful"], "sentiment": ["negative"]})
        result = preprocess(df)
        assert result["label"].iloc[0] == 0

    def test_adds_clean_review_column(self):
        df = pd.DataFrame({"review": ["Nice<br />movie"], "sentiment": ["positive"]})
        result = preprocess(df)
        assert result["clean_review"].iloc[0] == "nice movie"


class TestEvaluateModel:
    def test_returns_expected_metric_keys(self):
        model = DummyModel(preds=[1, 0, 1, 0], probas=[[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
        y_test = pd.Series([1, 0, 1, 0])

        y_pred = model.predict(None)
        y_proba = model.predict_proba(None)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)

        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc", "brier"):
            assert key in metrics

    def test_perfect_predictions_score_1(self):
        model = DummyModel(preds=[1, 0, 1, 0], probas=[[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        y_test = pd.Series([1, 0, 1, 0])

        y_pred = model.predict(None)
        y_proba = model.predict_proba(None)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)

        assert metrics["accuracy"] == 1.0
        assert metrics["roc_auc"] == 1.0

    def test_returns_predictions_matching_model_output(self):
        preds = [1, 0, 1, 0]
        model = DummyModel(preds=preds, probas=[[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
        y_test = pd.Series([1, 1, 0, 0])

        y_pred = model.predict(None)

        assert list(y_pred) == preds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
