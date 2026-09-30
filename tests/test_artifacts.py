import json

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from imdb_sentiment.artifacts import BaselineArtifactError, load_selected_baseline


def test_load_selected_baseline_loads_named_pipeline(tmp_path):
    model = Pipeline([("tfidf", TfidfVectorizer()), ("classifier", LogisticRegression())])
    model.fit(["excellent film", "awful film"], [1, 0])
    joblib.dump(model, tmp_path / "logistic_regression.joblib")
    (tmp_path / "results.json").write_text(json.dumps({
        "selected_model": "logistic_regression", "models": {"logistic_regression": {}},
    }), encoding="utf-8")

    name, loaded, results = load_selected_baseline(tmp_path)

    assert name == "logistic_regression"
    assert results["selected_model"] == name
    assert loaded.predict(["excellent"])[0] == 1


def test_load_selected_baseline_requires_results(tmp_path):
    with pytest.raises(BaselineArtifactError, match="Run 'python scripts/run_experiment.py"):
        load_selected_baseline(tmp_path)
