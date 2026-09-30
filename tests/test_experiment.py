from imdb_sentiment.experiment import build_pipeline, parameter_grid


def test_pipeline_keeps_tfidf_inside_estimator():
    pipeline = build_pipeline("logistic_regression", seed=42)
    assert list(pipeline.named_steps) == ["tfidf", "classifier"]


def test_grid_prefixes_all_preprocessing_parameters():
    config = {"tfidf": {"max_features": [10], "min_df": [1], "ngram_range": [[1, 2]]}, "models": {"logistic_regression": {"C": [1.0]}}}
    assert set(parameter_grid("logistic_regression", config)) == {"tfidf__max_features", "tfidf__min_df", "tfidf__ngram_range", "tfidf__sublinear_tf", "classifier__C"}
