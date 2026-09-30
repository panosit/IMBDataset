import numpy as np
from imdb_sentiment.metrics import bootstrap_confidence_intervals, calibration_summary, compute_metrics, confusion


def test_metrics_and_confusion_for_perfect_predictions():
    metrics = compute_metrics([0, 1, 0, 1], [0, 1, 0, 1], [0.01, 0.99, 0.1, 0.9])
    assert metrics["accuracy"] == metrics["roc_auc"] == metrics["pr_auc"] == 1.0
    assert confusion([0, 1, 0, 1], [0, 1, 0, 1]) == {"true_negative": 2, "false_positive": 0, "false_negative": 0, "true_positive": 2}


def test_bootstrap_intervals_are_bounded_and_deterministic():
    truth, pred, score = [0, 0, 1, 1], [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]
    first = bootstrap_confidence_intervals(truth, pred, score, iterations=20, seed=7)
    assert first == bootstrap_confidence_intervals(truth, pred, score, iterations=20, seed=7)
    assert 0 <= first["f1"]["low"] <= first["f1"]["high"] <= 1


def test_calibration_summary_has_requested_bin_bound():
    result = calibration_summary(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), bins=4)
    assert 0 <= result["expected_calibration_error"] <= 1
    assert len(result["bins"]) <= 4
