"""Evaluation, uncertainty, and calibration utilities."""
from __future__ import annotations
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, f1_score, matthews_corrcoef, precision_score, recall_score,
    roc_auc_score, confusion_matrix)


def compute_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_score)),
    }


def bootstrap_confidence_intervals(y_true, y_pred, y_score, iterations: int, seed: int) -> dict:
    """Stratified bootstrap 95% percentile intervals for all scalar metrics."""
    y_true, y_pred, y_score = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_score)
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(y_true == label) for label in (0, 1)]
    samples: dict[str, list[float]] = {}
    for _ in range(iterations):
        indices = np.concatenate([rng.choice(index, len(index), replace=True) for index in classes])
        values = compute_metrics(y_true[indices], y_pred[indices], y_score[indices])
        for name, value in values.items(): samples.setdefault(name, []).append(value)
    return {name: {"low": float(np.quantile(values, .025)), "high": float(np.quantile(values, .975))}
            for name, values in samples.items()}


def calibration_summary(y_true, y_score, bins: int = 10) -> dict:
    observed, predicted = calibration_curve(y_true, y_score, n_bins=bins, strategy="uniform")
    ece = float(np.mean(np.abs(observed - predicted)))
    return {"expected_calibration_error": ece, "bins": [
        {"mean_predicted_probability": float(p), "fraction_positive": float(o)}
        for p, o in zip(predicted, observed)]}


def confusion(y_true, y_pred) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)}
