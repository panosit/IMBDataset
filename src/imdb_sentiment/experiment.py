"""Leakage-safe nested-CV experiment runner with auditable artifacts."""
from __future__ import annotations
import json, platform, random, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from .data import file_sha256, load_data, preprocess
from .metrics import bootstrap_confidence_intervals, calibration_summary, compute_metrics, confusion


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)


def build_pipeline(name: str, seed: int) -> Pipeline:
    classifiers = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "linear_svc": CalibratedClassifierCV(LinearSVC(random_state=seed), cv=3),
        "multinomial_nb": MultinomialNB(),
    }
    if name not in classifiers: raise ValueError(f"Unknown model: {name}")
    return Pipeline([("tfidf", TfidfVectorizer()), ("classifier", classifiers[name])])


def parameter_grid(name: str, config: dict) -> dict:
    tfidf = config["tfidf"]
    grid = {"tfidf__max_features": tfidf["max_features"], "tfidf__min_df": tfidf["min_df"],
            "tfidf__ngram_range": [tuple(item) for item in tfidf["ngram_range"]],
            "tfidf__sublinear_tf": [tfidf.get("sublinear_tf", True)]}
    if name == "linear_svc": grid["classifier__estimator__C"] = config["models"][name]["C"]
    elif name == "logistic_regression": grid["classifier__C"] = config["models"][name]["C"]
    else: grid["classifier__alpha"] = config["models"][name]["alpha"]
    return grid


def score_predictions(model, texts):
    probabilities = model.predict_proba(texts)[:, 1]
    return model.predict(texts), probabilities


def nested_cv(texts, labels, name: str, config: dict) -> dict:
    outer = StratifiedKFold(config["outer_folds"], shuffle=True, random_state=config["seed"])
    records = []
    for fold, (train_idx, test_idx) in enumerate(outer.split(texts, labels), 1):
        inner = StratifiedKFold(config["inner_folds"], shuffle=True, random_state=config["seed"] + fold)
        search = GridSearchCV(build_pipeline(name, config["seed"]), parameter_grid(name, config),
                            scoring="roc_auc", cv=inner, n_jobs=-1, refit=True)
        search.fit(texts.iloc[train_idx], labels.iloc[train_idx])
        pred, score = score_predictions(search.best_estimator_, texts.iloc[test_idx])
        record = {"fold": fold, "best_params": search.best_params_, **compute_metrics(labels.iloc[test_idx], pred, score)}
        records.append(record)
    metric_names = [key for key in records[0] if key not in {"fold", "best_params"}]
    return {"folds": records, "mean": {key: float(np.mean([r[key] for r in records])) for key in metric_names},
            "std": {key: float(np.std([r[key] for r in records], ddof=1)) for key in metric_names}}


def environment_manifest(config: dict, data_path: Path) -> dict:
    try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError): commit = "unavailable"
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
            "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "dataset_sha256": file_sha256(data_path),
            "configuration": config}


def run_experiment(config: dict) -> Path:
    set_seed(config["seed"])
    root = Path(__file__).resolve().parents[2]
    data_path = root / config["data_path"]
    output = root / config["output_dir"]; output.mkdir(parents=True, exist_ok=True)
    df = preprocess(load_data(data_path))
    train, test = train_test_split(df, test_size=config["test_size"], random_state=config["seed"], stratify=df["label"])
    train, test = train.reset_index(names="source_index"), test.reset_index(names="source_index")
    train[["source_index", "label"]].to_csv(output / "train_split.csv", index=False)
    test[["source_index", "label"]].to_csv(output / "test_split.csv", index=False)
    manifest = environment_manifest(config, data_path); manifest.update({"raw_rows": len(df), "train_rows": len(train), "test_rows": len(test)})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {"models": {}, "test_set_protocol": "The test split was untouched during nested-CV tuning."}
    finalists = {}
    for name in config["models"]:
        started = time.perf_counter(); nested = nested_cv(train.clean_review, train.label, name, config)
        inner = StratifiedKFold(config["inner_folds"], shuffle=True, random_state=config["seed"])
        search = GridSearchCV(build_pipeline(name, config["seed"]), parameter_grid(name, config), scoring="roc_auc", cv=inner, n_jobs=-1, refit=True)
        search.fit(train.clean_review, train.label)
        pred, scores = score_predictions(search.best_estimator_, test.clean_review)
        finalists[name] = (pred, scores)
        metrics = compute_metrics(test.label, pred, scores)
        summary["models"][name] = {"nested_cv": nested, "final_best_params": search.best_params_,
          "test_metrics": metrics, "test_confidence_intervals": bootstrap_confidence_intervals(test.label, pred, scores, config["bootstrap_iterations"], config["seed"]),
          "calibration": calibration_summary(test.label, scores), "confusion_matrix": confusion(test.label, pred),
          "wall_time_seconds": round(time.perf_counter()-started, 3)}
        dump(search.best_estimator_, output / f"{name}.joblib")
    winner = max(summary["models"], key=lambda item: summary["models"][item]["nested_cv"]["mean"]["roc_auc"])
    summary["selected_model"] = winner
    summary["caveat"] = "Results are generated artifacts; do not report unexecuted configurations as findings."
    (output / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output
