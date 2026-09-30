"""Generate post-hoc baseline analysis from the saved experiment artifacts.

The script uses the exact row IDs written by ``run_experiment`` and loads the
selected fitted pipeline from ``outputs/baseline``. It never relies on the old
root-level model/vectorizer artifacts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from imdb_sentiment.artifacts import BaselineArtifactError, load_selected_baseline
from imdb_sentiment.data import load_data, preprocess
from imdb_sentiment.metrics import confusion

DATA_PATH = BASE_DIR / "data/raw/imdb_dataset.csv"
BASELINE_OUTPUT_DIR = BASE_DIR / "outputs/baseline"
RANDOM_STATE = 42
NEGATION_MARKERS = ["not ", "n't", " no ", " never ", " but ", " however ", " despite ", " although "]


def load_saved_split(df: pd.DataFrame, output_dir: Path = BASELINE_OUTPUT_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct the train/test dataframes from runner-emitted source indices."""
    try:
        train_ids = pd.read_csv(output_dir / "train_split.csv")["source_index"]
        test_ids = pd.read_csv(output_dir / "test_split.csv")["source_index"]
    except (FileNotFoundError, KeyError) as error:
        raise BaselineArtifactError(
            f"Split artifacts are missing or invalid in {output_dir}. Run the baseline experiment first."
        ) from error
    if train_ids.duplicated().any() or test_ids.duplicated().any() or set(train_ids).intersection(test_ids):
        raise BaselineArtifactError("Saved split identifiers are not disjoint.")
    if not set(train_ids).union(test_ids) <= set(df.index):
        raise BaselineArtifactError("Saved split identifiers do not match the current deduplicated dataset.")
    return df.loc[train_ids].copy(), df.loc[test_ids].copy()


def real_confusion_and_errors():
    df = preprocess(load_data(DATA_PATH))
    train, test = load_saved_split(df)
    model_name, model, _ = load_selected_baseline(BASELINE_OUTPUT_DIR)
    y_pred = model.predict(test["clean_review"])
    y_test = test["label"].reset_index(drop=True)
    text_arr = test["clean_review"].reset_index(drop=True)
    y_pred_arr = list(y_pred)

    fn_mask = [(y_test.iloc[i] == 1 and y_pred_arr[i] == 0) for i in range(len(y_test))]
    fp_mask = [(y_test.iloc[i] == 0 and y_pred_arr[i] == 1) for i in range(len(y_test))]
    correct_mask = [(y_test.iloc[i] == y_pred_arr[i]) for i in range(len(y_test))]

    def pct_with_negation(mask):
        subset = [text_arr.iloc[i] for i, selected in enumerate(mask) if selected]
        hits = sum(any(marker in f" {text} " for marker in NEGATION_MARKERS) for text in subset)
        return (100.0 * hits / len(subset), len(subset)) if subset else (0.0, 0)

    def avg_len(mask):
        subset = [text_arr.iloc[i] for i, selected in enumerate(mask) if selected]
        return sum(len(text.split()) for text in subset) / len(subset) if subset else 0.0

    fn_neg_pct, fn_count = pct_with_negation(fn_mask)
    fp_neg_pct, fp_count = pct_with_negation(fp_mask)
    correct_neg_pct, _ = pct_with_negation(correct_mask)
    n_pos, n_neg = int((y_test == 1).sum()), int((y_test == 0).sum())
    errors = {
        "selected_model": model_name, "n_test": len(y_test), "n_test_positive": n_pos, "n_test_negative": n_neg,
        "false_negatives": fn_count, "false_positives": fp_count,
        "false_negative_rate_of_positives_pct": round(100.0 * fn_count / n_pos, 2),
        "false_positive_rate_of_negatives_pct": round(100.0 * fp_count / n_neg, 2),
        "pct_false_negatives_containing_negation_marker": round(fn_neg_pct, 1),
        "pct_false_positives_containing_negation_marker": round(fp_neg_pct, 1),
        "pct_correct_containing_negation_marker": round(correct_neg_pct, 1),
        "avg_word_count_false_negatives": round(avg_len(fn_mask), 1),
        "avg_word_count_false_positives": round(avg_len(fp_mask), 1),
        "avg_word_count_correct": round(avg_len(correct_mask), 1),
        "sample_false_negative_excerpts": [text_arr.iloc[i][:200] for i, selected in enumerate(fn_mask) if selected][:3],
        "sample_false_positive_excerpts": [text_arr.iloc[i][:200] for i, selected in enumerate(fp_mask) if selected][:3],
    }
    return confusion(y_test, y_pred), errors, (train["clean_review"], train["label"])


def hyperparameter_sweep(train_texts, train_labels):
    """Exploratory, train-only 3-fold sensitivity sweep with pipeline-safe TF-IDF."""
    base = {"max_features": 20000, "min_df": 5, "ngram_range": (1, 2), "C": 1.0}
    grid = {"max_features": [10000, 20000, 30000], "min_df": [2, 5, 10],
            "ngram_range": [(1, 1), (1, 2), (1, 3)], "C": [0.1, 1.0, 5.0]}
    results = []
    for param, values in grid.items():
        for value in values:
            config = {**base, param: value}
            pipeline = Pipeline([("tfidf", TfidfVectorizer(max_features=config["max_features"], min_df=config["min_df"], ngram_range=config["ngram_range"])),
                                 ("classifier", LogisticRegression(max_iter=1000, C=config["C"], random_state=RANDOM_STATE))])
            scores = cross_val_score(pipeline, train_texts, train_labels, cv=3, scoring="roc_auc", n_jobs=-1)
            results.append({"varied_param": param, "value": str(value), "mean_cv_roc_auc": round(float(scores.mean()), 4), "std_cv_roc_auc": round(float(scores.std()), 4)})
    return results


def main() -> None:
    try:
        matrix, error_analysis, (train_texts, train_labels) = real_confusion_and_errors()
    except BaselineArtifactError as error:
        print(f"Error: {error}")
        sys.exit(1)
    output = {"confusion_matrix": matrix, "error_analysis": error_analysis,
              "hyperparameter_sensitivity": hyperparameter_sweep(train_texts, train_labels)}
    out_path = BASELINE_OUTPUT_DIR / "analysis_results.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
