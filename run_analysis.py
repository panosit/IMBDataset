"""
Real analysis script supporting the detailed thesis.

Produces analysis_results.json containing:
  - confusion_matrix (real, from the actual saved model on the actual test split)
  - error_analysis (real counts/patterns from actual misclassified test examples)
  - hyperparameter_sensitivity (real CV ROC-AUC from an actual one-at-a-time sweep,
    evaluated on the training set only via 3-fold cross-validation, exactly like
    the model-selection step in main.py)

This does NOT touch the test set for the hyperparameter sweep (avoids the same
leakage bug that main.py was fixed for). The confusion matrix / error analysis
use the test set exactly once, reusing the already-trained model artifacts.
"""

import json
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from main import load_data, preprocess, DATA_PATH, RANDOM_STATE, BASE_DIR

NEGATION_MARKERS = ["not ", "n't", " no ", " never ", " but ", " however ", " despite ", " although "]


def real_confusion_and_errors():
    df = load_data(DATA_PATH)
    df = preprocess(df)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_review"], df["label"], test_size=0.2,
        random_state=RANDOM_STATE, stratify=df["label"]
    )

    vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.joblib")
    model = joblib.load(BASE_DIR / "best_sentiment_model.joblib")

    X_test = vectorizer.transform(X_test_text)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    y_test_arr = y_test.reset_index(drop=True)
    y_pred_arr = list(y_pred)
    text_arr = X_test_text.reset_index(drop=True)

    fn_mask = [(y_test_arr[i] == 1 and y_pred_arr[i] == 0) for i in range(len(y_test_arr))]
    fp_mask = [(y_test_arr[i] == 0 and y_pred_arr[i] == 1) for i in range(len(y_test_arr))]
    correct_mask = [(y_test_arr[i] == y_pred_arr[i]) for i in range(len(y_test_arr))]

    def pct_with_negation(mask):
        subset = [text_arr[i] for i in range(len(text_arr)) if mask[i]]
        if not subset:
            return 0.0, 0
        hits = sum(1 for t in subset if any(m in (" " + t + " ") for m in NEGATION_MARKERS))
        return 100.0 * hits / len(subset), len(subset)

    def avg_len(mask):
        subset = [text_arr[i] for i in range(len(text_arr)) if mask[i]]
        if not subset:
            return 0.0
        return sum(len(t.split()) for t in subset) / len(subset)

    fn_neg_pct, fn_count = pct_with_negation(fn_mask)
    fp_neg_pct, fp_count = pct_with_negation(fp_mask)
    correct_neg_pct, correct_count = pct_with_negation(correct_mask)

    fn_examples = [text_arr[i] for i in range(len(text_arr)) if fn_mask[i]][:3]
    fp_examples = [text_arr[i] for i in range(len(text_arr)) if fp_mask[i]][:3]

    n_test = len(y_test_arr)
    n_pos = int((y_test_arr == 1).sum())
    n_neg = int((y_test_arr == 0).sum())

    error_analysis = {
        "n_test": n_test,
        "n_test_positive": n_pos,
        "n_test_negative": n_neg,
        "false_negatives": int(fn_count),
        "false_positives": int(fp_count),
        "false_negative_rate_of_positives_pct": round(100.0 * fn_count / n_pos, 2),
        "false_positive_rate_of_negatives_pct": round(100.0 * fp_count / n_neg, 2),
        "pct_false_negatives_containing_negation_marker": round(fn_neg_pct, 1),
        "pct_false_positives_containing_negation_marker": round(fp_neg_pct, 1),
        "pct_correct_containing_negation_marker": round(correct_neg_pct, 1),
        "avg_word_count_false_negatives": round(avg_len(fn_mask), 1),
        "avg_word_count_false_positives": round(avg_len(fp_mask), 1),
        "avg_word_count_correct": round(avg_len(correct_mask), 1),
        "sample_false_negative_excerpts": [t[:200] for t in fn_examples],
        "sample_false_positive_excerpts": [t[:200] for t in fp_examples],
    }

    confusion = {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    return confusion, error_analysis, (X_train_text, y_train)


def hyperparameter_sweep(X_train_text, y_train):
    """One-at-a-time sweep around the base config, 3-fold CV ROC-AUC on train only."""
    base = {"max_features": 20000, "min_df": 5, "ngram_range": (1, 2), "C": 1.0}
    grid = {
        "max_features": [10000, 20000, 30000],
        "min_df": [2, 5, 10],
        "ngram_range": [(1, 1), (1, 2), (1, 3)],
        "C": [0.1, 1.0, 5.0],
    }

    results = []
    for param, values in grid.items():
        for v in values:
            cfg = dict(base)
            cfg[param] = v
            vec = TfidfVectorizer(
                max_features=cfg["max_features"],
                min_df=cfg["min_df"],
                ngram_range=cfg["ngram_range"],
            )
            X_train = vec.fit_transform(X_train_text)
            clf = LogisticRegression(max_iter=1000, C=cfg["C"], random_state=RANDOM_STATE)
            scores = cross_val_score(clf, X_train, y_train, cv=3, scoring="roc_auc")
            results.append({
                "varied_param": param,
                "value": str(v),
                "mean_cv_roc_auc": round(float(scores.mean()), 4),
                "std_cv_roc_auc": round(float(scores.std()), 4),
            })
            print(f"{param}={v}: CV ROC-AUC={scores.mean():.4f} (+/- {scores.std():.4f})")

    return results


def main():
    confusion, error_analysis, (X_train_text, y_train) = real_confusion_and_errors()
    print("Confusion matrix:", confusion)
    print("Error analysis:", json.dumps(error_analysis, indent=2))

    sweep = hyperparameter_sweep(X_train_text, y_train)

    output = {
        "confusion_matrix": confusion,
        "error_analysis": error_analysis,
        "hyperparameter_sensitivity": sweep,
    }
    out_path = BASE_DIR / "analysis_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
