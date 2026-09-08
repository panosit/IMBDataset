"""
IMDB Movie Review Sentiment Analysis
======================================
End-to-end binary text classification pipeline on IMDB Dataset.csv.

Steps:
1. Load data
2. Clean text (strip HTML tags) & drop duplicates
3. Encode target (positive=1, negative=0)
4. Train/test split
5. TF-IDF vectorize (fit on train only)
6. Select between Logistic Regression and Multinomial Naive Bayes via
   cross-validation on the training set only
7. Evaluate the selected model once on the held-out test set
8. Save the best model + vectorizer
"""

import json
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "IMDB Dataset.csv"
RANDOM_STATE = 42

HTML_TAG_RE = re.compile(r"<.*?>")


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def clean_text(text: str) -> str:
    text = HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_review"] = df["review"].apply(clean_text)
    df["label"] = (df["sentiment"] == "positive").astype(int)
    return df


def run_eda(df: pd.DataFrame) -> None:
    print("\n=== Dataset shape ===")
    print(df.shape)

    print("\n=== Sentiment balance ===")
    print(df["sentiment"].value_counts())

    print("\n=== Missing values ===")
    print(df.isna().sum().sum(), "total missing values")

    df["review_len"] = df["clean_review"].str.split().str.len()

    plt.figure(figsize=(5, 4))
    sns.countplot(x="sentiment", data=df)
    plt.title("Sentiment Distribution")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "eda_sentiment_balance.png")
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.histplot(data=df, x="review_len", hue="sentiment", kde=True, element="step")
    plt.title("Review Length (word count) by Sentiment")
    plt.xlim(0, 800)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "eda_review_length.png")
    plt.close()

    print("\nEDA plots saved: eda_sentiment_balance.png, eda_review_length.png")


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k:>10}: {v:.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))

    return metrics, y_proba, y_pred


def plot_confusion_matrix(name: str, y_test, y_pred) -> None:
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["negative", "positive"], yticklabels=["negative", "positive"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "confusion_matrix.png")
    plt.close()
    print("\nSaved confusion_matrix.png")


def plot_roc_curves(results: dict, y_test) -> None:
    plt.figure(figsize=(6, 6))
    for name, y_proba in results.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "roc_curves.png")
    plt.close()
    print("\nSaved roc_curves.png")


def main():
    df = load_data(DATA_PATH)
    df = preprocess(df)
    run_eda(df)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_review"], df["label"], test_size=0.2,
        random_state=RANDOM_STATE, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=5)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # Model selection via cross-validation on the training set only.
    # The test set is never touched until the final, single evaluation below.
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Multinomial Naive Bayes": MultinomialNB(),
    }

    print("\n=== Cross-validated model selection (train set only) ===")
    cv_scores = {}
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
        cv_scores[name] = scores.mean()
        print(f"{name}: mean CV ROC-AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")

    best_name = max(cv_scores, key=cv_scores.get)
    print(f"\nSelected model based on CV: {best_name}")

    best_model = candidates[best_name]
    best_model.fit(X_train, y_train)
    best_metrics, best_proba, best_pred = evaluate_model(best_name, best_model, X_test, y_test)

    plot_roc_curves({best_name: best_proba}, y_test)
    plot_confusion_matrix(best_name, y_test, best_pred)

    joblib.dump(best_model, BASE_DIR / "best_sentiment_model.joblib")
    joblib.dump(vectorizer, BASE_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(best_metrics, BASE_DIR / "baseline_metrics.joblib")

    results = {
        "selected_model": best_name,
        "cv_scores": cv_scores,
        "test_metrics": {k: v for k, v in best_metrics.items() if k != "model"},
    }
    with open(BASE_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nBest model ('{best_name}') saved to best_sentiment_model.joblib")
    print("Vectorizer saved to tfidf_vectorizer.joblib")
    print("Metrics saved to baseline_metrics.joblib")
    print("Results summary saved to results.json")


if __name__ == "__main__":
    main()
