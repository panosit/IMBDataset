"""
IMDB Sentiment Analysis - DistilBERT Fine-Tuning
==================================================
Fine-tunes a pretrained DistilBERT model (distilbert-base-uncased) on the
full IMDB Dataset.csv for binary sentiment classification, and compares
its accuracy/ROC-AUC against the TF-IDF + Logistic Regression baseline
trained in main.py.

Runs on GPU automatically if available (torch.cuda.is_available()),
otherwise falls back to CPU. NOTE: fine-tuning on the full 50k reviews
on CPU is slow - expect a long run (likely multiple hours depending on
your machine). Progress is printed periodically by the Trainer.

Usage:
    python bert_finetune.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "IMDB Dataset.csv"
MODEL_NAME = "distilbert-base-uncased"
MODEL_OUT_DIR = BASE_DIR / "bert_sentiment_model"
RANDOM_STATE = 42
MAX_LENGTH = 256
NUM_EPOCHS = 1
BATCH_SIZE = 16

HTML_TAG_RE = re.compile(r"<.*?>")


def load_data(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates().reset_index(drop=True)
    df["review"] = df["review"].apply(lambda t: HTML_TAG_RE.sub(" ", t))
    df["label"] = (df["sentiment"] == "positive").astype(int)
    return df


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding="max_length", max_length=max_length
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs),
    }


def print_baseline_comparison(bert_metrics: dict) -> None:
    baseline_metrics_path = BASE_DIR / "baseline_metrics.joblib"
    if not baseline_metrics_path.exists():
        print("\n(No baseline metrics found - run main.py first to compare against TF-IDF + LogReg.)")
        return

    baseline_metrics = joblib.load(baseline_metrics_path)
    baseline_name = baseline_metrics.get("model", "Baseline")

    print(f"\n=== Comparison: DistilBERT vs {baseline_name} baseline ===")
    print(f"{'Metric':>10} | {'DistilBERT':>12} | {baseline_name[:18]:>18}")
    for metric, value in bert_metrics.items():
        base_val = baseline_metrics.get(metric)
        base_str = f"{base_val:.4f}" if base_val is not None else "n/a"
        print(f"{metric:>10} | {value:>12.4f} | {base_str:>18}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} reviews after de-duplication.")
    print("Splitting into train / validation / test (80% / 8% / 20% of the full set).")

    # Held-out test set is split off first and never used during training
    # or for epoch-level Trainer evaluation - only for the final metrics below.
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        df["review"], df["label"], test_size=0.2,
        random_state=RANDOM_STATE, stratify=df["label"]
    )
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels, test_size=0.1,
        random_state=RANDOM_STATE, stratify=train_val_labels
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    train_dataset = IMDBDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = IMDBDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    test_dataset = IMDBDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)

    training_args = TrainingArguments(
        output_dir=str(BASE_DIR / "bert_checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=50,
        report_to=[],
        use_cpu=(device == "cpu"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\n=== Final evaluation on held-out test set (never seen during training) ===")
    metrics = trainer.evaluate(eval_dataset=test_dataset)
    bert_metrics = {
        "accuracy": metrics["eval_accuracy"],
        "precision": metrics["eval_precision"],
        "recall": metrics["eval_recall"],
        "f1": metrics["eval_f1"],
        "roc_auc": metrics["eval_roc_auc"],
    }
    for k, v in bert_metrics.items():
        print(f"{k:>10}: {v:.4f}")

    print_baseline_comparison(bert_metrics)

    model.save_pretrained(MODEL_OUT_DIR)
    tokenizer.save_pretrained(MODEL_OUT_DIR)
    joblib.dump(bert_metrics, BASE_DIR / "bert_metrics.joblib")
    print(f"\nModel + tokenizer saved to {MODEL_OUT_DIR}")


if __name__ == "__main__":
    main()
