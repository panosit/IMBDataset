# IMDB Sentiment Analysis

Binary text classification (positive/negative) on the IMDB movie reviews
dataset (`IMDB Dataset.csv`, 50,000 reviews, perfectly balanced 25k/25k,
418 duplicate rows removed). Includes a classic TF-IDF + linear model
pipeline and an in-progress transformer fine-tuning comparison.

## Classic ML pipeline (`main.py`)

1. **Load & clean** — drop duplicates, strip HTML tags (`<br />` noise
   present in ~29,200 rows), strip characters other than letters/digits/
   apostrophes (preserves contractions like "wasn't" and ratings like
   "10/10"), lowercase.
2. **EDA** — sentiment balance, review length distribution by sentiment.
3. **Train/test split** — stratified 80/20 split. The test set is held
   out completely until the final evaluation step; it is never used for
   model selection.
4. **Vectorize** — TF-IDF, unigrams + bigrams, top 20,000 features,
   `min_df=5`, fitted on the training split only.
5. **Model selection** — Logistic Regression and Multinomial Naive Bayes
   are compared via 5-fold cross-validation (ROC-AUC) on the training
   set only. The winner is refit on the full training set.
6. **Final evaluation** — the selected model is evaluated exactly once
   on the held-out test set.
7. **Save** — best model (`best_sentiment_model.joblib`), fitted
   vectorizer (`tfidf_vectorizer.joblib`), test-set metrics
   (`baseline_metrics.joblib`, `results.json`).

### Results

Cross-validated model selection (training set, 5-fold, mean ROC-AUC):

| Model | CV ROC-AUC |
|---|---|
| **Logistic Regression** | **0.964** |
| Multinomial Naive Bayes | 0.946 |

Final held-out test-set performance of the selected model (Logistic
Regression):

| Metric | Score |
|---|---|
| Accuracy | 90.5% |
| Precision | 89.5% |
| Recall | 91.8% |
| F1 | 90.6% |
| ROC-AUC | 0.966 |

Exact numbers are written to `results.json` on every run of `main.py`.

## Transformer fine-tuning (`bert_finetune.py`) — in progress

Fine-tunes a pretrained `distilbert-base-uncased` checkpoint (Hugging Face
`transformers` `Trainer`) on the same 50k reviews (raw text, not the
cleaned text used for TF-IDF — BERT's tokenizer handles punctuation and
casing natively), for 1 epoch.

- Data is split 80/8/20 into train/validation/test. The validation split
  is used for epoch-level `Trainer` checkpointing; the test split is
  evaluated exactly once, after training finishes.
- Auto-detects GPU via `torch.cuda.is_available()`; falls back to CPU.
- Prints a comparison against the classic baseline by loading the real
  metrics `main.py` saved to `baseline_metrics.joblib` (no hardcoded
  numbers).
- **Status: script is complete and runnable, but has not yet been run to
  completion.** Full 50k/1-epoch training on CPU is expected to take on
  the order of hours.

## Usage

Train the classic baseline:

```
python main.py
```

Predict with the classic model — edit `review.json` with a review, then:

```
python predict.py
```

Fine-tune DistilBERT (long-running on CPU):

```
python bert_finetune.py
```

Predict with the fine-tuned transformer once trained:

```
python bert_predict.py
```

Run the unit tests:

```
python -m pytest test_main.py -v
```

## Files

- `main.py` / `predict.py` — TF-IDF + Logistic Regression pipeline
- `test_main.py` — unit tests for `main.py` (`clean_text`, `load_data`,
  `preprocess`, `evaluate_model`)
- `bert_finetune.py` / `bert_predict.py` — DistilBERT fine-tuning pipeline
- `review.json` — input for both `predict.py` and `bert_predict.py`
- `IMDB Dataset.csv` — dataset
- `best_sentiment_model.joblib`, `tfidf_vectorizer.joblib`,
  `baseline_metrics.joblib` — classic ML artifacts
- `results.json` — human-readable summary of the latest `main.py` run
  (CV scores + final test metrics)
- `bert_sentiment_model/` — fine-tuned DistilBERT weights + tokenizer
  (created after running `bert_finetune.py`)
- `eda_*.png`, `roc_curves.png`, `confusion_matrix.png` — generated plots
