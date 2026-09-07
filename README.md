# IMDB Sentiment Analysis

Binary text classification (positive/negative) on the IMDB movie reviews
dataset (`IMDB Dataset.csv`, 50,000 reviews, perfectly balanced 25k/25k,
418 duplicate rows removed). Includes a classic TF-IDF + linear model
pipeline and an in-progress transformer fine-tuning comparison.

## Classic ML pipeline (`main.py`)

1. **Load & clean** — drop duplicates, strip HTML tags (`<br />` noise
   present in ~29,200 rows), strip non-alphabetic characters, lowercase.
2. **EDA** — sentiment balance, review length distribution by sentiment.
3. **Vectorize** — TF-IDF, unigrams + bigrams, top 20,000 features,
   `min_df=5`.
4. **Models** — Logistic Regression and Multinomial Naive Bayes, on a
   stratified 80/20 train/test split.
5. **Save** — best model (`best_sentiment_model.joblib`) and the fitted
   vectorizer (`tfidf_vectorizer.joblib`).

### Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| **Logistic Regression** | **90.4%** | **0.966** |
| Multinomial Naive Bayes | 87.3% | 0.946 |

## Transformer fine-tuning (`bert_finetune.py`) — in progress

Fine-tunes a pretrained `distilbert-base-uncased` checkpoint (Hugging Face
`transformers` `Trainer`) on the same 50k reviews (raw text, not the
aggressively-cleaned text used for TF-IDF — BERT's tokenizer handles
punctuation and casing natively), for 1 epoch.

- Auto-detects GPU via `torch.cuda.is_available()`; falls back to CPU.
- **Status: script is complete and runnable, but has not yet been run to
  completion.** Full 50k/1-epoch training on CPU is expected to take on the
  order of hours. Final metrics and a head-to-head comparison against the
  90.4% / 0.966 baseline above will be added once the run finishes.

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

## Files

- `main.py` / `predict.py` — TF-IDF + Logistic Regression pipeline
- `bert_finetune.py` / `bert_predict.py` — DistilBERT fine-tuning pipeline
- `review.json` — input for both `predict.py` and `bert_predict.py`
- `IMDB Dataset.csv` — dataset
- `best_sentiment_model.joblib`, `tfidf_vectorizer.joblib` — classic ML
  artifacts
- `bert_sentiment_model/` — fine-tuned DistilBERT weights + tokenizer
  (created after running `bert_finetune.py`)
- `eda_*.png`, `roc_curves.png` — generated plots
