# Registered research protocol

## Research gap and questions
This project studies whether high-performing sentiment classifiers remain reliable when their input form changes, whether their probability estimates are calibrated, and which error categories explain disagreement. The contribution is a reproducible, leakage-safe protocol rather than a claim that a new model has already achieved a particular result.

- **RQ1:** How do sparse lexical and pretrained transformer models compare in discriminative performance, calibration, uncertainty, and compute cost on a fixed evaluation split?
- **RQ2:** How robust are those models to preregistered, label-preserving perturbations (HTML noise, punctuation variation, negation-preserving edits, and review truncation)?
- **RQ3:** Which pre-specified linguistic/error categories account for model failures, and do those rates differ by model?
- **RQ4:** Does the relative ranking transfer to held-out external review datasets?

The directional hypotheses must be finalized before executing the final analysis: transformer models are expected to improve discriminative performance under lexical variation; calibration and robustness are empirical questions rather than assumptions.

## Fixed protocol
The dataset is deduplicated before a stratified fixed 80/20 train/test partition using seed 42. All vocabulary learning and hyperparameter selection occurs inside training folds. Nested CV uses 5 outer folds and 3 inner folds. The final test split is used once per locked configuration. The exact rows are stored as `train_split.csv` and `test_split.csv`.

Primary metric: ROC-AUC. Secondary metrics: PR-AUC, accuracy, balanced accuracy, precision, recall, F1, MCC, Brier score, calibration bins, wall-clock time, and 95% stratified bootstrap intervals. Pairwise comparisons use paired predictions and a pre-declared test/interval procedure before results are inspected.

## Ethics and data governance
The IMDB CSV is treated as an externally sourced research dataset. Before submission, record its canonical provider, licence, acquisition date, checksum, and permitted quotation conditions. The labels represent binary review sentiment, not author identity, mental state, or a general measure of human opinion. The dataset is balanced and English-language; claims must not be generalized to other languages, genres, populations, or production deployment. Do not publish full review text unless the dataset licence permits it.

## Error taxonomy and annotation
Randomly sample errors per model after locking predictions. Two independent annotators apply a codebook covering mixed sentiment, negation/scope, sarcasm/irony, plot-summary dominance, world knowledge, annotation ambiguity, spelling/noise, and truncation. Report sampling frame, annotator training, disagreements, reconciliation policy, and Cohen's kappa. The simple marker analysis is exploratory only.
