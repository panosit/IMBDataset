# Reliability, Robustness, and Reproducibility in IMDB Sentiment Classification

## Status statement
This is a formal doctoral-thesis draft, not a claim that the listed experiments have been completed. Bracketed fields are evidence placeholders. Every numerical statement in the submission version must be regenerated from a committed configuration and linked manifest.

## Abstract
This thesis investigates reliable binary sentiment classification of movie reviews. Rather than treating benchmark accuracy as sufficient evidence, it evaluates discriminative performance, probability calibration, robustness under controlled input changes, explanation faithfulness, and external validity. The work compares leakage-safe sparse lexical baselines with pretrained transformer models using fixed data partitions, nested cross-validation, confidence intervals, and paired comparison procedures. The study contributes an auditable experimental protocol, a structured error-taxonomy method, and a reproducibility package that records data identity, code identity, configurations, environments, and split identifiers. Results are intentionally withheld until the protocol is executed.

## Chapter 1. Introduction and research questions
Automated sentiment classification is routinely evaluated by a single benchmark score. Such reporting is inadequate when estimates are selected after many modelling choices, probabilities are used for decisions, or inputs differ from their training form. This thesis asks whether performance rankings remain meaningful under these conditions. It frames reliability as a measurable combination of discrimination, calibration, robustness, and transparent failure analysis.

RQ1 asks how sparse lexical and pretrained transformer models compare on a locked evaluation set when selection occurs only within training data. RQ2 asks whether controlled, label-preserving perturbations change performance or confidence. RQ3 asks which pre-specified error categories explain failures and disagreements. RQ4 asks whether conclusions transfer to external review data. The contribution is a rigorous protocol and its empirically executed results, not novelty through benchmark reuse alone.

## Chapter 2. Literature review
The final version shall verify and synthesize scholarship on lexical sentiment classifiers, distributional and contextual language representations, pretrained transformers, calibration, robustness, explanation faithfulness, evaluation leakage, and reproducibility. It must distinguish explanatory claims from predictive claims; distinguish perturbation sensitivity from semantic robustness; and report inclusion/exclusion criteria for the review. Each claim must have a checked primary reference and an accurate citation.

Suggested anchor sources requiring verification before submission include Pang, Lee, and Vaithyanathan on sentiment classification; Devlin et al. on BERT; Sanh et al. on DistilBERT; Guo et al. on neural-network calibration; Ribeiro et al. on local explanations; and reproducibility guidance relevant to machine learning. This list is a starting bibliography, not a completed review.

## Chapter 3. Data, provenance, and ethics
The source data contain movie-review text and binary positive/negative labels. Exact duplicate rows are removed before partitioning. The submission must record canonical source, licence, retrieval date, SHA-256 digest, label construction, language coverage, and quotation policy. Binary sentiment labels are proxies for dataset annotator conventions; they do not capture mixed sentiment, sarcasm, intensity, demographic identity, or general human preference.

Preprocessing differs by model family and must be justified. Sparse models use explicit HTML removal and restricted normalization; tokenized transformers use their native tokenizer. All data transformations must occur after a split whenever fitted state is learned. Document risks include duplicate or near-duplicate leakage, benchmark overfitting, English-only coverage, historical bias, and misuse in domains with different stakes.

## Chapter 4. Methodology
A stratified fixed test partition is created with a recorded seed and row identifiers. Nested cross-validation is conducted on training data: inner folds tune parameters and outer folds estimate selection performance. TF-IDF vectorization belongs inside a pipeline, so vocabulary and document frequencies are learned only from each training fold. The final test partition is evaluated once per locked configuration.

The implemented baseline configuration currently compares word TF-IDF logistic regression, calibrated linear SVM, and multinomial Naive Bayes. The planned thesis suite additionally requires a majority reference, character TF-IDF logistic regression, and pretrained encoders before submission. All searches use a declared budget. The primary endpoint is ROC-AUC; secondary endpoints include PR-AUC, F1, balanced accuracy, precision, recall, MCC, Brier score, calibration error, latency, and resource use. Stratified bootstrap intervals and predeclared paired procedures quantify uncertainty.

## Chapter 5. Experiments
This chapter shall report only generated artifacts. For each configuration, include the configuration digest, code commit, data digest, hardware, seed list, elapsed time, peak memory if measured, outer-fold distribution, held-out metrics with 95% intervals, calibration curve, and paired model comparison. Do not select a method using the held-out test set. Any rerun after inspecting the test set constitutes a new exploratory analysis and must be labelled accordingly.

## Chapter 6. Analysis
Robustness tests are preregistered transformations applied after predictions are locked: HTML formatting changes, punctuation variation, whitespace noise, length truncation, and carefully validated negation-preserving edits. Report transformation validity checks and both score and confidence changes. Manual error analysis uses a random sample, a codebook, two annotators, agreement statistics, and a reconciliation rule. Explanation analyses report faithfulness checks, not merely attractive visualizations.

## Chapter 7. Discussion and threats to validity
The discussion will answer each research question with calibrated language, separating observed evidence from inference. Internal threats include selection bias, leakage, annotation ambiguity, and seed sensitivity. Construct threats include treating a binary label as complete sentiment. External threats include genre, language, platform, and time shift. Conclusion claims are limited to the evaluated datasets and protocols.

## Chapter 8. Reproducibility appendix
The repository supplies versioned configurations, scripts, tests, split identifiers, manifests, generated artifacts, and a data integrity check. The final appendix must archive command lines, lockfile, operating system, Python version, package versions, hardware, all random seeds, model identifiers, and artifact checksums.

## References to verify
Pang, B., Lee, L., and Vaithyanathan, S. (2002). Thumbs up? Sentiment classification using machine learning techniques.
Devlin, J. et al. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding.
Sanh, V. et al. (2019). DistilBERT, a distilled version of BERT.
Guo, C. et al. (2017). On calibration of modern neural networks.
Ribeiro, M. T. et al. (2016). Why should I trust you? Explaining the predictions of any classifier.
