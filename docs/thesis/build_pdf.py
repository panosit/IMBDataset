"""Build a polished, evidence-first doctoral dissertation draft without external tools.

The output deliberately remains a draft: it contains structured places for only
verified literature and generated experimental artifacts.  It must not be
submitted as a completed empirical thesis before those items are supplied.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "thesis_detailed.pdf"  # Keep the externally requested filename.
TITLE = "Reliability, Robustness, and Reproducibility in IMDB Sentiment Classification"
SUBTITLE = "A Doctoral Dissertation Draft and Reproducible Research Record"
AUTHOR = "[Candidate name]"
INSTITUTION = "[University and department]"


def esc(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def wrap(text: str, width: int = 92) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) > width and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    return lines + ([line] if line else [])


def body_page(chapter: str, heading: str, paragraphs: list[str], page_number: int, draft: bool = True) -> bytes:
    lines = []
    for paragraph in paragraphs:
        lines.extend(wrap(paragraph))
        lines.append("")
    commands = ["q", "0.16 0.20 0.30 rg", "45 760 522 1 re f", "Q", "BT", "/F2 8 Tf", "48 744 Td",
                f"({esc(chapter.upper())}) Tj", "/F2 15 Tf", "0 -28 Td", f"({esc(heading)}) Tj", "/F1 10 Tf", "0 -24 Td"]
    for line in lines[:43]:
        commands.append(f"({esc(line)}) Tj")
        commands.append("0 -13 Td")
    footer = "DRAFT - replace all placeholders with verified sources and generated artifacts" if draft else TITLE
    commands += ["ET", "BT", "/F1 7 Tf", "48 30 Td", f"({esc(footer)}) Tj", "468 30 Td", f"({page_number}) Tj", "ET"]
    return "\n".join(commands).encode("latin-1", "replace")


def cover_page() -> bytes:
    commands = ["q", "0.16 0.20 0.30 rg", "0 0 612 792 re f", "Q", "BT", "/F2 12 Tf", "72 670 Td",
                "0.88 0.90 0.94 rg", "(DOCTORAL DISSERTATION DRAFT) Tj", "/F2 25 Tf", "0 -88 Td"]
    for line in wrap(TITLE, 38):
        commands += [f"({esc(line)}) Tj", "0 -32 Td"]
    commands += ["/F1 13 Tf", "0 -32 Td", f"({esc(SUBTITLE)}) Tj", "0 -110 Td", f"({esc(AUTHOR)}) Tj", "0 -24 Td", f"({esc(INSTITUTION)}) Tj", "0 -24 Td", "([Month Year]) Tj", "/F1 8 Tf", "0 -90 Td", "(This document is a submission-style template and research record.) Tj", "0 -13 Td", "(It is not a completed empirical thesis until results and citations are verified.) Tj", "ET"]
    return "\n".join(commands).encode("latin-1", "replace")


def front_page(heading: str, text: str, number: int) -> bytes:
    return body_page("Front matter", heading, [text], number)


def records(title: str, prompts: list[str], count: int, start_page: int) -> list[tuple[bytes, int]]:
    pages = []
    for index in range(1, count + 1):
        content = [f"This controlled record is {index} of {count} in the {title.lower()} appendix.",
                   "Use one record per verified source, experiment, figure, annotation batch, or review decision. Do not treat an unfilled record as evidence."]
        for prompt in prompts:
            content.append(f"{prompt} [Complete with evidence, identifier, and date.]")
        pages.append((body_page("Appendix", f"{title} {index:03d}", content, start_page + index - 1), start_page + index - 1))
    return pages


def make_pages() -> list[bytes]:
    pages = [cover_page()]
    number = 2
    front = [
        ("Declaration", "I declare that the final submitted version will distinguish original work from prior scholarship, disclose assistance and data sources, and contain only results reproducible from the archived code and artifacts. Replace this text with the institution-approved declaration before submission."),
        ("Abstract", "This dissertation draft studies reliability in binary movie-review sentiment classification. The completed study will compare sparse lexical and pretrained transformer systems using a locked split, nested validation, uncertainty intervals, calibration analysis, perturbation tests, structured error analysis, and external validation. This draft intentionally reports no fabricated empirical findings."),
        ("Acknowledgements", "[Insert acknowledgements, funding statement, institutional support, collaborator roles, and declarations of conflicts of interest.]"),
        ("Reader's note", "Square-bracketed material is a required evidence placeholder. Replace it only with an approved institutional field, a verified primary source, or an artifact produced by the versioned experiment protocol."),
        ("Table of contents", "Chapter 1 Introduction; Chapter 2 Literature Review; Chapter 3 Data, Ethics, and Governance; Chapter 4 Methodology; Chapter 5 Experimental Results; Chapter 6 Analysis; Chapter 7 Discussion; Chapter 8 Conclusion; Appendices A through F."),
        ("List of tables and figures", "[Populate automatically from the final typesetting source. Each table and figure must cite its manifest, configuration, split identifier, and generated artifact path.]"),
    ]
    for heading, text in front:
        pages.append(front_page(heading, text, number)); number += 1

    chapters = [
        ("Chapter 1", "Introduction", [
            ("Problem statement", "Sentiment classification is often summarized by a single benchmark score. That practice can conceal selection bias, uncertain estimates, poorly calibrated probabilities, and brittle behavior under modest input changes. This dissertation treats reliability as an empirical question rather than an assumed consequence of high accuracy."),
            ("Research gap", "The research gap concerns the joint evaluation of discrimination, calibration, robustness, explanation faithfulness, and transfer using one locked and auditable protocol. A model ranking is useful only when the assumptions behind the ranking are transparent and its uncertainty is reported."),
            ("Research questions", "RQ1 compares sparse lexical and pretrained transformer systems on discrimination, calibration, uncertainty, and compute cost. RQ2 examines controlled label-preserving perturbations. RQ3 characterizes errors using a pre-specified taxonomy. RQ4 assesses whether conclusions transfer to external review datasets."),
            ("Contributions", "The intended contributions are a leakage-safe experiment architecture, a registered evaluation protocol, a structured error-analysis process, and an artifact-oriented reproducibility record. Contributions must be revised after results are available; they are not claims of completed novelty."),
            ("Scope", "The study is limited to binary sentiment labels and review-style text. It does not claim to infer author identity, mental state, ideology, or universal preference. Results must not be generalized beyond the evaluated language, domain, time period, and label construction."),
            ("Thesis roadmap", "The remaining chapters review verified scholarship, document data governance, specify the protocol, present generated results, analyze failures, and discuss limitations. Appendices preserve the evidence needed for independent reproduction."),
        ]),
        ("Chapter 2", "Literature Review", [
            ("Review method", "Before submission, define databases, search queries, publication-date boundaries, inclusion criteria, exclusion criteria, and the screening process. Store exported bibliographic records and a source-selection log in the archival package."),
            ("Lexical sentiment modelling", "Review classical bag-of-words, TF-IDF, linear classification, word n-grams, and character n-grams. Explain their assumptions, computational characteristics, interpretability strengths, and vulnerabilities to lexical variation using verified primary sources."),
            ("Pretrained language models", "Review contextual encoders, fine-tuning procedures, tokenizer choices, truncation, training stability, and compute requirements. Identify which claims concern general language understanding and which have been demonstrated specifically for sentiment classification."),
            ("Evaluation leakage", "Review why preprocessing, feature extraction, model selection, threshold selection, calibration, and error-category design must be separated from held-out evaluation. Tie every protocol safeguard to a cited methodological rationale."),
            ("Calibration", "Review probability calibration, reliability diagrams, Brier score, expected calibration error definitions, and post-hoc calibration. State the selected ECE binning and weighting rule before observing results."),
            ("Robustness", "Review semantic-preserving perturbations, adversarial evaluation, distribution shift, and the distinction between sensitivity and robustness. Every transformation in the final study requires a validity argument and documented examples."),
            ("Interpretability", "Review global coefficient analysis, local explanations, attribution methods, and faithfulness evaluation. Do not equate an attractive explanation with a causal or faithful explanation without empirical validation."),
            ("Synthesis", "End the chapter with a table mapping each research question to the reviewed gap, selected measure, competing explanations, and evidence required to support a conclusion."),
        ]),
        ("Chapter 3", "Data, Ethics, and Governance", [
            ("Dataset provenance", "Record the canonical source, licence, acquisition date, source URL, version identifier, SHA-256 checksum, and any conditions on storage or quotation. The current CSV must be treated as immutable after verification."),
            ("Labels and population", "Describe how positive and negative labels were produced by the source dataset. Binary labels are an operational target, not a complete representation of sentiment, irony, mixed affect, or audience response."),
            ("Preprocessing", "Specify normalization separately for sparse and transformer models. Any learned representation, including vocabulary, inverse document frequency, calibration mapping, or normalization statistic, must be fitted only on training data."),
            ("Ethical limits", "Discuss copyright, quotation limits, privacy expectations, representational harms, and deployment risks. Movie-review classification is low stakes relative to clinical, employment, credit, or legal domains; no result should be generalized to those settings."),
            ("Threats from duplicates", "Describe exact and near-duplicate handling, train-test leakage risks, and checks for document overlap. Report counts before and after each deduplication decision and retain a reproducible record of removed rows."),
        ]),
        ("Chapter 4", "Methodology", [
            ("Design", "Create a deterministic stratified hold-out split, archive its row identifiers, and prohibit its use during model selection. Use nested cross-validation on training data for parameter selection and model comparison."),
            ("Baselines", "The implemented configuration includes word TF-IDF logistic regression, calibrated linear SVM, and multinomial Naive Bayes. The final thesis protocol should add a majority baseline, character n-grams, and transformer models under comparable reporting rules."),
            ("Metrics", "Declare ROC-AUC as the primary endpoint. Report PR-AUC, accuracy, balanced accuracy, precision, recall, F1, MCC, Brier score, calibration summaries, confidence intervals, timing, and resource use as secondary outcomes."),
            ("Uncertainty and comparisons", "Use stratified bootstrap confidence intervals for each test metric. For paired model claims, add a predeclared paired bootstrap difference interval and a paired error test. State any multiplicity adjustment before comparing more than two models."),
            ("Reproducibility", "Record configuration, Git revision, environment, data digest, seeds, split files, hardware, and generated artifact paths. For GPU experiments additionally record library versions, device type, CUDA/driver details, tokenizer revision, checkpoint revision, and deterministic settings."),
            ("Stopping rule", "Lock the experiment matrix and analysis plan before the final test evaluation. Any analysis introduced after observing held-out results must be labelled exploratory and evaluated on a fresh partition or external dataset where feasible."),
        ]),
        ("Chapter 5", "Experimental Results", [
            ("Results policy", "This chapter contains no invented values. Insert each table and figure only after executing its locked configuration. Every result caption must name its dataset digest, split identifier, model configuration, seed set, and artifact path."),
            ("Primary comparison table", "[Insert generated table: model, outer-CV mean and standard deviation, held-out ROC-AUC with 95% confidence interval, PR-AUC, F1, Brier score, runtime, and manifest path.]"),
            ("Calibration results", "[Insert reliability diagrams and weighted-ECE/Brier results. Explain binning, calibration data separation, and whether a post-hoc method was applied.]"),
            ("Paired comparisons", "[Insert paired bootstrap difference intervals and predeclared paired tests. State the estimand, decision threshold, and multiple-comparison handling.]"),
            ("Compute reporting", "[Insert hardware table, wall-clock time, peak memory or accelerator memory, energy metric if available, and failed-run policy.]"),
        ]),
        ("Chapter 6", "Robustness and Error Analysis", [
            ("Perturbation protocol", "[Insert pre-registered transformations: HTML formatting, punctuation, whitespace, truncation, and validated negation-sensitive transformations. Provide examples and label-preservation checks.]"),
            ("Robustness results", "[Insert per-transformation metric and confidence-change tables, then distinguish observed sensitivity from a supported semantic-robustness claim.]"),
            ("Error taxonomy", "[Insert codebook, sampling frame, annotator training procedure, independent labels, Cohen's kappa, reconciliation policy, and category frequencies with uncertainty.]"),
            ("Interpretability", "[Insert global lexical features and local explanation examples only with a stated faithfulness evaluation. Avoid treating explanation scores as causal evidence.]"),
            ("External validation", "[Insert external dataset provenance, adaptation policy, performance results, and limitations. Do not call an internal test split external validation.]"),
        ]),
        ("Chapter 7", "Discussion", [
            ("Answering RQ1", "[Interpret the model comparison using generated confidence intervals, calibration results, and compute measurements. Avoid claims that exceed the evaluated data.]"),
            ("Answering RQ2", "[Interpret robustness findings by transformation type, label-preservation evidence, and uncertainty. Explain whether observed differences matter practically.]"),
            ("Answering RQ3", "[Interpret error-taxonomy findings with annotation agreement and sampling limitations. Separate descriptive error patterns from causal explanation.]"),
            ("Answering RQ4", "[Interpret cross-dataset evidence with careful attention to label, genre, language, and temporal differences.]"),
            ("Validity threats", "Discuss internal validity, construct validity, conclusion validity, and external validity. Include tuning bias, seed variation, measurement choices, annotation ambiguity, data drift, and unresolved limitations."),
        ]),
        ("Chapter 8", "Conclusion", [
            ("Conclusion", "Summarize only the claims supported by the final generated evidence. State the protocol contributions, empirical conclusions, limitations, and carefully bounded future research directions."),
            ("Archival statement", "Archive the final source document, bibliography, configuration files, lockfile, manifests, split identifiers, result tables, figure sources, code revision, and data-access instructions in accordance with the dataset licence and institutional policy."),
        ]),
    ]
    for chapter, title, sections in chapters:
        for heading, text in sections:
            pages.append(body_page(chapter, heading, [text, "Evidence status: [replace this field with a verified citation, a generated artifact identifier, or an institution-approved statement before submission.]"], number)); number += 1

    appendix_specs = [
        ("Appendix A: Literature evidence record", ["Bibliographic identifier", "Primary claim extracted", "Study context and limitations", "How this source informs the thesis", "Verification completed by"], 18),
        ("Appendix B: Experiment record", ["Configuration file and digest", "Git revision and environment", "Data digest and split identifier", "Seed, hardware, and runtime", "Generated result artifact and review"], 28),
        ("Appendix C: Figure and table record", ["Figure or table identifier", "Generating command and source artifact", "Caption and intended claim", "Statistical uncertainty displayed", "Reviewer approval"], 14),
        ("Appendix D: Error annotation record", ["Sampling frame and item identifier", "Independent annotator labels", "Codebook version", "Reconciliation outcome", "Quotation/licence check"], 16),
        ("Appendix E: External-validation record", ["Dataset provenance and licence", "Schema/label mapping", "Adaptation policy", "Results artifact", "Transfer limitation"], 12),
        ("Appendix F: Reproducibility checklist", ["Command and expected artifact", "Configuration and seed", "Environment or hardware", "Checksum or revision", "Independent reproduction status"], 12),
    ]
    for title, prompts, count in appendix_specs:
        for page_bytes, page_number in records(title, prompts, count, number):
            pages.append(page_bytes)
        number += count
    return pages


def write_pdf(pages: list[bytes]) -> None:
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", None,
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman >>",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Bold >>"]
    page_ids = []
    for stream in pages:
        content_id, page_id = len(objects) + 1, len(objects) + 2
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {content_id} 0 R >>".encode())
        page_ids.append(page_id)
    objects[1] = (f"<< /Type /Pages /Count {len(page_ids)} /Kids [{' '.join(f'{page_id} 0 R' for page_id in page_ids)}] >>").encode()
    data, offsets = b"%PDF-1.4\n", [0]
    for object_id, obj in enumerate(objects, 1):
        offsets.append(len(data)); data += f"{object_id} 0 obj\n".encode() + obj + b"\nendobj\n"
    start = len(data); data += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f\n".encode()
    for offset in offsets[1:]: data += f"{offset:010d} 00000 n\n".encode()
    data += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()
    OUT.write_bytes(data)


if __name__ == "__main__":
    pages = make_pages()
    write_pdf(pages)
    print(f"Wrote {OUT} with {len(pages)} pages.")
