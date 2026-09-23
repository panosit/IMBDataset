"""Dependency-free PDF builder for the formal thesis draft/evidence ledger."""
from pathlib import Path
from textwrap import wrap

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "thesis_detailed.pdf"
TITLE = "Reliability, Robustness, and Reproducibility in IMDB Sentiment Classification"
chapters = [
("Front matter", "Title page, declaration, abstract, acknowledgement, table of contents, and list of figures/tables. Replace institutional fields before submission."),
("Introduction and research questions", "State the gap, the four locked research questions, directional hypotheses, scope, claimed contribution, and thesis roadmap."),
("Literature review", "Synthesize verified primary sources on sentiment modelling, transformers, robustness, calibration, interpretability, leakage, and reproducibility. Do not retain unverified citations."),
("Data and ethics", "Record provenance, licence, retrieval date, digest, label definition, preprocessing, representational limits, risk assessment, and quotation policy."),
("Methodology", "Describe split generation, nested CV, model families, tuning budget, fixed seeds, metrics, bootstrap confidence intervals, paired tests, and stopping criteria."),
("Experiments", "Insert only executed results: manifests, per-seed results, confidence intervals, paired comparisons, compute measurements, calibration curves, and tables."),
("Analysis", "Report preregistered robustness transformations, manual error taxonomy, annotation procedure, agreement, explanation faithfulness, ablations, and external validation."),
("Discussion", "Answer each RQ, state limitations and validity threats, compare with verified literature, and identify bounded future work."),
("Reproducibility appendix", "Archive exact commands, commits, configuration files, environment details, split IDs, artifact map, and checksum table.")]

def esc(s): return s.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
def page(lines, number):
    content = ["BT", "/F1 9 Tf", "48 748 Td"]
    for i, line in enumerate(lines):
        if i: content.append("0 -13 Td")
        content.append(f"({esc(line)}) Tj")
    content.extend(["ET", "BT /F1 8 Tf 280 25 Td", f"({number}) Tj", "ET"])
    return "\n".join(content).encode('latin-1', 'replace')
def write_pdf(pages):
    objects=[b"<< /Type /Catalog /Pages 2 0 R >>", None, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    page_ids=[]
    for contents in pages:
        stream=page(contents[0], contents[1]); content_id=len(objects)+1; page_id=content_id+1
        objects.append(b"<< /Length %d >>\nstream\n"%len(stream)+stream+b"\nendstream")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>".encode())
        page_ids.append(page_id)
    objects[1]=(f"<< /Type /Pages /Count {len(page_ids)} /Kids [{' '.join(f'{x} 0 R' for x in page_ids)}] >>").encode()
    data=b"%PDF-1.4\n"; offsets=[0]
    for i,obj in enumerate(objects,1): offsets.append(len(data)); data+=f"{i} 0 obj\n".encode()+obj+b"\nendobj\n"
    start=len(data); data+=f"xref\n0 {len(objects)+1}\n0000000000 65535 f\n".encode()
    for off in offsets[1:]: data+=f"{off:010d} 00000 n\n".encode()
    data+=f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode(); OUT.write_bytes(data)

pages=[]
for n in range(1, 106):
    chapter, purpose = chapters[(n-1) % len(chapters)]
    lines=[TITLE, "FORMAL DRAFT AND EVIDENCE LEDGER - NOT A COMPLETED EMPIRICAL THESIS", "", f"{chapter} | evidence sheet {n:03d}", "", purpose, "", "Required evidence for this page:"]
    prompts=["1. Claim or research question:", "2. Configuration / source / citation identifier:", "3. Dataset digest, split identifier, and code commit:", "4. Method and predeclared decision rule:", "5. Generated result or verified literature note:", "6. Uncertainty, limitations, and threats to validity:", "7. Reviewer check and revision history:"]
    for prompt in prompts:
        lines += [prompt, "    " + "_"*82, "    " + "_"*82, ""]
    lines += ["Integrity rule: do not replace a missing result with a narrative assertion."]
    pages.append((lines,n))
write_pdf(pages)
print(f"Wrote {OUT} with {len(pages)} pages.")
