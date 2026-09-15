"""
Script to generate sample documents for DocuWiz AI testing and demonstration.
Creates:
1. sample_documents/legal_master_services_agreement.docx
2. sample_documents/academic_paper_rag_transformers.txt
3. sample_documents/enterprise_quarterly_report.txt
"""

import os
import docx
from pathlib import Path

os.makedirs("sample_documents", exist_ok=True)

# 1. Legal DOCX
doc = docx.Document()
doc.add_heading("MASTER SERVICES AGREEMENT", level=0)

doc.add_paragraph(
    "This Master Services Agreement (\"Agreement\") is entered into as of October 1, 2024, "
    "by and between Apex Enterprise Solutions Inc., a Delaware corporation (\"Provider\"), "
    "and Global Logistics Partners LLC, a California limited liability company (\"Customer\")."
)

doc.add_heading("1. Scope of Services & Deliverables", level=1)
doc.add_paragraph(
    "Provider shall provide enterprise document intelligence software, cloud deployment, and associated "
    "consulting services as described in Statement of Work No. 1. Provider will use commercially reasonable efforts "
    "to deliver the services within the designated timeline."
)

doc.add_heading("2. Payment Terms & Invoicing", level=1)
doc.add_paragraph(
    "Customer shall pay all invoices within thirty (30) days of receipt. In the event Customer fails to pay "
    "within said period, a late fee of 1.5% per month or the maximum rate permitted by law shall accrue on all "
    "outstanding balances."
)

doc.add_heading("3. Confidentiality & Non-Disclosure", level=1)
doc.add_paragraph(
    "Each party agrees that all code, inventions, business plans, technical processes, and financial terms disclosed "
    "under this Agreement constitute Confidential Information. The receiving party shall hold such Confidential Information "
    "in strict confidence using at least a reasonable standard of care and shall not disclose it to any third party "
    "without prior written consent."
)

doc.add_heading("4. Indemnification & Defense", level=1)
doc.add_paragraph(
    "Customer shall indemnify, defend, and hold harmless Provider, its officers, directors, and employees against any "
    "and all claims, liabilities, losses, damages, and reasonable attorney fees arising out of Customer breach of this "
    "Agreement, unauthorized use of Deliverables, or gross negligence."
)

doc.add_heading("5. Limitation of Liability", level=1)
doc.add_paragraph(
    "IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES "
    "ARISING OUT OF THIS AGREEMENT. PROVIDER TOTAL AGGREGATE LIABILITY UNDER THIS AGREEMENT SHALL IN NO EVENT EXCEED THE "
    "TOTAL FEES ACTUALLY PAID BY CUSTOMER IN THE PRECEDING TWELVE (12) MONTH PERIOD. NOTWITHSTANDING THE FOREGOING, "
    "CUSTOMER LIABILITIES REGARDING INDEMNITY OBLIGATIONS SHALL NOT BE LIMITED."
)

doc.add_heading("6. Term and Termination", level=1)
doc.add_paragraph(
    "This Agreement shall commence on the Effective Date and remain in effect for an initial period of two (2) years. "
    "This Agreement shall automatically renew for additional consecutive one-year terms unless either party provides "
    "written notice of non-renewal at least 60 days prior to expiration. Either party may terminate upon 10 days written "
    "notice in its sole discretion."
)

doc.add_heading("7. Governing Law & Dispute Resolution", level=1)
doc.add_paragraph(
    "This Agreement shall be governed by and construed in accordance with the laws of the State of New York, "
    "without regard to conflicts of law principles. Any dispute arising hereunder shall be submitted to binding arbitration "
    "in New York City under the rules of the American Arbitration Association."
)

docx_path = "sample_documents/legal_master_services_agreement.docx"
doc.save(docx_path)
print(f"Created: {docx_path}")

# 2. Academic Paper TXT
academic_text = """RETRIEVAL-AUGMENTED TRANSFORMERS FOR LONG-CONTEXT LEGAL AND ACADEMIC DOCUMENT INTELLIGENCE: AN EMPIRICAL STUDY

Dr. Elena Rostova, Prof. Marcus Vance, Dr. Sarah Lin
Department of Computer Science, Institute for Artificial Intelligence Research
Published: Journal of Computational Linguistics & Document Intelligence, 2024

ABSTRACT
Document intelligence systems face substantial difficulties when digesting long-form, dense text spanning complex legal contracts and academic literature. While traditional large language models (LLMs) struggle with token context limits and hallucination, retrieval-augmented generation (RAG) coupled with specialized hierarchical summarization offers a compelling paradigm. In this paper, we propose DocuWiz, a modular architecture integrating bidirectional auto-regressive transformers (BART) for map-reduce summarization, distilled transformer representations (DistilBERT) for multi-aspect sentiment and tone progression, and dense vector retrieval for source-grounded question answering. On empirical benchmarks comprising over 4,000 multi-page documents, DocuWiz reduces document processing latency by 62.4% while maintaining high factual fidelity with a ROUGE-2 score of 44.8 and an answer grounding accuracy of 94.2%.

1. INTRODUCTION & OBJECTIVES
Legal analysts and academic researchers routinely spend over 20 hours per week parsing dense manuscripts and agreements [Vaswani et al., 2017]. The cognitive load associated with cross-referencing multi-page clauses, detecting liabilities, and extracting key findings impedes organizational throughput. Our core research questions are:
(RQ1): How does hierarchical map-reduce chunking in BART compare to standard sliding-window extraction for long-context comprehension?
(RQ2): Can paragraph-level DistilBERT tone analysis reliably identify adverse liability clauses and academic limitations?
(RQ3): What is the optimal density-to-speed tradeoff in RAG pipelines across varied legal and scientific domains?

2. RELATED WORK & FOUNDATIONS
Previous work by Lewis et al. [2020] demonstrated the effectiveness of RAG for open-domain question answering. In the legal domain, Chalkidis et al. [2022] highlighted that clause detection requires domain-specific linguistic awareness due to syntactic nestedness. Furthermore, Sanh et al. [2019] showed that DistilBERT retains 97% of BERT language comprehension while executing 60% faster, making it exceptionally well suited for real-time document sentiment and tone assessment.

3. METHODOLOGY & ARCHITECTURE
We formulate our document intelligence pipeline into three decoupled modules:
3.1 Hierarchical Map-Reduce Summarizer: We deploy facebook/bart-large-cnn augmented with an adaptive chunking splitter. When document word count exceeds 600 tokens, the text is partitioned into overlapping windows (overlap = 50 tokens). Intermediate chunk representations are generated in parallel and subsequently merged through a recursive reduction step.
3.2 DistilBERT Tone Progression Engine: We track sentiment polarity and stylistic tone across discrete document sections. By scoring polarity between -1.0 (highly adverse/risk-laden) and +1.0 (affirmative/conclusive), we produce an emotional arc that visually identifies spikes in adversarial obligations or speculative findings.
3.3 Source-Grounded Vector Retrieval: We construct an in-memory sublinear term and semantic embedding matrix over document chunks. Chunks are tagged with persistent page references to guarantee provenance.

4. EXPERIMENTAL RESULTS & BENCHMARKS
We evaluate our system across three primary benchmarks: LegalBench (contracts), ArXivML (academic preprints), and SEC-10K (financial reports).
- Latency Reduction: Manual processing required an average of 42.5 minutes per 10-page document. DocuWiz reduced total time to 1.8 minutes, representing a 62.4% net efficiency improvement in end-user decision cycles.
- Summarization Quality: DocuWiz achieved ROUGE-1 of 48.2, ROUGE-2 of 44.8, and ROUGE-L of 46.1, outperforming baseline LexRank by +14.6 points.
- Question Answering Grounding: Grounding accuracy reached 94.2%, with citation precision at 96.8% [Brown et al., 2020].

5. LIMITATIONS & THREATS TO VALIDITY
Despite strong empirical results, our framework exhibits certain limitations:
First, optical character recognition (OCR) artifacts in scanned legacy PDFs can introduce tokenization noise, mildly impacting vector search precision.
Second, DistilBERT sentiment models pre-trained primarily on colloquial English occasionally misinterpret formal legal disclaimers as negative when they represent standard procedural defenses.
Third, hallucination risks persist when domain-specific jargon is absent from pre-training corpuses.

6. CONCLUSION & FUTURE DIRECTIONS
We presented DocuWiz, a document intelligence platform combining BART summarization, DistilBERT sentiment arcs, and source-grounded RAG. Empirical evaluations validate substantial productivity speedups and robust factual alignment. Future work will investigate multimodal table parsing and cross-lingual legal retrieval.

REFERENCES
[1] Vaswani et al., 2017. Attention is all you need. In NeurIPS.
[2] Lewis et al., 2020. Retrieval-augmented generation for knowledge-intensive NLP tasks. In NeurIPS.
[3] Chalkidis et al., 2022. LegalBench: A comprehensive legal NLP benchmark. In EMNLP.
[4] Sanh et al., 2019. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv:1910.01108.
[5] Brown et al., 2020. Language models are few-shot learners. In NeurIPS.
"""

academic_path = "sample_documents/academic_paper_rag_transformers.txt"
with open(academic_path, "w", encoding="utf-8") as f:
    f.write(academic_text.strip())
print(f"Created: {academic_path}")

# 3. Enterprise Report TXT
report_text = """AURORA TECHNOLOGIES INC. - Q3 2024 FINANCIAL & OPERATIONAL REPORT

Date: November 12, 2024
Prepared by: Office of Strategic Planning & Executive Operations

EXECUTIVE SUMMARY
Aurora Technologies demonstrated robust operational resilience in Q3 2024, achieving consolidated revenue of $184.2 million, representing an 18.5% year-over-year increase. Growth was primarily catalyzed by the enterprise adoption of our cloud data platform and autonomous AI workflow engines. Operating margin expanded by 340 basis points to 22.8%, driven by automated workflow efficiencies and disciplined infrastructure expenditure.

FINANCIAL HIGHLIGHTS
- Total Net Revenue: $184.2M (vs $155.4M in Q3 2023)
- Subscription Recurring Revenue (ARR): $142.6M (+24% YoY)
- Gross Profit Margin: 74.2%
- Net Operating Income: $42.0M
- Cash and Cash Equivalents: $310.5M

RISKS & REGULATORY FACTORS
The company continues to monitor supply chain stabilization regarding semiconductor accelerators. Foreign currency exchange fluctuations posed a 1.2% headwind on European sales. Cybersecurity posture remains top-tier with zero reported compliance breaches during the fiscal quarter.

STRATEGIC OUTLOOK FOR Q4
Management anticipates Q4 consolidated revenues in the range of $195M - $205M. Full-year capital expenditure will be allocated toward scaling distributed GPU clusters and expanding European data residency compliance.
"""

report_path = "sample_documents/enterprise_quarterly_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text.strip())
print(f"Created: {report_path}")
