"""
Generate sample PDF for DocuWiz testing using reportlab.
"""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

def create_sample_pdf(pdf_path: str):
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    # Page 1
    c.setFont("Helvetica-Bold", 16)
    c.drawString(54, height - 54, "RETRIEVAL-AUGMENTED TRANSFORMERS FOR DOCUMENT INTELLIGENCE")
    
    c.setFont("Helvetica-Bold", 11)
    c.drawString(54, height - 75, "Dr. Elena Rostova, Prof. Marcus Vance, Dr. Sarah Lin (2024)")
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(54, height - 105, "ABSTRACT")
    c.setFont("Helvetica", 10)
    abstract_text = (
        "Document intelligence systems face substantial difficulties when digesting long-form, dense text "
        "spanning complex legal contracts and academic literature. In this paper, we propose DocuWiz, a modular "
        "architecture integrating bidirectional auto-regressive transformers (BART) for map-reduce summarization, "
        "distilled transformer representations (DistilBERT) for multi-aspect sentiment and tone progression, and dense "
        "vector retrieval for source-grounded question answering. On empirical benchmarks comprising over 4,000 multi-page "
        "documents, DocuWiz reduces document processing latency by 62.4% while maintaining high factual fidelity."
    )
    y = height - 125
    for line in [abstract_text[i:i+85] for i in range(0, len(abstract_text), 85)]:
        c.drawString(54, y, line)
        y -= 14

    c.setFont("Helvetica-Bold", 12)
    y -= 15
    c.drawString(54, y, "1. INTRODUCTION & OBJECTIVES")
    y -= 18
    c.setFont("Helvetica", 10)
    intro_text = (
        "Legal analysts and academic researchers routinely spend over 20 hours per week parsing dense manuscripts "
        "and agreements. The cognitive load associated with cross-referencing multi-page clauses, detecting liabilities, "
        "and extracting key findings impedes organizational throughput. Our research explores hierarchical chunking "
        "in BART models and section-level sentiment arcs."
    )
    for line in [intro_text[i:i+85] for i in range(0, len(intro_text), 85)]:
        c.drawString(54, y, line)
        y -= 14

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(width - 100, 30, "Page 1 of 2")
    c.showPage()

    # Page 2
    c.setFont("Helvetica-Bold", 12)
    c.drawString(54, height - 54, "2. EXPERIMENTAL RESULTS & BENCHMARKS")
    c.setFont("Helvetica", 10)
    res_text = (
        "We evaluate our system across three primary benchmarks: LegalBench, ArXivML, and SEC-10K. "
        "Empirical evaluations show a 62.4% net efficiency improvement over manual document reading. "
        "Summarization achieved ROUGE-1 of 48.2 and ROUGE-2 of 44.8, outperforming LexRank baselines. "
        "Question answering grounding accuracy reached 94.2% with citation precision at 96.8%."
    )
    y = height - 75
    for line in [res_text[i:i+85] for i in range(0, len(res_text), 85)]:
        c.drawString(54, y, line)
        y -= 14

    y -= 15
    c.setFont("Helvetica-Bold", 12)
    c.drawString(54, y, "3. LIMITATIONS & CONCLUSION")
    y -= 18
    c.setFont("Helvetica", 10)
    limitations_text = (
        "We note that noisy OCR scans can introduce tokenization noise into vector search. Furthermore, "
        "DistilBERT tone classifiers require domain context to avoid flagging customary legal liability disclaimers "
        "as hostile. In conclusion, DocuWiz provides an effective end-to-end framework for document intelligence."
    )
    for line in [limitations_text[i:i+85] for i in range(0, len(limitations_text), 85)]:
        c.drawString(54, y, line)
        y -= 14

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(width - 100, 30, "Page 2 of 2")
    c.save()
    print(f"Sample PDF created: {pdf_path}")

if __name__ == "__main__":
    create_sample_pdf("sample_documents/academic_paper_rag_transformers.pdf")
