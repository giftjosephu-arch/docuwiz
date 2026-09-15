"""
Unit tests for NLP models: Summarizer, Sentiment, and Domain Intelligence.
"""

import pytest
from core.summarizer import summarize_document, extractive_summary_fallback
from core.sentiment import analyze_sentiment, detect_tone
from core.domain_intelligence import analyze_legal_document, analyze_academic_document, analyze_domain
from core.analytics import compute_productivity_benchmark, calculate_readability


def test_extractive_summary_fallback():
    long_text = (
        "Artificial intelligence is transforming document intelligence across multiple sectors. "
        "Natural language processing models like BART generate abstractive summaries of complex texts. "
        "DistilBERT offers high-speed sentiment analysis and risk detection. "
        "Retrieval-Augmented Generation allows users to chat with documents accurately. "
        "Empirical benchmarks demonstrate a 60% reduction in document processing time. "
        "Organizations adopt these automated tools to enhance legal and academic productivity."
    )
    res = extractive_summary_fallback(long_text, num_sentences=3)
    assert len(res) > 20
    assert "document" in res.lower()


def test_summarize_document():
    text = (
        "DocuWiz AI is an enterprise document intelligence platform. "
        "It leverages advanced NLP models like BART and DistilBERT for text extraction, summarization, and sentiment analysis. "
        "Integrated RAG and LLMs enable detailed insights and visualizations for PDF and DOCX analysis. "
        "Deployed as an interactive Streamlit application and FastAPI, reducing document processing time by 60%."
    )
    res = summarize_document(text, model_name="fast-extractive", style="executive", use_fallback_if_needed=True)
    assert "summary" in res
    assert "key_takeaways" in res
    assert res["stats"]["original_words"] > 10


def test_sentiment_analysis():
    sample_text = (
        "This agreement represents a highly favorable partnership with innovative solutions and compliant delivery.\n\n"
        "However, Customer shall indemnify Provider against all catastrophic liabilities, breaches, damages, and penalties."
    )
    res = analyze_sentiment(sample_text, model_name="fast-lexicon")
    assert "overall_label" in res
    assert "dominant_tone" in res
    assert len(res["timeline"]) >= 1
    assert "distribution" in res


def test_tone_detection():
    assert detect_tone("Client agrees to indemnify and hold harmless from all breaches and penalties.") == "Adversarial / Risk-Heavy"
    assert detect_tone("Results suggest that the hypothesis might possibly hold under certain assumptions.") == "Cautious / Speculative"
    assert detect_tone("Party shall strictly deliver deliverables on the specified date.") == "Assertive / Binding"


def test_legal_domain_intelligence():
    legal_text = (
        "Customer shall indemnify and hold harmless Provider against all claims. "
        "In no event shall aggregate liability exceed the fees paid. "
        "Notwithstanding the foregoing, Customer liability shall not be limited. "
        "Either party may terminate upon 10 days written notice in its sole discretion. "
        "Governed by the laws of the State of New York."
    )
    res = analyze_legal_document(legal_text)
    assert res["domain"] == "legal"
    assert res["risk_score"] > 30
    assert len(res["red_flags"]) >= 1
    assert "New York" in res["governing_jurisdiction"]


def test_academic_domain_intelligence():
    academic_text = (
        "ABSTRACT: We propose a novel architecture. "
        "METHODOLOGY: We formulate our model with transformer backbones. "
        "EXPERIMENTAL RESULTS: Our model achieved an accuracy of 94.2% and F1-score of 92.1. "
        "LIMITATIONS: Scanned OCR introduces noise into tokenization. "
        "References: [Vaswani et al., 2017]."
    )
    res = analyze_academic_document(academic_text)
    assert res["domain"] == "academic"
    assert res["rigor_score"] >= 50
    assert len(res["sections"]["Methodology & Architecture"]) >= 1


def test_productivity_benchmark():
    res = compute_productivity_benchmark(word_count=3000, doc_type="legal")
    assert res["processing_speedup_percent"] >= 55.0
    assert res["time_saved_minutes"] > 0
    assert "x" in res["productivity_multiplier"]
