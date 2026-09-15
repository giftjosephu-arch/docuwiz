"""
Unit tests for text extraction from PDF, DOCX, and TXT.
"""

import pytest
from pathlib import Path
from core.extractors import extract_document, calculate_metadata, clean_text_whitespace


def test_clean_text_whitespace():
    raw = "  Hello \r\n\r\n world \xa0 with    spaces  \n\n\n\n again "
    cleaned = clean_text_whitespace(raw)
    assert "Hello" in cleaned
    assert "again" in cleaned
    assert "\n\n\n" not in cleaned


def test_metadata_calculation():
    text = "Artificial intelligence and natural language processing revolutionize document intelligence and contract analysis."
    meta = calculate_metadata(text, pages_count=1)
    assert meta["word_count"] > 10
    assert meta["char_count"] > 50
    assert meta["lexical_diversity"] > 0.5
    assert meta["reading_time_minutes"] >= 1


def test_docx_extraction():
    p = Path("sample_documents/legal_master_services_agreement.docx")
    assert p.exists()
    res = extract_document(str(p), filename=p.name)
    assert res["file_type"] == "docx"
    assert "MASTER SERVICES AGREEMENT" in res["full_text"]
    assert res["metadata"]["word_count"] > 200
    assert len(res["pages"]) >= 1


def test_academic_txt_extraction():
    p = Path("sample_documents/academic_paper_rag_transformers.txt")
    assert p.exists()
    res = extract_document(str(p), filename=p.name)
    assert res["file_type"] == "txt"
    assert "TRANSFORMERS" in res["full_text"]
    assert res["metadata"]["word_count"] > 300
    assert len(res["pages"]) >= 1


def test_pdf_extraction():
    p = Path("sample_documents/academic_paper_rag_transformers.pdf")
    if p.exists():
        res = extract_document(str(p), filename=p.name)
        assert res["file_type"] == "pdf"
        assert res["metadata"]["word_count"] > 100
        assert len(res["pages"]) >= 1
