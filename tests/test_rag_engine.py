"""
Unit tests for RAG vector indexing, retrieval, and synthesis.
"""

import pytest
from core.rag_engine import build_rag_index, synthesize_local_answer, synthesize_llm_answer


def test_rag_indexing_and_search():
    doc_data = {
        "pages": [
            {"page_number": 1, "text": "The provider will deliver document intelligence software and consulting services."},
            {"page_number": 2, "text": "Either party may terminate this agreement upon thirty days written notice."},
            {"page_number": 3, "text": "The governing law shall be the State of California with binding arbitration."}
        ]
    }
    index = build_rag_index(doc_data, chunk_size=50, overlap=10)
    assert len(index.chunks) >= 3

    results = index.search("What is the termination notice period?", top_k=2)
    assert len(results) > 0
    assert results[0]["page_number"] == 2
    assert "terminate" in results[0]["text"].lower()


def test_rag_synthesis_local():
    doc_data = {
        "pages": [
            {"page_number": 1, "text": "The provider will deliver document intelligence software."},
            {"page_number": 2, "text": "Either party may terminate this agreement upon 30 days written notice to the other party."}
        ]
    }
    index = build_rag_index(doc_data, chunk_size=50, overlap=10)
    chunks = index.search("What are the termination terms?", top_k=2)

    synth = synthesize_local_answer("What are the termination terms?", chunks)
    assert "terminate" in synth["answer"].lower() or "Page 2" in synth["answer"]
    assert len(synth["citations"]) > 0
    assert synth["confidence"] in ["High", "Moderate"]
