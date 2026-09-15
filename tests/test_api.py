"""
Integration tests for FastAPI endpoints using Starlette TestClient.
"""

import pytest
from starlette.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "capabilities" in data


def test_summarize_endpoint():
    payload = {
        "text": (
            "DocuWiz AI is an enterprise document intelligence tool leveraging BART and DistilBERT models. "
            "It automates legal and academic document review, reducing processing time by 60%. "
            "The platform supports PDF and DOCX analysis with integrated RAG."
        ),
        "style": "executive",
        "model_name": "fast-extractive"
    }
    response = client.post("/api/nlp/summarize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "key_takeaways" in data


def test_sentiment_endpoint():
    payload = {
        "text": "The company delivered outstanding profit growth, while contract penalties remained strictly limited.",
        "model_name": "fast-lexicon"
    }
    response = client.post("/api/nlp/sentiment", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "overall_label" in data
    assert "timeline" in data


def test_domain_analysis_endpoint():
    payload = {
        "text": "Customer shall indemnify Provider. In no event shall aggregate liability exceed total fees paid.",
        "domain": "legal"
    }
    response = client.post("/api/nlp/domain-analysis", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["domain"] == "legal"
    assert "risk_score" in data["data"]


def test_rag_flow_endpoints():
    index_payload = {
        "document_id": "test_doc_1",
        "full_text": "Party A shall deliver software. Party B may terminate within 30 days written notice.",
        "pages": [{"page_number": 1, "text": "Party A shall deliver software. Party B may terminate within 30 days written notice."}]
    }
    r_idx = client.post("/api/rag/index", json=index_payload)
    assert r_idx.status_code == 200
    assert r_idx.json()["status"] == "indexed"

    query_payload = {
        "document_id": "test_doc_1",
        "query": "What is the notice period for termination?",
        "provider": "local"
    }
    r_query = client.post("/api/rag/query", json=query_payload)
    assert r_query.status_code == 200
    q_data = r_query.json()
    assert "answer" in q_data
    assert len(q_data["citations"]) > 0


def test_benchmark_endpoint():
    payload = {"word_count": 2500, "doc_type": "legal"}
    response = client.post("/api/nlp/benchmark", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["processing_speedup_percent"] >= 55.0
