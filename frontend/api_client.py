"""
DocuWiz AI - Frontend API Client & Local Fallback Router
Enables Streamlit to run either connected to the FastAPI backend or directly with the local core engine.
"""

import io
import requests
from typing import Dict, Any, Optional

# Core imports for local direct mode
from core.extractors import extract_document
from core.summarizer import summarize_document
from core.sentiment import analyze_sentiment
from core.domain_intelligence import analyze_domain
from core.analytics import compute_productivity_benchmark, extract_key_concepts, calculate_readability
from core.rag_engine import build_rag_index, synthesize_llm_answer, RAGIndex


class DocuWizClient:
    def __init__(self, backend_url: str = "http://127.0.0.1:8000", mode: str = "direct"):
        self.backend_url = backend_url.rstrip("/")
        self.mode = mode  # 'api' or 'direct'
        self.local_indices: Dict[str, RAGIndex] = {}

    def is_api_reachable(self) -> bool:
        """Check if FastAPI backend server is responsive."""
        try:
            r = requests.get(f"{self.backend_url}/api/health", timeout=1.5)
            return r.status_code == 200
        except Exception:
            return False

    def parse_document(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Extract and parse document."""
        if self.mode == "api":
            try:
                files = {"file": (filename, io.BytesIO(file_bytes))}
                r = requests.post(f"{self.backend_url}/api/documents/upload", files=files, timeout=30)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                pass  # Fall back to direct mode

        return extract_document(file_bytes, filename=filename)

    def summarize(self, text: str, model_name: str, style: str, max_len: int, min_len: int) -> Dict[str, Any]:
        """Generate BART summary."""
        if self.mode == "api":
            try:
                payload = {
                    "text": text,
                    "model_name": model_name,
                    "style": style,
                    "max_length": max_len,
                    "min_length": min_len
                }
                r = requests.post(f"{self.backend_url}/api/nlp/summarize", json=payload, timeout=60)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                pass

        return summarize_document(
            text=text,
            model_name=model_name,
            style=style,
            max_length=max_len,
            min_length=min_len
        )

    def sentiment(self, text: str, model_name: str) -> Dict[str, Any]:
        """Analyze DistilBERT sentiment and tone."""
        if self.mode == "api":
            try:
                payload = {"text": text, "model_name": model_name}
                r = requests.post(f"{self.backend_url}/api/nlp/sentiment", json=payload, timeout=45)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                pass

        return analyze_sentiment(text=text, model_name=model_name)

    def domain_analysis(self, text: str, domain: str) -> Dict[str, Any]:
        """Perform legal or academic domain analysis."""
        if self.mode == "api":
            try:
                payload = {"text": text, "domain": domain}
                r = requests.post(f"{self.backend_url}/api/nlp/domain-analysis", json=payload, timeout=30)
                if r.status_code == 200:
                    return r.json()["data"]
            except Exception:
                pass

        return analyze_domain(text=text, domain=domain)

    def index_for_rag(self, doc_id: str, doc_data: Dict[str, Any]) -> int:
        """Index document chunks for RAG."""
        if self.mode == "api":
            try:
                payload = {
                    "document_id": doc_id,
                    "full_text": doc_data.get("full_text", ""),
                    "pages": doc_data.get("pages", [])
                }
                r = requests.post(f"{self.backend_url}/api/rag/index", json=payload, timeout=30)
                if r.status_code == 200:
                    return r.json()["total_chunks"]
            except Exception:
                pass

        idx = build_rag_index(doc_data)
        self.local_indices[doc_id] = idx
        return len(idx.chunks)

    def query_rag(self, doc_id: str, query: str, provider: str = "local", api_key: Optional[str] = None) -> Dict[str, Any]:
        """Query document via RAG."""
        if self.mode == "api":
            try:
                payload = {
                    "document_id": doc_id,
                    "query": query,
                    "top_k": 4,
                    "provider": provider,
                    "api_key": api_key
                }
                r = requests.post(f"{self.backend_url}/api/rag/query", json=payload, timeout=40)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                pass

        # Direct local mode
        idx = self.local_indices.get(doc_id)
        if not idx:
            return {
                "query": query,
                "answer": "Document index not initialized.",
                "citations": [],
                "confidence": "None",
                "provider": provider
            }

        chunks = idx.search(query, top_k=4)
        synthesis = synthesize_llm_answer(query, chunks, api_key=api_key, provider=provider)
        return {
            "query": query,
            "answer": synthesis["answer"],
            "citations": synthesis["citations"],
            "confidence": synthesis["confidence"],
            "provider": synthesis["provider"]
        }
