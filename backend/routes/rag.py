"""
DocuWiz AI - RAG Route
Indexes documents into vector space and performs semantic retrieval with citations.
"""

from typing import Dict
from fastapi import APIRouter, HTTPException
from core.rag_engine import build_rag_index, synthesize_llm_answer, RAGIndex
from backend.schemas import RAGIndexRequest, RAGIndexResponse, RAGQueryRequest, RAGQueryResponse

router = APIRouter(prefix="/api/rag", tags=["RAG Document Q&A"])

# In-memory document indexes store
INDEX_STORE: Dict[str, RAGIndex] = {}


@router.post("/index", response_model=RAGIndexResponse)
async def index_document(req: RAGIndexRequest):
    """
    Build vector embeddings index for a document.
    """
    try:
        doc_data = {
            "full_text": req.full_text,
            "pages": req.pages or [{"page_number": 1, "text": req.full_text}]
        }
        idx = build_rag_index(
            document_data=doc_data,
            chunk_size=req.chunk_size or 180,
            overlap=req.overlap or 40
        )
        INDEX_STORE[req.document_id] = idx

        return RAGIndexResponse(
            document_id=req.document_id,
            total_chunks=len(idx.chunks),
            status="indexed"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index document: {str(e)}")


@router.post("/query", response_model=RAGQueryResponse)
async def query_rag(req: RAGQueryRequest):
    """
    Query indexed document with semantic search and source citations.
    """
    if req.document_id not in INDEX_STORE:
        raise HTTPException(status_code=404, detail=f"Document ID '{req.document_id}' not found in index. Please index first.")

    try:
        index = INDEX_STORE[req.document_id]
        retrieved_chunks = index.search(query=req.query, top_k=req.top_k or 4)

        synthesis = synthesize_llm_answer(
            query=req.query,
            retrieved_chunks=retrieved_chunks,
            api_key=req.api_key,
            provider=req.provider or "local"
        )

        return RAGQueryResponse(
            query=req.query,
            answer=synthesis["answer"],
            citations=synthesis["citations"],
            confidence=synthesis["confidence"],
            provider=synthesis["provider"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query error: {str(e)}")
