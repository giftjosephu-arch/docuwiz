"""
DocuWiz AI - FastAPI Application Server
Provides RESTful endpoints for document extraction, BART summarization,
DistilBERT sentiment analysis, and RAG document intelligence.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.routes.documents import router as documents_router
from backend.routes.nlp import router as nlp_router
from backend.routes.rag import router as rag_router

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description=(
        "DocuWiz AI is an enterprise document intelligence API leveraging "
        "advanced NLP models (BART, DistilBERT) and RAG for deep insights in "
        "legal and academic workflows."
    )
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(documents_router)
app.include_router(nlp_router)
app.include_router(rag_router)


@app.get("/api/health", tags=["Health"])
async def health_check():
    """System health check and capability status."""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.VERSION,
        "models": {
            "summarizer": settings.DEFAULT_BART_MODEL,
            "sentiment": settings.DEFAULT_SENTIMENT_MODEL,
            "device": settings.DEVICE
        },
        "capabilities": [
            "pdf_extraction",
            "docx_extraction",
            "bart_summarization",
            "distilbert_sentiment",
            "rag_qa",
            "legal_clause_audit",
            "academic_rigor_analysis",
            "productivity_benchmarks"
        ]
    }


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "DocuWiz AI Document Intelligence API is active.",
        "documentation": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=True)
