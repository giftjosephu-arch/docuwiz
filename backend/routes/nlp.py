"""
DocuWiz AI - NLP & Intelligence Route
Exposes BART summarization, DistilBERT sentiment, domain intelligence, and speedup benchmarks.
"""

from fastapi import APIRouter, HTTPException
from core.summarizer import summarize_document
from core.sentiment import analyze_sentiment
from core.domain_intelligence import analyze_domain
from core.analytics import compute_productivity_benchmark, extract_key_concepts, calculate_readability
from backend.schemas import (
    SummarizeRequest, SummarizeResponse,
    SentimentRequest, SentimentResponse,
    DomainAnalysisRequest, DomainAnalysisResponse,
    BenchmarkRequest, BenchmarkResponse
)
from backend.config import settings

router = APIRouter(prefix="/api/nlp", tags=["NLP Intelligence"])


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest):
    """
    Summarize text using BART with hierarchical chunking.
    """
    try:
        res = summarize_document(
            text=req.text,
            model_name=req.model_name or settings.DEFAULT_BART_MODEL,
            max_length=req.max_length or 140,
            min_length=req.min_length or 40,
            style=req.style or "executive",
            device=settings.DEVICE
        )
        return SummarizeResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization error: {str(e)}")


@router.post("/sentiment", response_model=SentimentResponse)
async def sentiment(req: SentimentRequest):
    """
    Analyze sentiment and tone progression using DistilBERT.
    """
    try:
        res = analyze_sentiment(
            text=req.text,
            model_name=req.model_name or settings.DEFAULT_SENTIMENT_MODEL,
            device=settings.DEVICE,
            granularity=req.granularity or "paragraph"
        )
        return SentimentResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis error: {str(e)}")


@router.post("/domain-analysis", response_model=DomainAnalysisResponse)
async def domain_analysis(req: DomainAnalysisRequest):
    """
    Deep legal audit (clauses, red-flag risks) or academic research synthesis.
    """
    try:
        res = analyze_domain(text=req.text, domain=req.domain or "auto")
        return DomainAnalysisResponse(domain=res.get("domain", "general"), data=res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Domain analysis error: {str(e)}")


@router.post("/benchmark", response_model=BenchmarkResponse)
async def benchmark(req: BenchmarkRequest):
    """
    Calculate time savings and the 60% processing speedup metrics.
    """
    try:
        res = compute_productivity_benchmark(word_count=req.word_count, doc_type=req.doc_type or "general")
        return BenchmarkResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark error: {str(e)}")
