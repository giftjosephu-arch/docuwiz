"""
DocuWiz AI - Pydantic Request & Response Schemas
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class PageItem(BaseModel):
    page_number: int
    text: str
    char_count: int
    word_count: int


class DocumentMetadata(BaseModel):
    word_count: int
    char_count: int
    unique_words: int
    lexical_diversity: float
    reading_time_minutes: float
    pages_count: int


class DocumentParsedResponse(BaseModel):
    filename: str
    file_type: str
    full_text: str
    pages: List[PageItem]
    metadata: DocumentMetadata


class SummarizeRequest(BaseModel):
    text: str
    model_name: Optional[str] = "sshleifer/distilbart-cnn-12-6"
    style: Optional[str] = "executive"
    max_length: Optional[int] = 140
    min_length: Optional[int] = 40


class SummarizeResponse(BaseModel):
    summary: str
    raw_summary: str
    key_takeaways: List[str]
    model_used: str
    stats: Dict[str, Any]


class SentimentRequest(BaseModel):
    text: str
    model_name: Optional[str] = "distilbert-base-uncased-finetuned-sst-2-english"
    granularity: Optional[str] = "paragraph"


class SentimentResponse(BaseModel):
    overall_label: str
    overall_score: float
    overall_polarity: float
    dominant_tone: str
    timeline: List[Dict[str, Any]]
    distribution: Dict[str, float]
    tone_distribution: Dict[str, int]
    critical_risk_passages: List[Dict[str, Any]]
    positive_highlights: List[Dict[str, Any]]
    model_used: str


class DomainAnalysisRequest(BaseModel):
    text: str
    domain: Optional[str] = "auto"  # 'legal', 'academic', or 'auto'


class DomainAnalysisResponse(BaseModel):
    domain: str
    data: Dict[str, Any]


class RAGIndexRequest(BaseModel):
    document_id: str
    full_text: str
    pages: Optional[List[Dict[str, Any]]] = None
    chunk_size: Optional[int] = 180
    overlap: Optional[int] = 40


class RAGIndexResponse(BaseModel):
    document_id: str
    total_chunks: int
    status: str


class RAGQueryRequest(BaseModel):
    document_id: str
    query: str
    top_k: Optional[int] = 4
    provider: Optional[str] = "local"
    api_key: Optional[str] = None


class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    confidence: str
    provider: str


class BenchmarkRequest(BaseModel):
    word_count: int
    doc_type: Optional[str] = "general"


class BenchmarkResponse(BaseModel):
    manual_review_minutes: float
    ai_processing_minutes: float
    time_saved_minutes: float
    processing_speedup_percent: float
    productivity_multiplier: str
    benchmark_breakdown: List[Dict[str, Any]]
