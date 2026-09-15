"""
DocuWiz AI - Retrieval-Augmented Generation (RAG) & LLM Q&A Engine
Indexes documents by chunks, performs semantic retrieval, and synthesizes answers
with precise page citations using either LLM APIs (Gemini, OpenAI, Groq) or Local Extractive Synthesis.
"""

import os
import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class DocumentChunk:
    def __init__(self, chunk_id: int, page_number: int, text: str, word_count: int):
        self.chunk_id = chunk_id
        self.page_number = page_number
        self.text = text
        self.word_count = word_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "page_number": self.page_number,
            "text": self.text,
            "word_count": self.word_count
        }


def stem_token(word: str) -> str:
    """Lightweight suffix stemmer for legal and academic terminology alignment."""
    w = word.lower()
    for suffix in ("ation", "ition", "tion", "sion", "ment", "ing", "ies", "ity", "able", "ible", "ed", "es", "ate", "al"):
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            w = w[:-len(suffix)]
            break
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        w = w[:-1]
    if (w.endswith("e") or w.endswith("a")) and len(w) > 3:
        w = w[:-1]
    return w


class RAGIndex:
    """In-memory semantic vector index for a document."""

    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.texts = [c.text for c in chunks]
        if self.texts:
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                stop_words="english",
                max_features=10000
            )
            self.matrix = self.vectorizer.fit_transform(self.texts)
        else:
            self.vectorizer = None
            self.matrix = None

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Retrieve top_k most relevant chunks with hybrid semantic and stem matching."""
        if self.matrix is None or self.matrix.shape[0] == 0 or not query.strip():
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.matrix).flatten()

        # Hybrid keyword stem overlap boost
        stopwords = {'the', 'and', 'what', 'when', 'where', 'which', 'who', 'how', 'does', 'with', 'from', 'this', 'that', 'are'}
        q_tokens = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()) if w not in stopwords]
        q_stems = {stem_token(w) for w in q_tokens}

        if q_stems:
            for idx, chunk in enumerate(self.chunks):
                chunk_words = re.findall(r'\b[a-zA-Z]{3,}\b', chunk.text.lower())
                chunk_stems = {stem_token(w) for w in chunk_words}
                overlap = len(q_stems.intersection(chunk_stems))
                if overlap > 0:
                    stem_boost = (overlap / len(q_stems)) * 0.5
                    similarities[idx] = max(float(similarities[idx]), float(stem_boost))

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []

        for idx in top_indices:
            score = float(similarities[idx])
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk.chunk_id,
                "page_number": chunk.page_number,
                "text": chunk.text,
                "similarity_score": round(score, 4),
                "relevance_pct": round(min(1.0, score) * 100, 1)
            })

        return results


def build_rag_index(document_data: Dict[str, Any], chunk_size: int = 180, overlap: int = 40) -> RAGIndex:
    """
    Chunk document by page and build vector index.
    """
    pages = document_data.get("pages", [])
    chunks: List[DocumentChunk] = []
    chunk_counter = 0

    if not pages:
        # Fallback to full_text
        full_text = document_data.get("full_text", "")
        pages = [{"page_number": 1, "text": full_text}]

    for page in pages:
        page_num = page.get("page_number", 1)
        p_text = page.get("text", "")
        words = p_text.split()
        if not words:
            continue

        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_str = " ".join(chunk_words)

            if len(chunk_str.strip()) > 30:
                chunks.append(DocumentChunk(
                    chunk_id=chunk_counter,
                    page_number=page_num,
                    text=chunk_str,
                    word_count=len(chunk_words)
                ))
                chunk_counter += 1

            if end >= len(words):
                break
            start += (chunk_size - overlap)

    return RAGIndex(chunks)


def synthesize_local_answer(query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    100% offline, privacy-safe, context-grounded extractive RAG synthesizer.
    Pulls most relevant sentences directly from retrieved chunks and links citations.
    """
    if not retrieved_chunks or retrieved_chunks[0]["similarity_score"] < 0.03:
        return {
            "answer": "I could not find sufficiently relevant information in the uploaded document to answer this query with high confidence.",
            "citations": [],
            "confidence": "Low",
            "provider": "Local Extractive RAG"
        }

    stopwords = {'the', 'and', 'what', 'when', 'where', 'which', 'who', 'how', 'does', 'with', 'from', 'this', 'that', 'are'}
    q_tokens = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()) if w not in stopwords]
    q_stems = {stem_token(w) for w in q_tokens}

    best_sentences: List[Tuple[str, int, float]] = []

    for chunk in retrieved_chunks:
        page_num = chunk["page_number"]
        chunk_text = chunk["text"]
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', chunk_text) if len(s.strip()) > 20]

        for s in sentences:
            s_words = re.findall(r'\b[a-zA-Z]{3,}\b', s.lower())
            s_stems = {stem_token(w) for w in s_words}
            overlap_count = len(q_stems.intersection(s_stems))
            if overlap_count > 0:
                score = (overlap_count / (len(q_stems) + 1)) * max(0.1, chunk["similarity_score"])
                best_sentences.append((s, page_num, score))

    best_sentences = sorted(best_sentences, key=lambda x: x[2], reverse=True)

    # Format synthesized response
    citations = []
    used_pages = set()
    for chunk in retrieved_chunks[:3]:
        citations.append({
            "page_number": chunk["page_number"],
            "snippet": chunk["text"][:160] + "...",
            "score": chunk["similarity_score"]
        })
        used_pages.add(chunk["page_number"])

    pages_str = ", ".join([f"Page {p}" for p in sorted(used_pages)])

    if best_sentences:
        # Take top 3 unique relevant sentences
        selected_sentences = []
        seen = set()
        for s, p, _ in best_sentences:
            cleaned = s.strip()
            if cleaned not in seen:
                selected_sentences.append(f"{cleaned} *(Source: Page {p})*")
                seen.add(cleaned)
            if len(selected_sentences) >= 3:
                break

        body = "\n\n".join(selected_sentences)
        answer = (
            f"Based on **{pages_str}** of the document:\n\n"
            f"{body}\n\n"
            f"> **Summary Context**: The relevant provisions indicate key stipulations aligned with your query."
        )
        confidence = "High" if retrieved_chunks[0]["similarity_score"] > 0.25 else "Moderate"
    else:
        # Contextual summary of top chunk
        top = retrieved_chunks[0]
        answer = (
            f"According to **Page {top['page_number']}**, the most pertinent passage states:\n\n"
            f"\"{top['text'][:350]}...\"\n\n"
            f"*(Extracted with {top['relevance_pct']}% semantic relevance)*"
        )
        confidence = "Moderate"

    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
        "provider": "Local Context Grounding"
    }


def synthesize_llm_answer(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    provider: str = "local"
) -> Dict[str, Any]:
    """
    Synthesize answer using either Google Gemini, OpenAI, Groq, or Local fallback.
    """
    if not retrieved_chunks:
        return {
            "answer": "No relevant document sections found to answer this question.",
            "citations": [],
            "confidence": "None",
            "provider": provider
        }

    # Format context with explicit page tags
    context_blocks = []
    for c in retrieved_chunks:
        context_blocks.append(f"--- [Page {c['page_number']} | Chunk {c['chunk_id']}] ---\n{c['text']}")
    context_str = "\n\n".join(context_blocks)

    system_prompt = (
        "You are DocuWiz AI, an elite document intelligence analyst. "
        "Answer the user's question using ONLY the provided document context. "
        "Cite the specific page numbers in your answer (e.g. '[Page 2]'). "
        "If the document does not contain enough information, state that clearly."
    )
    user_prompt = f"DOCUMENT CONTEXT:\n{context_str}\n\nQUESTION:\n{query}"

    # 1. Google Gemini
    if provider == "gemini" and api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(f"{system_prompt}\n\n{user_prompt}")
            return {
                "answer": response.text,
                "citations": retrieved_chunks,
                "confidence": "High",
                "provider": "Google Gemini"
            }
        except Exception as e:
            logger.warning(f"Gemini API call failed: {e}. Falling back to local synthesizer.")

    # 2. OpenAI / Groq
    if provider in ("openai", "groq") and api_key:
        try:
            from openai import OpenAI
            base_url = "https://api.groq.com/openai/v1" if provider == "groq" else None
            client = OpenAI(api_key=api_key, base_url=base_url)
            model_name = "llama-3.1-70b-versatile" if provider == "groq" else "gpt-4o-mini"
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2
            )
            return {
                "answer": resp.choices[0].message.content,
                "citations": retrieved_chunks,
                "confidence": "High",
                "provider": provider.capitalize()
            }
        except Exception as e:
            logger.warning(f"{provider} API call failed: {e}. Falling back to local synthesizer.")

    # 3. Local Extractive Grounding Fallback
    return synthesize_local_answer(query, retrieved_chunks)
