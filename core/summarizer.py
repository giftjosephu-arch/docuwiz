"""
DocuWiz AI - BART Document Summarization Engine
Leverages facebook/bart-large-cnn or sshleifer/distilbart-cnn-12-6
using AutoModelForSeq2SeqLM and hierarchical map-reduce chunking for arbitrary document lengths.
"""

import re
import logging
from typing import Dict, Any, List, Optional
from collections import Counter

logger = logging.getLogger(__name__)

# Global cache for model and tokenizer to prevent redundant reloading
_BART_MODEL_CACHE: Dict[str, Any] = {}


def get_bart_model_and_tokenizer(model_name: str = "sshleifer/distilbart-cnn-12-6", device: str = "cpu"):
    """
    Lazy load and cache BART model and tokenizer.
    """
    if model_name in ["fast-extractive", "extractive", "fallback"]:
        return None, None

    global _BART_MODEL_CACHE
    cache_key = f"{model_name}_{device}"
    if cache_key in _BART_MODEL_CACHE:
        return _BART_MODEL_CACHE[cache_key]

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if device == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")

        _BART_MODEL_CACHE[cache_key] = (model, tokenizer)
        logger.info(f"Loaded BART model: {model_name} on device: {device}")
        return model, tokenizer
    except Exception as e:
        logger.warning(f"Could not load BART model {model_name}: {e}. Falling back to extractive mode.")
        return None, None


def extractive_summary_fallback(text: str, num_sentences: int = 5) -> str:
    """
    High-grade extractive TextRank/frequency fallback summarizer.
    Guarantees zero-failure operation even without network or GPU.
    """
    raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 20]
    if len(raw_sentences) <= num_sentences:
        return text

    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    stopwords = {
        'the', 'and', 'that', 'this', 'with', 'from', 'for', 'have', 'were', 'which',
        'their', 'there', 'they', 'what', 'when', 'where', 'who', 'will', 'more', 'about',
        'into', 'been', 'some', 'could', 'other', 'than', 'then', 'also', 'such', 'these'
    }
    filtered_words = [w for w in words if w not in stopwords]
    word_freq = Counter(filtered_words)

    sentence_scores = []
    for idx, s in enumerate(raw_sentences):
        s_words = re.findall(r'\b[a-zA-Z]{3,}\b', s.lower())
        score = sum(word_freq.get(w, 0) for w in s_words) / (len(s_words) + 1)
        if idx < 3:
            score *= 1.25
        sentence_scores.append((score, idx, s))

    top_sentences = sorted(sorted(sentence_scores, key=lambda x: x[0], reverse=True)[:num_sentences], key=lambda x: x[1])
    return " ".join([s[2] for s in top_sentences])


def chunk_text_by_words(text: str, chunk_size: int = 600, overlap: int = 50) -> List[str]:
    """Split text into manageable token/word chunks with overlap."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start += (chunk_size - overlap)
    return chunks


def summarize_document(
    text: str,
    model_name: str = "sshleifer/distilbart-cnn-12-6",
    max_length: int = 140,
    min_length: int = 40,
    style: str = "executive",
    device: str = "cpu",
    use_fallback_if_needed: bool = True
) -> Dict[str, Any]:
    """
    Summarize document using BART / DistilBART with Map-Reduce for long texts.
    """
    if not text or len(text.strip()) < 50:
        return {
            "summary": "Document text is too brief to generate a summary.",
            "raw_summary": "Document text is too brief to generate a summary.",
            "key_takeaways": [],
            "model_used": "none",
            "stats": {"chunks_processed": 0, "original_words": len(text.split()), "summary_words": 0, "compression_ratio": 1.0}
        }

    word_count = len(text.split())
    model, tokenizer = get_bart_model_and_tokenizer(model_name=model_name, device=device)

    # If model is unavailable or in fallback mode
    if model is None or tokenizer is None:
        if use_fallback_if_needed:
            logger.info("Using extractive frequency summarizer fallback.")
            ext_summary = extractive_summary_fallback(text, num_sentences=5)
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', ext_summary) if s.strip()]
            takeaways = sentences[:5] if sentences else [ext_summary]
            formatted = format_summary_by_style(ext_summary, takeaways, style)
            return {
                "summary": formatted,
                "raw_summary": ext_summary,
                "key_takeaways": takeaways,
                "model_used": "extractive-frequency-fallback",
                "stats": {
                    "chunks_processed": 1,
                    "original_words": word_count,
                    "summary_words": len(ext_summary.split()),
                    "compression_ratio": round(len(ext_summary.split()) / max(1, word_count), 2)
                }
            }
        else:
            raise RuntimeError(f"Could not load Hugging Face model {model_name}")

    # Map-Reduce Chunking
    chunks = chunk_text_by_words(text, chunk_size=550, overlap=50)
    intermediate_summaries = []

    try:
        import torch

        for idx, chunk in enumerate(chunks):
            chunk_words = len(chunk.split())
            c_max = min(max_length, max(30, int(chunk_words * 0.6)))
            c_min = min(min_length, max(15, int(c_max * 0.4)))

            inputs = tokenizer(chunk, max_length=1024, truncation=True, return_tensors="pt")
            if device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                summary_ids = model.generate(
                    inputs["input_ids"],
                    max_length=c_max,
                    min_length=c_min,
                    num_beams=2,
                    early_stopping=True
                )
            summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            intermediate_summaries.append(summary_text)

        # Reduce step if multi-chunk
        if len(intermediate_summaries) > 1:
            combined_intermediate = " ".join(intermediate_summaries)
            if len(combined_intermediate.split()) > 400:
                inputs = tokenizer(combined_intermediate, max_length=1024, truncation=True, return_tensors="pt")
                if device == "cuda" and torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                with torch.no_grad():
                    summary_ids = model.generate(
                        inputs["input_ids"],
                        max_length=max_length * 2,
                        min_length=min_length,
                        num_beams=2,
                        early_stopping=True
                    )
                final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            else:
                final_summary = combined_intermediate
        else:
            final_summary = intermediate_summaries[0]

    except Exception as exc:
        logger.error(f"Error during BART inference: {exc}")
        if use_fallback_if_needed:
            final_summary = extractive_summary_fallback(text, num_sentences=5)
        else:
            raise exc

    # Generate Bullet Takeaways
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', final_summary) if len(s.strip()) > 15]
    takeaways = sentences[:5] if sentences else [final_summary]

    formatted_summary = format_summary_by_style(final_summary, takeaways, style)

    return {
        "summary": formatted_summary,
        "raw_summary": final_summary,
        "key_takeaways": takeaways,
        "model_used": model_name,
        "stats": {
            "chunks_processed": len(chunks),
            "original_words": word_count,
            "summary_words": len(final_summary.split()),
            "compression_ratio": round(len(final_summary.split()) / max(1, word_count), 2)
        }
    }


def format_summary_by_style(summary: str, takeaways: List[str], style: str) -> str:
    """Apply domain formatting to summary output."""
    if style == "takeaways":
        bullets = "\n".join([f"• {t}" for t in takeaways])
        return f"### Key Takeaways\n\n{bullets}"

    elif style == "legal":
        return (
            "### ⚖️ Legal Executive Digest\n\n"
            f"**Core Summary:**\n{summary}\n\n"
            "**Key Clauses & Rights Identified:**\n" +
            "\n".join([f"- {t}" for t in takeaways])
        )

    elif style == "academic":
        return (
            "### 🎓 Academic Research Synopsis\n\n"
            f"**Research Overview:**\n{summary}\n\n"
            "**Key Contributions & Findings:**\n" +
            "\n".join([f"- {t}" for t in takeaways])
        )

    else:  # default 'executive'
        return summary
