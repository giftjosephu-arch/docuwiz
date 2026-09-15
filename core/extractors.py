"""
DocuWiz AI - Document Text Extraction Engine
Supports PDF, DOCX, and TXT with structured metadata and page-level mapping.
"""

import io
import re
from typing import Dict, Any, List, Optional, Union
from pathlib import Path


def clean_text_whitespace(text: str) -> str:
    """Normalize whitespace and strip extraneous empty lines while preserving paragraphs."""
    if not text:
        return ""
    # Replace non-breaking spaces and carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    # Replace sequences of more than 2 newlines with 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Replace sequences of whitespace on the same line with a single space
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def calculate_metadata(full_text: str, pages_count: int = 1) -> Dict[str, Any]:
    """Calculate descriptive metrics for extracted text."""
    words = re.findall(r"\b\w+\b", full_text)
    total_words = len(words)
    total_chars = len(full_text)
    unique_words = len(set(word.lower() for word in words))
    lexical_diversity = round(unique_words / total_words, 3) if total_words > 0 else 0.0
    # Average adult reading speed: ~225 words per minute
    reading_time_minutes = max(1, round(total_words / 225, 1)) if total_words > 0 else 0

    return {
        "word_count": total_words,
        "char_count": total_chars,
        "unique_words": unique_words,
        "lexical_diversity": lexical_diversity,
        "reading_time_minutes": reading_time_minutes,
        "pages_count": max(1, pages_count)
    }


def extract_from_pdf(source: Union[bytes, io.BytesIO, str, Path]) -> Dict[str, Any]:
    """
    Extract text and metadata from a PDF document page by page.
    """
    from pypdf import PdfReader

    stream: io.BytesIO
    if isinstance(source, (str, Path)):
        with open(source, "rb") as f:
            stream = io.BytesIO(f.read())
    elif isinstance(source, bytes):
        stream = io.BytesIO(source)
    else:
        stream = source

    reader = PdfReader(stream)
    pages: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        page_text = page.extract_text() or ""
        cleaned_page_text = clean_text_whitespace(page_text)
        pages.append({
            "page_number": page_num,
            "text": cleaned_page_text,
            "char_count": len(cleaned_page_text),
            "word_count": len(re.findall(r"\b\w+\b", cleaned_page_text))
        })
        if cleaned_page_text:
            full_text_parts.append(cleaned_page_text)

    full_text = "\n\n".join(full_text_parts)
    meta = calculate_metadata(full_text, pages_count=len(reader.pages))

    return {
        "full_text": full_text,
        "pages": pages,
        "metadata": meta
    }


def extract_from_docx(source: Union[bytes, io.BytesIO, str, Path]) -> Dict[str, Any]:
    """
    Extract text, headings, and tables from a DOCX document.
    """
    import docx

    stream: io.BytesIO
    if isinstance(source, (str, Path)):
        with open(source, "rb") as f:
            stream = io.BytesIO(f.read())
    elif isinstance(source, bytes):
        stream = io.BytesIO(source)
    else:
        stream = source

    doc = docx.Document(stream)
    paragraphs: List[str] = []
    headings: List[Dict[str, str]] = []

    for p in doc.paragraphs:
        txt = p.text.strip()
        if not txt:
            continue
        if p.style and p.style.name and p.style.name.startswith("Heading"):
            headings.append({"level": p.style.name, "text": txt})
        paragraphs.append(txt)

    # Extract tables
    table_texts: List[str] = []
    for t_idx, table in enumerate(doc.tables):
        rows_data = []
        for row in table.rows:
            row_vals = [cell.text.strip() for cell in row.cells]
            if any(row_vals):
                rows_data.append(" | ".join(row_vals))
        if rows_data:
            table_texts.append(f"[Table {t_idx + 1}]\n" + "\n".join(rows_data))

    all_content_parts = paragraphs + table_texts
    full_text = clean_text_whitespace("\n\n".join(all_content_parts))

    # Approximate pages (standard manuscript ~350 words per page)
    words = re.findall(r"\b\w+\b", full_text)
    approx_pages = max(1, (len(words) + 349) // 350)

    # Segment into pseudo-pages for unified RAG citations
    pages: List[Dict[str, Any]] = []
    chunk_size = max(1, len(paragraphs) // approx_pages) if paragraphs else 1
    for p_idx in range(approx_pages):
        start = p_idx * chunk_size
        end = start + chunk_size if p_idx < approx_pages - 1 else len(paragraphs)
        page_paragraphs = paragraphs[start:end]
        p_text = clean_text_whitespace("\n\n".join(page_paragraphs))
        pages.append({
            "page_number": p_idx + 1,
            "text": p_text,
            "char_count": len(p_text),
            "word_count": len(re.findall(r"\b\w+\b", p_text))
        })

    meta = calculate_metadata(full_text, pages_count=approx_pages)

    return {
        "full_text": full_text,
        "pages": pages,
        "headings": headings,
        "metadata": meta
    }


def extract_from_txt(source: Union[bytes, io.BytesIO, str, Path]) -> Dict[str, Any]:
    """
    Extract text and metadata from plain text or markdown files.
    """
    raw_str: str
    if isinstance(source, (str, Path)):
        if isinstance(source, str) and not Path(source).exists() and len(source) > 200:
            raw_str = source
        else:
            with open(source, "r", encoding="utf-8", errors="replace") as f:
                raw_str = f.read()
    elif isinstance(source, bytes):
        raw_str = source.decode("utf-8", errors="replace")
    else:
        raw_str = source.read().decode("utf-8", errors="replace")

    full_text = clean_text_whitespace(raw_str)
    words = re.findall(r"\b\w+\b", full_text)
    approx_pages = max(1, (len(words) + 349) // 350)

    # Segment into pseudo-pages
    paras = [p.strip() for p in full_text.split("\n\n") if p.strip()]
    pages: List[Dict[str, Any]] = []
    chunk_size = max(1, len(paras) // approx_pages) if paras else 1
    for p_idx in range(approx_pages):
        start = p_idx * chunk_size
        end = start + chunk_size if p_idx < approx_pages - 1 else len(paras)
        p_text = "\n\n".join(paras[start:end])
        pages.append({
            "page_number": p_idx + 1,
            "text": p_text,
            "char_count": len(p_text),
            "word_count": len(re.findall(r"\b\w+\b", p_text))
        })

    meta = calculate_metadata(full_text, pages_count=approx_pages)

    return {
        "full_text": full_text,
        "pages": pages,
        "metadata": meta
    }


def extract_document(
    content: Union[bytes, io.BytesIO, str, Path],
    filename: str = "document.txt"
) -> Dict[str, Any]:
    """
    Auto-detect document type by file extension and extract unified content.
    Returns:
        Dict containing full_text, pages list, metadata, filename, and file_type.
    """
    fn_lower = filename.lower()
    if fn_lower.endswith(".pdf"):
        res = extract_from_pdf(content)
        file_type = "pdf"
    elif fn_lower.endswith(".docx") or fn_lower.endswith(".doc"):
        res = extract_from_docx(content)
        file_type = "docx"
    else:
        res = extract_from_txt(content)
        file_type = "txt"

    res["filename"] = filename
    res["file_type"] = file_type
    return res
