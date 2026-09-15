"""
DocuWiz AI - Documents Route
Handles file uploading, parsing, and metadata extraction.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from core.extractors import extract_document
from backend.schemas import DocumentParsedResponse

router = APIRouter(prefix="/api/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentParsedResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and parse PDF, DOCX, or TXT file into structured pages and metadata.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    try:
        content = await file.read()
        res = extract_document(content, filename=file.filename)
        return DocumentParsedResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract document: {str(e)}")
