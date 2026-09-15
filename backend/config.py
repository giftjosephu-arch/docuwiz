"""
DocuWiz AI - Backend Configuration Settings
"""

import os
from pydantic import BaseModel


class Settings(BaseModel):
    APP_NAME: str = "DocuWiz AI Document Intelligence API"
    VERSION: str = "1.0.0"
    HOST: str = os.getenv("DOCUWIZ_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("DOCUWIZ_PORT", "8000"))
    DEFAULT_BART_MODEL: str = os.getenv("BART_MODEL", "sshleifer/distilbart-cnn-12-6")
    DEFAULT_SENTIMENT_MODEL: str = os.getenv("SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
    DEVICE: str = os.getenv("DOCUWIZ_DEVICE", "cpu")
    MAX_UPLOAD_SIZE_MB: int = 50


settings = Settings()
