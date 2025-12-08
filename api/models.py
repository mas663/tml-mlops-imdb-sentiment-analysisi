"""
Pydantic models untuk request dan response validation
"""

from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request body untuk endpoint predict"""
    text: str = Field(
        ..., 
        min_length=1,
        max_length=5000,
        description="Teks review yang akan diprediksi",
        example="This movie was absolutely fantastic! Great story and acting."
    )
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text tidak boleh kosong')
        return v.strip()


class PredictionResponse(BaseModel):
    """Response body dari endpoint predict"""
    sentiment: str = Field(
        ...,
        description="Hasil prediksi: positive atau negative"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score dari model (0-1)"
    )
    probabilities: dict = Field(
        ...,
        description="Probability untuk setiap class"
    )
    text_length: int = Field(
        ...,
        description="Panjang karakter input text"
    )
    timestamp: str = Field(
        ...,
        description="Waktu prediksi dilakukan"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "sentiment": "positive",
                "confidence": 0.95,
                "probabilities": {
                    "negative": 0.05,
                    "positive": 0.95
                },
                "text_length": 52,
                "timestamp": "2024-12-08T10:30:00"
            }
        }


class HealthResponse(BaseModel):
    """Response untuk health check"""
    status: str
    model_loaded: bool
    version: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Response untuk model info"""
    model_type: str
    vectorizer_type: str
    max_features: int
    test_accuracy: float
    test_f1_score: float
    test_precision: float
    test_recall: float
    trained_date: Optional[str]
    model_version: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "LogisticRegression",
                "vectorizer_type": "TfidfVectorizer",
                "max_features": 20000,
                "test_accuracy": 0.8997,
                "test_f1_score": 0.9002,
                "test_precision": 0.8955,
                "test_recall": 0.9050,
                "trained_date": "2024-12-08",
                "model_version": "v1.0.0"
            }
        }


class ErrorResponse(BaseModel):
    """Response untuk error"""
    error: str
    detail: Optional[str] = None
    timestamp: str
