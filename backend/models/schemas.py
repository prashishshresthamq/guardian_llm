"""
Guardian LLM - Data Schemas and Models
Defines data structures for API requests/responses and internal processing
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, validator


class RiskLevel(str, Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategory(str, Enum):
    """Categories of detected risks"""
    VIOLENCE = "violence"
    SELF_HARM = "self-harm"
    HATE_SPEECH = "hate-speech"
    HARASSMENT = "harassment"
    ADULT_CONTENT = "adult-content"
    MISINFORMATION = "misinformation"
    SPAM = "spam"
    SAFE = "safe"


class SentimentType(str, Enum):
    """Sentiment classifications"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


# Pydantic Models for API
class AnalysisRequest(BaseModel):
    """Request model for text analysis"""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")
    user_id: Optional[int] = Field(None, description="User ID for tracking")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Analysis options")
    
    @validator('text')
    def validate_text(cls, v):
        """Validate and clean input text"""
        v = v.strip()
        if not v:
            raise ValueError("Text cannot be empty")
        return v


class BatchAnalysisRequest(BaseModel):
    """Request model for batch text analysis"""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to analyze")
    user_id: Optional[int] = Field(None, description="User ID for tracking")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Analysis options")


class RiskAssessment(BaseModel):
    """Risk assessment details"""
    category: RiskCategory
    level: RiskLevel
    score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    keywords: List[str] = Field(default_factory=list)
    context: Optional[str] = None


class SentimentAnalysis(BaseModel):
    """Sentiment analysis results"""
    type: SentimentType
    score: float = Field(..., ge=-1.0, le=1.0)
    positive_score: float = Field(..., ge=0.0, le=1.0)
    negative_score: float = Field(..., ge=0.0, le=1.0)
    neutral_score: float = Field(..., ge=0.0, le=1.0)
    keywords: Dict[str, List[str]] = Field(default_factory=dict)


class TextStatistics(BaseModel):
    """Text statistics"""
    word_count: int = Field(..., ge=0)
    character_count: int = Field(..., ge=0)
    sentence_count: int = Field(..., ge=0)
    avg_word_length: float = Field(..., ge=0.0)
    high_risk_keywords: int = Field(..., ge=0)
    medium_risk_keywords: int = Field(..., ge=0)


class Recommendation(BaseModel):
    """Action recommendation"""
    type: str
    message: str
    action: str
    priority: str = Field(..., pattern="^(low|medium|high|critical)$")
    metadata: Optional[Dict[str, Any]] = None


class AnalysisResponse(BaseModel):
    """Response model for text analysis"""
    id: Optional[str] = None
    text: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    analysis: Dict[str, Any] = Field(..., description="Analysis results")
    risk_assessments: List[RiskAssessment]
    sentiment: SentimentAnalysis
    statistics: TextStatistics
    recommendations: List[Recommendation]
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BatchAnalysisResponse(BaseModel):
    """Response model for batch text analysis"""
    count: int
    results: List[AnalysisResponse]
    summary: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Dataclasses for internal use
@dataclass
class ProcessedText:
    """Processed text data"""
    original: str
    cleaned: str
    tokens: List[str]
    sentences: List[str]
    language: str = "en"
    metadata: Dict[str, Any] = None


@dataclass
class RiskIndicator:
    """Risk indicator data"""
    keyword: str
    category: RiskCategory
    severity: float
    context_weight: float = 1.0


@dataclass
class AnalysisConfig:
    """Configuration for analysis"""
    enable_sentiment: bool = True
    enable_risk_detection: bool = True
    enable_keyword_extraction: bool = True
    enable_recommendations: bool = True
    risk_threshold: float = 0.5
    confidence_threshold: float = 0.7
    max_recommendations: int = 5
    custom_keywords: Dict[str, List[str]] = None


# Database Schema Models (for SQLAlchemy)
class AnalysisStatus(str, Enum):
    """Analysis status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# Error Response Models
class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ValidationError(BaseModel):
    """Validation error details"""
    field: str
    message: str
    value: Optional[Any] = None


# WebSocket Message Models
class WSMessage(BaseModel):
    """WebSocket message model"""
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WSAnalysisUpdate(BaseModel):
    """Real-time analysis update"""
    analysis_id: str
    status: AnalysisStatus
    progress: float = Field(..., ge=0.0, le=1.0)
    partial_results: Optional[Dict[str, Any]] = None


# Export Models
class ExportFormat(str, Enum):
    """Export format options"""
    JSON = "json"
    CSV = "csv"
    PDF = "pdf"
    XLSX = "xlsx"


class ExportRequest(BaseModel):
    """Export request model"""
    analysis_ids: List[str]
    format: ExportFormat
    include_metadata: bool = True
    filters: Optional[Dict[str, Any]] = None


# Utility functions
def risk_level_from_score(score: float) -> RiskLevel:
    """Convert risk score to risk level"""
    if score >= 0.8:
        return RiskLevel.CRITICAL
    elif score >= 0.6:
        return RiskLevel.HIGH
    elif score >= 0.4:
        return RiskLevel.MEDIUM
    else:
        return RiskLevel.LOW


def sentiment_type_from_scores(positive: float, negative: float, neutral: float) -> SentimentType:
    """Determine sentiment type from scores"""
    scores = {"positive": positive, "negative": negative, "neutral": neutral}
    max_score = max(scores.values())
    
    if max_score < 0.4:  # No strong sentiment
        return SentimentType.MIXED
    
    return SentimentType(max(scores, key=scores.get).upper())


# Validation schemas for configuration
class ModelConfig(BaseModel):
    """Model configuration schema"""
    model_name: str
    version: str
    parameters: Dict[str, Any]
    enabled: bool = True


class ServiceConfig(BaseModel):
    """Service configuration schema"""
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    timeout: int = 30
    retry_count: int = 3
    enabled: bool = True