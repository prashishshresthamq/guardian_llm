from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum

class RiskCategory(Enum):
    BIAS_FAIRNESS = "bias_fairness"
    PRIVACY_DATA = "privacy_data"
    SAFETY_SECURITY = "safety_security"
    DUAL_USE = "dual_use"
    SOCIETAL_IMPACT = "societal_impact"
    TRANSPARENCY = "transparency"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskAssessment:
    """Data class for risk assessment results"""
    category: RiskCategory
    score: float  # 0.0 to 10.0
    confidence: float  # 0.0 to 1.0
    level: RiskLevel
    explanation: str
    evidence: List[str]
    recommendations: List[str]

@dataclass
class AnalysisRequest:
    """Data class for analysis requests"""
    title: str
    abstract: Optional[str] = ""
    content: str = ""
    authors: Optional[List[str]] = None

@dataclass
class AnalysisResponse:
    """Data class for analysis responses"""
    paper_id: str
    title: str
    overall_risk_score: float
    processing_time: float
    risk_assessments: List[Dict]
    timestamp: str
    status: str

def validate_analysis_request(data: dict) -> tuple[bool, str]:
    """Validate analysis request data"""
    if not data:
        return False, "No data provided"
    
    title = data.get('title', '').strip()
    content = data.get('content', '').strip()
    
    if not title and not content:
        return False, "Either title or content is required"
    
    if content and len(content) < 50:
        return False, "Content too short for meaningful analysis (minimum 50 characters)"
    
    if content and len(content) > 1000000:  # 1MB limit
        return False, "Content too large (maximum 1MB)"
    
    return True, ""