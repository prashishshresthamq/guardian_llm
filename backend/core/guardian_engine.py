"""
Guardian LLM - Core Analysis Engine
Main engine for text analysis, risk detection, and sentiment analysis
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import nltk
from textblob import TextBlob
import numpy as np

# Import local modules
from backend.models.schemas import (
    RiskLevel, 
    RiskCategory, 
    SentimentType,
    RiskAssessment,
    risk_level_from_score,
    sentiment_type_from_scores
)
from backend.config.setting import Config
from backend.core.text_processors import TextProcessor
from backend.core.risk_analyzers import RiskAnalyzer

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass


class GuardianEngine:
    """Main analysis engine for Guardian LLM"""
    
    def __init__(self):
        """Initialize the Guardian Engine"""
        self.config = Config()
        self.text_processor = TextProcessor()
        self.risk_analyzer = RiskAnalyzer()
        self.is_initialized = True
        
    def get_status(self) -> Dict[str, Any]:
        """Get engine status"""
        return {
            'initialized': self.is_initialized,
            'version': '1.0.0',
            'components': {
                'text_processor': 'active',
                'risk_analyzer': 'active',
                'sentiment_analyzer': 'active'
            }
        }
    
    def analyze_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform complete text analysis
        
        Args:
            text: Text to analyze
            options: Optional analysis configuration
            
        Returns:
            Dictionary containing analysis results
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Process text
        processed_text = self.text_processor.process(text)
        
        # Perform risk analysis
        risk_analysis = self._analyze_risks(processed_text)
        
        # Perform sentiment analysis
        sentiment_analysis = self._analyze_sentiment(text)
        
        # Generate statistics
        statistics = self._generate_statistics(processed_text, risk_analysis)
        
        # Generate risk assessments
        risk_assessments = self._generate_risk_assessments(processed_text, risk_analysis)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(risk_analysis, sentiment_analysis)
        
        return {
            'text': text,
            'processed_text': processed_text,
            'risk_analysis': risk_analysis,
            'sentiment': sentiment_analysis,
            'statistics': statistics,
            'risk_assessments': risk_assessments,
            'recommendations': recommendations,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _analyze_risks(self, processed_text: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze text for various risk factors"""
        
        # Get risk keywords from config
        high_risk_keywords = self.config.RISK_KEYWORDS['high_risk']
        medium_risk_keywords = self.config.RISK_KEYWORDS['medium_risk']
        low_risk_keywords = self.config.RISK_KEYWORDS['low_risk']
        
        # Count risk keywords
        text_lower = processed_text['cleaned'].lower()
        tokens_lower = [t.lower() for t in processed_text['tokens']]
        
        high_risk_count = sum(1 for keyword in high_risk_keywords if keyword in text_lower)
        medium_risk_count = sum(1 for keyword in medium_risk_keywords if keyword in text_lower)
        low_risk_count = sum(1 for keyword in low_risk_keywords if keyword in text_lower)
        
        # Calculate risk scores
        total_words = len(processed_text['tokens'])
        if total_words == 0:
            critical_risk_score = 0
            overall_risk_score = 0
        else:
            # Weighted risk calculation
            critical_risk_score = min(1.0, (high_risk_count * 0.3 + medium_risk_count * 0.1) / max(1, total_words / 100))
            overall_risk_score = min(1.0, (high_risk_count * 0.2 + medium_risk_count * 0.1 + low_risk_count * 0.05) / max(1, total_words / 100))
        
        # Detect specific risk categories
        risk_categories = self._detect_risk_categories(text_lower, tokens_lower)
        
        # Determine risk levels
        critical_risk_level = risk_level_from_score(critical_risk_score)
        overall_risk_level = risk_level_from_score(overall_risk_score)
        
        return {
            'critical_risk': {
                'score': critical_risk_score,
                'level': critical_risk_level,
                'count': high_risk_count
            },
            'overall_risk': {
                'score': overall_risk_score,
                'level': overall_risk_level
            },
            'risk_categories': risk_categories,
            'keyword_counts': {
                'high': high_risk_count,
                'medium': medium_risk_count,
                'low': low_risk_count
            }
        }
    
    def _detect_risk_categories(self, text_lower: str, tokens_lower: List[str]) -> Dict[str, float]:
        """Detect specific risk categories in text"""
        
        categories = {
            RiskCategory.VIOLENCE: {
                'keywords': ['violence', 'violent', 'attack', 'assault', 'fight', 'weapon', 'gun', 'knife', 'bomb'],
                'patterns': [r'\b(kill|hurt|harm|attack)\s+(someone|people|them|him|her)\b']
            },
            RiskCategory.SELF_HARM: {
                'keywords': ['suicide', 'suicidal', 'self-harm', 'cut myself', 'end my life', 'kill myself'],
                'patterns': [r'\b(want|going)\s+to\s+(die|kill\s+myself)\b']
            },
            RiskCategory.HATE_SPEECH: {
                'keywords': ['hate', 'racist', 'racism', 'discrimination', 'bigot'],
                'patterns': [r'\b(hate|despise)\s+(all|every)\s+\w+\b']
            },
            RiskCategory.HARASSMENT: {
                'keywords': ['harass', 'bully', 'threaten', 'stalk', 'intimidate'],
                'patterns': [r'\b(going\s+to|will)\s+(find|get|hurt)\s+you\b']
            }
        }
        
        detected_categories = {}
        
        for category, config in categories.items():
            score = 0
            
            # Check keywords
            keyword_matches = sum(1 for keyword in config['keywords'] if keyword in text_lower)
            score += keyword_matches * 0.2
            
            # Check patterns
            for pattern in config['patterns']:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    score += 0.3
            
            if score > 0:
                detected_categories[category.value] = min(1.0, score)
        
        return detected_categories
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of the text"""
        try:
            # Use TextBlob for sentiment analysis
            blob = TextBlob(text)
            
            # Get polarity and subjectivity
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Calculate sentiment scores
            if polarity > 0.1:
                positive_score = min(1.0, polarity)
                negative_score = 0
                neutral_score = 1 - positive_score
            elif polarity < -0.1:
                positive_score = 0
                negative_score = min(1.0, abs(polarity))
                neutral_score = 1 - negative_score
            else:
                positive_score = 0.1
                negative_score = 0.1
                neutral_score = 0.8
            
            # Determine sentiment type
            sentiment_type = sentiment_type_from_scores(positive_score, negative_score, neutral_score)
            
            # Extract emotional keywords
            emotional_keywords = self._extract_emotional_keywords(text)
            
            return {
                'score': polarity,
                'type': sentiment_type.value,
                'positive_score': positive_score,
                'negative_score': negative_score,
                'neutral_score': neutral_score,
                'subjectivity': subjectivity,
                'keywords': emotional_keywords
            }
            
        except Exception as e:
            # Return neutral sentiment on error
            return {
                'score': 0,
                'type': SentimentType.NEUTRAL.value,
                'positive_score': 0,
                'negative_score': 0,
                'neutral_score': 1.0,
                'subjectivity': 0,
                'keywords': {'positive': [], 'negative': []}
            }
    
    def _extract_emotional_keywords(self, text: str) -> Dict[str, List[str]]:
        """Extract emotional keywords from text"""
        positive_words = ['happy', 'joy', 'love', 'excellent', 'good', 'wonderful', 'fantastic', 'amazing']
        negative_words = ['sad', 'angry', 'hate', 'terrible', 'awful', 'horrible', 'bad', 'worst']
        
        text_lower = text.lower()
        words = text_lower.split()
        
        found_positive = [word for word in positive_words if word in words]
        found_negative = [word for word in negative_words if word in words]
        
        return {
            'positive': found_positive,
            'negative': found_negative
        }
    
    def _generate_statistics(self, processed_text: Dict[str, Any], risk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text statistics"""
        return {
            'word_count': len(processed_text['tokens']),
            'character_count': len(processed_text['original']),
            'sentence_count': len(processed_text['sentences']),
            'avg_word_length': np.mean([len(token) for token in processed_text['tokens']]) if processed_text['tokens'] else 0,
            'high_risk_keywords': risk_analysis['keyword_counts']['high'],
            'medium_risk_keywords': risk_analysis['keyword_counts']['medium']
        }
    
    def _generate_risk_assessments(self, processed_text: Dict[str, Any], risk_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate detailed risk assessments"""
        assessments = []
        
        # Add assessments for detected risk categories
        for category, score in risk_analysis['risk_categories'].items():
            level = risk_level_from_score(score)
            assessment = {
                'category': category,
                'level': level.value,
                'score': score,
                'confidence': min(0.9, score + 0.3),  # Simplified confidence calculation
                'keywords': [],  # Would be populated with actual detected keywords
                'context': None
            }
            assessments.append(assessment)
        
        # Add overall risk assessment if significant
        if risk_analysis['critical_risk']['score'] > 0.3:
            assessments.append({
                'category': 'overall',
                'level': risk_analysis['critical_risk']['level'],
                'score': risk_analysis['critical_risk']['score'],
                'confidence': 0.8,
                'keywords': [],
                'context': 'Multiple risk factors detected'
            })
        
        return assessments
    
    def _generate_recommendations(self, risk_analysis: Dict[str, Any], sentiment_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Critical risk recommendations
        if risk_analysis['critical_risk']['level'] in ['high', 'critical']:
            recommendations.append({
                'type': 'critical',
                'message': 'High risk content detected. Immediate review required.',
                'action': 'Flag for immediate human review and potential intervention',
                'priority': 'critical'
            })
        
        # Sentiment-based recommendations
        if sentiment_analysis['score'] < -0.5:
            recommendations.append({
                'type': 'sentiment',
                'message': 'Highly negative sentiment detected.',
                'action': 'Consider providing emotional support resources',
                'priority': 'high'
            })
        
        # Category-specific recommendations
        if RiskCategory.SELF_HARM.value in risk_analysis['risk_categories']:
            recommendations.append({
                'type': 'intervention',
                'message': 'Self-harm indicators detected.',
                'action': 'Provide crisis helpline information and support resources',
                'priority': 'critical'
            })
        
        if RiskCategory.VIOLENCE.value in risk_analysis['risk_categories']:
            recommendations.append({
                'type': 'safety',
                'message': 'Violence-related content detected.',
                'action': 'Review for potential threats and take appropriate action',
                'priority': 'high'
            })
        
        # Default recommendation if no issues
        if not recommendations:
            recommendations.append({
                'type': 'safe',
                'message': 'No significant risks detected.',
                'action': 'Continue normal monitoring',
                'priority': 'low'
            })
        
        return recommendations