"""
Guardian LLM - Risk Analyzers
Specialized analyzers for different types of risks
"""

import re
from typing import Dict, List, Tuple
from backend.models.schemas import RiskCategory
import logging

logger = logging.getLogger(__name__)


class KeywordAnalyzer:
    """Keyword-based risk analysis"""
    
    def __init__(self):
        self.risk_keywords = {
            RiskCategory.VIOLENCE: [
                'violence', 'violent', 'attack', 'assault', 'fight', 'weapon', 'gun', 'knife', 
                'bomb', 'kill', 'murder', 'hurt', 'harm', 'abuse', 'beat', 'punch', 'stab'
            ],
            RiskCategory.SELF_HARM: [
                'suicide', 'suicidal', 'self-harm', 'self harm', 'cut myself', 'end my life', 
                'kill myself', 'want to die', 'better off dead', 'overdose', 'slit wrists'
            ],
            RiskCategory.HATE_SPEECH: [
                'hate', 'racist', 'racism', 'discrimination', 'bigot', 'prejudice', 
                'supremacist', 'nazi', 'slur', 'derogatory', 'offensive'
            ],
            RiskCategory.HARASSMENT: [
                'harass', 'bully', 'threaten', 'stalk', 'intimidate', 'torment', 
                'humiliate', 'shame', 'doxx', 'blackmail', 'extort'
            ],
            RiskCategory.ADULT_CONTENT: [
                'sexual', 'explicit', 'nude', 'pornographic', 'adult content', 'nsfw',
                'erotic', 'xxx', 'sex', 'fetish'
            ],
            RiskCategory.MISINFORMATION: [
                'fake news', 'conspiracy', 'hoax', 'false claim', 'misinformation',
                'disinformation', 'propaganda', 'misleading', 'fabricated'
            ],
            RiskCategory.SPAM: [
                'spam', 'scam', 'phishing', 'clickbait', 'promotional', 'advertisement',
                'buy now', 'free money', 'winner', 'congratulations', 'click here'
            ]
        }
        
        self.high_risk_patterns = [
            r'going to kill',
            r'want to die',
            r'end it all',
            r'shoot up',
            r'bomb threat',
            r'death threat',
            r'kill myself',
            r'self harm'
        ]
    
    def analyze(self, text: str) -> Dict[RiskCategory, float]:
        """Analyze text for risk keywords"""
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in self.risk_keywords.items():
            score = 0.0
            keyword_matches = 0
            
            # Check individual keywords
            for keyword in keywords:
                if keyword in text_lower:
                    score += 0.5
                    keyword_matches += 1
            
            # Check high-risk patterns
            for pattern in self.high_risk_patterns:
                if re.search(pattern, text_lower):
                    score += 2.0
            
            # Normalize score to 0-1 range
            scores[category] = min(score / 10.0, 1.0)
        
        return scores


class ContextAnalyzer:
    """Analyze context and severity of risks"""
    
    def __init__(self):
        self.severity_modifiers = {
            'immediate': 2.0,
            'planning': 1.5,
            'going to': 1.5,
            'will': 1.3,
            'want to': 1.2,
            'thinking about': 1.1,
            'might': 0.8,
            'could': 0.8
        }
        
        self.mitigation_terms = [
            'help', 'support', 'therapy', 'counseling', 'treatment',
            'recovery', 'healing', 'better', 'improve', 'cope'
        ]
    
    def analyze_context(self, text: str, base_scores: Dict[RiskCategory, float]) -> Dict[RiskCategory, float]:
        """Adjust risk scores based on context"""
        text_lower = text.lower()
        adjusted_scores = base_scores.copy()
        
        # Check for severity modifiers
        for modifier, multiplier in self.severity_modifiers.items():
            if modifier in text_lower:
                for category in adjusted_scores:
                    if adjusted_scores[category] > 0:
                        adjusted_scores[category] = min(adjusted_scores[category] * multiplier, 1.0)
        
        # Check for mitigation terms (reduce score if discussing help/recovery)
        mitigation_count = sum(1 for term in self.mitigation_terms if term in text_lower)
        if mitigation_count > 0:
            mitigation_factor = 0.8 ** mitigation_count  # Reduce by 20% for each mitigation term
            for category in adjusted_scores:
                if category in [RiskCategory.SELF_HARM, RiskCategory.VIOLENCE]:
                    adjusted_scores[category] *= mitigation_factor
        
        return adjusted_scores


class PatternAnalyzer:
    """Analyze patterns and combinations of risks"""
    
    def __init__(self):
        self.dangerous_combinations = [
            ([RiskCategory.SELF_HARM, RiskCategory.VIOLENCE], 1.5),
            ([RiskCategory.HATE_SPEECH, RiskCategory.HARASSMENT], 1.3),
            ([RiskCategory.VIOLENCE, RiskCategory.HATE_SPEECH], 1.4)
        ]
    
    def analyze_patterns(self, scores: Dict[RiskCategory, float]) -> Dict[str, any]:
        """Analyze patterns in risk scores"""
        results = {
            'combined_risk': 0.0,
            'risk_pattern': None,
            'escalation_risk': False
        }
        
        # Check for dangerous combinations
        for categories, multiplier in self.dangerous_combinations:
            if all(scores.get(cat, 0) > 0.3 for cat in categories):
                results['combined_risk'] = max(results['combined_risk'], 
                                             max(scores[cat] for cat in categories) * multiplier)
                results['risk_pattern'] = [cat.value for cat in categories]
        
        # Check for escalation risk (multiple high scores)
        high_risk_count = sum(1 for score in scores.values() if score > 0.6)
        if high_risk_count >= 2:
            results['escalation_risk'] = True
        
        return results


class RiskAnalyzer:
    """Main risk analyzer that combines all analysis methods"""
    
    def __init__(self):
        """Initialize all analyzer components"""
        self.keyword_analyzer = KeywordAnalyzer()
        self.context_analyzer = ContextAnalyzer()
        self.pattern_analyzer = PatternAnalyzer()
    
    def analyze_text(self, text: str) -> Dict[str, any]:
        """
        Analyze text for various risk factors
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing risk analysis results
        """
        results = {
            'risk_scores': {},
            'adjusted_scores': {},
            'patterns': {},
            'high_risk_categories': [],
            'critical_categories': [],
            'overall_risk': 0.0,
            'risk_level': 'low'
        }
        
        # Step 1: Keyword-based analysis
        keyword_scores = self.keyword_analyzer.analyze(text)
        results['risk_scores'] = {cat.value: score for cat, score in keyword_scores.items()}
        
        # Step 2: Context analysis
        adjusted_scores = self.context_analyzer.analyze_context(text, keyword_scores)
        results['adjusted_scores'] = {cat.value: score for cat, score in adjusted_scores.items()}
        
        # Step 3: Pattern analysis
        pattern_results = self.pattern_analyzer.analyze_patterns(adjusted_scores)
        results['patterns'] = pattern_results
        
        # Step 4: Identify high-risk categories
        for category, score in adjusted_scores.items():
            if score > 0.6:
                results['high_risk_categories'].append(category.value)
            if score > 0.8:
                results['critical_categories'].append(category.value)
        
        # Step 5: Calculate overall risk
        if results['critical_categories']:
            results['overall_risk'] = max(adjusted_scores.values())
            results['risk_level'] = 'critical'
        elif results['high_risk_categories']:
            results['overall_risk'] = max(adjusted_scores.values())
            results['risk_level'] = 'high'
        elif any(score > 0.3 for score in adjusted_scores.values()):
            results['overall_risk'] = max(adjusted_scores.values())
            results['risk_level'] = 'medium'
        else:
            results['overall_risk'] = max(adjusted_scores.values()) if adjusted_scores else 0.0
            results['risk_level'] = 'low'
        
        # Add combined risk from patterns
        if pattern_results['combined_risk'] > results['overall_risk']:
            results['overall_risk'] = min(pattern_results['combined_risk'], 1.0)
            if results['overall_risk'] > 0.8:
                results['risk_level'] = 'critical'
            elif results['overall_risk'] > 0.6:
                results['risk_level'] = 'high'
        
        return results
    
    def get_risk_evidence(self, text: str, category: RiskCategory) -> List[str]:
        """Get specific evidence for a risk category"""
        evidence = []
        text_lower = text.lower()
        
        if category in self.keyword_analyzer.risk_keywords:
            keywords = self.keyword_analyzer.risk_keywords[category]
            for keyword in keywords:
                if keyword in text_lower:
                    # Find context around keyword
                    index = text_lower.find(keyword)
                    start = max(0, index - 30)
                    end = min(len(text), index + len(keyword) + 30)
                    context = text[start:end]
                    if start > 0:
                        context = "..." + context
                    if end < len(text):
                        context = context + "..."
                    evidence.append(f"Found '{keyword}': {context}")
        
        return evidence