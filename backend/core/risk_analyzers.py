import re
import torch
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple
from backend.models.schemas import RiskCategory
import logging

logger = logging.getLogger(__name__)

class KeywordAnalyzer:
    """Keyword-based risk analysis"""
    
    def __init__(self):
        self.risk_keywords = {
            RiskCategory.BIAS_FAIRNESS: [
                'bias', 'fairness', 'discrimination', 'demographic', 'gender', 'race', 'ethnicity',
                'inequality', 'disparity', 'prejudice', 'stereotyp', 'unfair', 'equit', 'inclusive'
            ],
            RiskCategory.PRIVACY_DATA: [
                'privacy', 'personal data', 'gdpr', 'consent', 'anonymiz', 'pseudonym',
                'confidential', 'sensitive data', 'pii', 'data protection', 'surveillance',
                'biometric', 'facial recognition', 'tracking'
            ],
            RiskCategory.SAFETY_SECURITY: [
                'safety', 'security', 'vulnerability', 'attack', 'adversarial', 'robust',
                'failure', 'risk', 'harm', 'dangerous', 'malicious', 'threat', 'breach'
            ],
            RiskCategory.DUAL_USE: [
                'military', 'weapon', 'warfare', 'surveillance', 'reconnaissance', 'dual-use',
                'defense', 'combat', 'targeting', 'autonomous weapon', 'drone', 'lethal'
            ],
            RiskCategory.SOCIETAL_IMPACT: [
                'employment', 'job', 'social', 'economic impact', 'displacement', 'automation',
                'democracy', 'political', 'manipulation', 'influence', 'society', 'inequality'
            ],
            RiskCategory.TRANSPARENCY: [
                'explainab', 'interpretab', 'black box', 'opacity', 'transparent', 'accountab',
                'auditable', 'understandab', 'explain', 'interpret', 'opaque'
            ]
        }
        
        self.high_risk_patterns = [
            r'facial recognition',
            r'emotion detection',
            r'behavioral prediction',
            r'social credit',
            r'mass surveillance',
            r'autonomous weapon',
            r'deepfake',
            r'synthetic media',
            r'predictive policing'
        ]
    
    def analyze(self, text: str) -> Dict[RiskCategory, float]:
        """Analyze text for ethical risk keywords"""
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in self.risk_keywords.items():
            score = 0.0
            keyword_matches = 0
            
            # Check individual keywords
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword)
                matches = len(re.findall(pattern, text_lower))
                if matches > 0:
                    score += min(matches * 0.5, 2.0)  # Cap individual keyword contribution
                    keyword_matches += 1
            
            # Check high-risk patterns
            for pattern in self.high_risk_patterns:
                if re.search(pattern, text_lower):
                    score += 3.0
            
            # Bonus for multiple keyword types in same category
            if keyword_matches > 3:
                score += 1.0
            
            # Normalize score to 0-10 range
            scores[category] = min(score, 10.0)
        
        return scores

class BiasAnalyzer:
    """Specialized bias detection"""
    
    def __init__(self):
        self.bias_indicators = [
            'underrepresented', 'overrepresented', 'demographic bias',
            'gender bias', 'racial bias', 'algorithmic discrimination',
            'unfair treatment', 'disparate impact', 'protected class',
            'algorithmic fairness', 'equitable', 'inclusive design'
        ]
        
        self.demographic_terms = [
            'gender', 'race', 'ethnicity', 'age', 'disability', 'religion',
            'sexual orientation', 'nationality', 'socioeconomic', 'cultural'
        ]
        
        self.fairness_terms = [
            'bias', 'fair', 'equitable', 'discrimination', 'inclusive',
            'diverse', 'representative', 'balanced'
        ]
    
    def analyze(self, text: str) -> Tuple[float, List[str]]:
        """Detect bias-related risks"""
        text_lower = text.lower()
        evidence = []
        score = 0.0
        
        # Check for explicit bias indicators
        for indicator in self.bias_indicators:
            if indicator in text_lower:
                evidence.append(f"Bias indicator found: '{indicator}'")
                score += 1.5
        
        # Check demographic mentions vs fairness considerations
        demo_count = sum(1 for term in self.demographic_terms if term in text_lower)
        fairness_count = sum(1 for term in self.fairness_terms if term in text_lower)
        
        if demo_count >= 2:
            if fairness_count == 0:
                evidence.append("Multiple demographic groups mentioned without fairness considerations")
                score += 3.0
            elif fairness_count < demo_count / 2:
                evidence.append("Limited fairness discussion relative to demographic complexity")
                score += 1.5
        
        # Check for dataset bias indicators
        dataset_patterns = [
            r'dataset.*bias', r'training.*data.*bias', r'sample.*bias',
            r'selection.*bias', r'representation.*gap'
        ]
        
        for pattern in dataset_patterns:
            if re.search(pattern, text_lower):
                evidence.append("Dataset bias considerations mentioned")
                score += 1.0
        
        return min(score, 10.0), evidence

class PrivacyAnalyzer:
    """Privacy and data protection analysis"""
    
    def __init__(self):
        self.privacy_risks = [
            'personal information', 'sensitive data', 'biometric data',
            'location data', 'tracking', 'profiling', 'identification',
            'facial recognition', 'voice recognition', 'behavioral data'
        ]
        
        self.data_collection_terms = [
            'collect', 'gather', 'acquire', 'obtain', 'scrape',
            'harvest', 'mine', 'extract', 'capture'
        ]
        
        self.consent_terms = [
            'consent', 'permission', 'authorization', 'gdpr', 'ccpa',
            'opt-in', 'opt-out', 'privacy policy', 'terms of service'
        ]
    
    def analyze(self, text: str) -> Tuple[float, List[str]]:
        """Analyze privacy risks"""
        text_lower = text.lower()
        evidence = []
        score = 0.0
        
        # Check for privacy risk indicators
        for risk in self.privacy_risks:
            if risk in text_lower:
                evidence.append(f"Privacy risk detected: '{risk}'")
                score += 1.5
        
        # Check data collection vs consent
        has_data_collection = any(term in text_lower for term in self.data_collection_terms)
        has_consent_mention = any(term in text_lower for term in self.consent_terms)
        
        if has_data_collection:
            if not has_consent_mention:
                evidence.append("Data collection mentioned without consent procedures")
                score += 2.5
            else:
                evidence.append("Data collection with consent considerations")
                score += 0.5
        
        # Check for specific privacy violations
        violation_patterns = [
            r'without.*consent', r'involuntary.*data', r'covert.*collection',
            r'unauthorized.*access', r'data.*breach'
        ]
        
        for pattern in violation_patterns:
            if re.search(pattern, text_lower):
                evidence.append("Potential privacy violation detected")
                score += 3.0
        
        return min(score, 10.0), evidence

class TransformerAnalyzer:
    """Advanced transformer-based analysis"""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            logger.info(f"Loaded transformer model: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load transformer model: {e}")
    
    def analyze(self, text: str) -> Dict[str, float]:
        """Classify text using transformer models"""
        if not self.tokenizer or not self.model:
            return {category.value: 0.0 for category in RiskCategory}
        
        try:
            # Chunk text for processing
            max_length = 512
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            
            risk_scores = {category.value: 0.0 for category in RiskCategory}
            
            for chunk in chunks[:3]:  # Process first 3 chunks
                inputs = self.tokenizer(
                    chunk, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True, 
                    max_length=512
                )
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Extract embeddings and compute risk scores
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
                # Simple risk scoring (in production, use trained classifiers)
                for i, category in enumerate(RiskCategory):
                    idx = i % embeddings.shape[1]
                    risk_scores[category.value] += float(embeddings[0][idx].abs())
            
            # Normalize scores
            for category in risk_scores:
                risk_scores[category] = min(abs(risk_scores[category]) * 2, 10.0)
            
            return risk_scores
            
        except Exception as e:
            logger.error(f"Transformer analysis error: {e}")
            return {category.value: 0.0 for category in RiskCategory}