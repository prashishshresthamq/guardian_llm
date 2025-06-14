# core/semantic_analyzer.py
"""
Guardian LLM - SVD-based Semantic Risk Analyzer
Implements dimensionality reduction for semantic risk pattern detection
Based on COMP8420 Week 11 - Matrix and Vector in NLP
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SemanticRiskAnalyzer:
    """
    Semantic risk analyzer using SVD for dimensionality reduction
    and pattern detection in latent semantic space
    """
    
    def __init__(self, n_components: int = 10, risk_threshold: float = 0.7):  
        """
        Initialize semantic analyzer
        
        Args:
            n_components: Number of latent dimensions for SVD
            risk_threshold: Threshold for risk detection
        """
        self.n_components = n_components
        self.risk_threshold = risk_threshold
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        
        # Risk concept vectors (will be populated during training)
        self.risk_concepts = {}
        self.is_fitted = False
        
        # Predefined risk patterns for each category
        self.risk_patterns = {
            'bias_fairness': [
                "algorithmic bias discrimination",
                "unfair treatment demographic groups",
                "biased decision making",
                "discriminatory outcomes",
                "fairness violation"
            ],
            'privacy_data': [
                "privacy violation personal data",
                "unauthorized data collection",
                "surveillance without consent",
                "data breach exposure",
                "personal information misuse"
            ],
            'safety_security': [
                "system vulnerability exploit",
                "security breach attack",
                "safety critical failure",
                "malicious code injection",
                "unauthorized access control"
            ],
            'dual_use': [
                "military application weapon",
                "dual use technology misuse",
                "weaponization potential harm",
                "malicious repurposing",
                "harmful application"
            ],
            'societal_impact': [
                "job displacement automation",
                "social inequality amplification",
                "community harm disruption",
                "economic disadvantage",
                "societal disruption"
            ],
            'transparency': [
                "black box unexplainable",
                "lack transparency accountability",
                "opaque decision process",
                "uninterpretable model",
                "accountability deficit"
            ]
        }
    
    def fit(self, training_texts: List[str], risk_labels: Optional[Dict[str, List[int]]] = None):
        """
        Fit the semantic analyzer on training data
        
        Args:
            training_texts: List of training documents
            risk_labels: Optional risk labels for supervised learning
        """
        logger.info(f"Fitting semantic analyzer on {len(training_texts)} documents")
        
        # Create term-document matrix
        X = self.vectorizer.fit_transform(training_texts)
        
        # Apply SVD for dimensionality reduction
        X_reduced = self.svd.fit_transform(X)
        
        # Normalize for cosine similarity
        X_reduced = normalize(X_reduced, norm='l2')
        
        # Learn risk concept vectors
        self._learn_risk_concepts(X_reduced, risk_labels)
        
        self.is_fitted = True
        logger.info(f"Semantic analyzer fitted with {self.n_components} components")
    
    def analyze_semantic_risk(self, text: str) -> Dict[str, float]:
        """
        Analyze semantic risk using SVD-based approach
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary of risk scores for each category
        """
        if not self.is_fitted:
            # Use pre-trained patterns if model not fitted
            return self._analyze_with_patterns(text)
        
        # Transform text to term-document matrix
        X = self.vectorizer.transform([text])
        
        # Project to latent semantic space
        X_reduced = self.svd.transform(X)
        X_reduced = normalize(X_reduced, norm='l2')
        
        # Calculate similarity to risk concepts
        risk_scores = {}
        for category, concept_vector in self.risk_concepts.items():
            similarity = 1 - cosine(X_reduced[0], concept_vector)
            risk_scores[category] = max(0, min(1, similarity))
        
        return risk_scores
    
    def _learn_risk_concepts(self, X_reduced: np.ndarray, risk_labels: Optional[Dict[str, List[int]]]):
        """
        Learn risk concept vectors from training data
        
        Args:
            X_reduced: Reduced document matrix
            risk_labels: Risk labels for documents
        """
        # If no labels provided, use pattern-based initialization
        if risk_labels is None:
            self._initialize_with_patterns()
            return
        
        # Learn concept vectors from labeled data
        for category, document_indices in risk_labels.items():
            if document_indices:
                # Average vectors of documents with this risk
                concept_vector = np.mean(X_reduced[document_indices], axis=0)
                self.risk_concepts[category] = normalize(concept_vector.reshape(1, -1))[0]
    
    def _initialize_with_patterns(self):
        """Initialize risk concepts using predefined patterns"""
        all_patterns = []
        pattern_to_category = {}
        
        # Collect all patterns
        for category, patterns in self.risk_patterns.items():
            for pattern in patterns:
                all_patterns.append(pattern)
                pattern_to_category[len(all_patterns) - 1] = category
        
        # Fit on patterns if not already fitted
        if not self.is_fitted:
            X = self.vectorizer.fit_transform(all_patterns)
            
            # Adaptive n_components based on available features
            n_features = X.shape[1]
            adaptive_components = min(self.n_components, n_features - 1)
            
            self.svd = TruncatedSVD(n_components=adaptive_components, random_state=42)
            X_reduced = self.svd.fit_transform(X)
        else:
            X = self.vectorizer.transform(all_patterns)
            X_reduced = self.svd.transform(X)
        
        X_reduced = normalize(X_reduced, norm='l2')
        
        # Average patterns for each category
        for category in self.risk_patterns:
            indices = [i for i, cat in pattern_to_category.items() if cat == category]
            if indices:
                self.risk_concepts[category] = np.mean(X_reduced[indices], axis=0)
    
    def _analyze_with_patterns(self, text: str) -> Dict[str, float]:
        """Fallback analysis using pattern matching"""
        risk_scores = {}
        text_lower = text.lower()
        
        for category, patterns in self.risk_patterns.items():
            score = 0.0
            for pattern in patterns:
                # Simple keyword matching
                keywords = pattern.split()
                matches = sum(1 for keyword in keywords if keyword in text_lower)
                score += matches / len(keywords)
            
            # Normalize score
            risk_scores[category] = min(1.0, score / len(patterns))
        
        return risk_scores
    
    def extract_semantic_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract semantic features using SVD
        
        Args:
            texts: List of texts
            
        Returns:
            Feature matrix of shape (n_texts, n_components)
        """
        X = self.vectorizer.transform(texts)
        X_reduced = self.svd.transform(X)
        return normalize(X_reduced, norm='l2')
    
    def find_similar_risks(self, text: str, risk_database: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Find similar risk patterns from database
        
        Args:
            text: Query text
            risk_database: Database of known risks
            top_k: Number of similar risks to return
            
        Returns:
            List of similar risk cases
        """
        # Extract query features
        query_features = self.extract_semantic_features([text])[0]
        
        # Extract database features
        db_texts = [item['text'] for item in risk_database]
        db_features = self.extract_semantic_features(db_texts)
        
        # Calculate similarities
        similarities = []
        for i, features in enumerate(db_features):
            sim = 1 - cosine(query_features, features)
            similarities.append((sim, risk_database[i]))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        return [item[1] for item in similarities[:top_k]]
    
    def explain_risk_detection(self, text: str, category: str) -> Dict[str, any]:
        """
        Explain why a text was flagged for a specific risk
        
        Args:
            text: Input text
            category: Risk category
            
        Returns:
            Explanation dictionary
        """
        # Get important terms
        X = self.vectorizer.transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get term scores
        tfidf_scores = X.toarray()[0]
        important_terms = []
        
        for idx in np.argsort(tfidf_scores)[-10:][::-1]:
            if tfidf_scores[idx] > 0:
                important_terms.append({
                    'term': feature_names[idx],
                    'score': float(tfidf_scores[idx])
                })
        
        # Project to latent space
        X_reduced = self.svd.transform(X)
        
        # Get most influential latent dimensions
        influential_dims = np.argsort(np.abs(X_reduced[0]))[-5:][::-1]
        
        explanation = {
            'category': category,
            'important_terms': important_terms,
            'influential_dimensions': influential_dims.tolist(),
            'semantic_similarity': float(self.analyze_semantic_risk(text).get(category, 0))
        }
        
        return explanation
    
    def get_latent_topics(self, n_topics: int = 10) -> List[Dict[str, List[str]]]:
        """
        Extract latent topics from SVD components
        
        Args:
            n_topics: Number of topics to extract
            
        Returns:
            List of topics with top terms
        """
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        
        for i in range(min(n_topics, self.n_components)):
            # Get top terms for this component
            component = self.svd.components_[i]
            top_indices = np.argsort(np.abs(component))[-10:][::-1]
            
            top_terms = [feature_names[idx] for idx in top_indices]
            topics.append({
                'topic_id': i,
                'terms': top_terms,
                'variance_explained': self.svd.explained_variance_ratio_[i]
            })
        
        return topics


class SemanticRiskIntegrator:
    """
    Integrates semantic risk analysis with Guardian Engine
    """
    
    def __init__(self, semantic_analyzer: SemanticRiskAnalyzer):
        self.semantic_analyzer = semantic_analyzer
    
    def enhance_risk_analysis(self, text: str, traditional_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Enhance traditional risk scores with semantic analysis
        
        Args:
            text: Input text
            traditional_scores: Scores from traditional analysis
            
        Returns:
            Enhanced risk scores
        """
        # Get semantic risk scores
        semantic_scores = self.semantic_analyzer.analyze_semantic_risk(text)
        
        # Combine scores (weighted average)
        enhanced_scores = {}
        semantic_weight = 0.4  # 40% semantic, 60% traditional
        
        for category in traditional_scores:
            trad_score = traditional_scores.get(category, 0)
            sem_score = semantic_scores.get(category, 0)
            
            # Weighted combination
            enhanced_scores[category] = (
                (1 - semantic_weight) * trad_score + 
                semantic_weight * sem_score
            )
            
            # Boost score if both methods agree on high risk
            if trad_score > 0.6 and sem_score > 0.6:
                enhanced_scores[category] = min(1.0, enhanced_scores[category] * 1.2)
        
        return enhanced_scores
    
    def generate_semantic_evidence(self, text: str, category: str) -> List[str]:
        """
        Generate evidence based on semantic analysis
        
        Args:
            text: Input text
            category: Risk category
            
        Returns:
            List of evidence statements
        """
        explanation = self.semantic_analyzer.explain_risk_detection(text, category)
        evidence = []
        
        # Add term-based evidence
        top_terms = explanation['important_terms'][:3]
        if top_terms:
            terms_str = ', '.join([t['term'] for t in top_terms])
            evidence.append(f"Semantic analysis detected risk-related terms: {terms_str}")
        
        # Add similarity-based evidence
        similarity = explanation['semantic_similarity']
        if similarity > 0.7:
            evidence.append(f"High semantic similarity ({similarity:.2f}) to known {category} patterns")
        elif similarity > 0.5:
            evidence.append(f"Moderate semantic similarity ({similarity:.2f}) to {category} patterns")
        
        # Add latent dimension evidence
        if explanation['influential_dimensions']:
            evidence.append(f"Activated risk-relevant latent semantic dimensions")
        
        return evidence


# Utility functions for integration
def create_semantic_analyzer(training_corpus: Optional[List[str]] = None) -> SemanticRiskAnalyzer:
    """
    Create and initialize a semantic risk analyzer
    
    Args:
        training_corpus: Optional training documents
        
    Returns:
        Initialized SemanticRiskAnalyzer
    """
    analyzer = SemanticRiskAnalyzer(n_components=100)
    
    if training_corpus:
        analyzer.fit(training_corpus)
    else:
        # Initialize with patterns
        analyzer._initialize_with_patterns()
    
    return analyzer