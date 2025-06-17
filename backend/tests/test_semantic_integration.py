# tests/test_semantic_integration.py
"""
Test the SVD-based semantic analysis integration
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.semantic_analyzer import SemanticRiskAnalyzer, SemanticRiskIntegrator
from core.guardian_engine import GuardianEngine


class TestSemanticIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = SemanticRiskAnalyzer(n_components=50)
        self.integrator = SemanticRiskIntegrator(self.analyzer)
        
        # Initialize with patterns for testing
        self.analyzer._initialize_with_patterns()
    
    def test_svd_risk_detection(self):
        """Test SVD-based risk detection"""
        test_cases = [
            {
                'text': "The hiring algorithm shows systematic bias against women candidates",
                'expected_category': 'bias_fairness',
                'min_score': 0.3
            },
            {
                'text': "Collecting user location data without explicit consent violates privacy",
                'expected_category': 'privacy_data',
                'min_score': 0.3
            },
            {
                'text': "Security vulnerabilities could lead to unauthorized system access",
                'expected_category': 'safety_security',
                'min_score': 0.3
            }
        ]
        
        for case in test_cases:
            scores = self.analyzer.analyze_semantic_risk(case['text'])
            
            # Check that expected category has high score
            self.assertGreater(
                scores[case['expected_category']], 
                case['min_score'],
                f"Expected {case['expected_category']} score > {case['min_score']} for: {case['text'][:50]}..."
            )
            
            # Check that scores are normalized
            for score in scores.values():
                self.assertGreaterEqual(score, 0)
                self.assertLessEqual(score, 1)
    
    def test_semantic_feature_extraction(self):
        """Test semantic feature extraction"""
        texts = [
            "AI bias in decision making",
            "Privacy violation concerns",
            "Safety critical system failure"
        ]
        
        features = self.analyzer.extract_semantic_features(texts)
        
        # Check feature matrix shape
        self.assertEqual(features.shape[0], len(texts))
        self.assertEqual(features.shape[1], self.analyzer.n_components)
        
        # Check features are normalized
        norms = np.linalg.norm(features, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(len(texts)), decimal=5)
    
    def test_risk_score_enhancement(self):
        """Test enhancement of traditional scores with semantic analysis"""
        text = "The system exhibits algorithmic bias and lacks transparency"
        
        # Mock traditional scores
        traditional_scores = {
            'bias_fairness': 0.5,
            'privacy_data': 0.1,
            'transparency': 0.4,
            'safety_security': 0.0
        }
        
        # Enhance with semantic analysis
        enhanced_scores = self.integrator.enhance_risk_analysis(text, traditional_scores)
        
        # Check that bias and transparency scores are boosted
        self.assertGreater(enhanced_scores['bias_fairness'], traditional_scores['bias_fairness'])
        self.assertGreater(enhanced_scores['transparency'], traditional_scores['transparency'])
        
        # Check all scores are valid
        for score in enhanced_scores.values():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)
    
    def test_semantic_evidence_generation(self):
        """Test semantic evidence generation"""
        text = "Personal health data is being collected and sold to third parties without user consent"
        category = 'privacy_data'
        
        evidence = self.integrator.generate_semantic_evidence(text, category)
        
        # Check evidence is generated
        self.assertIsInstance(evidence, list)
        self.assertGreater(len(evidence), 0)
        
        # Check evidence mentions semantic analysis
        evidence_text = ' '.join(evidence).lower()
        self.assertIn('semantic', evidence_text)
    
    def test_latent_topic_extraction(self):
        """Test latent topic extraction"""
        # First, fit the analyzer with some data
        training_texts = [
            "AI bias discrimination unfair treatment",
            "privacy data collection surveillance tracking",
            "security vulnerability exploit attack risk",
            "transparency explainability accountability black box",
            "automation job displacement economic impact"
        ]
        
        self.analyzer.fit(training_texts)
        
        # Extract topics
        topics = self.analyzer.get_latent_topics(n_topics=3)
        
        # Check topic structure
        self.assertEqual(len(topics), 3)
        for topic in topics:
            self.assertIn('topic_id', topic)
            self.assertIn('terms', topic)
            self.assertIn('variance_explained', topic)
            self.assertIsInstance(topic['terms'], list)
            self.assertGreater(len(topic['terms']), 0)
    
    def test_similarity_search(self):
        """Test finding similar risks"""
        # Mock risk database
        risk_database = [
            {'text': 'Facial recognition bias against minorities', 'category': 'bias_fairness'},
            {'text': 'Unauthorized data collection practices', 'category': 'privacy_data'},
            {'text': 'System vulnerability to injection attacks', 'category': 'safety_security'},
            {'text': 'Lack of model interpretability', 'category': 'transparency'},
            {'text': 'Potential weaponization of AI technology', 'category': 'dual_use'}
        ]
        
        # Query text
        query = "Algorithm shows racial bias in predictions"
        
        # Find similar risks
        similar = self.analyzer.find_similar_risks(query, risk_database, top_k=2)
        
        # Check results
        self.assertEqual(len(similar), 2)
        # Most similar should be bias-related
        self.assertEqual(similar[0]['category'], 'bias_fairness')
    
    def test_risk_explanation(self):
        """Test risk detection explanation"""
        text = "The black-box AI system makes decisions without any transparency or accountability"
        category = 'transparency'
        
        explanation = self.analyzer.explain_risk_detection(text, category)
        
        # Check explanation structure
        self.assertIn('category', explanation)
        self.assertIn('important_terms', explanation)
        self.assertIn('influential_dimensions', explanation)
        self.assertIn('semantic_similarity', explanation)
        
        # Check important terms include relevant keywords
        terms = [t['term'] for t in explanation['important_terms']]
        # At least one transparency-related term should be present
        transparency_terms = ['black', 'box', 'transparency', 'accountability', 'decisions']
        self.assertTrue(any(term in terms for term in transparency_terms))
    
    def test_guardian_engine_integration(self):
        """Test full integration with Guardian Engine"""
        # Initialize engine with semantic analyzer
        engine = GuardianEngine()
        
        # Test text with multiple risks
        text = """
        Our facial recognition system has shown significant bias against certain ethnic groups,
        while also collecting biometric data without proper consent. The lack of transparency
        in the decision-making process makes it impossible to audit for these issues.
        """
        
        # Analyze text
        results = engine.analyze_text(text)
        
        # Check that semantic analysis was performed
        self.assertIn('risk_analysis', results)
        if 'analysis_methods' in results['risk_analysis']:
            self.assertIn('semantic_svd', results['risk_analysis']['analysis_methods'])
        
        # Check risk categories detected
        risk_categories = results['risk_analysis']['risk_categories']
        self.assertGreater(risk_categories.get('bias_fairness', 0), 0)
        self.assertGreater(risk_categories.get('privacy_data', 0), 0)
        self.assertGreater(risk_categories.get('transparency', 0), 0)
        
        # Check semantic metadata if available
        if 'semantic_metadata' in results:
            self.assertIn('latent_topics', results['semantic_metadata'])


class TestSVDPerformance(unittest.TestCase):
    """Test performance aspects of SVD implementation"""
    
    def test_dimensionality_reduction(self):
        """Test that SVD effectively reduces dimensionality"""
        analyzer = SemanticRiskAnalyzer(n_components=10)
        
        # Create a larger corpus
        corpus = [
            f"This is document {i} about {category} risks and concerns"
            for i in range(100)
            for category in ['bias', 'privacy', 'security', 'transparency']
        ]
        
        # Fit the analyzer
        analyzer.fit(corpus)
        
        # Check dimensionality reduction
        features = analyzer.extract_semantic_features(corpus[:10])
        
        # Features should be in reduced space
        self.assertEqual(features.shape[1], 10)
        
        # Check explained variance
        total_variance = sum(analyzer.svd.explained_variance_ratio_)
        self.assertGreater(total_variance, 0)
        self.assertLessEqual(total_variance, 1)
    
    def test_semantic_similarity(self):
        """Test semantic similarity in latent space"""
        analyzer = SemanticRiskAnalyzer(n_components=20)
        analyzer._initialize_with_patterns()
        
        # Similar texts should have high similarity in latent space
        text1 = "The algorithm exhibits racial bias"
        text2 = "Discriminatory outcomes based on ethnicity"
        text3 = "Privacy violation through data collection"
        
        features = analyzer.extract_semantic_features([text1, text2, text3])
        
        # Calculate cosine similarities
        from scipy.spatial.distance import cosine
        sim_1_2 = 1 - cosine(features[0], features[1])  # Similar (both about bias)
        sim_1_3 = 1 - cosine(features[0], features[2])  # Different topics
        
        # Similar texts should have higher similarity
        self.assertGreater(sim_1_2, sim_1_3)


if __name__ == '__main__':
    unittest.main()