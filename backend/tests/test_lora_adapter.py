"""
Tests for LoRA adapter functionality
"""

import unittest
import numpy as np
from core.lora_adapter import LoRAAdapter, LoRAConfig


class TestLoRAAdapter(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = LoRAConfig(rank=8, alpha=16.0)
        self.adapter = LoRAAdapter(config=self.config)
        
    def test_initialization(self):
        """Test LoRA adapter initialization"""
        self.assertIsNotNone(self.adapter)
        self.assertEqual(self.adapter.config.rank, 8)
        self.assertFalse(self.adapter.is_trained)
        
    def test_feature_extraction(self):
        """Test feature extraction"""
        texts = ["This is a test text", "Another test document"]
        features = self.adapter.extract_features(texts)
        
        self.assertEqual(features.shape[0], 2)
        self.assertEqual(len(features.shape), 2)
        
    def test_risk_embedding_computation(self):
        """Test risk embedding computation"""
        text = "This AI system has privacy concerns and bias issues"
        risk_categories = ['bias_fairness', 'privacy_data']
        
        scores = self.adapter.compute_risk_embeddings(text, risk_categories)
        
        self.assertIn('bias_fairness', scores)
        self.assertIn('privacy_data', scores)
        self.assertTrue(0 <= scores['bias_fairness'] <= 1)
        self.assertTrue(0 <= scores['privacy_data'] <= 1)
        
    def test_domain_adaptation(self):
        """Test domain adaptation training"""
        domain_texts = [
            "Medical AI for diagnosis",
            "Clinical decision support system",
            "Patient data privacy"
        ]
        
        # This should not raise an exception
        self.adapter.adapt_for_domain(
            domain_texts=domain_texts,
            domain_name='biomedical',
            epochs=1,
            batch_size=2
        )
        
        self.assertTrue(self.adapter.is_trained)


if __name__ == '__main__':
    unittest.main()