"""
Novel Evaluation Metrics for Ethical Risk Assessment
Goes beyond standard accuracy to measure real-world impact
"""

import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import torch
from typing import Dict, List, Tuple, Optional
class EthicalRiskEvaluator:
    """
    Novel evaluation framework specifically for ethical risk assessment
    """
    
    def __init__(self):
        self.risk_weights = {
            'bias_fairness': 1.2,      # Higher weight for critical risks
            'privacy_data': 1.1,
            'safety_security': 1.5,    # Highest weight
            'dual_use': 1.3,
            'societal_impact': 1.0,
            'transparency': 0.9
        }
    
    def calculate_risk_aware_f1(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               risk_category: str) -> float:
        """
        Novel metric: Risk-weighted F1 that penalizes false negatives more
        for high-risk categories
        """
        weight = self.risk_weights[risk_category]
        
        # Custom F1 with asymmetric penalties
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        # Penalize false negatives more heavily for dangerous categories
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + weight * fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1
    
    def calculate_calibration_error(self, probabilities: np.ndarray, 
                                  labels: np.ndarray, n_bins: int = 10) -> float:
        """
        Novel: Expected Calibration Error for risk assessment
        Measures if predicted probabilities match actual risk frequencies
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0
        
        for i in range(n_bins):
            mask = (probabilities > bin_boundaries[i]) & (probabilities <= bin_boundaries[i + 1])
            if np.sum(mask) > 0:
                bin_accuracy = np.mean(labels[mask])
                bin_confidence = np.mean(probabilities[mask])
                bin_weight = np.sum(mask) / len(probabilities)
                ece += bin_weight * np.abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def calculate_risk_coverage(self, predictions: Dict[str, np.ndarray], 
                              ground_truth: Dict[str, np.ndarray]) -> float:
        """
        Novel metric: Measures how well the model covers different risk combinations
        """
        # Create risk profiles (which risks co-occur)
        pred_profiles = self._create_risk_profiles(predictions)
        true_profiles = self._create_risk_profiles(ground_truth)
        
        # Calculate Jaccard similarity of risk profile coverage
        intersection = len(pred_profiles.intersection(true_profiles))
        union = len(pred_profiles.union(true_profiles))
        
        return intersection / union if union > 0 else 0
    
    def statistical_significance_test(self, model_a_scores: np.ndarray, 
                                    model_b_scores: np.ndarray) -> Tuple[float, float]:
        """
        Perform McNemar's test for statistical significance
        """
        # Create contingency table
        a_correct_b_wrong = np.sum((model_a_scores == 1) & (model_b_scores == 0))
        a_wrong_b_correct = np.sum((model_a_scores == 0) & (model_b_scores == 1))
        
        # McNemar's test
        statistic = (abs(a_correct_b_wrong - a_wrong_b_correct) - 1)**2 / (a_correct_b_wrong + a_wrong_b_correct)
        p_value = 1 - stats.chi2.cdf(statistic, 1)
        
        return statistic, p_value