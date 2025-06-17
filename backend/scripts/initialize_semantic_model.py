# scripts/initialize_semantic_model.py
"""
Initialize semantic model with proper training data
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.semantic_analyzer import SemanticRiskAnalyzer
import pickle

def initialize_semantic_model():
    """Initialize and save a properly trained semantic model"""
    
    # Training data covering all risk categories
    training_texts = [
        # Bias & Fairness
        "The algorithm shows systematic bias against minority groups in hiring decisions",
        "Facial recognition accuracy varies significantly across different ethnic demographics",
        "AI system discriminates based on gender in loan approval processes",
        
        # Privacy & Data
        "Personal data collected without explicit user consent violating GDPR",
        "Sensitive medical information exposed through inadequate security measures",
        "User tracking continues even after opting out of data collection",
        
        # Safety & Security
        "Critical vulnerability allows unauthorized system access and data manipulation",
        "Autonomous vehicle AI fails to detect pedestrians in low-light conditions",
        "Medical AI provides incorrect diagnoses leading to patient harm",
        
        # Dual Use
        "Technology easily adaptable for military surveillance applications",
        "AI system could be weaponized for autonomous targeting",
        "Research has clear dual-use potential in biological weapons",
        
        # Societal Impact
        "Widespread deployment will eliminate millions of jobs in manufacturing",
        "AI system reinforces existing social inequalities through biased recommendations",
        "Technology disrupts traditional community structures and relationships",
        
        # Transparency
        "Black box model provides no explanation for critical decisions",
        "AI system lacks interpretability making auditing impossible",
        "No transparency in training data or decision-making process",
        
        # Neutral examples
        "This research improves efficiency in data processing tasks",
        "The system uses standard machine learning techniques",
        "Implementation follows established best practices"
    ]
    
    # Create risk labels
    risk_labels = {
        'bias_fairness': [0, 1, 2],
        'privacy_data': [3, 4, 5],
        'safety_security': [6, 7, 8],
        'dual_use': [9, 10, 11],
        'societal_impact': [12, 13, 14],
        'transparency': [15, 16, 17]
    }
    
    # Initialize and train analyzer
    analyzer = SemanticRiskAnalyzer(n_components=50)
    analyzer.fit(training_texts, risk_labels)
    
    # Save the model
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    save_path = os.path.join(model_dir, 'semantic_model.pkl')
    save_data = {
        'vectorizer': analyzer.vectorizer,
        'svd': analyzer.svd,
        'risk_concepts': analyzer.risk_concepts,
        'is_fitted': True
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"✅ Semantic model initialized and saved to {save_path}")
    return analyzer

if __name__ == '__main__':
    initialize_semantic_model()