# scripts/train_semantic_model.py
"""
Script to train the SVD-based semantic risk model
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.semantic_analyzer import SemanticRiskAnalyzer
from core.guardian_engine import GuardianEngine
import json
import logging
import pickle
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data(data_path: str) -> tuple:
    """Load training data for semantic model"""
    
    # Example training data structure
    # In practice, this would load from a real dataset
    training_texts = []
    risk_labels = {
        'bias_fairness': [],
        'privacy_data': [],
        'safety_security': [],
        'dual_use': [],
        'societal_impact': [],
        'transparency': []
    }
    
    # Sample training data with risk examples
    risk_examples = {
        'bias_fairness': [
            "The AI system exhibits significant algorithmic bias against minority groups, leading to discriminatory outcomes in loan approvals.",
            "Our facial recognition model shows demographic bias with lower accuracy for certain ethnic groups.",
            "The hiring algorithm demonstrates gender bias by systematically ranking male candidates higher.",
            "Biased training data has resulted in unfair treatment of protected demographic categories.",
            "The recommendation system perpetuates societal biases through its filtering mechanisms."
        ],
        'privacy_data': [
            "The system collects personal data without explicit user consent, violating privacy regulations.",
            "Sensitive medical information is stored without proper encryption or access controls.",
            "User behavioral data is being tracked and sold to third parties without notification.",
            "The AI model can be used to de-anonymize supposedly private datasets.",
            "Location tracking continues even after users opt out of data collection."
        ],
        'safety_security': [
            "The autonomous system has critical safety vulnerabilities that could lead to physical harm.",
            "Security flaws in the AI model allow for adversarial attacks that compromise system integrity.",
            "The model fails catastrophically under edge cases, posing safety risks to users.",
            "Inadequate testing has left dangerous bugs in the safety-critical components.",
            "The system lacks proper failsafe mechanisms for high-risk scenarios."
        ],
        'dual_use': [
            "This facial recognition technology could be easily repurposed for mass surveillance.",
            "The AI system has clear military applications for autonomous weapons development.",
            "The technology can be weaponized for targeted disinformation campaigns.",
            "Dual-use potential exists for converting this research into offensive cyber weapons.",
            "The model could be misused for creating deepfakes for malicious purposes."
        ],
        'societal_impact': [
            "Widespread deployment will likely result in massive job displacement in the service sector.",
            "The technology exacerbates existing social inequalities through differential access.",
            "Implementation causes significant disruption to traditional community structures.",
            "Economic impacts disproportionately affect vulnerable populations.",
            "Social fragmentation increases as the technology replaces human interactions."
        ],
        'transparency': [
            "The black-box nature of the model prevents any meaningful accountability.",
            "Decision-making processes are completely opaque with no interpretability.",
            "Users cannot understand or challenge the AI's recommendations.",
            "The lack of explainability makes it impossible to audit for bias or errors.",
            "No transparency exists regarding the training data or model architecture."
        ]
    }
    
    # Add examples to training data
    doc_id = 0
    for category, examples in risk_examples.items():
        for example in examples:
            training_texts.append(example)
            risk_labels[category].append(doc_id)
            doc_id += 1
    
    # Add some neutral examples
    neutral_examples = [
        "This research presents a new approach to natural language processing.",
        "The system improves efficiency in data processing tasks.",
        "Our methodology follows established best practices in the field.",
        "Results show improved performance over baseline methods.",
        "The implementation uses standard machine learning techniques."
    ]
    
    for example in neutral_examples:
        training_texts.append(example)
        # Don't add to any risk category
        doc_id += 1
    
    return training_texts, risk_labels


def train_semantic_model():
    """Train and save the semantic risk model"""
    
    logger.info("Starting semantic model training...")
    
    # Load training data
    training_texts, risk_labels = load_training_data('data/')
    logger.info(f"Loaded {len(training_texts)} training documents")
    
    # Initialize semantic analyzer
    analyzer = SemanticRiskAnalyzer(n_components=50)  # Reduced for small dataset
    
    # Fit the model
    analyzer.fit(training_texts, risk_labels)
    
    # Test the model on sample texts
    test_texts = [
        "The AI system shows clear bias against women in hiring decisions.",
        "Personal health data is being collected without user awareness.",
        "The model has potential military applications for autonomous drones.",
        "Implementation will eliminate thousands of jobs in manufacturing."
    ]
    
    logger.info("\nTesting semantic model on sample texts:")
    for text in test_texts:
        scores = analyzer.analyze_semantic_risk(text)
        logger.info(f"\nText: {text[:60]}...")
        logger.info("Risk scores:")
        for category, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            if score > 0.1:
                logger.info(f"  {category}: {score:.3f}")
    
    # Extract and display latent topics
    topics = analyzer.get_latent_topics(n_topics=10)
    logger.info("\nLatent topics discovered:")
    for topic in topics[:5]:
        logger.info(f"\nTopic {topic['topic_id']} (variance: {topic['variance_explained']:.3f}):")
        logger.info(f"  Terms: {', '.join(topic['terms'][:5])}")
    
    # Save the model
    save_path = os.path.join('models', 'semantic_model.pkl')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    save_data = {
        'vectorizer': analyzer.vectorizer,
        'svd': analyzer.svd,
        'risk_concepts': analyzer.risk_concepts,
        'config': {
            'n_components': analyzer.n_components,
            'risk_threshold': analyzer.risk_threshold
        }
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    logger.info(f"\nSemantic model saved to {save_path}")
    
    # Generate model statistics
    logger.info("\nModel statistics:")
    logger.info(f"  Vocabulary size: {len(analyzer.vectorizer.vocabulary_)}")
    logger.info(f"  Number of components: {analyzer.n_components}")
    logger.info(f"  Risk categories: {len(analyzer.risk_concepts)}")
    logger.info(f"  Total variance explained: {sum(analyzer.svd.explained_variance_ratio_):.3f}")


def evaluate_semantic_model(analyzer: SemanticRiskAnalyzer):
    """Evaluate the semantic model performance"""
    
    # Evaluation test cases
    test_cases = [
        {
            'text': "The algorithm systematically discriminates against protected groups based on race and gender.",
            'expected': 'bias_fairness',
            'severity': 'high'
        },
        {
            'text': "User location data is collected and stored indefinitely without consent or notification.",
            'expected': 'privacy_data',
            'severity': 'high'
        },
        {
            'text': "Critical safety vulnerabilities could lead to system failures endangering human lives.",
            'expected': 'safety_security',
            'severity': 'high'
        },
        {
            'text': "The technology can be easily weaponized for military surveillance and targeting.",
            'expected': 'dual_use',
            'severity': 'high'
        },
        {
            'text': "Mass automation will displace millions of workers without retraining opportunities.",
            'expected': 'societal_impact',
            'severity': 'high'
        },
        {
            'text': "The decision process is a complete black box with zero explainability.",
            'expected': 'transparency',
            'severity': 'high'
        }
    ]
    
    logger.info("\n" + "="*60)
    logger.info("SEMANTIC MODEL EVALUATION")
    logger.info("="*60)
    
    correct_predictions = 0
    
    for i, test_case in enumerate(test_cases):
        text = test_case['text']
        expected = test_case['expected']
        
        # Get risk scores
        scores = analyzer.analyze_semantic_risk(text)
        
        # Find highest scoring category
        predicted = max(scores.items(), key=lambda x: x[1])[0]
        is_correct = predicted == expected
        
        if is_correct:
            correct_predictions += 1
        
        logger.info(f"\nTest Case {i+1}:")
        logger.info(f"Text: {text[:80]}...")
        logger.info(f"Expected: {expected}")
        logger.info(f"Predicted: {predicted} {'✓' if is_correct else '✗'}")
        logger.info("All scores:")
        for category, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {category}: {score:.3f}")
    
    accuracy = correct_predictions / len(test_cases)
    logger.info(f"\nOverall Accuracy: {accuracy:.1%} ({correct_predictions}/{len(test_cases)})")
    
    return accuracy


def generate_augmented_training_data():
    """Generate additional training data through augmentation"""
    
    logger.info("\nGenerating augmented training data...")
    
    # Templates for data augmentation
    templates = {
        'bias_fairness': [
            "The {system} exhibits {adjective} bias against {group}",
            "{adjective} discrimination in {domain} affecting {group}",
            "Algorithmic bias leads to {outcome} for {group}",
            "Unfair treatment of {group} in {system} decisions"
        ],
        'privacy_data': [
            "{data_type} data collected without {consent_type} consent",
            "Privacy violation through {method} of {data_type}",
            "Unauthorized {action} of personal {data_type}",
            "{system} tracks {data_type} despite user preferences"
        ],
        'safety_security': [
            "{vulnerability} in {component} poses {risk_type} risk",
            "Security flaw allows {attack_type} attacks",
            "{system} failure could cause {consequence}",
            "Lack of {safety_feature} in critical {component}"
        ]
    }
    
    # Substitution values
    substitutions = {
        'system': ['AI model', 'algorithm', 'system', 'application', 'platform'],
        'adjective': ['significant', 'systematic', 'persistent', 'severe', 'documented'],
        'group': ['minorities', 'women', 'elderly', 'disabled individuals', 'low-income users'],
        'domain': ['hiring', 'lending', 'healthcare', 'education', 'criminal justice'],
        'outcome': ['unfair rejection', 'discrimination', 'exclusion', 'disadvantage'],
        'data_type': ['location', 'health', 'financial', 'biometric', 'behavioral'],
        'consent_type': ['explicit', 'informed', 'proper', 'user', 'documented'],
        'method': ['collection', 'storage', 'sharing', 'analysis', 'processing'],
        'action': ['access', 'sharing', 'retention', 'analysis', 'distribution'],
        'vulnerability': ['Buffer overflow', 'SQL injection', 'Authentication bypass', 'Memory leak'],
        'component': ['authentication system', 'data processor', 'API endpoint', 'core module'],
        'risk_type': ['safety', 'security', 'operational', 'catastrophic', 'critical'],
        'attack_type': ['adversarial', 'injection', 'denial-of-service', 'privilege escalation'],
        'consequence': ['data breach', 'system crash', 'physical harm', 'service disruption'],
        'safety_feature': ['validation', 'error handling', 'failsafe', 'monitoring', 'backup']
    }
    
    augmented_data = []
    
    for category, category_templates in templates.items():
        for template in category_templates:
            # Generate variations
            for _ in range(5):  # 5 variations per template
                text = template
                # Replace placeholders
                import re
                placeholders = re.findall(r'\{(\w+)\}', template)
                for placeholder in placeholders:
                    if placeholder in substitutions:
                        import random
                        replacement = random.choice(substitutions[placeholder])
                        text = text.replace(f'{{{placeholder}}}', replacement, 1)
                
                augmented_data.append({
                    'text': text,
                    'category': category
                })
    
    logger.info(f"Generated {len(augmented_data)} augmented training examples")
    return augmented_data


if __name__ == "__main__":
    # Train the model
    train_semantic_model()
    
    # Load and evaluate the model
    logger.info("\n" + "="*60)
    logger.info("Loading and evaluating saved model...")
    logger.info("="*60)
    
    # Load the saved model
    with open('models/semantic_model.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    
    # Recreate analyzer
    eval_analyzer = SemanticRiskAnalyzer(n_components=saved_data['config']['n_components'])
    eval_analyzer.vectorizer = saved_data['vectorizer']
    eval_analyzer.svd = saved_data['svd']
    eval_analyzer.risk_concepts = saved_data['risk_concepts']
    eval_analyzer.is_fitted = True
    
    # Evaluate
    accuracy = evaluate_semantic_model(eval_analyzer)
    
    # Generate augmented data for future training
    augmented_data = generate_augmented_training_data()
    
    # Save augmented data
    augmented_path = 'data/augmented_training_data.json'
    os.makedirs(os.path.dirname(augmented_path), exist_ok=True)
    with open(augmented_path, 'w') as f:
        json.dump(augmented_data, f, indent=2)
    
    logger.info(f"\nAugmented training data saved to {augmented_path}")
    logger.info("\nSemantic model training complete!")