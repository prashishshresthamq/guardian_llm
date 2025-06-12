"""
Script to train LoRA adapters for different domains
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.lora_adapter import LoRAAdapter, LoRAConfig
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_domain_data(domain: str, data_path: str) -> list:
    """Load domain-specific training data"""
    # This is a placeholder - in practice, you'd load real domain data
    domain_samples = {
        'biomedical': [
            "This study investigates the use of AI for early cancer detection...",
            "Machine learning models for predicting patient outcomes...",
            "Privacy concerns in medical AI systems include...",
        ],
        'legal': [
            "AI systems in legal decision-making raise fairness concerns...",
            "Algorithmic bias in criminal justice predictions...",
            "Transparency requirements for legal AI applications...",
        ],
        'technical': [
            "The proposed neural architecture introduces potential security vulnerabilities...",
            "Adversarial attacks on deep learning systems...",
            "Model interpretability challenges in complex AI systems...",
        ]
    }
    
    return domain_samples.get(domain, [])


def train_domain_adapters():
    """Train LoRA adapters for each domain"""
    domains = ['biomedical', 'legal', 'technical']
    
    # Configuration for LoRA
    config = LoRAConfig(
        rank=16,
        alpha=32.0,
        dropout=0.1,
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj']
    )
    
    save_dir = 'models/lora_adapters'
    os.makedirs(save_dir, exist_ok=True)
    
    for domain in domains:
        logger.info(f"Training adapter for domain: {domain}")
        
        # Initialize adapter
        adapter = LoRAAdapter(config=config)
        
        # Load domain data
        domain_texts = load_domain_data(domain, 'data/')
        
        if domain_texts:
            # Train adapter
            adapter.adapt_for_domain(
                domain_texts=domain_texts,
                domain_name=domain,
                epochs=5,
                batch_size=4,
                learning_rate=5e-5
            )
            
            # Save adapter
            adapter.save_adapter(save_dir, domain)
            logger.info(f"Saved adapter for domain: {domain}")
        else:
            logger.warning(f"No training data found for domain: {domain}")


if __name__ == "__main__":
    train_domain_adapters()