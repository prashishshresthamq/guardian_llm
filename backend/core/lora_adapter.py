"""
Guardian LLM - LoRA Adapter
Low-Rank Adaptation for efficient domain-specific fine-tuning
Based on COMP8420 Week 11 - Matrix and Vector in NLP
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import AutoModel, AutoTokenizer
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation"""
    rank: int = 16
    alpha: float = 32.0  # Scaling factor
    dropout: float = 0.1
    target_modules: List[str] = None  # Which layers to apply LoRA to
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default to attention layers
            self.target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj']


class LoRALayer(nn.Module):
    """
    LoRA layer implementation
    W' = W + BA where B ∈ R^{d×r} and A ∈ R^{r×k}
    """
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Create low-rank matrices A and B
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        
        # Initialize A with Gaussian, B with zeros (as per paper)
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.merged = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation"""
        # Regular forward pass happens in the base model
        # We only compute the low-rank adaptation
        if self.training:
            x = self.dropout(x)
        
        # Compute BA multiplication
        lora_output = x @ self.lora_A.T @ self.lora_B.T
        return lora_output * self.scaling
    
    def merge_weights(self, base_weight: torch.Tensor) -> torch.Tensor:
        """Merge LoRA weights with base weights for inference"""
        if not self.merged:
            # W' = W + BA * scaling
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            return base_weight + delta_w
        return base_weight


class LoRAAdapter:
    """
    LoRA Adapter for Guardian LLM
    Implements Low-Rank Adaptation for efficient domain-specific fine-tuning
    """
    
    def __init__(self, base_model_name: str = "bert-base-uncased", config: Optional[LoRAConfig] = None):
        """
        Initialize LoRA Adapter
        
        Args:
            base_model_name: Name of the pre-trained model
            config: LoRA configuration
        """
        self.config = config or LoRAConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load base model and tokenizer
        logger.info(f"Loading base model: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModel.from_pretrained(base_model_name).to(self.device)
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Add LoRA layers
        self.lora_layers = nn.ModuleDict()
        self._inject_lora_layers()
        
        # Training components
        self.optimizer = None
        self.is_trained = False
        
    def _inject_lora_layers(self):
        """Inject LoRA layers into the base model"""
        logger.info(f"Injecting LoRA layers with rank={self.config.rank}")
        
        for name, module in self.base_model.named_modules():
            # Check if this module should have LoRA
            if any(target in name for target in self.config.target_modules):
                if isinstance(module, nn.Linear):
                    # Create LoRA layer
                    lora_layer = LoRALayer(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        rank=self.config.rank,
                        alpha=self.config.alpha,
                        dropout=self.config.dropout
                    ).to(self.device)
                    
                    # Store the LoRA layer
                    self.lora_layers[name] = lora_layer
                    
                    # Modify forward pass to include LoRA
                    self._modify_forward(module, lora_layer)
                    
        logger.info(f"Injected {len(self.lora_layers)} LoRA layers")
    
    def _modify_forward(self, module: nn.Module, lora_layer: LoRALayer):
        """Modify module's forward pass to include LoRA adaptation"""
        original_forward = module.forward
        
        def forward_with_lora(x):
            # Original computation
            base_output = original_forward(x)
            # Add LoRA adaptation
            lora_output = lora_layer(x)
            return base_output + lora_output
        
        module.forward = forward_with_lora
    
    def adapt_for_domain(self, domain_texts: List[str], domain_name: str = "general", 
                        epochs: int = 3, batch_size: int = 8, learning_rate: float = 1e-4):
        """
        Fine-tune LoRA weights for specific research domain
        
        Args:
            domain_texts: List of domain-specific texts
            domain_name: Name of the domain (e.g., 'biomedical', 'legal', 'technical')
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
        """
        logger.info(f"Adapting for domain: {domain_name} with {len(domain_texts)} texts")
        
        # Prepare optimizer (only optimize LoRA parameters)
        lora_params = []
        for layer in self.lora_layers.values():
            lora_params.extend([layer.lora_A, layer.lora_B])
        
        self.optimizer = torch.optim.AdamW(lora_params, lr=learning_rate)
        
        # Training loop
        self.base_model.train()
        for layer in self.lora_layers.values():
            layer.train()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # Process texts in batches
            for i in range(0, len(domain_texts), batch_size):
                batch_texts = domain_texts[i:i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                # Forward pass
                outputs = self.base_model(**inputs)
                
                # Compute domain-specific loss (contrastive learning)
                loss = self._compute_domain_loss(outputs, domain_name)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        logger.info(f"Domain adaptation completed for: {domain_name}")
    
    def _compute_domain_loss(self, outputs, domain_name: str) -> torch.Tensor:
        """
        Compute domain-specific loss
        Using contrastive learning to make embeddings domain-specific
        """
        # Get embeddings from the last hidden state
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling
        
        # Normalize embeddings
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        
        # Create labels (assuming batch items are from same domain)
        batch_size = embeddings.size(0)
        labels = torch.arange(batch_size).to(self.device)
        
        # Contrastive loss (simplified version)
        loss = nn.functional.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def extract_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract features using LoRA-adapted model
        
        Args:
            texts: List of texts to extract features from
            
        Returns:
            Feature matrix of shape (n_texts, embedding_dim)
        """
        self.base_model.eval()
        features = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = self.base_model(**inputs)
                
                # Use CLS token embedding as feature
                feature = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                features.append(feature)
        
        return np.vstack(features)
    
    def compute_risk_embeddings(self, text: str, risk_categories: List[str]) -> Dict[str, float]:
        """
        Compute risk scores using adapted embeddings
        
        Args:
            text: Input text
            risk_categories: List of risk categories
            
        Returns:
            Dictionary of risk scores
        """
        # Extract text embedding
        text_embedding = self.extract_features([text])[0]
        
        # Define risk category embeddings (these could be learned)
        risk_templates = {
            'bias_fairness': "This text contains bias and fairness issues",
            'privacy_data': "This text has privacy and data protection concerns",
            'safety_security': "This text presents safety and security risks",
            'dual_use': "This text describes dual-use technology",
            'societal_impact': "This text has negative societal impact",
            'transparency': "This text lacks transparency and explainability"
        }
        
        risk_scores = {}
        
        for category in risk_categories:
            if category in risk_templates:
                # Extract risk template embedding
                risk_embedding = self.extract_features([risk_templates[category]])[0]
                
                # Compute cosine similarity
                similarity = np.dot(text_embedding, risk_embedding) / (
                    np.linalg.norm(text_embedding) * np.linalg.norm(risk_embedding)
                )
                
                # Convert to risk score (0-1 range)
                risk_scores[category] = max(0, min(1, (similarity + 1) / 2))
        
        return risk_scores
    
    def save_adapter(self, save_path: str, domain_name: str):
        """Save LoRA adapter weights"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        save_dict = {
            'config': self.config.__dict__,
            'domain': domain_name,
            'lora_weights': {}
        }
        
        for name, layer in self.lora_layers.items():
            save_dict['lora_weights'][name] = {
                'lora_A': layer.lora_A.cpu().detach().numpy(),
                'lora_B': layer.lora_B.cpu().detach().numpy()
            }
        
        torch.save(save_dict, os.path.join(save_path, f'lora_adapter_{domain_name}.pt'))
        logger.info(f"Saved LoRA adapter for domain: {domain_name}")
    
    def load_adapter(self, load_path: str):
        """Load LoRA adapter weights"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        for name, layer in self.lora_layers.items():
            if name in checkpoint['lora_weights']:
                layer.lora_A.data = torch.tensor(
                    checkpoint['lora_weights'][name]['lora_A']
                ).to(self.device)
                layer.lora_B.data = torch.tensor(
                    checkpoint['lora_weights'][name]['lora_B']
                ).to(self.device)
        
        self.is_trained = True
        logger.info(f"Loaded LoRA adapter from: {load_path}")
    
    def merge_and_export(self, export_path: str):
        """Merge LoRA weights with base model and export"""
        # This would merge the LoRA adaptations into the base model
        # for more efficient inference
        pass


class DomainSpecificAnalyzer:
    """
    Analyzer that uses LoRA-adapted models for domain-specific risk assessment
    """
    
    def __init__(self):
        self.adapters = {}
        self.domains = ['biomedical', 'legal', 'financial', 'technical', 'social']
        
    def initialize_adapters(self):
        """Initialize LoRA adapters for different domains"""
        for domain in self.domains:
            self.adapters[domain] = LoRAAdapter()
    
    def analyze_with_domain_adaptation(self, text: str, detected_domain: str) -> Dict[str, float]:
        """Analyze text using domain-specific adapter"""
        if detected_domain in self.adapters:
            adapter = self.adapters[detected_domain]
            return adapter.compute_risk_embeddings(text, 
                ['bias_fairness', 'privacy_data', 'safety_security', 
                 'dual_use', 'societal_impact', 'transparency'])
        else:
            # Use general adapter
            return self.adapters.get('general', LoRAAdapter()).compute_risk_embeddings(text, [])