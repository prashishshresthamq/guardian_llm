"""
Actual Model Training for Guardian LLM
Implements real training loops with logging and checkpointing
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import wandb
from tqdm import tqdm
import numpy as np
import inspect

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class GuardianDataset(Dataset):
    """Real dataset implementation"""
    
    def __init__(self, papers, tokenizer, max_length=512):
        self.papers = papers
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.papers)
    
    def __getitem__(self, idx):
        paper = self.papers[idx]
        
        # Combine title and abstract for text
        text = f"{paper['title']}. {paper['abstract']}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Get labels - try different possible structures
        try:
            # Option 1: Direct attributes
            labels = torch.tensor([
                paper.get('bias_fairness', 0),
                paper.get('privacy_data', 0),
                paper.get('safety_security', 0),
                paper.get('dual_use', 0),
                paper.get('societal_impact', 0),
                paper.get('transparency', 0)
            ]) / 10.0
        except:
            # Option 2: Nested under different key
            try:
                labels = torch.tensor([
                    paper['labels']['bias_fairness'],
                    paper['labels']['privacy_data'],
                    paper['labels']['safety_security'],
                    paper['labels']['dual_use'],
                    paper['labels']['societal_impact'],
                    paper['labels']['transparency']
                ]) / 10.0
            except:
                # Option 3: Default values if no annotations
                labels = torch.zeros(6)  # Default to zeros
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels
        }

class ModelTrainer:
    """Real training implementation with all the bells and whistles"""
    
    def __init__(self, model, config, base_model=None):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Store base model for getting embeddings if needed
        self.base_model = base_model
        if self.base_model:
            self.base_model.to(self.device)
            self.base_model.eval()  # Set to eval mode
        
        # Check model type by inspecting its forward method and class name
        try:
            forward_params = inspect.signature(model.forward).parameters
            param_names = list(forward_params.keys())
            
            # Check if it's a GNN (has edge_index and batch parameters)
            self.is_gnn = 'edge_index' in param_names and 'batch' in param_names
            
            # Check if it's HRPN or similar (expects tensor input, not dict)
            self.expects_tensor = (
                model.__class__.__name__ == 'HierarchicalRiskPropagationNetwork' or
                (len(param_names) >= 1 and param_names[0] in ['x', 'input', 'inputs'] and 
                 'edge_index' in param_names)
            )
        except:
            self.is_gnn = False
            self.expects_tensor = False
        
        # Initialize wandb for experiment tracking
        wandb.init(project="guardian-llm", config=config)
        
    def train(self, train_dataset, val_dataset, epochs=10):
        """Full training loop with validation, checkpointing, and early stopping"""
        
        # Create dataloaders with no multiprocessing to avoid tokenizer issues
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0  # Changed from 4 to 0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0  # Changed from 4 to 0
        )
        
        # Optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Loss function - weighted BCE for imbalanced risks
        risk_weights = torch.tensor([1.2, 1.1, 1.5, 1.3, 1.0, 0.9]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=risk_weights)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            
            for batch in train_pbar:
                # Get labels
                labels = batch['labels'].to(self.device)
                
                # Forward pass based on model type
                optimizer.zero_grad()
                
                if self.is_gnn:
                    outputs = self._forward_gnn(batch)
                else:
                    outputs = self._forward_standard(batch)
                
                # Extract tensor from potentially complex output
                outputs = self._extract_output_tensor(outputs)
                
                # Handle shape mismatch
                if outputs.shape[0] != labels.shape[0]:
                    # If batch size mismatch, try to fix it
                    if outputs.shape[0] == 1:
                        # Model returned single output for entire batch
                        if outputs.dim() == 3:
                            # If 3D tensor [1, 1, 6], first squeeze then expand
                            outputs = outputs.squeeze(0).squeeze(0)  # Now it's [6]
                            outputs = outputs.unsqueeze(0).expand(labels.shape[0], -1)  # Now [batch_size, 6]
                        elif outputs.dim() == 2:
                            # If 2D tensor [1, 6], expand first dimension
                            outputs = outputs.expand(labels.shape[0], -1)
                        else:
                            raise ValueError(f"Unexpected output dimensions: {outputs.shape}")
                    else:
                        raise ValueError(f"Batch size mismatch: outputs {outputs.shape} vs labels {labels.shape}")
                
                # Final shape check
                if outputs.shape != labels.shape:
                    raise ValueError(f"Final shape mismatch: outputs {outputs.shape} vs labels {labels.shape}")
                
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({'loss': loss.item()})
                
                # Log to wandb
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': scheduler.get_last_lr()[0]
                })
            
            # Validation phase
            val_loss, val_metrics = self.validate(val_loader, criterion)
            
            # Log epoch metrics
            wandb.log({
                'epoch': epoch,
                'train_loss_avg': train_loss / len(train_loader),
                'val_loss': val_loss,
                **val_metrics
            })
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(epoch, val_loss)
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        return self.model
    
    def _forward_standard(self, batch):
        """Forward pass for standard models"""
        batch_dict = {
            'input_ids': batch['input_ids'].to(self.device),
            'attention_mask': batch['attention_mask'].to(self.device)
        }
        return self.model(batch_dict)
    
    def _forward_gnn(self, batch):
        """Forward pass for GNN models"""
        batch_size = batch['input_ids'].size(0)
        
        # Move tensors to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Create minimal graph structure
        # For HRPN, we might need to process each sample separately
        # or create a proper batch graph structure
        
        # HRPN expects embeddings, not token IDs
        if self.expects_tensor:
            # Get embeddings from base model if available
            if self.base_model is not None:
                with torch.no_grad():
                    # Get embeddings from base model (e.g., BERT)
                    outputs = self.base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
                    # Use CLS token embedding or mean pooling
                    if hasattr(outputs, 'pooler_output'):
                        embeddings = outputs.pooler_output
                    else:
                        # Mean pooling over sequence
                        hidden_states = outputs.last_hidden_state
                        mask = attention_mask.unsqueeze(-1).float()
                        embeddings = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                # Fallback: create simple embeddings
                # Get vocabulary size (approximate)
                vocab_size = 50000  # Common vocab size
                embed_dim = 768  # Standard embedding dimension
                
                # Create embedding layer if not exists
                if not hasattr(self, 'fallback_embedding'):
                    self.fallback_embedding = nn.Embedding(vocab_size, embed_dim).to(self.device)
                
                # Get embeddings and average pool
                token_embeds = self.fallback_embedding(input_ids).float()
                mask = attention_mask.unsqueeze(-1).float()
                embeddings = (token_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            
            # For HRPN, it might expect to process the entire batch as one graph
            # Create edge index for a disconnected graph (no edges between samples)
            edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)
            
            # Batch assignment - all samples in same batch
            batch_assignment = torch.zeros(batch_size, dtype=torch.long).to(self.device)
            
            # Try different approaches for HRPN
            try:
                # Approach 1: Pass embeddings directly
                outputs = self.model(embeddings, edge_index, batch_assignment)
            except Exception as e:
                print(f"First approach failed: {e}")
                try:
                    # Approach 2: Process each sample separately and concatenate
                    outputs_list = []
                    for i in range(batch_size):
                        single_embed = embeddings[i:i+1]
                        single_edge = torch.empty((2, 0), dtype=torch.long).to(self.device)
                        single_batch = torch.zeros(1, dtype=torch.long).to(self.device)
                        single_output = self.model(single_embed, single_edge, single_batch)
                        outputs_list.append(single_output)
                    outputs = torch.cat(outputs_list, dim=0)
                except Exception as e2:
                    print(f"Second approach failed: {e2}")
                    # Approach 3: Return outputs as is
                    outputs = self.model(embeddings, edge_index, batch_assignment)
            
            return outputs
        else:
            # Other GNNs might expect a dict
            batch_dict = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)
            batch_assignment = torch.zeros(batch_size, dtype=torch.long).to(self.device)
            return self.model(batch_dict, edge_index, batch_assignment)
    
    def _extract_output_tensor(self, outputs):
        """Extract tensor from model output, handling various formats"""
        if isinstance(outputs, torch.Tensor):
            return outputs  # Don't modify here, let the training loop handle shape issues
        
        if isinstance(outputs, dict):
            # Try common keys
            for key in ['logits', 'output', 'predictions', 'out', 'y', 'risk_scores']:
                if key in outputs:
                    return outputs[key]
            
            # Handle risk_0, risk_1, ... format
            risk_keys = [k for k in outputs.keys() if k.startswith('risk_')]
            if risk_keys:
                # Sort to ensure correct order (risk_0, risk_1, ...)
                risk_keys = sorted(risk_keys, key=lambda x: int(x.split('_')[1]))
                if len(risk_keys) == 6:  # Expecting 6 risk categories
                    # Stack individual risk tensors into a single tensor
                    risk_tensors = []
                    for k in risk_keys:
                        tensor = outputs[k]
                        risk_tensors.append(tensor)
                    
                    # Stack along the last dimension
                    try:
                        # First try stacking
                        result = torch.stack(risk_tensors, dim=-1)
                        # If result is [1, 1, 6] for a batch, we'll handle it in training loop
                        return result
                    except:
                        # If stacking fails, try concatenating
                        # This might give us [batch_size * 6] which we need to reshape
                        concat = torch.cat(risk_tensors, dim=-1)
                        if concat.dim() == 1:
                            # Reshape to [1, 6] if needed
                            return concat.view(-1, 6)
                        return concat
            
            # Find first tensor with 6 in last dimension
            for value in outputs.values():
                if isinstance(value, torch.Tensor) and value.shape[-1] == 6:
                    return value
            
            # If no tensor found, raise error with helpful message
            raise ValueError(f"Model returned dict but no valid output tensor found. Keys: {list(outputs.keys())}")
        
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            return outputs[0]
        
        raise ValueError(f"Unexpected output type: {type(outputs)}")
    
    def validate(self, val_loader, criterion):
        """Validation loop with metric calculation"""
        self.model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                labels = batch['labels'].to(self.device)
                
                # Forward pass based on model type
                if self.is_gnn:
                    outputs = self._forward_gnn(batch)
                else:
                    outputs = self._forward_standard(batch)
                
                # Extract tensor from potentially complex output
                outputs = self._extract_output_tensor(outputs)
                
                # Handle shape mismatch (same as in training)
                if outputs.shape[0] != labels.shape[0]:
                    if outputs.shape[0] == 1:
                        # Model returned single output for entire batch
                        if outputs.dim() == 3:
                            # If 3D tensor [1, 1, 6], first squeeze then expand
                            outputs = outputs.squeeze(0).squeeze(0)  # Now it's [6]
                            outputs = outputs.unsqueeze(0).expand(labels.shape[0], -1)  # Now [batch_size, 6]
                        elif outputs.dim() == 2:
                            # If 2D tensor [1, 6], expand first dimension
                            outputs = outputs.expand(labels.shape[0], -1)
                        else:
                            raise ValueError(f"Unexpected output dimensions: {outputs.shape}")
                    else:
                        raise ValueError(f"Batch size mismatch: outputs {outputs.shape} vs labels {labels.shape}")
                
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                all_preds.append(torch.sigmoid(outputs).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Calculate metrics
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        metrics = self.calculate_metrics(all_preds, all_labels)
        
        return val_loss / len(val_loader), metrics
    
    def calculate_metrics(self, preds, labels):
        """Calculate comprehensive metrics"""
        from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
        
        metrics = {}
        risk_names = ['bias_fairness', 'privacy_data', 'safety_security', 
                     'dual_use', 'societal_impact', 'transparency']
        
        for i, risk in enumerate(risk_names):
            # Convert to binary predictions
            binary_preds = (preds[:, i] > 0.5).astype(int)
            binary_labels = (labels[:, i] > 0.5).astype(int)
            
            # Skip if all labels are the same (can't calculate AUC)
            if len(np.unique(binary_labels)) < 2:
                metrics[f'{risk}_auc'] = 0.5
                metrics[f'{risk}_f1'] = 0.0
                metrics[f'{risk}_avg_precision'] = 0.0
                continue
            
            # Calculate metrics
            try:
                metrics[f'{risk}_auc'] = roc_auc_score(binary_labels, preds[:, i])
            except:
                metrics[f'{risk}_auc'] = 0.5
                
            metrics[f'{risk}_f1'] = f1_score(binary_labels, binary_preds)
            
            # Calculate precision at different recall levels
            try:
                precision, recall, _ = precision_recall_curve(binary_labels, preds[:, i])
                metrics[f'{risk}_avg_precision'] = np.mean(precision)
            except:
                metrics[f'{risk}_avg_precision'] = 0.0
        
        return metrics
    
    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint"""
        # Create checkpoints directory if it doesn't exist
        os.makedirs('checkpoints', exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        path = f'checkpoints/guardian_llm_epoch_{epoch}_loss_{val_loss:.4f}.pt'
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")