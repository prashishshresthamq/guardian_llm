"""
Active Learning Dataset Builder for Guardian LLM
Implements uncertainty sampling for efficient annotation
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import json
from sentence_transformers import SentenceTransformer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import requests

class ActiveLearningDatasetBuilder:
    """
    Novel approach: Use active learning to build dataset efficiently
    """
    
    def __init__(self):
        self.risk_categories = ['harassment', 'misinformation', 'harmful_content', 'privacy_violation']
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.papers_annotated = []
        self.annotation_history = []
        
        # Gaussian Process for uncertainty estimation (NOVEL)
        self.uncertainty_models = {
            category: GaussianProcessClassifier(
                kernel=1.0 * RBF(1.0),
                optimizer="fmin_l_bfgs_b",
                n_restarts_optimizer=5,
                random_state=42
            ) for category in self.risk_categories
        }
        
    def fetch_arxiv_papers(self, query, max_results=2000):
        """Fetch papers from ArXiv with proper pagination"""
        import arxiv
        import numpy as np
        
        papers = []
        
        # Create search object without 'start' parameter
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        try:
            for result in search.results():
                paper_data = {
                    'id': result.entry_id,
                    'title': result.title,
                    'abstract': result.summary,
                    'authors': [author.name for author in result.authors],
                    'published': result.published,
                    'categories': result.categories,
                    # Add mock annotations
                    'bias_fairness': np.random.randint(0, 10),
                    'privacy_data': np.random.randint(0, 10),
                    'safety_security': np.random.randint(0, 10),
                    'dual_use': np.random.randint(0, 10),
                    'societal_impact': np.random.randint(0, 10),
                    'transparency': np.random.randint(0, 10),
                }
                papers.append(paper_data)
                
        except arxiv.UnexpectedEmptyPageError:
            print(f"Reached end of available papers. Total fetched: {len(papers)}")
        except Exception as e:
            print(f"Error fetching papers: {e}")
        
        print(f"Successfully fetched {len(papers)} papers")
        return papers
    
    def select_papers_for_annotation(self, papers: List[Dict], n_select: int = 10) -> List[Dict]:
        """
        Novel: Use uncertainty sampling to select most informative papers
        """
        # Encode all papers
        embeddings = self.encoder.encode([p['abstract'] for p in papers])
        
        # Calculate uncertainty for each paper
        uncertainties = []
        for i, paper in enumerate(papers):
            paper_uncertainty = 0
            
            for category, model in self.uncertainty_models.items():
                if len(self.papers_annotated) > 0:
                    # Predict probability and get uncertainty
                    prob = model.predict_proba([embeddings[i]])[0]
                    # Entropy as uncertainty measure
                    entropy = -np.sum(prob * np.log(prob + 1e-10))
                    paper_uncertainty += entropy
                else:
                    # Random for first batch
                    paper_uncertainty += np.random.random()
            
            uncertainties.append(paper_uncertainty)
        
        # Select papers with highest uncertainty
        selected_indices = np.argsort(uncertainties)[-n_select:]
        return [papers[i] for i in selected_indices]
    
    def create_annotation_interface(self):
        """
        Create a simple Gradio interface for annotation
        """
        import gradio as gr
        
        def annotate_paper(paper_text, bias_risk, privacy_risk, safety_risk, 
                          dual_use_risk, societal_risk, transparency_risk):
            # Save annotation
            annotation = {
                'paper_id': self.current_paper['id'],
                'text': paper_text,
                'annotations': {
                    'bias_fairness': float(bias_risk),
                    'privacy_data': float(privacy_risk),
                    'safety_security': float(safety_risk),
                    'dual_use': float(dual_use_risk),
                    'societal_impact': float(societal_risk),
                    'transparency': float(transparency_risk)
                },
                'annotator': 'expert_1',
                'timestamp': str(datetime.now())
            }
            
            self.papers_annotated.append(annotation)
            self.update_uncertainty_models(annotation)
            
            return "Annotation saved! " + f"Total annotated: {len(self.papers_annotated)}"
        
        interface = gr.Interface(
            fn=annotate_paper,
            inputs=[
                gr.Textbox(lines=10, label="Paper Abstract"),
                gr.Slider(0, 10, label="Bias & Fairness Risk"),
                gr.Slider(0, 10, label="Privacy & Data Risk"),
                gr.Slider(0, 10, label="Safety & Security Risk"),
                gr.Slider(0, 10, label="Dual-Use Risk"),
                gr.Slider(0, 10, label="Societal Impact Risk"),
                gr.Slider(0, 10, label="Transparency Risk")
            ],
            outputs="text",
            title="Guardian LLM - Active Learning Annotation Tool"
        )
        
        return interface