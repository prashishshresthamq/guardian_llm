"""
Guardian LLM - Core Analysis Engine
Main engine for text analysis, risk detection, and sentiment analysis
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import nltk
from textblob import TextBlob
import numpy as np
from core.lora_adapter import LoRAAdapter, DomainSpecificAnalyzer
from core.cot_analyzer import ChainOfThoughtAnalyzer, CoTIntegrator  # New import
import time  
import logging

from core.semantic_analyzer import (
    SemanticRiskAnalyzer, 
    SemanticRiskIntegrator
    
)
from core.vector_db_postgres import PgVectorDatabase, create_pgvector_analyzer  # Add this
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import local modules
from models.schemas import (
    RiskLevel, 
    RiskCategory, 
    SentimentType,
    RiskAssessment,
    risk_level_from_score,
    sentiment_type_from_scores
)
from config.setting import Config
from core.text_processors import TextProcessor
from core.risk_analyzers import RiskAnalyzer

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass


class GuardianEngine:
    """Main analysis engine for Guardian LLM"""
    
    def __init__(self):
        """Initialize the Guardian Engine"""
        self.config = Config()
        self.text_processor = TextProcessor()
        self.risk_analyzer = RiskAnalyzer()
        self.is_initialized = True
        
        # Initialize LoRA adapter
        self.lora_adapter = None
        self.domain_analyzer = DomainSpecificAnalyzer()
        self._initialize_lora()
        
         # Initialize SVD-based semantic analyzer
        self.semantic_analyzer = SemanticRiskAnalyzer(n_components=100)
        self.semantic_integrator = SemanticRiskIntegrator(self.semantic_analyzer)
        self._initialize_semantic_analyzer()
        
         # Initialize Chain of Thought analyzer
        self.cot_analyzer = ChainOfThoughtAnalyzer()
        self.cot_integrator = CoTIntegrator(self.cot_analyzer)
        logger.info("Guardian Engine initialized with Chain of Thought reasoning")


        
    def get_status(self) -> Dict[str, Any]:
        """Get engine status including semantic analyzer type"""
        analyzer_info = 'inactive'
        if hasattr(self, 'analyzer_type'):
            analyzer_info = f'active_{self.analyzer_type}'
        
        return {
            'initialized': self.is_initialized,
            'version': '1.0.0',
            'components': {
                'text_processor': 'active',
                'risk_analyzer': 'active',
                'sentiment_analyzer': 'active',
                'semantic_analyzer': analyzer_info,
                'lora_adapter': 'active' if self.lora_adapter else 'inactive',
                'cot_analyzer': 'active'
            },
            'semantic_details': {
                'type': getattr(self, 'analyzer_type', 'unknown'),
                'enhanced': getattr(self, 'analyzer_type', 'svd') in ['pgvector', 'faiss'],
                'vector_db': getattr(self, 'analyzer_type', 'none') if hasattr(self, 'analyzer_type') else 'none'
            }
        }
        
    def _initialize_semantic_analyzer(self):
        """Initialize the semantic analyzer with enhanced vector database support"""
        try:
            # Check if we should use enhanced analyzer
            use_enhanced = os.getenv('USE_ENHANCED_SEMANTIC', 'true').lower() == 'true'
            use_postgres = bool(os.getenv('DATABASE_URL')) or bool(os.getenv('PGVECTOR_URL'))
            
            if use_enhanced:
                if use_postgres:
                    # Use PostgreSQL with pgvector
                    connection_string = os.getenv('PGVECTOR_URL') or os.getenv('DATABASE_URL')
                    self.semantic_analyzer = create_pgvector_analyzer(connection_string)
                    self.analyzer_type = 'pgvector'
                    logger.info("Using PostgreSQL pgvector for semantic analysis")
                else:
                    # Use FAISS-based enhanced analyzer
                    db_path = os.path.join(self.config.MODEL_CACHE_DIR, 'vector_db')
                    self.semantic_analyzer = create_enhanced_semantic_analyzer(db_path=db_path)
                    self.analyzer_type = 'faiss'
                    logger.info("Using FAISS for enhanced semantic analysis")
                
                # Create integrator for enhanced analyzer
                self.semantic_integrator = SemanticRiskIntegrator(self.semantic_analyzer)
                self.semantic_analyzer.is_fitted = True
                
            else:
                # Use original SVD-based analyzer
                model_path = os.path.join(self.config.MODEL_CACHE_DIR, 'semantic_model.pkl')
                os.makedirs(self.config.MODEL_CACHE_DIR, exist_ok=True)
                
                self.semantic_analyzer = SemanticRiskAnalyzer(n_components=100)
                
                if os.path.exists(model_path):
                    # Load pre-trained model
                    import pickle
                    with open(model_path, 'rb') as f:
                        saved_data = pickle.load(f)
                        self.semantic_analyzer.vectorizer = saved_data.get('vectorizer')
                        self.semantic_analyzer.svd = saved_data.get('svd')
                        self.semantic_analyzer.risk_concepts = saved_data.get('risk_concepts', {})
                        self.semantic_analyzer.is_fitted = saved_data.get('is_fitted', False)
                    logger.info("Loaded pre-trained SVD semantic model")
                else:
                    # Initialize with patterns if no saved model
                    logger.info("No saved semantic model found, initializing with patterns")
                    self.semantic_analyzer._initialize_with_patterns()
                    self.semantic_analyzer.is_fitted = True
                
                self.semantic_integrator = SemanticRiskIntegrator(self.semantic_analyzer)
                self.analyzer_type = 'svd'
                
        except Exception as e:
            logger.error(f"Failed to initialize semantic analyzer: {e}")
            # Fallback to pattern-based SVD analyzer
            self.semantic_analyzer = SemanticRiskAnalyzer(n_components=100)
            self.semantic_analyzer._initialize_with_patterns()
            self.semantic_analyzer.is_fitted = True
            self.semantic_integrator = SemanticRiskIntegrator(self.semantic_analyzer)
            self.analyzer_type = 'svd'
            
            
    def add_custom_risk_pattern(self, category: str, pattern_text: str, severity: float = 0.8):
        """Add custom risk pattern to the semantic analyzer"""
        if hasattr(self, 'analyzer_type'):
            if self.analyzer_type == 'pgvector':
                self.semantic_analyzer.add_custom_patterns_from_feedback(category, pattern_text, severity)
            elif self.analyzer_type == 'faiss':
                # Enhanced analyzer method
                self.semantic_analyzer.add_custom_patterns(category, [pattern_text], [severity])
            else:
                # SVD analyzer doesn't support dynamic patterns
                logger.warning("SVD analyzer doesn't support adding custom patterns")
        
        logger.info(f"Added custom pattern for {category}")    
        
    def find_similar_analyses(self, text: str, top_k: int = 5) -> List[Dict]:
        """Find similar previously analyzed texts"""
        if hasattr(self, 'analyzer_type'):
            if self.analyzer_type == 'pgvector':
                return self.semantic_analyzer.find_similar_analyzed_texts(text, top_k)
            elif self.analyzer_type == 'faiss':
                return self.semantic_analyzer.get_similar_texts(text, top_k=top_k)
            else:
                # SVD analyzer doesn't have similarity search
                logger.warning("SVD analyzer doesn't support similarity search")
                return []
        return []

    def update_pattern_severity_from_feedback(self, pattern_id: str, category: str, new_severity: float):
        """Update pattern severity based on user feedback"""
        if hasattr(self, 'analyzer_type'):
            if self.analyzer_type == 'pgvector':
                self.semantic_analyzer.update_pattern_severity(pattern_id, new_severity)
            elif self.analyzer_type == 'faiss':
                self.semantic_analyzer.update_pattern_severity(category, pattern_id, new_severity)
            else:
                logger.warning("SVD analyzer doesn't support updating pattern severity")
                
                
    def analyze_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform complete text analysis with enhanced semantic analysis support
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Check if CoT analysis is enabled in options
        enable_cot = options.get('enable_cot', True) if options else True
        cot_mode = options.get('cot_mode', 'enhanced') if options else 'enhanced'
        
        # Process text
        processed_text = self.text_processor.process(text)
        
        # Detect domain
        domain = self._detect_domain(text)
        
        # Perform traditional risk analysis
        risk_analysis = self._analyze_risks(processed_text)
        
        # Enhanced semantic analysis based on analyzer type
        if hasattr(self, 'analyzer_type') and self.analyzer_type in ['pgvector', 'faiss']:
            # Use enhanced semantic analysis
            if self.analyzer_type == 'pgvector':
                semantic_scores = self.semantic_analyzer.analyze_text_semantic_risk(text, store_results=True)
            else:  # faiss
                semantic_scores = self.semantic_analyzer.analyze_semantic_risk(text)
            
            # Get semantic evidence for each category
            for category in risk_analysis['risk_categories']:
                if self.analyzer_type == 'pgvector':
                    semantic_evidence = self.semantic_analyzer.find_evidence_sentences(text, category, num_sentences=3)
                else:  # faiss
                    semantic_evidence = self.semantic_analyzer.find_evidence_with_embeddings(text, category, num_sentences=3)
                
                # Store semantic evidence for later use
                risk_analysis[f'{category}_semantic_evidence'] = semantic_evidence
            
            # Merge scores
            for category, sem_score in semantic_scores.items():
                if category in risk_analysis['risk_categories']:
                    trad_score = risk_analysis['risk_categories'][category]
                    risk_analysis['risk_categories'][category] = 0.6 * trad_score + 0.4 * sem_score
            
            risk_analysis['analysis_methods'] = ['keyword', f'semantic_{self.analyzer_type}']
            
        else:
            # Use original SVD-based semantic analysis
            traditional_scores = {cat: score for cat, score in risk_analysis['risk_categories'].items()}
            enhanced_scores = self.semantic_integrator.enhance_risk_analysis(text, traditional_scores)
            risk_analysis['risk_categories'] = enhanced_scores
            risk_analysis['analysis_methods'] = ['keyword', 'semantic_svd']
            
        # Enhance with LoRA if available
        if self.lora_adapter and domain:
            lora_risks = self.domain_analyzer.analyze_with_domain_adaptation(text, domain)
            risk_analysis = self._merge_risk_scores(risk_analysis, lora_risks)
            if 'analysis_methods' in risk_analysis:
                risk_analysis['analysis_methods'].append('lora_domain')
        
        # Perform Chain of Thought analysis if enabled
        cot_analysis = None
        if enable_cot and cot_mode != 'disabled':
            try:
                if cot_mode == 'standalone':
                    # Standalone CoT analysis
                    cot_result = self.cot_analyzer.analyze_with_reasoning(text)
                    cot_analysis = {
                        'reasoning_chain': self.cot_analyzer.get_detailed_reasoning(cot_result),
                        'cot_risk_scores': cot_result.final_risk_scores,
                        'reasoning_summary': cot_result.reasoning_summary,
                        'overall_cot_risk': cot_result.overall_risk,
                        'cot_confidence': cot_result.confidence,
                        'mode': 'standalone'
                    }
                else:  # enhanced mode
                    # Integrate CoT with traditional analysis
                    enhanced_analysis = self.cot_integrator.enhance_analysis_with_cot(
                        text, {'risk_categories': risk_analysis['risk_categories']}
                    )
                    cot_analysis = enhanced_analysis['chain_of_thought']
                    cot_analysis['mode'] = 'enhanced'
                    # Update risk scores with CoT enhancement
                    risk_analysis['risk_categories'] = enhanced_analysis['risk_categories']
                    if 'analysis_methods' in risk_analysis:
                        risk_analysis['analysis_methods'].append('chain_of_thought')
                
                logger.info(f"CoT analysis completed in {cot_mode} mode")
            except Exception as e:
                logger.error(f"CoT analysis failed: {e}")
                cot_analysis = {'error': str(e), 'mode': cot_mode}
        
        # Rest of the analysis...
        sentiment_analysis = self._analyze_sentiment(text)
        statistics = self._generate_statistics(processed_text, risk_analysis)
        
        # Generate risk assessments with CoT insights
        if cot_analysis and 'error' not in cot_analysis:
            risk_assessments = self._generate_risk_assessments_with_cot(
                text, processed_text, risk_analysis, cot_analysis
            )
        else:
            risk_assessments = self._generate_risk_assessments_with_semantic(
                text, processed_text, risk_analysis
            )
        
        # Generate recommendations with CoT insights
        recommendations = self._generate_recommendations_with_cot(
            risk_analysis, sentiment_analysis, cot_analysis
        )
        
        # Add semantic analysis metadata
        semantic_metadata = {}
        if hasattr(self, 'semantic_analyzer') and self.semantic_analyzer.is_fitted:
            # Extract latent topics
            topics = self.semantic_analyzer.get_latent_topics(n_topics=5)
            semantic_metadata['latent_topics'] = topics
        
        # Build final response
        response = {
            'text': text,
            'domain': domain,
            'domain_confidence': self._get_domain_confidence(text, domain),
            'processed_text': processed_text,
            'risk_analysis': risk_analysis,
            'sentiment': sentiment_analysis,
            'statistics': statistics,
            'risk_assessments': risk_assessments,
            'recommendations': recommendations,
            'semantic_metadata': semantic_metadata,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Add CoT analysis to response if available
        if cot_analysis:
            response['chain_of_thought'] = cot_analysis
        
        return response
    
    def _initialize_lora(self):
        """Initialize LoRA adapter with pre-trained weights if available"""
        try:
            self.lora_adapter = LoRAAdapter()
            # Load pre-trained domain adapters if they exist
            import os
            adapter_dir = os.path.join(self.config.MODEL_CACHE_DIR, 'lora_adapters')
            if os.path.exists(adapter_dir):
                for domain in ['biomedical', 'legal', 'technical']:
                    adapter_path = os.path.join(adapter_dir, f'lora_adapter_{domain}.pt')
                    if os.path.exists(adapter_path):
                        adapter = LoRAAdapter()
                        adapter.load_adapter(adapter_path)
                        self.domain_analyzer.adapters[domain] = adapter
            logger.info("LoRA adapters initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LoRA: {e}")
            self.lora_adapter = None
    
    # Add new method for enhanced risk assessments:
    def _generate_risk_assessments_with_semantic(self, text: str, processed_text: Dict[str, Any], 
                                            risk_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate detailed risk assessments with semantic evidence"""
        assessments = []
        
        for category, score in risk_analysis['risk_categories'].items():
            level = risk_level_from_score(score)
            
            # Get traditional evidence
            traditional_evidence = self.risk_analyzer.get_risk_evidence(text, category)
            
            # Get semantic evidence
            semantic_evidence = []
            
            # Check if we have cached semantic evidence from enhanced analyzer
            cached_evidence_key = f'{category}_semantic_evidence'
            if cached_evidence_key in risk_analysis:
                cached_evidence = risk_analysis[cached_evidence_key]
                for ev in cached_evidence[:3]:
                    if isinstance(ev, dict):
                        text_snippet = ev.get('text', '')
                        relevance = ev.get('relevance', 0)
                        semantic_evidence.append(f"[Semantic {relevance:.2f}] {text_snippet}")
                    else:
                        semantic_evidence.append(str(ev))
            else:
                # Use semantic integrator for SVD-based evidence
                if hasattr(self, 'semantic_integrator'):
                    semantic_evidence = self.semantic_integrator.generate_semantic_evidence(text, category)
            
            # Combine evidence
            all_evidence = traditional_evidence[:3] + semantic_evidence[:2]
            
            # Ensure we have evidence
            if not all_evidence:
                all_evidence = [
                    f"Analysis indicates {level.value} risk for {category.replace('_', ' ')}",
                    f"Risk score: {score:.2f}/10 based on content analysis"
                ]
            
            assessment = {
                'category': category,
                'level': level.value,
                'score': score,
                'confidence': min(0.9, score + 0.3),
                'keywords': self._extract_category_keywords(text, category),
                'evidence': all_evidence[:5],
                'context': self._get_risk_context(text, category),
                'analysis_type': f'hybrid_{self.analyzer_type}' if hasattr(self, 'analyzer_type') else 'hybrid'
            }
            assessments.append(assessment)
        
        return assessments

    def _get_recommendations(self, category: str, score: float) -> List[str]:
        """
        Generate basic recommendations for a risk category
        This is the missing method that was causing the AttributeError
        """
        recommendations = []
        
        # Define risk categories if not already available
        if not hasattr(self, 'risk_categories'):
            self.risk_categories = {
                'bias_fairness': {
                    'name': 'Bias and Fairness',
                    'description': 'Potential bias or fairness issues in AI systems'
                },
                'privacy_data': {
                    'name': 'Privacy and Data Protection',
                    'description': 'Privacy violations or data protection concerns'
                },
                'safety_security': {
                    'name': 'Safety and Security',
                    'description': 'Safety risks or security vulnerabilities'
                },
                'dual_use': {
                    'name': 'Dual Use',
                    'description': 'Technology that could be used for harmful purposes'
                },
                'societal_impact': {
                    'name': 'Societal Impact',
                    'description': 'Negative impacts on society or communities'
                },
                'transparency': {
                    'name': 'Transparency',
                    'description': 'Lack of transparency or explainability'
                }
            }
        
        # Generate recommendations based on score severity
        if score >= 7.5:
            recommendations.append(f"URGENT: Immediate review and mitigation required for {category.replace('_', ' ')}")
            recommendations.append("Implement comprehensive safeguards before deployment")
            recommendations.append("Conduct thorough risk assessment with domain experts")
        elif score >= 5.0:
            recommendations.append(f"HIGH PRIORITY: Address {category.replace('_', ' ')} concerns")
            recommendations.append("Develop mitigation strategies and monitoring systems")
            recommendations.append("Consider additional testing and validation")
        elif score >= 2.5:
            recommendations.append(f"MODERATE: Monitor {category.replace('_', ' ')} risks")
            recommendations.append("Implement standard safeguards and best practices")
            recommendations.append("Regular review and assessment recommended")
        else:
            recommendations.append(f"LOW: Continue monitoring {category.replace('_', ' ')}")
            recommendations.append("Maintain current safeguards")
        
        # Category-specific recommendations
        if category == 'bias_fairness' and score >= 2.5:
            recommendations.append("Conduct bias testing across demographic groups")
            recommendations.append("Implement fairness metrics and regular audits")
        elif category == 'privacy_data' and score >= 2.5:
            recommendations.append("Review data handling and privacy policies")
            recommendations.append("Ensure GDPR/privacy regulation compliance")
        elif category == 'safety_security' and score >= 2.5:
            recommendations.append("Perform security vulnerability assessment")
            recommendations.append("Implement fail-safe mechanisms")
        elif category == 'dual_use' and score >= 2.5:
            recommendations.append("Establish use-case restrictions")
            recommendations.append("Monitor for potential misuse")
        elif category == 'societal_impact' and score >= 2.5:
            recommendations.append("Assess community and workforce impacts")
            recommendations.append("Develop transition/support programs if needed")
        elif category == 'transparency' and score >= 2.5:
            recommendations.append("Improve model interpretability")
            recommendations.append("Provide clear documentation and explanations")
        
        return recommendations[:5]  # Return top 5 recommendations


    def analyze(self, text: str, title: Optional[str] = None, 
           use_cot: bool = True, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main analysis method with enhanced evidence extraction
        """
        start_time = time.time()
        options = options or {}
        
        logger.info(f"Starting analysis with use_cot={use_cot}, text_length={len(text)}")
        
        # Process text
        processed_text = self.text_processor.process(text)
        
        # Perform risk analysis
        risk_analysis = self.risk_analyzer.analyze_risks(text, processed_text)
        
        # CRITICAL: Add text and sentences for evidence extraction
        risk_analysis['text'] = text
        risk_analysis['sentences'] = self.text_processor.get_sentences(text)
        
        # Domain detection
        domain = self._detect_domain(text, processed_text)
        domain_confidence = 0.85
        
        # Apply domain-specific adapter if available
        if domain and hasattr(self, 'lora_adapter') and self.lora_adapter:
            try:
                adapted_analysis = self.lora_adapter.apply_domain_adapter(
                    text, domain, risk_analysis
                )
                risk_analysis.update(adapted_analysis)
                domain_confidence = 0.95
                if 'lora_domain' not in risk_analysis.get('analysis_methods', []):
                    risk_analysis.setdefault('analysis_methods', []).append('lora_domain')
            except Exception as e:
                logger.warning(f"Failed to apply LoRA adapter: {e}")
        
        # Semantic analysis integration
        semantic_scores = {}
        if hasattr(self, 'semantic_integrator') and self.semantic_integrator:
            try:
                semantic_scores = self.semantic_integrator.analyze(text)
                # Merge semantic scores
                for category, score in semantic_scores.items():
                    if category in risk_analysis['risk_categories']:
                        keyword_score = risk_analysis['risk_categories'][category]
                        risk_analysis['risk_categories'][category] = (
                            0.6 * keyword_score + 0.4 * score
                        )
                if 'semantic_svd' not in risk_analysis.get('analysis_methods', []):
                    risk_analysis.setdefault('analysis_methods', []).append('semantic_svd')
            except Exception as e:
                logger.warning(f"Semantic analysis failed: {e}")
        
        # Chain of Thought analysis
        cot_analysis = {}
        if use_cot and self.cot_analyzer:
            try:
                cot_analysis = self.cot_analyzer.analyze_with_cot(
                    text=text,
                    risk_analysis=risk_analysis,
                    domain=domain
                )
                if 'chain_of_thought' not in risk_analysis.get('analysis_methods', []):
                    risk_analysis.setdefault('analysis_methods', []).append('chain_of_thought')
                
                # Update risk scores with CoT insights
                if 'cot_risk_scores' in cot_analysis:
                    for category, cot_score in cot_analysis['cot_risk_scores'].items():
                        if category in risk_analysis['risk_categories']:
                            current_score = risk_analysis['risk_categories'][category]
                            risk_analysis['risk_categories'][category] = (
                                0.5 * current_score + 0.5 * cot_score
                            )
            except Exception as e:
                logger.error(f"CoT analysis failed: {e}")
                use_cot = False
        
        # Calculate overall risk score
        overall_risk_score = self._calculate_overall_risk(risk_analysis['risk_categories'])
        
        # CRITICAL: Always use enhanced assessment methods
        if use_cot and cot_analysis:
            risk_assessments = self._generate_risk_assessments_with_cot(
                text, processed_text, risk_analysis, cot_analysis
            )
        else:
            # Use enhanced method even without CoT
            risk_assessments = self._generate_enhanced_risk_assessments(
                text, processed_text, risk_analysis, semantic_scores
            )
        
        # Sentiment analysis
        sentiment = self._analyze_sentiment(text)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Compile final results
        result = {
            'text': text[:1000] + '...' if len(text) > 1000 else text,
            'title': title or 'Untitled',
            'timestamp': datetime.utcnow().isoformat(),
            'domain': domain,
            'domain_confidence': domain_confidence,
            'overall_risk_score': overall_risk_score,
            'risk_level': self._get_overall_risk_level(overall_risk_score),
            'risk_categories': risk_analysis['risk_categories'],
            'detected_keywords': risk_analysis.get('detected_keywords', {}),
            'analysis_methods': risk_analysis.get('analysis_methods', ['keyword']),
            'risk_assessments': risk_assessments,
            'sentiment': sentiment,
            'statistics': {
                'word_count': processed_text['statistics']['word_count'],
                'sentence_count': processed_text['statistics']['sentence_count'],
                'processing_time': round(processing_time, 2)
            },
            'recommendations': self._generate_overall_recommendations(risk_assessments),
            'metadata': {
                'version': '1.0.0',
                'analysis_date': datetime.utcnow().isoformat()
            }
        }
        
        # Add CoT-specific results if available
        if cot_analysis:
            result['cot_analysis'] = {
                'reasoning_chain': cot_analysis.get('reasoning_chain', {}),
                'confidence': cot_analysis.get('cot_confidence', 0.5),
                'risk_scores': cot_analysis.get('cot_risk_scores', {})
            }
        
        logger.info(f"Analysis completed in {processing_time:.2f}s")
        return result
    
    def _calculate_overall_risk(self, risk_categories: Dict[str, float]) -> float:
        """Calculate overall risk score from category scores"""
        if not risk_categories:
            return 0.0
        
        # Weighted average with higher weight for critical categories
        weights = {
            'safety_security': 1.5,
            'dual_use': 1.4,
            'bias_fairness': 1.2,
            'privacy_data': 1.2,
            'societal_impact': 1.0,
            'transparency': 0.8
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for category, score in risk_categories.items():
            weight = weights.get(category, 1.0)
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _get_overall_risk_level(self, score: float) -> str:
        """Get risk level string from overall score"""
        if score >= 7.5:
            return 'critical'
        elif score >= 5.0:
            return 'high'
        elif score >= 2.5:
            return 'moderate'
        elif score >= 1.0:
            return 'low'
        else:
            return 'minimal'
    def _generate_overall_recommendations(self, risk_assessments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate overall recommendations from risk assessments"""
        recommendations = []
        
        # Find highest risk level
        highest_risk_score = 0
        critical_categories = []
        
        for assessment in risk_assessments:
            if assessment['score'] > highest_risk_score:
                highest_risk_score = assessment['score']
            if assessment['score'] >= 7.5:
                critical_categories.append(assessment['category'])
        
        # Generate overall recommendations
        if critical_categories:
            recommendations.append({
                'type': 'critical',
                'message': f'Critical risks detected in: {", ".join(critical_categories)}',
                'action': 'Immediate review and comprehensive mitigation required',
                'priority': 'critical'
            })
        
        if highest_risk_score >= 5.0:
            recommendations.append({
                'type': 'review',
                'message': 'Significant AI safety risks identified',
                'action': 'Conduct thorough safety review before deployment',
                'priority': 'high'
            })
        
        # Add general best practice recommendation
        recommendations.append({
            'type': 'best_practice',
            'message': 'Continue monitoring AI safety best practices',
            'action': 'Regular assessment and updates recommended',
            'priority': 'medium'
        })
        
        return recommendations
    
    def extract_paper_metadata(self, text: str) -> Dict[str, str]:
        """
        Extract title, authors, and abstract from research paper text
        """
        metadata = {
            'title': '',
            'authors': '',
            'abstract': ''
        }
        
        # Split text into lines for processing
        lines = text.split('\n')
        
        # Try to find title (usually in first few lines, often in larger font or all caps)
        title_found = False
        for i, line in enumerate(lines[:50]):  # Check first 50 lines
            line = line.strip()
            if not line:
                continue
                
            # Heuristics for title detection
            if (len(line) > 20 and len(line) < 200 and 
                not line.startswith(('Abstract', 'ABSTRACT', 'Introduction', 'Keywords')) and
                not any(char in line for char in ['@', 'http', 'www', '.com', '.edu'])):
                
                # Check if line looks like a title (no period at end, reasonable length)
                if not line.endswith('.') and not title_found:
                    metadata['title'] = line
                    title_found = True
                    break
        
        # Try to find authors (often after title, before abstract)
        authors_found = False
        author_patterns = [
            r'^[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+',  # John A. Smith
            r'^[A-Z][a-z]+ [A-Z][a-z]+',  # John Smith
            r'^[A-Z]\. [A-Z][a-z]+',  # J. Smith
        ]
        
        for i, line in enumerate(lines[:100]):
            line = line.strip()
            if not line or authors_found:
                continue
                
            # Look for author patterns
            if any(re.match(pattern, line) for pattern in author_patterns):
                # Collect authors (might be multiple lines)
                author_lines = [line]
                for j in range(i+1, min(i+5, len(lines))):
                    next_line = lines[j].strip()
                    if (next_line and 
                        any(re.match(pattern, next_line) for pattern in author_patterns) or
                        next_line.startswith(('and', ',', ';'))):
                        author_lines.append(next_line)
                    else:
                        break
                
                metadata['authors'] = ' '.join(author_lines)
                authors_found = True
        
        # Try to find abstract
        abstract_keywords = ['Abstract', 'ABSTRACT', 'Summary', 'SUMMARY']
        abstract_found = False
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            if any(line_stripped.startswith(keyword) for keyword in abstract_keywords):
                # Found abstract section
                abstract_lines = []
                
                # Collect abstract text (usually until Introduction or Keywords)
                for j in range(i+1, min(i+50, len(lines))):
                    next_line = lines[j].strip()
                    
                    # Stop conditions
                    if (next_line.startswith(('Introduction', 'INTRODUCTION', 'Keywords', 
                                            'KEYWORDS', '1.', '1 Introduction')) or
                        (j > i+5 and not next_line)):  # Empty line after some content
                        break
                        
                    if next_line:
                        abstract_lines.append(next_line)
                
                if abstract_lines:
                    metadata['abstract'] = ' '.join(abstract_lines)[:1000]  # Limit length
                    abstract_found = True
                    break
        
        # Fallback: if no title found, use first non-empty line
        if not metadata['title']:
            for line in lines[:20]:
                line = line.strip()
                if len(line) > 20 and not any(char in line for char in ['@', 'http']):
                    metadata['title'] = line[:200]
                    break
        
        # Clean up extracted data
        metadata['title'] = metadata['title'].strip()
        metadata['authors'] = metadata['authors'].strip()
        metadata['abstract'] = metadata['abstract'].strip()
        
        return metadata

    def _generate_risk_assessments(self, risk_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Enhanced version of basic risk assessment with dynamic evidence
        """
        # Get text from risk_analysis
        text = risk_analysis.get('text', '')
        sentences = risk_analysis.get('sentences', [])
        
        # If we have text, use enhanced method
        if text:
            processed_text = {'sentences': sentences}
            return self._generate_enhanced_risk_assessments(
                text, processed_text, risk_analysis, {}
            )
        
        # Fallback to original method if no text
        assessments = []
        for category, score in risk_analysis['risk_categories'].items():
            level = risk_level_from_score(score)
            
            assessment = {
                'category': category,
                'level': level.value,
                'score': score,
                'confidence': min(0.9, score + 0.3),
                'keywords': risk_analysis.get('detected_keywords', {}).get(category, []),
                'evidence': [
                    f"Analysis indicates {level.value} risk for {self.risk_categories[category]['name']}",
                    f"Pattern detection for {category.replace('_', ' ')} factors"
                ],
                'explanation': self.risk_categories[category]['description'],
                'recommendations': self._get_recommendations(category, score)
            }
            assessments.append(assessment)
        
        return assessments
    def _generate_enhanced_risk_assessments(self, text: str, processed_text: Dict[str, Any],
                                       risk_analysis: Dict[str, Any], 
                                       semantic_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Generate enhanced risk assessments with dynamic evidence
        """
        assessments = []
        sentences = processed_text.get('sentences', self.text_processor.get_sentences(text))
        
        for category, score in risk_analysis['risk_categories'].items():
            level = risk_level_from_score(score)
            
            # 1. Extract keyword-based evidence with actual text
            keyword_evidence = []
            if self.risk_analyzer:
                keyword_evidence = self.risk_analyzer.get_risk_evidence(text, category)
            
            # 2. Extract text snippets showing risks
            text_snippets = self._extract_risk_snippets(text, category, processed_text)
            
            # 3. Get semantic insights if available
            semantic_evidence = []
            if category in semantic_scores and semantic_scores[category] > 0.3:
                semantic_evidence.append(
                    f"Semantic analysis: detected {level.value}-level patterns similar to {category.replace('_', ' ')} risks"
                )
            
            # 4. Combine evidence intelligently
            all_evidence = []
            
            # Add most specific evidence first
            if text_snippets:
                all_evidence.extend(text_snippets[:2])
            
            if keyword_evidence:
                # Filter out generic evidence
                specific_evidence = [e for e in keyword_evidence 
                                if 'indicates' not in e.lower() and '"' in e]
                all_evidence.extend(specific_evidence[:3-len(all_evidence)])
            
            if semantic_evidence and len(all_evidence) < 3:
                all_evidence.extend(semantic_evidence)
            
            # Only use generic evidence if we found nothing specific
            if not all_evidence:
                all_evidence = [
                    f"Analysis indicates {level.value} risk for {self.risk_categories[category]['name']}",
                    f"Risk score: {score:.2f}/10 based on content analysis"
                ]
            
            # 5. Generate dynamic recommendations
            recommendations = self._generate_dynamic_recommendations(
                category, score, all_evidence, {}
            )
            
            # 6. Create detailed explanation
            explanation = self._generate_dynamic_explanation(category, all_evidence)
            
            assessment = {
                'category': category,
                'level': level.value,
                'score': round(score, 2),
                'confidence': min(0.9, score/10 + 0.4),
                'keywords': risk_analysis.get('detected_keywords', {}).get(category, [])[:5],
                'evidence': all_evidence[:5],
                'explanation': explanation,
                'recommendations': recommendations,
                'context': self._get_detailed_risk_context(text, category, all_evidence),
                'analysis_type': 'enhanced'
            }
            assessments.append(assessment)
        
        return assessments
    def _generate_risk_assessments_with_cot(self, text: str, processed_text: Dict[str, Any], 
                                        risk_analysis: Dict[str, Any], 
                                        cot_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate detailed risk assessments with evidence from all analyzers"""
        assessments = []
        
        # Get CoT reasoning for context
        reasoning_chain = cot_analysis.get('reasoning_chain', {})
        cot_risk_scores = cot_analysis.get('cot_risk_scores', {})
        
        # Get sentences for context extraction
        sentences = processed_text.get('sentences', self.text_processor.get_sentences(text))
        
        for category, score in risk_analysis['risk_categories'].items():
            level = risk_level_from_score(score)
            
            # 1. Get keyword-based evidence from risk analyzer
            keyword_evidence = []
            if self.risk_analyzer:
                keyword_evidence = self.risk_analyzer.get_risk_evidence(text, category)
            
            # 2. Get semantic evidence with actual text snippets
            semantic_evidence = []
            if hasattr(self, 'semantic_analyzer') and self.semantic_analyzer:
                try:
                    explanation = self.semantic_analyzer.explain_risk_detection(text, category)
                    
                    # Extract sentences containing important terms
                    important_terms = [term['term'] for term in explanation.get('important_terms', [])[:5]]
                    for term in important_terms[:3]:
                        for sentence in sentences:
                            if term.lower() in sentence.lower():
                                semantic_evidence.append(f'Semantic match: "{term}" found in: "{sentence[:100]}..."')
                                break
                except:
                    pass
            
            # 3. Get CoT reasoning evidence
            cot_evidence = []
            reasoning_steps = reasoning_chain.get('reasoning_steps', [])
            
            for step in reasoning_steps:
                if step.get('step') in ['risk_identification', 'misuse_analysis', 'severity_assessment']:
                    findings = step.get('findings', [])
                    # Find findings related to this category
                    category_keywords = category.replace('_', ' ').split()
                    for finding in findings:
                        if any(keyword in finding.lower() for keyword in category_keywords):
                            cot_evidence.append(f"Reasoning: {finding[:150]}...")
                            if len(cot_evidence) >= 2:
                                break
            
            # 4. Extract actual text snippets showing risks
            text_snippets = self._extract_risk_snippets(text, category, processed_text)
            
            # 5. Combine all evidence sources intelligently
            all_evidence = []
            
            # Add the most specific evidence first
            if text_snippets:
                all_evidence.extend(text_snippets[:2])
            
            # Add semantic findings if they provide new information
            if semantic_evidence:
                all_evidence.extend([e for e in semantic_evidence 
                                if not any(e in existing for existing in all_evidence)][:2])
            
            # Add CoT insights
            if cot_evidence:
                all_evidence.extend(cot_evidence[:1])
            
            # Add specific keyword evidence
            if len(all_evidence) < 3 and keyword_evidence:
                specific_keyword_evidence = [e for e in keyword_evidence if '"' in e]
                all_evidence.extend(specific_keyword_evidence[:3-len(all_evidence)])
            
            # 6. Generate dynamic recommendations based on findings
            dynamic_recommendations = self._generate_dynamic_recommendations(
                category, score, all_evidence, cot_analysis
            )
            
            # Enhanced confidence calculation
            base_confidence = min(0.9, score/10 + 0.3)
            cot_confidence = cot_analysis.get('cot_confidence', 0.5)
            combined_confidence = (base_confidence + cot_confidence) / 2
            
            assessment = {
                'category': category,
                'level': level.value,
                'score': round(score, 2),
                'confidence': round(combined_confidence, 2),
                'keywords': risk_analysis.get('detected_keywords', {}).get(category, [])[:5],
                'evidence': all_evidence[:5],
                'context': self._get_detailed_risk_context(text, category, all_evidence),
                'analysis_type': 'comprehensive',
                'explanation': self._generate_dynamic_explanation(category, all_evidence),
                'recommendations': dynamic_recommendations
            }
            assessments.append(assessment)
        
        return assessments
    
    def _generate_dynamic_explanation(self, category: str, evidence: List[str]) -> str:
        """Generate a dynamic explanation based on actual findings"""
        if not evidence:
            return self.risk_categories[category]['description']
        
        # Count types of evidence
        text_evidence = [e for e in evidence if '"' in e and '[detected:' in e]
        semantic_evidence = [e for e in evidence if 'Semantic' in e]
        reasoning_evidence = [e for e in evidence if 'Reasoning:' in e or 'CoT Analysis:' in e]
        
        explanation_parts = []
        
        if text_evidence:
            explanation_parts.append(f"{len(text_evidence)} specific text patterns detected")
        
        if semantic_evidence:
            explanation_parts.append(f"semantic analysis identified risk indicators")
        
        if reasoning_evidence:
            explanation_parts.append("logical reasoning confirmed concerns")
        
        # Build base explanation
        if explanation_parts:
            base_explanation = f"Analysis found {', '.join(explanation_parts)} related to {category.replace('_', ' ')}. "
        else:
            base_explanation = f"{self.risk_categories[category]['description']}. "
        
        # Add specific insights based on evidence content
        evidence_text = ' '.join(evidence).lower()
        
        if category == 'bias_fairness' and any(term in evidence_text for term in ['demographic', 'gender', 'racial']):
            base_explanation += "Specific demographic or identity-based bias patterns were identified."
        elif category == 'privacy_data' and any(term in evidence_text for term in ['consent', 'personal data', 'gdpr']):
            base_explanation += "Privacy compliance and data protection issues require attention."
        elif category == 'safety_security' and any(term in evidence_text for term in ['vulnerability', 'exploit', 'critical']):
            base_explanation += "Critical security vulnerabilities need immediate remediation."
        elif category == 'dual_use' and any(term in evidence_text for term in ['military', 'weapon', 'surveillance']):
            base_explanation += "Potential for harmful applications or misuse detected."
        elif category == 'societal_impact' and any(term in evidence_text for term in ['job', 'inequality', 'community']):
            base_explanation += "Significant societal disruption potential identified."
        elif category == 'transparency' and any(term in evidence_text for term in ['black box', 'explainable', 'audit']):
            base_explanation += "Lack of interpretability poses accountability challenges."
        
        return base_explanation
    
    def _generate_dynamic_recommendations(self, category: str, score: float, 
                                    evidence: List[str], cot_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on actual findings"""
        recommendations = []
        
        # Extract key issues from evidence
        issues_found = set()
        evidence_text = ' '.join(evidence).lower()
        
        # Category-specific issue detection
        issue_keywords = {
            'bias_fairness': {
                'bias': ['demographic bias', 'algorithmic bias', 'gender bias', 'racial bias'],
                'discrimination': ['discriminatory', 'unfair treatment'],
                'fairness': ['fairness issues', 'unfair advantage']
            },
            'privacy_data': {
                'consent': ['without consent', 'consent not', 'no consent'],
                'data_collection': ['collect data', 'personal data', 'user data'],
                'breach': ['data breach', 'privacy breach', 'data leak']
            },
            'safety_security': {
                'vulnerability': ['security vulnerability', 'security flaw'],
                'failure': ['system failure', 'catastrophic failure'],
                'exploit': ['exploit', 'attack vector']
            },
            'dual_use': {
                'military': ['military application', 'military use'],
                'weapon': ['weapon', 'weaponization'],
                'surveillance': ['surveillance system', 'surveillance capability']
            },
            'societal_impact': {
                'employment': ['job loss', 'job displacement', 'unemployment'],
                'inequality': ['social inequality', 'economic inequality'],
                'disruption': ['disruption', 'community impact']
            },
            'transparency': {
                'blackbox': ['black box', 'black-box'],
                'explainability': ['not explainable', 'unexplainable', 'cannot explain'],
                'transparency': ['lack transparency', 'no transparency']
            }
        }
        
        # Detect specific issues
        if category in issue_keywords:
            for issue_type, keywords in issue_keywords[category].items():
                if any(keyword in evidence_text for keyword in keywords):
                    issues_found.add(issue_type)
        
        # Generate specific recommendations based on findings
        if category == 'bias_fairness':
            if 'bias' in issues_found:
                recommendations.append("Implement comprehensive bias testing across all demographic groups")
                recommendations.append("Use debiasing techniques and balanced training datasets")
            if 'discrimination' in issues_found:
                recommendations.append("Establish fairness metrics and regular auditing procedures")
                recommendations.append("Implement algorithmic fairness constraints")
        
        elif category == 'privacy_data':
            if 'consent' in issues_found:
                recommendations.append("Implement explicit opt-in consent mechanisms")
                recommendations.append("Provide clear data usage policies and user control options")
            if 'data_collection' in issues_found:
                recommendations.append("Minimize data collection to essential information only")
                recommendations.append("Implement data anonymization and encryption")
        
        elif category == 'safety_security':
            if 'vulnerability' in issues_found:
                recommendations.append("Conduct immediate security audit and penetration testing")
                recommendations.append("Implement security patches and regular updates")
            if 'failure' in issues_found:
                recommendations.append("Design robust failsafe mechanisms")
                recommendations.append("Implement comprehensive error handling and recovery")
        
        elif category == 'dual_use':
            if 'military' in issues_found or 'weapon' in issues_found:
                recommendations.append("Establish strict use-case restrictions and licensing")
                recommendations.append("Implement technical safeguards against weaponization")
            if 'surveillance' in issues_found:
                recommendations.append("Add privacy-preserving features to prevent mass surveillance")
                recommendations.append("Require transparency in deployment contexts")
        
        elif category == 'societal_impact':
            if 'employment' in issues_found:
                recommendations.append("Develop workforce transition and retraining programs")
                recommendations.append("Consider phased implementation to minimize job displacement")
            if 'inequality' in issues_found:
                recommendations.append("Ensure equitable access across socioeconomic groups")
                recommendations.append("Monitor and address disparate impact")
        
        elif category == 'transparency':
            if 'blackbox' in issues_found or 'explainability' in issues_found:
                recommendations.append("Implement explainable AI methods (LIME, SHAP)")
                recommendations.append("Provide clear documentation of decision processes")
            if 'transparency' in issues_found:
                recommendations.append("Create user-friendly explanations of system behavior")
                recommendations.append("Enable audit trails and decision logging")
        
        # Add severity-based recommendations
        if score >= 7.5:
            recommendations.insert(0, f"CRITICAL: Immediate action required for {category.replace('_', ' ')} risks")
        elif score >= 5.0:
            recommendations.insert(0, f"HIGH PRIORITY: Develop comprehensive mitigation plan for {category.replace('_', ' ')}")
        elif score >= 2.5:
            recommendations.insert(0, f"MODERATE: Review and enhance {category.replace('_', ' ')} safeguards")
        
        # Get CoT mitigation strategies if available
        if cot_analysis and 'reasoning_chain' in cot_analysis:
            mitigation_step = next((step for step in cot_analysis['reasoning_chain'].get('reasoning_steps', [])
                                if step.get('step') == 'mitigation_strategies'), None)
            if mitigation_step and 'strategies' in mitigation_step:
                for strategy in mitigation_step['strategies'][:2]:
                    if not any(strategy.lower() in rec.lower() for rec in recommendations):
                        recommendations.append(strategy)
        
        # Ensure we have recommendations
        if not recommendations:
            recommendations = self._get_recommendations(category, score)
        
        return recommendations[:5]

    def _extract_risk_snippets(self, text: str, category: str, processed_text: Dict[str, Any]) -> List[str]:
        """Extract actual text snippets that indicate risks"""
        snippets = []
        sentences = processed_text.get('sentences', text.split('. '))
        
        # Category-specific patterns
        risk_patterns = {
            'bias_fairness': [
                r'bias(?:ed)?\s+(?:against|toward)',
                r'discriminat\w+',
                r'unfair\w*\s+(?:treatment|advantage)',
                r'demographic\s+(?:bias|disparity)',
                r'protected\s+(?:class|group)',
                r'gender\s+(?:bias|discrimination)',
                r'racial\s+(?:bias|profiling)',
                r'algorithmic\s+(?:bias|fairness)'
            ],
            'privacy_data': [
                r'personal\s+(?:data|information)',
                r'privacy\s+(?:violation|concern|breach)',
                r'(?:collect|store|share)\s+(?:user|personal)\s+data',
                r'consent\s+(?:not|without)',
                r'surveillance',
                r'data\s+(?:leak|breach|exposure)',
                r'unauthorized\s+access',
                r'GDPR\s+(?:violation|compliance)'
            ],
            'safety_security': [
                r'security\s+(?:vulnerability|flaw|risk)',
                r'safety\s+(?:concern|risk|critical)',
                r'(?:system|catastrophic)\s+failure',
                r'exploit\w*',
                r'attack\s+vector',
                r'critical\s+(?:bug|flaw|issue)',
                r'malicious\s+(?:code|actor)',
                r'zero[- ]day'
            ],
            'dual_use': [
                r'military\s+(?:application|use)',
                r'weapon\w*',
                r'dual[\s-]use',
                r'surveillance\s+(?:system|capability)',
                r'malicious\s+(?:use|actor)',
                r'autonomous\s+weapon',
                r'warfare',
                r'defense\s+application'
            ],
            'societal_impact': [
                r'job\s+(?:loss|displacement)',
                r'economic\s+(?:impact|disruption)',
                r'social\s+(?:inequality|divide)',
                r'community\s+(?:impact|disruption)',
                r'workforce\s+(?:reduction|displacement)',
                r'unemployment',
                r'societal\s+(?:harm|impact)',
                r'digital\s+divide'
            ],
            'transparency': [
                r'black[\s-]box',
                r'(?:lack|no)\s+(?:transparency|explanation)',
                r'(?:not|un)\s*explainable',
                r'(?:cannot|difficult to)\s+(?:understand|interpret)',
                r'opaque\s+(?:system|process)',
                r'accountability',
                r'audit\w*',
                r'interpretability'
            ]
        }
        
        patterns = risk_patterns.get(category, [])
        
        # Search for patterns in sentences
        for sentence in sentences[:50]:  # Check first 50 sentences for performance
            for pattern in patterns:
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    # Clean and format the snippet
                    snippet = sentence.strip()
                    if len(snippet) > 150:
                        # Find match position and extract surrounding text
                        start = max(0, match.start() - 50)
                        end = min(len(sentence), match.end() + 100)
                        snippet = sentence[start:end].strip()
                        if start > 0:
                            snippet = "..." + snippet
                        if end < len(sentence):
                            snippet = snippet + "..."
                    
                    # Highlight the matched part
                    matched_text = match.group()
                    snippets.append(f'"{snippet}" [detected: {matched_text}]')
                    break
            
            if len(snippets) >= 3:
                break
        
        return snippets
    
    def _get_detailed_risk_context(self, text: str, category: str, evidence: List[str]) -> str:
        """Get detailed context about the risk based on evidence"""
        if not evidence:
            return "Limited context available"
        
        # Count evidence types
        text_snippets = len([e for e in evidence if '"' in e])
        analysis_findings = len(evidence) - text_snippets
        
        # Calculate text coverage
        sentences = self.text_processor.get_sentences(text) if hasattr(self.text_processor, 'get_sentences') else text.split('. ')
        total_sentences = len(sentences)
        
        context_parts = []
        
        if text_snippets > 0:
            context_parts.append(f"{text_snippets} specific text passages")
        
        if analysis_findings > 0:
            context_parts.append(f"{analysis_findings} analytical indicators")
        
        # Add scope information
        if text_snippets > 5:
            context_parts.append("pervasive throughout document")
        elif text_snippets > 2:
            context_parts.append("multiple occurrences")
        else:
            context_parts.append("limited occurrences")
        
        # Add confidence indicator
        if len(evidence) >= 4:
            context_parts.append("high confidence")
        elif len(evidence) >= 2:
            context_parts.append("moderate confidence")
        else:
            context_parts.append("preliminary finding")
        
        return f"Risk context: {', '.join(context_parts)}"
    
    
    def _get_analysis_methods_description(self, risk_analysis: Dict[str, Any]) -> str:
        """Get human-readable description of analysis methods used"""
        methods = risk_analysis.get('analysis_methods', ['keyword'])
        
        method_descriptions = {
            'keyword': 'Pattern matching',
            'semantic_svd': 'Semantic analysis',
            'lora_domain': 'Domain expertise',
            'chain_of_thought': 'Logical reasoning'
        }
        
        active_methods = [method_descriptions.get(m, m) for m in methods]
        
        if len(active_methods) == 1:
            return f"Analysis method: {active_methods[0]}"
        else:
            return f"Analysis methods: {', '.join(active_methods)}"
        
    def _generate_recommendations_with_cot(self, risk_analysis: Dict[str, Any], 
                                          sentiment_analysis: Dict[str, Any],
                                          cot_analysis: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate recommendations enhanced with CoT insights"""
        recommendations = []
        
        # Get traditional recommendations
        traditional_recommendations = self._generate_recommendations(risk_analysis, sentiment_analysis)
        
        # Enhance with CoT insights if available
        if cot_analysis and 'error' not in cot_analysis:
            reasoning_chain = cot_analysis.get('reasoning_chain', {})
            
            # Extract mitigation strategies from CoT
            cot_mitigations = self._extract_cot_mitigations(reasoning_chain)
            
            # Merge traditional and CoT recommendations
            for trad_rec in traditional_recommendations:
                enhanced_rec = trad_rec.copy()
                enhanced_rec['reasoning'] = f"Traditional analysis + CoT reasoning: {trad_rec.get('action', '')}"
                recommendations.append(enhanced_rec)
            
            # Add CoT-specific recommendations
            for mitigation in cot_mitigations:
                cot_rec = {
                    'type': 'cot_mitigation',
                    'message': f'CoT Analysis suggests: {mitigation}',
                    'action': mitigation,
                    'priority': self._determine_mitigation_priority(mitigation),
                    'source': 'chain_of_thought'
                }
                recommendations.append(cot_rec)
        else:
            recommendations = traditional_recommendations
        
        return recommendations
    
    
    def _extract_cot_evidence_for_category(self, category: str, reasoning_chain: Dict[str, Any]) -> List[str]:
        """Extract CoT evidence specific to a risk category"""
        evidence = []
        reasoning_steps = reasoning_chain.get('reasoning_steps', [])
        
        for step in reasoning_steps:
            if step['step'] in ['misuse_analysis', 'severity_assessment']:
                findings = step.get('findings', [])
                category_findings = [f for f in findings if category.replace('_', ' ') in f.lower()]
                if category_findings:
                    evidence.append(f"CoT {step['step']}: {category_findings[0]}")
        
        return evidence
    
    def _get_category_cot_reasoning(self, category: str, reasoning_chain: Dict[str, Any]) -> str:
        """Get CoT reasoning specific to a category"""
        reasoning_steps = reasoning_chain.get('reasoning_steps', [])
        
        for step in reasoning_steps:
            if step['step'] == 'severity_assessment':
                reasoning = step.get('reasoning', '')
                if category.replace('_', ' ') in reasoning.lower():
                    # Extract relevant portion
                    lines = reasoning.split('\n')
                    relevant_lines = [line for line in lines if category.replace('_', ' ') in line.lower()]
                    return ' '.join(relevant_lines[:2])  # First 2 relevant lines
        
        return "CoT analysis considered this category in systematic risk assessment"
    
    def _extract_cot_mitigations(self, reasoning_chain: Dict[str, Any]) -> List[str]:
        """Extract mitigation strategies from CoT reasoning"""
        mitigations = []
        reasoning_steps = reasoning_chain.get('reasoning_steps', [])
        
        for step in reasoning_steps:
            if step['step'] == 'mitigation_strategies':
                findings = step.get('findings', [])
                mitigations.extend(findings[:5])  # Top 5 mitigations
                break
        
        return mitigations
    
    def _determine_mitigation_priority(self, mitigation: str) -> str:
        """Determine priority level for a mitigation strategy"""
        high_priority_keywords = ['immediate', 'critical', 'urgent', 'essential', 'mandatory']
        medium_priority_keywords = ['important', 'significant', 'recommended', 'should']
        
        mitigation_lower = mitigation.lower()
        
        if any(keyword in mitigation_lower for keyword in high_priority_keywords):
            return 'critical'
        elif any(keyword in mitigation_lower for keyword in medium_priority_keywords):
            return 'high'
        else:
            return 'medium'
    
    def analyze_text_with_detailed_cot(self, text: str, 
                                      options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Specialized method for detailed Chain of Thought analysis
        Returns comprehensive reasoning breakdown
        """
        # Enable detailed CoT mode
        if not options:
            options = {}
        options['enable_cot'] = True
        options['cot_mode'] = 'standalone'
        
        # Perform analysis
        result = self.analyze_text(text, options)
        
        # Add detailed CoT breakdown
        if 'chain_of_thought' in result:
            cot_data = result['chain_of_thought']
            result['detailed_cot'] = {
                'step_by_step_reasoning': self._format_step_by_step_reasoning(cot_data),
                'risk_reasoning_matrix': self._create_risk_reasoning_matrix(cot_data),
                'confidence_analysis': self._analyze_reasoning_confidence(cot_data),
                'reasoning_quality_score': self._calculate_reasoning_quality(cot_data)
            }
        
        return result
    
    def _format_step_by_step_reasoning(self, cot_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format CoT reasoning into step-by-step breakdown"""
        formatted_steps = []
        reasoning_steps = cot_data.get('reasoning_chain', {}).get('reasoning_steps', [])
        
        for i, step in enumerate(reasoning_steps, 1):
            formatted_step = {
                'step_number': i,
                'step_name': step['step'].replace('_', ' ').title(),
                'objective': self._get_step_objective(step['step']),
                'reasoning': step['reasoning'],
                'key_findings': step['findings'][:5],  # Top 5 findings
                'confidence': step['confidence'],
                'impact_on_final_score': self._calculate_step_impact(step, cot_data)
            }
            formatted_steps.append(formatted_step)
        
        return formatted_steps
    
    def _create_risk_reasoning_matrix(self, cot_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a matrix showing how CoT reasoning led to each risk score"""
        matrix = {}
        cot_risk_scores = cot_data.get('cot_risk_scores', {})
        reasoning_steps = cot_data.get('reasoning_chain', {}).get('reasoning_steps', [])
        
        for category, score in cot_risk_scores.items():
            matrix[category] = {
                'final_score': score,
                'contributing_steps': [],
                'reasoning_path': []
            }
            
            # Trace which steps contributed to this category's score
            for step in reasoning_steps:
                step_findings = step.get('findings', [])
                category_mentions = [f for f in step_findings if category.replace('_', ' ') in f.lower()]
                
                if category_mentions:
                    matrix[category]['contributing_steps'].append({
                        'step': step['step'],
                        'contribution': category_mentions[0],
                        'confidence': step['confidence']
                    })
                    matrix[category]['reasoning_path'].append(step['step'])
        
        return matrix
    
    def _analyze_reasoning_confidence(self, cot_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the confidence levels throughout the reasoning chain"""
        reasoning_steps = cot_data.get('reasoning_chain', {}).get('reasoning_steps', [])
        
        confidences = [step['confidence'] for step in reasoning_steps]
        
        return {
            'average_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'confidence_trend': self._calculate_confidence_trend(confidences),
            'lowest_confidence_step': min(reasoning_steps, key=lambda x: x['confidence'])['step'] if reasoning_steps else None,
            'highest_confidence_step': max(reasoning_steps, key=lambda x: x['confidence'])['step'] if reasoning_steps else None,
            'confidence_variance': np.var(confidences) if confidences else 0
        }
    
    def _calculate_reasoning_quality(self, cot_data: Dict[str, Any]) -> float:
        """Calculate overall quality score for the reasoning process"""
        reasoning_steps = cot_data.get('reasoning_chain', {}).get('reasoning_steps', [])
        
        if not reasoning_steps:
            return 0.0
        
        # Factors contributing to reasoning quality
        factors = {
            'completeness': len(reasoning_steps) / 6.0,  # Expected 6 steps
            'avg_confidence': sum(step['confidence'] for step in reasoning_steps) / len(reasoning_steps),
            'finding_richness': sum(len(step['findings']) for step in reasoning_steps) / (len(reasoning_steps) * 5),
            'reasoning_depth': sum(len(step['reasoning'].split()) for step in reasoning_steps) / (len(reasoning_steps) * 50)
        }
        
        # Weight the factors
        weights = {'completeness': 0.3, 'avg_confidence': 0.3, 'finding_richness': 0.2, 'reasoning_depth': 0.2}
        
        quality_score = sum(min(factors[factor], 1.0) * weight for factor, weight in weights.items())
        return min(quality_score, 1.0)
    
    def _get_step_objective(self, step_type: str) -> str:
        """Get the objective description for each reasoning step"""
        objectives = {
            'technique_identification': 'Identify the AI techniques and methodologies discussed',
            'misuse_analysis': 'Analyze potential ways the technology could be misused',
            'severity_assessment': 'Evaluate the severity and likelihood of identified risks',
            'stakeholder_impact': 'Assess impact on different groups and communities',
            'mitigation_strategies': 'Develop strategies to mitigate identified risks',
            'final_recommendation': 'Synthesize analysis into actionable recommendations'
        }
        return objectives.get(step_type, 'Process information for risk assessment')
    
    def _calculate_step_impact(self, step: Dict[str, Any], cot_data: Dict[str, Any]) -> float:
        """Calculate how much this step impacted the final risk scores"""
        # Simplified impact calculation based on step type and confidence
        step_weights = {
            'technique_identification': 0.1,
            'misuse_analysis': 0.25,
            'severity_assessment': 0.35,
            'stakeholder_impact': 0.15,
            'mitigation_strategies': 0.1,
            'final_recommendation': 0.05
        }
        
        base_weight = step_weights.get(step['step'], 0.1)
        confidence_modifier = step['confidence']
        findings_modifier = min(len(step['findings']) / 5.0, 1.0)
        
        return base_weight * confidence_modifier * findings_modifier
    
    def _calculate_confidence_trend(self, confidences: List[float]) -> str:
        """Calculate the trend in confidence levels across reasoning steps"""
        if len(confidences) < 2:
            return 'insufficient_data'
        
        # Simple trend calculation
        differences = [confidences[i+1] - confidences[i] for i in range(len(confidences)-1)]
        avg_change = sum(differences) / len(differences)
        
        if avg_change > 0.05:
            return 'increasing'
        elif avg_change < -0.05:
            return 'decreasing'
        else:
            return 'stable'
    
# Add helper methods:
    def _extract_category_keywords(self, text: str, category: str) -> List[str]:
        """Extract specific keywords found in text for a category"""
        keywords_found = []
        text_lower = text.lower()
        
        # Get category-specific keywords
        category_keywords = self.risk_analyzer.risk_keywords.get(category, {})
        
        for severity, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    keywords_found.append(keyword)
        
        # Remove duplicates and limit
        return list(set(keywords_found))[:10]

    def _get_risk_context(self, text: str, category: str) -> str:
        """Get contextual information about the risk"""
        # Extract sentences around risk indicators
        sentences = self.text_processor.get_sentences(text)
        # This is simplified - in practice would be more sophisticated
        return f"Risk detected across {len(sentences)} sentences"

    def _get_domain_confidence(self, text: str, domain: str) -> float:
        """Calculate confidence in domain detection"""
        # Use semantic similarity to domain patterns
        if hasattr(self, 'semantic_analyzer'):
            domain_patterns = {
                'biomedical': "medical clinical patient diagnosis treatment healthcare",
                'legal': "law legal court justice regulation compliance",
                'technical': "algorithm model system architecture implementation",
                # ... etc
            }
            if domain in domain_patterns:
                # This is simplified - would use proper similarity calculation
                return 0.85
        return 0.7

    # Add method to save semantic model:
    def save_semantic_model(self, save_path: str):
        """Save the semantic model (only works for SVD analyzer)"""
        if hasattr(self, 'analyzer_type') and self.analyzer_type == 'svd':
            import pickle
            import os
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            save_data = {
                'vectorizer': self.semantic_analyzer.vectorizer,
                'svd': self.semantic_analyzer.svd,
                'risk_concepts': self.semantic_analyzer.risk_concepts,
                'is_fitted': self.semantic_analyzer.is_fitted
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            logger.info(f"Saved SVD semantic model to {save_path}")
        else:
            logger.warning(f"Cannot save {getattr(self, 'analyzer_type', 'unknown')} analyzer model using this method")
        
    def _analyze_risks(self, processed_text: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze text for various risk factors"""
        
        # Get risk keywords from config
        high_risk_keywords = self.config.RISK_KEYWORDS['high_risk']
        medium_risk_keywords = self.config.RISK_KEYWORDS['medium_risk']
        low_risk_keywords = self.config.RISK_KEYWORDS['low_risk']
        
        # Count risk keywords
        text_lower = processed_text['cleaned'].lower()
        tokens_lower = [t.lower() for t in processed_text['tokens']]
        
        high_risk_count = sum(1 for keyword in high_risk_keywords if keyword in text_lower)
        medium_risk_count = sum(1 for keyword in medium_risk_keywords if keyword in text_lower)
        low_risk_count = sum(1 for keyword in low_risk_keywords if keyword in text_lower)
        
        # Calculate risk scores
        total_words = len(processed_text['tokens'])
        if total_words == 0:
            critical_risk_score = 0
            overall_risk_score = 0
        else:
            # Weighted risk calculation
            critical_risk_score = min(1.0, (high_risk_count * 0.3 + medium_risk_count * 0.1) / max(1, total_words / 100))
            overall_risk_score = min(1.0, (high_risk_count * 0.2 + medium_risk_count * 0.1 + low_risk_count * 0.05) / max(1, total_words / 100))
        
        # Detect specific risk categories
        risk_categories = self._detect_risk_categories(text_lower, tokens_lower)
        
        # Determine risk levels
        critical_risk_level = risk_level_from_score(critical_risk_score)
        overall_risk_level = risk_level_from_score(overall_risk_score)
        
        return {
            'critical_risk': {
                'score': critical_risk_score,
                'level': critical_risk_level,
                'count': high_risk_count
            },
            'overall_risk': {
                'score': overall_risk_score,
                'level': overall_risk_level
            },
            'risk_categories': risk_categories,
            'keyword_counts': {
                'high': high_risk_count,
                'medium': medium_risk_count,
                'low': low_risk_count
            }
        }
    
    def _detect_risk_categories(self, text_lower: str, tokens_lower: List[str]) -> Dict[str, float]:
        """Detect specific risk categories in text"""
        
        categories = {
            RiskCategory.VIOLENCE: {
                'keywords': ['violence', 'violent', 'attack', 'assault', 'fight', 'weapon', 'gun', 'knife', 'bomb'],
                'patterns': [r'\b(kill|hurt|harm|attack)\s+(someone|people|them|him|her)\b']
            },
            RiskCategory.SELF_HARM: {
                'keywords': ['suicide', 'suicidal', 'self-harm', 'cut myself', 'end my life', 'kill myself'],
                'patterns': [r'\b(want|going)\s+to\s+(die|kill\s+myself)\b']
            },
            RiskCategory.HATE_SPEECH: {
                'keywords': ['hate', 'racist', 'racism', 'discrimination', 'bigot'],
                'patterns': [r'\b(hate|despise)\s+(all|every)\s+\w+\b']
            },
            RiskCategory.HARASSMENT: {
                'keywords': ['harass', 'bully', 'threaten', 'stalk', 'intimidate'],
                'patterns': [r'\b(going\s+to|will)\s+(find|get|hurt)\s+you\b']
            }
        }
        
        detected_categories = {}
        
        for category, config in categories.items():
            score = 0
            
            # Check keywords
            keyword_matches = sum(1 for keyword in config['keywords'] if keyword in text_lower)
            score += keyword_matches * 0.2
            
            # Check patterns
            for pattern in config['patterns']:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    score += 0.3
            
            if score > 0:
                detected_categories[category.value] = min(1.0, score)
        
        return detected_categories
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of the text"""
        try:
            # Use TextBlob for sentiment analysis
            blob = TextBlob(text)
            
            # Get polarity and subjectivity
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Calculate sentiment scores
            if polarity > 0.1:
                positive_score = min(1.0, polarity)
                negative_score = 0
                neutral_score = 1 - positive_score
            elif polarity < -0.1:
                positive_score = 0
                negative_score = min(1.0, abs(polarity))
                neutral_score = 1 - negative_score
            else:
                positive_score = 0.1
                negative_score = 0.1
                neutral_score = 0.8
            
            # Determine sentiment type
            sentiment_type = sentiment_type_from_scores(positive_score, negative_score, neutral_score)
            
            # Extract emotional keywords
            emotional_keywords = self._extract_emotional_keywords(text)
            
            return { 
                'score': polarity,
                'type': sentiment_type.value,
                'positive_score': positive_score,
                'negative_score': negative_score,
                'neutral_score': neutral_score,
                'subjectivity': subjectivity,
                'keywords': emotional_keywords
            }
           
        except Exception as e:
            # Return neutral sentiment on error
            return {
                'score': 0,
                'type': SentimentType.NEUTRAL.value,
                'positive_score': 0,
                'negative_score': 0,
                'neutral_score': 1.0,
                'subjectivity': 0,
                'keywords': {'positive': [], 'negative': []}
            }
    def _extract_emotional_keywords(self, text: str) -> Dict[str, List[str]]:
       """Extract emotional keywords from text"""
       positive_words = ['happy', 'joy', 'love', 'excellent', 'good', 'wonderful', 'fantastic', 'amazing']
       negative_words = ['sad', 'angry', 'hate', 'terrible', 'awful', 'horrible', 'bad', 'worst']
       
       text_lower = text.lower()
       words = text_lower.split()
       
       found_positive = [word for word in positive_words if word in words]
       found_negative = [word for word in negative_words if word in words]
       
       return {
           'positive': found_positive,
           'negative': found_negative
       }
    
    def _generate_statistics(self, processed_text: Dict[str, Any], risk_analysis: Dict[str, Any]) -> Dict[str, Any]:
       """Generate text statistics"""
       return {
           'word_count': len(processed_text['tokens']),
           'character_count': len(processed_text['original']),
           'sentence_count': len(processed_text['sentences']),
           'avg_word_length': np.mean([len(token) for token in processed_text['tokens']]) if processed_text['tokens'] else 0,
           'high_risk_keywords': risk_analysis['keyword_counts']['high'],
           'medium_risk_keywords': risk_analysis['keyword_counts']['medium']
       }
   
    def _generate_risk_assessments_with_semantic(self, text: str, processed_text: Dict[str, Any], 
                                            risk_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate detailed risk assessments with semantic evidence"""
        assessments = []
        
        # Add assessments for detected risk categories
        for category, score in risk_analysis['risk_categories'].items():
            level = risk_level_from_score(score)
            
            # Get traditional evidence
            traditional_evidence = self.risk_analyzer.get_risk_evidence(text, category)
            
            # Get semantic evidence
            semantic_evidence = []
            if hasattr(self, 'semantic_integrator'):
                semantic_evidence = self.semantic_integrator.generate_semantic_evidence(
                    text, category
                )
            
            # Combine evidence
            all_evidence = traditional_evidence[:3] + semantic_evidence[:2]
            
            assessment = {
                'category': category,
                'level': level.value,
                'score': score,
                'confidence': min(0.9, score + 0.3),
                'keywords': self._extract_category_keywords(text, category),
                'evidence': all_evidence,
                'context': self._get_risk_context(text, category),
                'analysis_type': 'hybrid'  # keyword + semantic
            }
            assessments.append(assessment)
        
        # Add overall risk assessment if significant
        if risk_analysis['critical_risk']['score'] > 0.3:
            assessments.append({
                'category': 'overall',
                'level': risk_analysis['critical_risk']['level'],
                'score': risk_analysis['critical_risk']['score'],
                'confidence': 0.85,
                'keywords': [],
                'evidence': ['Multiple risk factors detected through semantic and keyword analysis'],
                'context': 'Comprehensive analysis indicates elevated risk levels',
                'analysis_type': 'combined'
            })
        
        return assessments

    def _extract_category_keywords(self, text: str, category: str) -> List[str]:
        """Extract keywords relevant to a risk category"""
        # This would use the semantic analyzer to find relevant terms
        if hasattr(self, 'semantic_analyzer') and self.semantic_analyzer.is_fitted:
            explanation = self.semantic_analyzer.explain_risk_detection(text, category)
            return [term['term'] for term in explanation['important_terms'][:5]]
        return []

    def _get_risk_context(self, text: str, category: str) -> str:
        """Get contextual information about the risk"""
        # Extract sentences around risk indicators
        sentences = self.text_processor.get_sentences(text)
        # This is simplified - in practice would be more sophisticated
        return f"Risk detected across {len(sentences)} sentences"

    def _get_domain_confidence(self, text: str, domain: str) -> float:
        """Calculate confidence in domain detection"""
        # Use semantic similarity to domain patterns
        if hasattr(self, 'semantic_analyzer'):
            domain_patterns = {
                'biomedical': "medical clinical patient diagnosis treatment healthcare",
                'legal': "law legal court justice regulation compliance",
                'technical': "algorithm model system architecture implementation",
                # ... etc
            }
            if domain in domain_patterns:
                # This is simplified - would use proper similarity calculation
                return 0.85
        return 0.7

    def _generate_recommendations(self, risk_analysis: Dict[str, Any], sentiment_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Critical risk recommendations
        if risk_analysis['critical_risk']['level'] in ['high', 'critical']:
            recommendations.append({
                'type': 'critical',
                'message': 'High risk content detected. Immediate review required.',
                'action': 'Flag for immediate human review and potential intervention',
                'priority': 'critical'
            })
        
        # Sentiment-based recommendations
        if sentiment_analysis['score'] < -0.5:
            recommendations.append({
                'type': 'sentiment',
                'message': 'Highly negative sentiment detected.',
                'action': 'Consider providing emotional support resources',
                'priority': 'high'
            })
        
        # Category-specific recommendations
        if RiskCategory.SELF_HARM.value in risk_analysis['risk_categories']:
            recommendations.append({
                'type': 'intervention',
                'message': 'Self-harm indicators detected.',
                'action': 'Provide crisis helpline information and support resources',
                'priority': 'critical'
            })
        
        if RiskCategory.VIOLENCE.value in risk_analysis['risk_categories']:
            recommendations.append({
                'type': 'safety',
                'message': 'Violence-related content detected.',
                'action': 'Review for potential threats and take appropriate action',
                'priority': 'high'
            })
        
        # Default recommendation if no issues
        if not recommendations:
            recommendations.append({
                'type': 'safe',
                'message': 'No significant risks detected.',
                'action': 'Continue normal monitoring',
                'priority': 'low'
            })
        
        return recommendations
   
    def _detect_domain(self, text: str) -> str:
        """Detect the domain of the research paper"""
        domain_keywords = {
            'biomedical': ['patient', 'medical', 'clinical', 'disease', 'treatment', 'diagnosis'],
            'legal': ['law', 'legal', 'court', 'justice', 'regulation', 'compliance'],
            'financial': ['financial', 'banking', 'investment', 'market', 'trading'],
            'technical': ['algorithm', 'model', 'system', 'architecture', 'implementation'],
            'social': ['social', 'society', 'community', 'behavior', 'interaction']
        }
        
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            domain_scores[domain] = score
        
        # Return domain with highest score
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        return 'general'
   
    def _merge_risk_scores(self, traditional_risks: Dict, lora_risks: Dict) -> Dict:
        """Merge traditional and LoRA-based risk scores"""
        merged = traditional_risks.copy()
        
        if 'risk_categories' not in merged:
            merged['risk_categories'] = {}
        
        # Weight: 60% traditional, 40% LoRA
        for category, lora_score in lora_risks.items():
            if category in merged['risk_categories']:
                traditional_score = merged['risk_categories'][category]
                merged['risk_categories'][category] = 0.6 * traditional_score + 0.4 * lora_score
            else:
                merged['risk_categories'][category] = lora_score
        
        return merged

   # Save semantic model method
    def save_semantic_model(self, save_path: str):
        """Save the trained semantic model"""
        import pickle
        import os
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        save_data = {
            'vectorizer': self.semantic_analyzer.vectorizer,
            'svd': self.semantic_analyzer.svd,
            'risk_concepts': self.semantic_analyzer.risk_concepts
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Saved semantic model to {save_path}")