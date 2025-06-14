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
from core.semantic_analyzer import SemanticRiskAnalyzer, SemanticRiskIntegrator
from core.cot_analyzer import ChainOfThoughtAnalyzer, CoTIntegrator  # New import

import logging

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
        """Get engine status including CoT analyzer"""
        return {
            'initialized': self.is_initialized,
            'version': '1.0.0',
            'components': {
                'text_processor': 'active',
                'risk_analyzer': 'active',
                'sentiment_analyzer': 'active',
                'semantic_analyzer': 'active',
                'lora_adapter': 'active' if self.lora_adapter else 'inactive',
                'cot_analyzer': 'active'  # New component
            }
        }
        
    def _initialize_semantic_analyzer(self):
        """Initialize the semantic analyzer with training data or patterns"""
        try:
            import os
            model_path = os.path.join(self.config.MODEL_CACHE_DIR, 'semantic_model.pkl')
            
            # Create models directory if it doesn't exist
            os.makedirs(self.config.MODEL_CACHE_DIR, exist_ok=True)
            
            if os.path.exists(model_path):
                # Load pre-trained model
                import pickle
                with open(model_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.semantic_analyzer.vectorizer = saved_data.get('vectorizer')
                    self.semantic_analyzer.svd = saved_data.get('svd')
                    self.semantic_analyzer.risk_concepts = saved_data.get('risk_concepts', {})
                    self.semantic_analyzer.is_fitted = saved_data.get('is_fitted', False)
                logger.info("Loaded pre-trained semantic model")
            else:
                # Initialize with patterns if no saved model
                logger.info("No saved semantic model found, initializing with patterns")
                self.semantic_analyzer._initialize_with_patterns()
                self.semantic_analyzer.is_fitted = True
                
        except Exception as e:
            logger.error(f"Failed to initialize semantic analyzer: {e}")
            # Fallback to pattern-based initialization
            self.semantic_analyzer._initialize_with_patterns()
            self.semantic_analyzer.is_fitted = True
                    
    def analyze_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform complete text analysis with LoRA, SVD, and Chain of Thought enhancement
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Check if CoT analysis is enabled in options
        enable_cot = options.get('enable_cot', True) if options else True
        cot_mode = options.get('cot_mode', 'enhanced') if options else 'enhanced'  # 'enhanced', 'standalone', 'disabled'
        
        # Process text
        processed_text = self.text_processor.process(text)
        
        # Detect domain
        domain = self._detect_domain(text)
        
        # Perform traditional risk analysis
        risk_analysis = self._analyze_risks(processed_text)
        
        # Enhance with semantic analysis (SVD)
        if hasattr(self, 'semantic_integrator'):
            traditional_scores = {
                cat: score for cat, score in risk_analysis['risk_categories'].items()
            }
            enhanced_scores = self.semantic_integrator.enhance_risk_analysis(
                text, traditional_scores
            )
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

    def _generate_risk_assessments_with_cot(self, text: str, processed_text: Dict[str, Any], 
                                           risk_analysis: Dict[str, Any], 
                                           cot_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate detailed risk assessments enhanced with CoT reasoning"""
        assessments = []
        
        # Get CoT reasoning for context
        reasoning_chain = cot_analysis.get('reasoning_chain', {})
        cot_risk_scores = cot_analysis.get('cot_risk_scores', {})
        
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
            
            # Get CoT reasoning for this category
            cot_evidence = self._extract_cot_evidence_for_category(category, reasoning_chain)
            
            # Combine all evidence sources
            all_evidence = traditional_evidence[:2] + semantic_evidence[:2] + cot_evidence[:2]
            
            # Enhanced confidence calculation using CoT
            base_confidence = min(0.9, score + 0.3)
            cot_confidence = cot_analysis.get('cot_confidence', 0.5)
            combined_confidence = (base_confidence + cot_confidence) / 2
            
            assessment = {
                'category': category,
                'level': level.value,
                'score': score,
                'confidence': combined_confidence,
                'keywords': self._extract_category_keywords(text, category),
                'evidence': all_evidence,
                'context': self._get_risk_context(text, category),
                'analysis_type': 'hybrid_cot',  # keyword + semantic + CoT
                'cot_reasoning': self._get_category_cot_reasoning(category, reasoning_chain)
            }
            assessments.append(assessment)
        
        # Add overall risk assessment if significant
        if risk_analysis['critical_risk']['score'] > 0.3:
            overall_reasoning = reasoning_chain.get('reasoning_steps', [])
            final_recommendation = next(
                (step for step in overall_reasoning if step['step'] == 'final_recommendation'), 
                None
            )
            
            assessments.append({
                'category': 'overall',
                'level': risk_analysis['critical_risk']['level'],
                'score': risk_analysis['critical_risk']['score'],
                'confidence': combined_confidence,
                'keywords': [],
                'evidence': ['Multi-layered analysis including Chain of Thought reasoning'],
                'context': 'Comprehensive analysis with systematic ethical reasoning',
                'analysis_type': 'comprehensive_cot',
                'cot_reasoning': final_recommendation['reasoning'] if final_recommendation else ''
            })
        
        return assessments
    
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

    # Add method to save semantic model:
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