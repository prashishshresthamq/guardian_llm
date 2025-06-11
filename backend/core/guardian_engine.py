# backend/core/guardian_engine.py
"""
Guardian LLM Engine - Main AI Analysis Engine
============================================

This is the core Guardian LLM engine that orchestrates all risk analysis 
components. It follows the separated architecture pattern where the engine
focuses purely on AI analysis logic, separate from web/API concerns.

Author: Prashish Shrestha
Course: COMP8420 - Advanced Topics in AI
Project: Guardian LLM - Automated Ethical Risk Auditor
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import asdict

# Import from other modules in the separated architecture
from backend.models.schemas import RiskAssessment, RiskCategory, RiskLevel, AnalysisRequest, AnalysisResponse
from backend.core.risk_analyzers import KeywordAnalyzer, BiasAnalyzer, PrivacyAnalyzer, TransformerAnalyzer
from backend.core.text_processors import TextProcessor

logger = logging.getLogger(__name__)

class GuardianEngine:
    """
    Main Guardian LLM Analysis Engine
    
    This class orchestrates all risk analysis components to provide comprehensive
    ethical risk assessment for AI research papers. It's designed to be independent
    of web frameworks and can be used via API, CLI, or direct Python integration.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Guardian Engine with configuration
        
        Args:
            config: Optional configuration dictionary with settings
        """
        self.config = config or {}
        
        # Initialize text processor
        self.text_processor = TextProcessor()
        
        # Initialize all risk analyzers
        self._initialize_analyzers()
        
        # Set up risk thresholds and weights
        self._setup_configuration()
        
        logger.info("🚀 Guardian Engine initialized successfully")
    
    def _initialize_analyzers(self) -> None:
        """Initialize all risk analysis components"""
        try:
            logger.info("📡 Initializing analyzers...")
            
            self.keyword_analyzer = KeywordAnalyzer()
            logger.info("✅ Keyword analyzer ready")
            
            self.bias_analyzer = BiasAnalyzer()
            logger.info("✅ Bias analyzer ready")
            
            self.privacy_analyzer = PrivacyAnalyzer()
            logger.info("✅ Privacy analyzer ready")
            
            # Transformer analyzer might fail if models not available
            try:
                model_name = self.config.get('transformer_model', 'bert-base-uncased')
                self.transformer_analyzer = TransformerAnalyzer(model_name=model_name)
                logger.info("✅ Transformer analyzer ready")
            except Exception as e:
                logger.warning(f"⚠️ Transformer analyzer failed to load: {e}")
                self.transformer_analyzer = None
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize analyzers: {e}")
            raise
    
    def _setup_configuration(self) -> None:
        """Setup risk thresholds and analyzer weights"""
        # Risk level thresholds
        self.risk_thresholds = self.config.get('risk_thresholds', {
            'low': (0.0, 2.5),
            'medium': (2.5, 5.0),
            'high': (5.0, 7.5),
            'critical': (7.5, 10.0)
        })
        
        # Analyzer weights for combining scores
        self.analyzer_weights = self.config.get('analyzer_weights', {
            'keyword': 0.3,
            'specialized': 0.4,  # bias and privacy analyzers
            'transformer': 0.3
        })
        
        # Performance settings
        self.max_text_length = self.config.get('max_text_length', 100000)  # 100KB
        self.min_text_length = self.config.get('min_text_length', 50)
        
        logger.info(f"⚙️ Configuration loaded - thresholds: {len(self.risk_thresholds)} levels")
    
    def analyze_paper(self, request: AnalysisRequest) -> AnalysisResponse:
        """
        Main entry point for paper analysis
        
        Args:
            request: AnalysisRequest containing paper data
            
        Returns:
            AnalysisResponse with complete risk assessment
        """
        start_time = datetime.now()
        
        # Validate input
        if not self._validate_input(request):
            raise ValueError("Invalid analysis request")
        
        # Generate unique paper ID
        paper_id = f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Prepare text for analysis
        full_text = self._prepare_text(request.title, request.abstract or "", request.content)
        
        logger.info(f"📄 Analyzing paper: '{request.title[:50]}...' ({len(full_text):,} characters)")
        
        # Perform comprehensive analysis
        risk_assessments = self._perform_analysis(full_text, request.title)
        
        # Calculate overall metrics
        overall_risk_score = self._calculate_overall_score(risk_assessments)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create response
        response = AnalysisResponse(
            paper_id=paper_id,
            title=request.title,
            overall_risk_score=round(overall_risk_score, 2),
            processing_time=round(processing_time, 2),
            risk_assessments=[asdict(assessment) for assessment in risk_assessments],
            timestamp=datetime.now().isoformat(),
            status='completed'
        )
        
        # Log completion
        self._log_analysis_summary(request.title, overall_risk_score, risk_assessments, processing_time)
        
        return response
    
    def analyze_text(self, title: str, abstract: str = "", content: str = "") -> List[RiskAssessment]:
        """
        Direct text analysis method (backward compatibility)
        
        Args:
            title: Paper title
            abstract: Paper abstract (optional)
            content: Full paper content
            
        Returns:
            List of RiskAssessment objects
        """
        request = AnalysisRequest(
            title=title,
            abstract=abstract,
            content=content
        )
        
        response = self.analyze_paper(request)
        
        # Convert dict assessments back to RiskAssessment objects
        assessments = []
        for assessment_dict in response.risk_assessments:
            assessment = RiskAssessment(
                category=RiskCategory(assessment_dict['category']),
                score=assessment_dict['score'],
                confidence=assessment_dict['confidence'],
                level=RiskLevel(assessment_dict['level']),
                explanation=assessment_dict['explanation'],
                evidence=assessment_dict['evidence'],
                recommendations=assessment_dict['recommendations']
            )
            assessments.append(assessment)
        
        return assessments
    
    def _validate_input(self, request: AnalysisRequest) -> bool:
        """Validate analysis request"""
        if not request.title and not request.content:
            logger.error("No title or content provided")
            return False
        
        full_text = f"{request.title} {request.abstract or ''} {request.content}"
        
        if len(full_text.strip()) < self.min_text_length:
            logger.error(f"Text too short: {len(full_text)} < {self.min_text_length}")
            return False
        
        if len(full_text) > self.max_text_length:
            logger.error(f"Text too long: {len(full_text)} > {self.max_text_length}")
            return False
        
        return True
    
    def _prepare_text(self, title: str, abstract: str, content: str) -> str:
        """Prepare and clean text for analysis"""
        # Combine all text
        full_text = f"{title}\n\n{abstract}\n\n{content}"
        
        # Clean and preprocess
        clean_text = self.text_processor.clean_text(full_text)
        
        # Truncate if too long (safety measure)
        if len(clean_text) > self.max_text_length:
            clean_text = clean_text[:self.max_text_length]
            logger.warning(f"Text truncated to {self.max_text_length} characters")
        
        return clean_text
    
    def _perform_analysis(self, text: str, title: str) -> List[RiskAssessment]:
        """Perform comprehensive risk analysis"""
        
        # Extract paper sections for enhanced analysis
        sections = self.text_processor.extract_sections(text)
        
        # Run all analyzers
        logger.info("🔍 Running keyword analysis...")
        keyword_scores = self.keyword_analyzer.analyze(text)
        
        logger.info("⚖️ Running bias analysis...")
        bias_score, bias_evidence = self.bias_analyzer.analyze(text)
        
        logger.info("🔒 Running privacy analysis...")
        privacy_score, privacy_evidence = self.privacy_analyzer.analyze(text)
        
        # Run transformer analysis if available
        transformer_scores = {}
        if self.transformer_analyzer:
            logger.info("🤖 Running transformer analysis...")
            transformer_scores = self.transformer_analyzer.analyze(text)
        else:
            logger.info("⏭️ Skipping transformer analysis (not available)")
            transformer_scores = {category.value: 0.0 for category in RiskCategory}
        
        # Generate assessments for each category
        assessments = []
        
        for category in RiskCategory:
            assessment = self._create_category_assessment(
                category=category,
                keyword_scores=keyword_scores,
                bias_score=bias_score,
                bias_evidence=bias_evidence,
                privacy_score=privacy_score,
                privacy_evidence=privacy_evidence,
                transformer_scores=transformer_scores,
                text=text,
                sections=sections
            )
            assessments.append(assessment)
        
        return assessments
    
    def _create_category_assessment(
        self,
        category: RiskCategory,
        keyword_scores: Dict,
        bias_score: float,
        bias_evidence: List[str],
        privacy_score: float,
        privacy_evidence: List[str],
        transformer_scores: Dict,
        text: str,
        sections: Dict[str, str]
    ) -> RiskAssessment:
        """Create detailed assessment for a specific risk category"""
        
        # Get base scores
        keyword_score = keyword_scores.get(category, 0.0)
        transformer_score = transformer_scores.get(category.value, 0.0)
        
        # Category-specific scoring and evidence
        evidence = []
        specialized_score = 0.0
        
        if category == RiskCategory.BIAS_FAIRNESS:
            specialized_score = bias_score
            evidence = bias_evidence.copy()
        elif category == RiskCategory.PRIVACY_DATA:
            specialized_score = privacy_score
            evidence = privacy_evidence.copy()
        
        # Weighted combination of scores
        if specialized_score > 0:
            combined_score = (
                keyword_score * self.analyzer_weights['keyword'] +
                specialized_score * self.analyzer_weights['specialized'] +
                transformer_score * self.analyzer_weights['transformer']
            )
        else:
            # No specialized analyzer for this category
            combined_score = (keyword_score + transformer_score) / 2
        
        # Apply contextual adjustments
        combined_score = self._apply_contextual_adjustments(
            score=combined_score,
            category=category,
            sections=sections,
            text=text
        )
        
        # Add general evidence if scores are significant
        if keyword_score > 2.0:
            evidence.append(f"Keyword relevance: {keyword_score:.1f}/10")
        if transformer_score > 2.0:
            evidence.append(f"Semantic analysis: {transformer_score:.1f}/10")
        
        # Determine risk level and confidence
        risk_level = self._determine_risk_level(combined_score)
        confidence = self._calculate_confidence(combined_score, evidence, category)
        
        # Generate explanation and recommendations
        explanation = self._generate_explanation(category, combined_score, evidence, sections)
        recommendations = self._generate_recommendations(category, risk_level, evidence)
        
        return RiskAssessment(
            category=category,
            score=round(combined_score, 2),
            confidence=round(confidence, 2),
            level=risk_level,
            explanation=explanation,
            evidence=evidence[:5],  # Limit evidence items
            recommendations=recommendations[:8]  # Limit recommendations
        )
    
    def _apply_contextual_adjustments(
        self,
        score: float,
        category: RiskCategory,
        sections: Dict[str, str],
        text: str
    ) -> float:
        """Apply context-aware score adjustments"""
        
        # Abstract vs full paper adjustment
        abstract_length = len(sections.get('abstract', ''))
        content_length = len(text)
        
        if abstract_length > 0 and content_length > abstract_length * 3:
            # Full paper available - higher confidence in scores
            score *= 1.1
        elif abstract_length > 0 and content_length < abstract_length * 2:
            # Likely only abstract - reduce confidence
            score *= 0.8
        
        # Research stage indicators
        stage_indicators = {
            'proposal': ['propose', 'plan to', 'will investigate', 'future work'],
            'ongoing': ['preliminary', 'ongoing', 'in progress', 'developing'],
            'completed': ['results', 'conclusion', 'demonstrated', 'achieved']
        }
        
        text_lower = text.lower()
        
        for stage, indicators in stage_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                if stage == 'proposal':
                    score *= 0.7  # Proposed work has lower immediate risk
                elif stage == 'completed':
                    score *= 1.2  # Completed work has higher risk
                break
        
        # Mitigation mentions
        mitigation_terms = [
            'mitigate', 'address', 'prevent', 'avoid', 'reduce risk',
            'safety measure', 'protection', 'safeguard', 'ethical review'
        ]
        
        if any(term in text_lower for term in mitigation_terms):
            score *= 0.9  # Slight reduction for acknowledged mitigation
        
        return min(score, 10.0)
    
    def _determine_risk_level(self, score: float) -> RiskLevel:
        """Determine risk level based on score"""
        for level_name, (min_score, max_score) in self.risk_thresholds.items():
            if min_score <= score < max_score:
                return RiskLevel(level_name)
        return RiskLevel.CRITICAL
    
    def _calculate_confidence(self, score: float, evidence: List[str], category: RiskCategory) -> float:
        """Calculate confidence in the risk assessment"""
        base_confidence = 0.7
        
        # Higher scores generally mean higher confidence
        score_confidence = min(score / 10.0 * 0.2, 0.2)
        
        # More evidence means higher confidence
        evidence_confidence = min(len(evidence) * 0.05, 0.15)
        
        # Category-specific confidence adjustments
        category_adjustments = {
            RiskCategory.BIAS_FAIRNESS: 0.05,  # Well-established detection
            RiskCategory.PRIVACY_DATA: 0.05,   # Clear indicators
            RiskCategory.DUAL_USE: -0.05,      # More subjective
            RiskCategory.SOCIETAL_IMPACT: -0.05,  # Harder to predict
            RiskCategory.TRANSPARENCY: 0.0,
            RiskCategory.SAFETY_SECURITY: 0.0
        }
        
        category_adj = category_adjustments.get(category, 0.0)
        
        final_confidence = base_confidence + score_confidence + evidence_confidence + category_adj
        return min(max(final_confidence, 0.5), 1.0)  # Clamp between 0.5 and 1.0
    
    def _generate_explanation(
        self,
        category: RiskCategory,
        score: float,
        evidence: List[str],
        sections: Dict[str, str]
    ) -> str:
        """Generate human-readable explanation for risk assessment"""
        
        category_names = {
            RiskCategory.BIAS_FAIRNESS: "Bias and Fairness",
            RiskCategory.PRIVACY_DATA: "Privacy and Data Protection", 
            RiskCategory.SAFETY_SECURITY: "Safety and Security",
            RiskCategory.DUAL_USE: "Dual-Use Potential",
            RiskCategory.SOCIETAL_IMPACT: "Societal Impact",
            RiskCategory.TRANSPARENCY: "Transparency and Explainability"
        }
        
        category_name = category_names.get(category, category.value.replace('_', ' ').title())
        
        # Base explanation with score context
        if score < 2.5:
            risk_desc = "minimal risk indicators"
        elif score < 5.0:
            risk_desc = "moderate risk considerations"
        elif score < 7.5:
            risk_desc = "significant risk factors"
        else:
            risk_desc = "critical risk concerns"
        
        explanation = f"{category_name} analysis detected {risk_desc} with a score of {score:.1f}/10. "
        
        # Add specific findings
        if evidence:
            if len(evidence) == 1:
                explanation += f"Key finding: {evidence[0]}."
            else:
                explanation += f"Key findings include: {'; '.join(evidence[:3])}."
        
        # Add section-specific insights
        relevant_sections = self._get_relevant_sections(category, sections)
        if relevant_sections:
            explanation += f" Analysis focused on {', '.join(relevant_sections)} sections."
        
        return explanation
    
    def _get_relevant_sections(self, category: RiskCategory, sections: Dict[str, str]) -> List[str]:
        """Identify paper sections most relevant to each risk category"""
        section_relevance = {
            RiskCategory.BIAS_FAIRNESS: ['methodology', 'results', 'discussion'],
            RiskCategory.PRIVACY_DATA: ['methodology', 'abstract'],
            RiskCategory.SAFETY_SECURITY: ['methodology', 'discussion', 'conclusion'],
            RiskCategory.DUAL_USE: ['abstract', 'introduction', 'conclusion'],
            RiskCategory.SOCIETAL_IMPACT: ['introduction', 'discussion', 'conclusion'],
            RiskCategory.TRANSPARENCY: ['methodology', 'results']
        }
        
        relevant = section_relevance.get(category, [])
        return [section for section in relevant if sections.get(section, '').strip()]
    
    def _generate_recommendations(
        self,
        category: RiskCategory,
        level: RiskLevel,
        evidence: List[str]
    ) -> List[str]:
        """Generate specific, actionable recommendations"""
        
        base_recommendations = {
            RiskCategory.BIAS_FAIRNESS: [
                "Conduct bias testing across demographic groups",
                "Implement fairness metrics and monitoring systems",
                "Ensure diverse and representative training datasets",
                "Consider algorithmic fairness techniques (e.g., demographic parity)",
                "Perform intersectional bias analysis",
                "Establish bias audit protocols"
            ],
            RiskCategory.PRIVACY_DATA: [
                "Review data collection and consent procedures",
                "Implement privacy-preserving techniques (differential privacy, federated learning)",
                "Ensure GDPR/CCPA/regional privacy law compliance",
                "Apply data minimization principles",
                "Conduct Privacy Impact Assessments (PIA)",
                "Implement user consent and data control mechanisms"
            ],
            RiskCategory.SAFETY_SECURITY: [
                "Conduct comprehensive safety testing and validation",
                "Implement robust security measures and access controls",
                "Plan for failure modes and recovery procedures",
                "Perform adversarial robustness testing",
                "Establish monitoring and alert systems",
                "Create incident response protocols"
            ],
            RiskCategory.DUAL_USE: [
                "Assess potential misuse scenarios and attack vectors",
                "Implement access controls and usage restrictions",
                "Engage with institutional ethics committees",
                "Consider publication and dissemination limitations",
                "Establish responsible disclosure practices",
                "Monitor for misuse and unauthorized applications"
            ],
            RiskCategory.SOCIETAL_IMPACT: [
                "Evaluate broader societal implications and consequences",
                "Engage with affected communities and stakeholders",
                "Consider economic and social displacement effects",
                "Plan for responsible and gradual deployment",
                "Assess impact on vulnerable populations",
                "Develop mitigation strategies for negative outcomes"
            ],
            RiskCategory.TRANSPARENCY: [
                "Improve model interpretability and explainability",
                "Provide comprehensive system documentation",
                "Implement algorithmic auditing mechanisms",
                "Ensure decision-making transparency",
                "Create user-friendly explanations of system behavior",
                "Establish accountability frameworks"
            ]
        }
        
        recommendations = base_recommendations.get(category, ["Conduct additional ethical review"]).copy()
        
        # Add urgency-based recommendations
        if level == RiskLevel.CRITICAL:
            recommendations.extend([
                "🚨 Seek immediate expert ethics review before proceeding",
                "🚨 Consider halting development until risks are addressed",
                "🚨 Engage with regulatory bodies and oversight committees"
            ])
        elif level == RiskLevel.HIGH:
            recommendations.extend([
                "⚠️ Obtain additional expert review before publication",
                "⚠️ Implement enhanced safety and monitoring measures",
                "⚠️ Consider staged or limited deployment"
            ])
        
        # Evidence-specific recommendations
        if any('without consent' in ev.lower() for ev in evidence):
            recommendations.insert(0, "📋 Immediately establish proper consent procedures")
        
        if any('bias' in ev.lower() for ev in evidence):
            recommendations.insert(0, "📊 Conduct immediate bias assessment and testing")
        
        if any('weapon' in ev.lower() or 'military' in ev.lower() for ev in evidence):
            recommendations.insert(0, "🔒 Restrict access and consider dual-use implications")
        
        return recommendations[:8]  # Return top 8 recommendations
    
    def _calculate_overall_score(self, assessments: List[RiskAssessment]) -> float:
        """Calculate weighted overall risk score"""
        if not assessments:
            return 0.0
        
        # Category weights for overall score
        category_weights = {
            RiskCategory.BIAS_FAIRNESS: 1.2,    # High importance
            RiskCategory.PRIVACY_DATA: 1.2,     # High importance  
            RiskCategory.SAFETY_SECURITY: 1.1,  # Medium-high importance
            RiskCategory.DUAL_USE: 1.3,         # Highest importance
            RiskCategory.SOCIETAL_IMPACT: 1.0,  # Medium importance
            RiskCategory.TRANSPARENCY: 0.9      # Lower importance
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for assessment in assessments:
            weight = category_weights.get(assessment.category, 1.0)
            weighted_sum += assessment.score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _log_analysis_summary(
        self,
        title: str,
        overall_score: float,
        assessments: List[RiskAssessment],
        processing_time: float
    ) -> None:
        """Log comprehensive analysis summary"""
        
        high_risk_categories = [
            a.category.value for a in assessments 
            if a.level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        ]
        
        critical_categories = [
            a.category.value for a in assessments 
            if a.level == RiskLevel.CRITICAL
        ]
        
        logger.info(f"📊 Analysis Summary for '{title[:30]}...':")
        logger.info(f"   Overall Score: {overall_score:.1f}/10")
        logger.info(f"   Processing Time: {processing_time:.2f}s")
        logger.info(f"   High Risk Categories: {len(high_risk_categories)}")
        
        if critical_categories:
            logger.warning(f"🚨 CRITICAL RISKS: {', '.join(critical_categories)}")
        elif high_risk_categories:
            logger.warning(f"⚠️ High risks: {', '.join(high_risk_categories)}")
        else:
            logger.info("✅ No high-risk issues detected")
    
    def get_system_info(self) -> Dict[str, any]:
        """Get information about the Guardian Engine system"""
        return {
            'version': '1.0.0',
            'analyzers': {
                'keyword': True,
                'bias': True,
                'privacy': True,
                'transformer': self.transformer_analyzer is not None
            },
            'risk_categories': [category.value for category in RiskCategory],
            'risk_levels': [level.value for level in RiskLevel],
            'configuration': {
                'max_text_length': self.max_text_length,
                'min_text_length': self.min_text_length,
                'risk_thresholds': self.risk_thresholds,
                'analyzer_weights': self.analyzer_weights
            }
        }
    
    def validate_system(self) -> Dict[str, bool]:
        """Validate that all system components are working"""
        validation_results = {}
        
        # Test text processor
        try:
            test_text = "This is a test sentence for validation."
            cleaned = self.text_processor.clean_text(test_text)
            validation_results['text_processor'] = len(cleaned) > 0
        except Exception as e:
            logger.error(f"Text processor validation failed: {e}")
            validation_results['text_processor'] = False
        
        # Test keyword analyzer
        try:
            test_scores = self.keyword_analyzer.analyze("bias discrimination privacy")
            validation_results['keyword_analyzer'] = isinstance(test_scores, dict)
        except Exception as e:
            logger.error(f"Keyword analyzer validation failed: {e}")
            validation_results['keyword_analyzer'] = False
        
        # Test bias analyzer
        try:
            score, evidence = self.bias_analyzer.analyze("demographic bias testing")
            validation_results['bias_analyzer'] = isinstance(score, (int, float))
        except Exception as e:
            logger.error(f"Bias analyzer validation failed: {e}")
            validation_results['bias_analyzer'] = False
        
        # Test privacy analyzer
        try:
            score, evidence = self.privacy_analyzer.analyze("personal data collection")
            validation_results['privacy_analyzer'] = isinstance(score, (int, float))
        except Exception as e:
            logger.error(f"Privacy analyzer validation failed: {e}")
            validation_results['privacy_analyzer'] = False
        
        # Test transformer analyzer
        if self.transformer_analyzer:
            try:
                test_scores = self.transformer_analyzer.analyze("test analysis")
                validation_results['transformer_analyzer'] = isinstance(test_scores, dict)
            except Exception as e:
                logger.error(f"Transformer analyzer validation failed: {e}")
                validation_results['transformer_analyzer'] = False
        else:
            validation_results['transformer_analyzer'] = False
        
        return validation_results

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_guardian_engine(config: Optional[Dict] = None) -> GuardianEngine:
    """
    Factory function to create a configured Guardian Engine
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured GuardianEngine instance
    """
    return GuardianEngine(config)

def quick_analyze(text: str, title: str = "Untitled") -> Dict[str, any]:
    """
    Quick analysis function for simple use cases
    
    Args:
        text: Text to analyze
        title: Optional title
        
    Returns:
        Simplified analysis results dictionary
    """
    engine = GuardianEngine()
    
    request = AnalysisRequest(title=title, content=text)
    response = engine.analyze_paper(request)
    
    # Simplify output
    return {
        'overall_score': response.overall_risk_score,
        'processing_time': response.processing_time,
        'high_risk_categories': [
            assessment['category'] for assessment in response.risk_assessments
            if assessment['level'] in ['high', 'critical']
        ],
        'recommendations': [
            rec for assessment in response.risk_assessments
            for rec in assessment['recommendations'][:2]  # Top 2 per category
        ]
    }

# =============================================================================
# MAIN EXECUTION (for testing)
# =============================================================================

if __name__ == "__main__":
    # Test the Guardian Engine
    print("🧪 Testing Guardian Engine...")
    
    # Create engine
    engine = GuardianEngine()
    
    # Validate system
    validation = engine.validate_system()
    print(f"System validation: {validation}")
    
    # Test analysis
    test_text = """
    This paper presents a facial recognition system for identifying individuals
    in crowded spaces. The system uses demographic data including race and gender
    for improved accuracy. We collected biometric data from social media profiles
    without explicit consent. The system has potential military applications
    for surveillance and crowd control.
    """
    
    result = quick_analyze(test_text, "Test Facial Recognition Paper")
    print(f"Test analysis result: {result}")
    
    print("✅ Guardian Engine test completed!")