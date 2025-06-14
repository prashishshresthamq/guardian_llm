# core/cot_analyzer.py
"""
Guardian LLM - Chain of Thought Analyzer
Implements step-by-step ethical reasoning for complex risk assessment
Based on COMP8420 Advanced Topics in AI - Reasoning and Inference
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ReasoningStep(str, Enum):
    """Types of reasoning steps in the chain"""
    TECHNIQUE_IDENTIFICATION = "technique_identification"
    MISUSE_ANALYSIS = "misuse_analysis" 
    SEVERITY_ASSESSMENT = "severity_assessment"
    STAKEHOLDER_IMPACT = "stakeholder_impact"
    MITIGATION_STRATEGIES = "mitigation_strategies"
    FINAL_RECOMMENDATION = "final_recommendation"


@dataclass
class ReasoningStepResult:
    """Result of a single reasoning step"""
    step_type: ReasoningStep
    reasoning: str
    findings: List[str]
    confidence: float
    metadata: Dict[str, Any] = None


@dataclass
class ChainOfThoughtResult:
    """Complete chain of thought analysis result"""
    text: str
    reasoning_chain: List[ReasoningStepResult]
    final_risk_scores: Dict[str, float]
    overall_risk: float
    confidence: float
    reasoning_summary: str
    timestamp: datetime


class ChainOfThoughtAnalyzer:
    """
    Chain of Thought analyzer for ethical risk assessment
    Implements step-by-step reasoning for complex ethical analysis
    """
    
    def __init__(self):
        """Initialize the CoT analyzer"""
        self.ai_techniques = {
            'machine_learning': ['ml', 'machine learning', 'neural network', 'deep learning', 'training data'],
            'natural_language_processing': ['nlp', 'language model', 'text processing', 'sentiment analysis'],
            'computer_vision': ['image recognition', 'facial recognition', 'object detection', 'computer vision'],
            'reinforcement_learning': ['reinforcement learning', 'rl', 'reward function', 'policy learning'],
            'generative_ai': ['gpt', 'generative', 'llm', 'large language model', 'text generation'],
            'robotics': ['robot', 'autonomous', 'control system', 'actuator', 'sensor'],
            'recommendation_systems': ['recommendation', 'collaborative filtering', 'content filtering'],
            'anomaly_detection': ['anomaly detection', 'outlier detection', 'fraud detection'],
            'predictive_modeling': ['prediction', 'forecasting', 'time series', 'regression']
        }
        
        self.risk_categories = {
            'bias_fairness': {
                'description': 'Algorithmic bias and fairness issues',
                'indicators': ['bias', 'fairness', 'discrimination', 'demographic', 'protected group'],
                'severity_factors': ['systematic', 'widespread', 'harmful outcomes']
            },
            'privacy_data': {
                'description': 'Privacy violations and data protection concerns',
                'indicators': ['privacy', 'personal data', 'consent', 'gdpr', 'data collection'],
                'severity_factors': ['sensitive data', 'unauthorized access', 'data breach']
            },
            'safety_security': {
                'description': 'Safety risks and security vulnerabilities',
                'indicators': ['safety', 'security', 'vulnerability', 'attack', 'failure'],
                'severity_factors': ['critical system', 'physical harm', 'life-threatening']
            },
            'dual_use': {
                'description': 'Potential for malicious or military use',
                'indicators': ['military', 'weapon', 'surveillance', 'dual-use', 'misuse'],
                'severity_factors': ['weaponization', 'mass surveillance', 'authoritarian use']
            },
            'societal_impact': {
                'description': 'Broader societal and economic implications',
                'indicators': ['job displacement', 'economic impact', 'social inequality', 'disruption'],
                'severity_factors': ['mass unemployment', 'social unrest', 'inequality amplification']
            },
            'transparency': {
                'description': 'Lack of transparency and explainability',
                'indicators': ['black box', 'explainability', 'interpretability', 'transparency'],
                'severity_factors': ['high-stakes decisions', 'regulatory compliance', 'accountability']
            }
        }
        
        self.stakeholder_groups = [
            'end_users', 'vulnerable_populations', 'society', 'organizations', 
            'governments', 'researchers', 'developers'
        ]
    
    def analyze_with_reasoning(self, text: str) -> ChainOfThoughtResult:
        """
        Perform complete chain of thought analysis
        
        Args:
            text: Research paper or text to analyze
            
        Returns:
            Complete CoT analysis result
        """
        logger.info("Starting Chain of Thought analysis")
        
        reasoning_chain = []
        
        # Step 1: Identify AI techniques used
        step1 = self._identify_techniques(text)
        reasoning_chain.append(step1)
        
        # Step 2: Analyze potential misuse scenarios
        step2 = self._analyze_misuse_scenarios(text, step1.findings)
        reasoning_chain.append(step2)
        
        # Step 3: Assess severity of identified risks
        step3 = self._assess_risk_severity(text, step2.findings)
        reasoning_chain.append(step3)
        
        # Step 4: Analyze stakeholder impact
        step4 = self._analyze_stakeholder_impact(text, step3.findings)
        reasoning_chain.append(step4)
        
        # Step 5: Generate mitigation strategies
        step5 = self._generate_mitigation_strategies(text, step4.findings)
        reasoning_chain.append(step5)
        
        # Step 6: Final recommendation
        step6 = self._generate_final_recommendation(reasoning_chain)
        reasoning_chain.append(step6)
        
        # Synthesize results
        final_scores = self._synthesize_risk_scores(reasoning_chain)
        overall_risk = max(final_scores.values()) if final_scores else 0.0
        overall_confidence = sum(step.confidence for step in reasoning_chain) / len(reasoning_chain)
        
        # Generate reasoning summary
        summary = self._generate_reasoning_summary(reasoning_chain, final_scores)
        
        result = ChainOfThoughtResult(
            text=text,
            reasoning_chain=reasoning_chain,
            final_risk_scores=final_scores,
            overall_risk=overall_risk,
            confidence=overall_confidence,
            reasoning_summary=summary,
            timestamp=datetime.utcnow()
        )
        
        logger.info(f"CoT analysis completed with overall risk: {overall_risk:.3f}")
        return result
    
    def _identify_techniques(self, text: str) -> ReasoningStepResult:
        """Step 1: Identify AI techniques used in the paper"""
        text_lower = text.lower()
        identified_techniques = []
        reasoning_parts = []
        
        for technique, keywords in self.ai_techniques.items():
            matches = [kw for kw in keywords if kw in text_lower]
            if matches:
                identified_techniques.append(technique)
                reasoning_parts.append(f"Detected {technique} based on keywords: {', '.join(matches)}")
        
        if not identified_techniques:
            identified_techniques.append('general_ai')
            reasoning_parts.append("No specific AI technique clearly identified, treating as general AI research")
        
        reasoning = "Let me identify the main AI techniques discussed in this paper:\n" + \
                   "\n".join(f"- {part}" for part in reasoning_parts)
        
        confidence = min(0.9, len(identified_techniques) * 0.3)
        
        return ReasoningStepResult(
            step_type=ReasoningStep.TECHNIQUE_IDENTIFICATION,
            reasoning=reasoning,
            findings=identified_techniques,
            confidence=confidence,
            metadata={'keyword_matches': reasoning_parts}
        )
    
    def _analyze_misuse_scenarios(self, text: str, techniques: List[str]) -> ReasoningStepResult:
        """Step 2: Analyze potential misuse scenarios"""
        text_lower = text.lower()
        misuse_scenarios = []
        reasoning_parts = []
        
        # Define misuse scenarios for each technique
        technique_misuse = {
            'machine_learning': [
                'Biased decision-making in hiring or lending',
                'Discriminatory profiling based on demographics',
                'Privacy invasion through inference attacks'
            ],
            'natural_language_processing': [
                'Deepfake text generation for disinformation',
                'Automated harassment or hate speech generation',
                'Privacy breaches through text analysis'
            ],
            'computer_vision': [
                'Mass surveillance and facial recognition abuse',
                'Deepfake image/video creation',
                'Unauthorized biometric data collection'
            ],
            'reinforcement_learning': [
                'Manipulation of human behavior',
                'Autonomous weapons development',
                'Gaming of regulatory systems'
            ],
            'generative_ai': [
                'Misinformation and propaganda generation',
                'Academic dishonesty and plagiarism',
                'Impersonation and fraud'
            ],
            'robotics': [
                'Autonomous weapons systems',
                'Replacement of human workers',
                'Privacy invasion through surveillance robots'
            ]
        }
        
        for technique in techniques:
            if technique in technique_misuse:
                scenarios = technique_misuse[technique]
                misuse_scenarios.extend(scenarios)
                reasoning_parts.append(f"For {technique}, potential misuse includes: {', '.join(scenarios)}")
        
        # Check for explicit risk indicators in text
        risk_indicators = []
        for category, info in self.risk_categories.items():
            indicators = [ind for ind in info['indicators'] if ind in text_lower]
            if indicators:
                risk_indicators.append(f"{category}: {', '.join(indicators)}")
                reasoning_parts.append(f"Text mentions {category} indicators: {', '.join(indicators)}")
        
        reasoning = "Now let me consider potential misuse scenarios based on the identified techniques:\n" + \
                   "\n".join(f"- {part}" for part in reasoning_parts)
        
        confidence = min(0.9, (len(misuse_scenarios) + len(risk_indicators)) * 0.1)
        
        return ReasoningStepResult(
            step_type=ReasoningStep.MISUSE_ANALYSIS,
            reasoning=reasoning,
            findings=misuse_scenarios + risk_indicators,
            confidence=confidence,
            metadata={'techniques_analyzed': techniques, 'risk_indicators': risk_indicators}
        )
    
    def _assess_risk_severity(self, text: str, misuse_findings: List[str]) -> ReasoningStepResult:
        """Step 3: Assess severity of identified risks"""
        text_lower = text.lower()
        severity_assessments = []
        reasoning_parts = []
        
        # Analyze each risk category
        for category, info in self.risk_categories.items():
            base_score = 0.0
            category_reasoning = []
            
            # Check for indicators
            indicators_found = [ind for ind in info['indicators'] if ind in text_lower]
            if indicators_found:
                base_score += len(indicators_found) * 0.2
                category_reasoning.append(f"Found indicators: {', '.join(indicators_found)}")
            
            # Check for severity factors
            severity_factors = [sf for sf in info['severity_factors'] if sf in text_lower]
            if severity_factors:
                base_score += len(severity_factors) * 0.3
                category_reasoning.append(f"Severity factors present: {', '.join(severity_factors)}")
            
            # Context analysis
            if 'critical' in text_lower or 'high-risk' in text_lower:
                base_score += 0.2
                category_reasoning.append("High-risk context detected")
            
            if 'safety' in text_lower and category == 'safety_security':
                base_score += 0.3
                category_reasoning.append("Safety-critical application indicated")
            
            # Scale applications increase risk
            if any(term in text_lower for term in ['large-scale', 'widespread', 'mass deployment']):
                base_score += 0.2
                category_reasoning.append("Large-scale deployment mentioned")
            
            base_score = min(base_score, 1.0)  # Cap at 1.0
            
            if base_score > 0.1:  # Only include if there's meaningful risk
                severity_level = self._score_to_severity_level(base_score)
                assessment = f"{category}: {severity_level} risk (score: {base_score:.2f})"
                severity_assessments.append(assessment)
                
                reasoning_parts.append(f"{category} ({severity_level}): {'; '.join(category_reasoning)}")
        
        reasoning = "Let me assess the severity of each identified risk:\n" + \
                   "\n".join(f"- {part}" for part in reasoning_parts)
        
        confidence = 0.8 if reasoning_parts else 0.3
        
        return ReasoningStepResult(
            step_type=ReasoningStep.SEVERITY_ASSESSMENT,
            reasoning=reasoning,
            findings=severity_assessments,
            confidence=confidence,
            metadata={'category_scores': {cat: self._extract_score_from_assessment(assess) 
                     for assess in severity_assessments for cat in self.risk_categories.keys() if cat in assess}}
        )
    
    def _analyze_stakeholder_impact(self, text: str, severity_findings: List[str]) -> ReasoningStepResult:
        """Step 4: Analyze impact on different stakeholders"""
        text_lower = text.lower()
        stakeholder_impacts = []
        reasoning_parts = []
        
        stakeholder_keywords = {
            'end_users': ['user', 'consumer', 'individual', 'person', 'citizen'],
            'vulnerable_populations': ['vulnerable', 'minority', 'elderly', 'disabled', 'children'],
            'society': ['society', 'community', 'public', 'social', 'cultural'],
            'organizations': ['company', 'organization', 'business', 'enterprise', 'institution'],
            'governments': ['government', 'policy', 'regulation', 'law', 'authority'],
            'researchers': ['researcher', 'scientist', 'academic', 'scholar', 'study'],
            'developers': ['developer', 'engineer', 'programmer', 'technologist', 'ai team']
        }
        
        for stakeholder, keywords in stakeholder_keywords.items():
            mentions = [kw for kw in keywords if kw in text_lower]
            if mentions:
                impact_analysis = self._assess_stakeholder_specific_impact(text_lower, stakeholder, severity_findings)
                if impact_analysis:
                    stakeholder_impacts.append(f"{stakeholder}: {impact_analysis}")
                    reasoning_parts.append(f"Impact on {stakeholder} (mentioned: {', '.join(mentions)}): {impact_analysis}")
        
        # Default analysis if no specific stakeholders mentioned
        if not stakeholder_impacts:
            general_impact = "General public may be affected through indirect consequences"
            stakeholder_impacts.append(f"general_public: {general_impact}")
            reasoning_parts.append(general_impact)
        
        reasoning = "Now let me analyze the impact on different stakeholder groups:\n" + \
                   "\n".join(f"- {part}" for part in reasoning_parts)
        
        confidence = min(0.9, len(stakeholder_impacts) * 0.2)
        
        return ReasoningStepResult(
            step_type=ReasoningStep.STAKEHOLDER_IMPACT,
            reasoning=reasoning,
            findings=stakeholder_impacts,
            confidence=confidence,
            metadata={'stakeholders_mentioned': list(stakeholder_keywords.keys())}
        )
    
    def _generate_mitigation_strategies(self, text: str, stakeholder_findings: List[str]) -> ReasoningStepResult:
        """Step 5: Generate mitigation strategies"""
        mitigation_strategies = []
        reasoning_parts = []
        
        # Extract risk categories from previous findings
        identified_risks = []
        for finding in stakeholder_findings:
            for category in self.risk_categories.keys():
                if category.replace('_', ' ') in finding.lower():
                    identified_risks.append(category)
        
        # Generate category-specific mitigations
        mitigation_templates = {
            'bias_fairness': [
                "Implement bias testing and fairness metrics",
                "Use diverse and representative training data",
                "Regular algorithmic auditing and bias monitoring",
                "Stakeholder involvement in design and testing"
            ],
            'privacy_data': [
                "Implement privacy-by-design principles",
                "Use differential privacy techniques",
                "Obtain explicit informed consent",
                "Regular privacy impact assessments"
            ],
            'safety_security': [
                "Implement robust testing and validation",
                "Use formal verification methods",
                "Deploy fail-safe mechanisms",
                "Regular security audits and penetration testing"
            ],
            'dual_use': [
                "Establish use-case restrictions and licensing",
                "Implement access controls and monitoring",
                "Engage with policy makers on regulation",
                "Consider export controls and restrictions"
            ],
            'societal_impact': [
                "Conduct thorough impact assessments",
                "Engage with affected communities",
                "Develop transition and retraining programs",
                "Implement gradual deployment strategies"
            ],
            'transparency': [
                "Develop explainable AI methods",
                "Provide clear documentation and disclosure",
                "Implement audit trails and logging",
                "Offer user-friendly explanations of decisions"
            ]
        }
        
        for risk_category in set(identified_risks):
            if risk_category in mitigation_templates:
                strategies = mitigation_templates[risk_category]
                mitigation_strategies.extend(strategies)
                reasoning_parts.append(f"For {risk_category} risks: {', '.join(strategies)}")
        
        # General mitigations if no specific risks identified
        if not mitigation_strategies:
            general_mitigations = [
                "Follow AI ethics guidelines and best practices",
                "Implement comprehensive testing and validation",
                "Ensure transparency and accountability measures",
                "Regular monitoring and evaluation post-deployment"
            ]
            mitigation_strategies.extend(general_mitigations)
            reasoning_parts.append(f"General mitigations: {', '.join(general_mitigations)}")
        
        reasoning = "Based on the identified risks, here are recommended mitigation strategies:\n" + \
                   "\n".join(f"- {part}" for part in reasoning_parts)
        
        confidence = 0.85 if identified_risks else 0.6
        
        return ReasoningStepResult(
            step_type=ReasoningStep.MITIGATION_STRATEGIES,
            reasoning=reasoning,
            findings=mitigation_strategies,
            confidence=confidence,
            metadata={'addressed_risks': identified_risks}
        )
    
    def _generate_final_recommendation(self, reasoning_chain: List[ReasoningStepResult]) -> ReasoningStepResult:
        """Step 6: Generate final recommendation based on complete analysis"""
        # Extract key information from previous steps
        techniques = reasoning_chain[0].findings if reasoning_chain else []
        misuse_scenarios = reasoning_chain[1].findings if len(reasoning_chain) > 1 else []
        severity_assessments = reasoning_chain[2].findings if len(reasoning_chain) > 2 else []
        stakeholder_impacts = reasoning_chain[3].findings if len(reasoning_chain) > 3 else []
        mitigations = reasoning_chain[4].findings if len(reasoning_chain) > 4 else []
        
        # Determine overall risk level
        high_risk_indicators = sum(1 for assessment in severity_assessments 
                                 if 'high' in assessment.lower() or 'critical' in assessment.lower())
        
        if high_risk_indicators >= 2:
            risk_level = "HIGH RISK"
            recommendation = "Requires immediate ethical review and comprehensive risk mitigation before deployment"
        elif high_risk_indicators >= 1:
            risk_level = "MODERATE RISK" 
            recommendation = "Requires ethical review and targeted risk mitigation strategies"
        elif severity_assessments:
            risk_level = "LOW-MODERATE RISK"
            recommendation = "Implement standard ethical safeguards and monitoring"
        else:
            risk_level = "LOW RISK"
            recommendation = "Follow standard AI ethics guidelines and best practices"
        
        # Generate comprehensive recommendation
        recommendation_parts = [
            f"Overall Assessment: {risk_level}",
            f"Primary Recommendation: {recommendation}",
            f"Key Risks Identified: {len(severity_assessments)} categories",
            f"Stakeholders Affected: {len(stakeholder_impacts)} groups",
            f"Mitigation Strategies: {len(mitigations)} recommended approaches"
        ]
        
        reasoning = "Final recommendation based on complete chain of thought analysis:\n" + \
                   "\n".join(f"- {part}" for part in recommendation_parts)
        
        # Calculate overall confidence
        avg_confidence = sum(step.confidence for step in reasoning_chain) / len(reasoning_chain) if reasoning_chain else 0.5
        
        return ReasoningStepResult(
            step_type=ReasoningStep.FINAL_RECOMMENDATION,
            reasoning=reasoning,
            findings=[recommendation, risk_level],
            confidence=avg_confidence,
            metadata={
                'risk_level': risk_level,
                'high_risk_count': high_risk_indicators,
                'total_assessments': len(severity_assessments)
            }
        )
    
    def _synthesize_risk_scores(self, reasoning_chain: List[ReasoningStepResult]) -> Dict[str, float]:
        """Synthesize final risk scores from the reasoning chain"""
        risk_scores = {}
        
        # Get severity assessment step
        severity_step = next((step for step in reasoning_chain 
                            if step.step_type == ReasoningStep.SEVERITY_ASSESSMENT), None)
        
        if severity_step and severity_step.metadata and 'category_scores' in severity_step.metadata:
            risk_scores = severity_step.metadata['category_scores']
        else:
            # Fallback: assign default scores based on presence of risks
            for category in self.risk_categories.keys():
                risk_scores[category] = 0.3 if any(category in finding.lower() 
                                                 for step in reasoning_chain 
                                                 for finding in step.findings) else 0.1
        
        return risk_scores
    
    def _generate_reasoning_summary(self, reasoning_chain: List[ReasoningStepResult], 
                                  risk_scores: Dict[str, float]) -> str:
        """Generate a concise summary of the reasoning process"""
        summary_parts = []
        
        for step in reasoning_chain:
            step_summary = f"{step.step_type.value}: {len(step.findings)} findings (confidence: {step.confidence:.2f})"
            summary_parts.append(step_summary)
        
        highest_risk = max(risk_scores.items(), key=lambda x: x[1]) if risk_scores else ("none", 0.0)
        
        summary = f"CoT Analysis Summary: {len(reasoning_chain)} reasoning steps completed. " + \
                 f"Highest risk category: {highest_risk[0]} ({highest_risk[1]:.2f}). " + \
                 "Steps: " + " → ".join([step.step_type.value for step in reasoning_chain])
        
        return summary
    
    def _assess_stakeholder_specific_impact(self, text_lower: str, stakeholder: str, 
                                          severity_findings: List[str]) -> str:
        """Assess impact on a specific stakeholder group"""
        impact_templates = {
            'end_users': "Direct impact through system decisions and interactions",
            'vulnerable_populations': "Disproportionate negative effects due to existing disadvantages",
            'society': "Broad societal implications affecting community structures",
            'organizations': "Operational and liability impacts for implementing entities",
            'governments': "Policy and regulatory implications requiring oversight",
            'researchers': "Academic integrity and research ethics considerations",
            'developers': "Technical and ethical responsibilities in development"
        }
        
        # Check for specific impact indicators
        if any('harm' in finding.lower() for finding in severity_findings):
            return f"Potential harm - {impact_templates.get(stakeholder, 'Impact requires assessment')}"
        elif any('high' in finding.lower() for finding in severity_findings):
            return f"Significant impact - {impact_templates.get(stakeholder, 'Impact requires assessment')}"
        else:
            return impact_templates.get(stakeholder, "Impact requires further assessment")
    
    def _score_to_severity_level(self, score: float) -> str:
        """Convert numerical score to severity level"""
        if score >= 0.8:
            return "Critical"
        elif score >= 0.6:
            return "High"
        elif score >= 0.4:
            return "Medium"
        elif score >= 0.2:
            return "Low"
        else:
            return "Minimal"
    
    def _extract_score_from_assessment(self, assessment: str) -> float:
        """Extract numerical score from assessment string"""
        import re
        match = re.search(r'score: ([\d.]+)', assessment)
        return float(match.group(1)) if match else 0.0
    
    def generate_reasoning_chain(self, prompt: str) -> str:
        """Generate a reasoning chain for a given prompt (legacy method)"""
        # This method provides backwards compatibility
        lines = prompt.split('\n')
        text = ""
        for line in lines:
            if line.strip() and not line.strip().startswith(('1.', '2.', '3.', 'Let\'s', 'Paper:')):
                if 'Paper:' in line:
                    text = line.replace('Paper:', '').strip()
                    break
        
        if not text:
            return "Unable to extract text from prompt"
        
        result = self.analyze_with_reasoning(text)
        return result.reasoning_summary
    
    def get_detailed_reasoning(self, cot_result: ChainOfThoughtResult) -> Dict[str, Any]:
        """Get detailed reasoning breakdown from CoT result"""
        return {
            'reasoning_steps': [
                {
                    'step': step.step_type.value,
                    'reasoning': step.reasoning,
                    'findings': step.findings,
                    'confidence': step.confidence,
                    'metadata': step.metadata
                }
                for step in cot_result.reasoning_chain
            ],
            'final_scores': cot_result.final_risk_scores,
            'overall_risk': cot_result.overall_risk,
            'confidence': cot_result.confidence,
            'summary': cot_result.reasoning_summary
        }
    
    def explain_reasoning_step(self, step: ReasoningStepResult) -> str:
        """Provide human-readable explanation of a reasoning step"""
        explanations = {
            ReasoningStep.TECHNIQUE_IDENTIFICATION: "Identifying what AI techniques are discussed",
            ReasoningStep.MISUSE_ANALYSIS: "Analyzing how the technology could be misused",
            ReasoningStep.SEVERITY_ASSESSMENT: "Evaluating how severe each risk is",
            ReasoningStep.STAKEHOLDER_IMPACT: "Understanding who would be affected",
            ReasoningStep.MITIGATION_STRATEGIES: "Suggesting ways to reduce risks",
            ReasoningStep.FINAL_RECOMMENDATION: "Making overall recommendations"
        }
        
        base_explanation = explanations.get(step.step_type, "Processing step")
        return f"{base_explanation}: Found {len(step.findings)} key insights with {step.confidence:.1%} confidence"


# Integration with Guardian Engine
class CoTIntegrator:
    """Integrates Chain of Thought analysis with Guardian Engine"""
    
    def __init__(self, cot_analyzer: ChainOfThoughtAnalyzer):
        self.cot_analyzer = cot_analyzer
    
    def enhance_analysis_with_cot(self, text: str, traditional_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance traditional analysis with Chain of Thought reasoning"""
        
        # Perform CoT analysis
        cot_result = self.cot_analyzer.analyze_with_reasoning(text)
        
        # Merge with traditional analysis
        enhanced_analysis = traditional_analysis.copy()
        enhanced_analysis['chain_of_thought'] = {
            'reasoning_chain': self.cot_analyzer.get_detailed_reasoning(cot_result),
            'cot_risk_scores': cot_result.final_risk_scores,
            'reasoning_summary': cot_result.reasoning_summary,
            'overall_cot_risk': cot_result.overall_risk,
            'cot_confidence': cot_result.confidence
        }
        
        # Weight combination: 70% traditional, 30% CoT
        if 'risk_categories' in enhanced_analysis:
            for category in enhanced_analysis['risk_categories']:
                traditional_score = enhanced_analysis['risk_categories'][category]
                cot_score = cot_result.final_risk_scores.get(category, 0.0)
                enhanced_analysis['risk_categories'][category] = (
                    0.7 * traditional_score + 0.3 * cot_score
                )
        
        return enhanced_analysis


# Example usage and testing
if __name__ == "__main__":
    # Initialize analyzer
    cot_analyzer = ChainOfThoughtAnalyzer()
    
    # Test text
    test_text = """
    This paper presents a novel facial recognition system using deep learning 
    that can identify individuals with 99% accuracy. The system uses a large 
    dataset of facial images collected from social media platforms. We discuss 
    potential applications in security and surveillance, though we acknowledge 
    privacy concerns may arise from widespread deployment.
    """
    
    # Perform analysis
    result = cot_analyzer.analyze_with_reasoning(test_text)
    
    # Print results
    print("=== Chain of Thought Analysis ===")
    print(f"Overall Risk: {result.overall_risk:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"\nRisk Scores:")
    for category, score in result.final_risk_scores.items():
        print(f"  {category}: {score:.3f}")
    
    print(f"\nReasoning Summary:")
    print(result.reasoning_summary)
    
    print(f"\nDetailed Reasoning Chain:")
    for i, step in enumerate(result.reasoning_chain, 1):
        print(f"\nStep {i}: {step.step_type.value}")
        print(f"Confidence: {step.confidence:.3f}")
        print(f"Findings: {len(step.findings)} items")
        print(f"Reasoning: {step.reasoning[:200]}...")