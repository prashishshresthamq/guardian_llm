import re
import spacy
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict

class DynamicRiskAnalyzer:
    def __init__(self):
        # Load models
        self.nlp = spacy.load("en_core_web_sm")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Risk indicators for each category
        self.risk_indicators = {
            'bias_fairness': {
                'keywords': ['bias', 'discrimination', 'fairness', 'demographic', 'gender', 'race', 'ethnicity', 
                           'minority', 'underrepresented', 'disparity', 'inequality', 'prejudice'],
                'patterns': [
                    r'accuracy.{0,20}(varies|differs|gap).{0,20}(demographic|group|race|gender)',
                    r'(performance|accuracy).{0,30}(lower|worse|degraded).{0,20}(minority|women|black)',
                    r'dataset.{0,30}(imbalanced|skewed|biased)',
                    r'(exclude|underrepresent|overrepresent).{0,20}(population|group|demographic)'
                ],
                'risk_phrases': [
                    'demographic parity', 'algorithmic bias', 'discriminatory outcomes',
                    'fairness metrics', 'protected attributes', 'disparate impact'
                ]
            },
            'privacy_data': {
                'keywords': ['privacy', 'personal data', 'PII', 'GDPR', 'confidential', 'sensitive',
                           'tracking', 'surveillance', 'data collection', 'consent', 'anonymization'],
                'patterns': [
                    r'collect.{0,20}(personal|private|sensitive).{0,20}(data|information)',
                    r'(track|monitor|surveillance).{0,20}(user|individual|person)',
                    r'(store|retain|keep).{0,30}(indefinitely|permanent|long-term)',
                    r'(identify|re-identify|deanonymize).{0,20}(user|individual)'
                ],
                'risk_phrases': [
                    'personal information', 'data breach', 'unauthorized access',
                    'third-party sharing', 'data retention', 'user tracking'
                ]
            },
            'safety_security': {
                'keywords': ['vulnerability', 'exploit', 'attack', 'malicious', 'security', 'safety',
                           'failure', 'risk', 'threat', 'breach', 'compromise', 'adversarial'],
                'patterns': [
                    r'(vulnerable|susceptible).{0,20}(attack|exploit|manipulation)',
                    r'(safety|security).{0,20}(critical|concern|risk|issue)',
                    r'(failure|malfunction).{0,20}(catastrophic|severe|critical)',
                    r'adversarial.{0,20}(attack|example|input)'
                ],
                'risk_phrases': [
                    'security vulnerability', 'system failure', 'adversarial attacks',
                    'critical infrastructure', 'safety-critical', 'exploitation risk'
                ]
            },
            'dual_use': {
                'keywords': ['military', 'weapon', 'warfare', 'surveillance', 'dual-use', 'misuse',
                           'malicious', 'harmful', 'offensive', 'defense', 'intelligence'],
                'patterns': [
                    r'(military|defense|warfare).{0,20}(application|use|purpose)',
                    r'(weapon|weaponize).{0,20}(system|technology|capability)',
                    r'(surveillance|monitoring).{0,20}(mass|population|citizen)',
                    r'(misuse|abuse).{0,20}(potential|risk|concern)'
                ],
                'risk_phrases': [
                    'dual-use technology', 'military application', 'weaponization potential',
                    'mass surveillance', 'offensive capabilities', 'misuse potential'
                ]
            },
            'societal_impact': {
                'keywords': ['society', 'employment', 'job', 'displacement', 'inequality', 'social',
                           'economic', 'community', 'workforce', 'automation', 'impact'],
                'patterns': [
                    r'(job|employment).{0,20}(loss|displacement|elimination)',
                    r'(increase|widen|exacerbate).{0,20}(inequality|gap|divide)',
                    r'(replace|automate).{0,20}(human|worker|employee)',
                    r'(social|societal).{0,20}(impact|consequence|effect)'
                ],
                'risk_phrases': [
                    'job displacement', 'economic inequality', 'social disruption',
                    'workforce automation', 'digital divide', 'societal consequences'
                ]
            },
            'transparency': {
                'keywords': ['black box', 'explainable', 'interpretable', 'transparency', 'accountability',
                           'opaque', 'unexplainable', 'audit', 'oversight', 'governance'],
                'patterns': [
                    r'(black.?box|opaque|unexplainable).{0,20}(model|system|algorithm)',
                    r'(lack|absence|without).{0,20}(transparency|explanation|interpretability)',
                    r'(cannot|unable|difficult).{0,20}(explain|interpret|understand)',
                    r'(accountability|oversight|governance).{0,20}(lacking|absent|needed)'
                ],
                'risk_phrases': [
                    'lack of transparency', 'black box system', 'unexplainable decisions',
                    'accountability gap', 'algorithmic opacity', 'interpretability issues'
                ]
            }
        }

    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]

    def find_evidence_sentences(self, text: str, category: str) -> List[Dict]:
        """Find specific evidence sentences for a risk category"""
        sentences = self.extract_sentences(text)
        evidence = []
        
        indicators = self.risk_indicators.get(category, {})
        keywords = indicators.get('keywords', [])
        patterns = indicators.get('patterns', [])
        risk_phrases = indicators.get('risk_phrases', [])
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            relevance_score = 0
            matched_terms = []
            
            # Check keywords
            for keyword in keywords:
                if keyword.lower() in sentence_lower:
                    relevance_score += 1
                    matched_terms.append(keyword)
            
            # Check patterns
            for pattern in patterns:
                if re.search(pattern, sentence_lower):
                    relevance_score += 2
                    matched_terms.append(f"pattern: {pattern[:30]}...")
            
            # Check risk phrases
            for phrase in risk_phrases:
                if phrase.lower() in sentence_lower:
                    relevance_score += 1.5
                    matched_terms.append(phrase)
            
            # If relevant, add to evidence
            if relevance_score > 0:
                # Get context (previous and next sentence if available)
                context = []
                if i > 0:
                    context.append(sentences[i-1])
                context.append(sentence)
                if i < len(sentences) - 1:
                    context.append(sentences[i+1])
                
                evidence.append({
                    'sentence': sentence,
                    'context': ' '.join(context),
                    'relevance_score': relevance_score,
                    'matched_terms': matched_terms,
                    'position': i / len(sentences)  # Relative position in document
                })
        
        # Sort by relevance and return top evidence
        evidence.sort(key=lambda x: x['relevance_score'], reverse=True)
        return evidence[:5]

    def generate_specific_recommendations(self, evidence: List[Dict], category: str, risk_score: float) -> List[str]:
        """Generate specific recommendations based on found evidence"""
        recommendations = []
        
        if not evidence:
            return self.get_generic_recommendations(category, risk_score)
        
        # Analyze the specific risks found
        risk_aspects = defaultdict(list)
        for ev in evidence:
            for term in ev['matched_terms']:
                if 'pattern:' not in term:
                    risk_aspects[term].append(ev['sentence'])
        
        # Generate specific recommendations based on evidence
        if category == 'bias_fairness':
            recommendations.extend(self._generate_bias_recommendations(risk_aspects, evidence))
        elif category == 'privacy_data':
            recommendations.extend(self._generate_privacy_recommendations(risk_aspects, evidence))
        elif category == 'safety_security':
            recommendations.extend(self._generate_security_recommendations(risk_aspects, evidence))
        elif category == 'dual_use':
            recommendations.extend(self._generate_dual_use_recommendations(risk_aspects, evidence))
        elif category == 'societal_impact':
            recommendations.extend(self._generate_societal_recommendations(risk_aspects, evidence))
        elif category == 'transparency':
            recommendations.extend(self._generate_transparency_recommendations(risk_aspects, evidence))
        
        # Add severity-based general recommendations
        if risk_score >= 7.5:
            recommendations.insert(0, f"CRITICAL: Immediate action required - {len(evidence)} high-risk indicators identified")
        elif risk_score >= 5.0:
            recommendations.insert(0, f"HIGH PRIORITY: Address {len(evidence)} significant risk factors identified in the analysis")
        
        return recommendations[:5]  # Return top 5 recommendations

    def _generate_bias_recommendations(self, risk_aspects: Dict, evidence: List[Dict]) -> List[str]:
        """Generate bias-specific recommendations"""
        recommendations = []
        
        # Check for specific bias types mentioned
        if any('demographic' in term for terms in risk_aspects.values() for term in terms):
            recommendations.append("Conduct demographic parity analysis across all identified groups and implement bias mitigation techniques")
        
        if any('accuracy' in ev['sentence'].lower() and 'varies' in ev['sentence'].lower() for ev in evidence):
            recommendations.append("Implement fairness-aware training methods to reduce accuracy disparities between groups")
        
        if any('dataset' in term or 'data' in term for term in risk_aspects):
            recommendations.append("Audit training data for representation bias and augment with balanced samples from underrepresented groups")
        
        # Add specific evidence-based recommendation
        if evidence:
            top_evidence = evidence[0]['sentence']
            if 'gender' in top_evidence.lower():
                recommendations.append("Implement gender-debiasing techniques and test for equal performance across gender identities")
            elif 'race' in top_evidence.lower() or 'ethnicity' in top_evidence.lower():
                recommendations.append("Apply racial bias testing frameworks and ensure equitable outcomes across ethnic groups")
        
        return recommendations

    def _generate_privacy_recommendations(self, risk_aspects: Dict, evidence: List[Dict]) -> List[str]:
        """Generate privacy-specific recommendations"""
        recommendations = []
        
        if any('personal data' in term or 'PII' in term for term in risk_aspects):
            recommendations.append("Implement data minimization principles and collect only essential personal information")
        
        if any('gdpr' in ev['sentence'].lower() for ev in evidence):
            recommendations.append("Ensure GDPR compliance with explicit consent mechanisms and right-to-deletion capabilities")
        
        if any('track' in term or 'surveillance' in term for term in risk_aspects):
            recommendations.append("Limit tracking capabilities and implement privacy-preserving alternatives like differential privacy")
        
        if any('store' in ev['sentence'].lower() or 'retain' in ev['sentence'].lower() for ev in evidence):
            recommendations.append("Define clear data retention policies with automatic deletion timelines for sensitive information")
        
        # Context-specific recommendation
        for ev in evidence[:2]:
            if 're-identify' in ev['sentence'].lower() or 'deanonymize' in ev['sentence'].lower():
                recommendations.append("Strengthen anonymization techniques using k-anonymity or l-diversity to prevent re-identification")
                break
        
        return recommendations

    def _generate_security_recommendations(self, risk_aspects: Dict, evidence: List[Dict]) -> List[str]:
        """Generate security-specific recommendations"""
        recommendations = []
        
        if any('adversarial' in term for term in risk_aspects):
            recommendations.append("Implement adversarial training and robustness testing against known attack vectors")
        
        if any('vulnerability' in term or 'exploit' in term for term in risk_aspects):
            recommendations.append("Conduct comprehensive security audit and penetration testing to identify vulnerabilities")
        
        if any('failure' in ev['sentence'].lower() for ev in evidence):
            recommendations.append("Design fail-safe mechanisms and graceful degradation strategies for system failures")
        
        # Extract specific security concerns
        for ev in evidence:
            if 'attack' in ev['sentence'].lower():
                attack_context = ev['sentence'].lower()
                if 'injection' in attack_context:
                    recommendations.append("Implement input validation and sanitization to prevent injection attacks")
                elif 'ddos' in attack_context or 'denial' in attack_context:
                    recommendations.append("Deploy rate limiting and DDoS protection mechanisms")
        
        return recommendations

    def _generate_dual_use_recommendations(self, risk_aspects: Dict, evidence: List[Dict]) -> List[str]:
        """Generate dual-use specific recommendations"""
        recommendations = []
        
        if any('military' in term or 'weapon' in term for term in risk_aspects):
            recommendations.append("Establish clear ethical guidelines and usage restrictions for military/defense applications")
        
        if any('surveillance' in term for term in risk_aspects):
            recommendations.append("Implement strict access controls and audit logs for surveillance capabilities")
        
        if any('misuse' in ev['sentence'].lower() for ev in evidence):
            recommendations.append("Develop technical safeguards and monitoring systems to detect and prevent misuse")
        
        # Context-aware recommendations
        for ev in evidence:
            if 'dual-use' in ev['sentence'].lower():
                recommendations.append("Create comprehensive dual-use risk assessment and establish review board for sensitive applications")
                break
        
        return recommendations

    def _generate_societal_recommendations(self, risk_aspects: Dict, evidence: List[Dict]) -> List[str]:
        """Generate societal impact recommendations"""
        recommendations = []
        
        if any('job' in term or 'employment' in term for term in risk_aspects):
            recommendations.append("Develop reskilling programs and transition support for workers affected by automation")
        
        if any('inequality' in term for term in risk_aspects):
            recommendations.append("Assess distributional impacts and implement measures to prevent widening inequality")
        
        if any('automat' in ev['sentence'].lower() for ev in evidence):
            recommendations.append("Create human-in-the-loop systems and preserve meaningful human oversight in automated decisions")
        
        # Specific societal concerns
        for ev in evidence:
            if 'community' in ev['sentence'].lower():
                recommendations.append("Engage with affected communities and incorporate stakeholder feedback in deployment")
            elif 'economic' in ev['sentence'].lower():
                recommendations.append("Conduct economic impact assessment and develop mitigation strategies for negative effects")
        
        return recommendations

    def _generate_transparency_recommendations(self, risk_aspects: Dict, evidence: List[Dict]) -> List[str]:
        """Generate transparency-specific recommendations"""
        recommendations = []
        
        if any('black box' in term or 'opaque' in term for term in risk_aspects):
            recommendations.append("Replace black-box models with interpretable alternatives or implement explanation methods like SHAP/LIME")
        
        if any('explainable' in term or 'interpretable' in term for term in risk_aspects):
            recommendations.append("Develop user-friendly explanation interfaces showing how decisions are made")
        
        if any('accountability' in term for term in risk_aspects):
            recommendations.append("Establish clear accountability framework with designated responsible parties for system decisions")
        
        if any('audit' in ev['sentence'].lower() for ev in evidence):
            recommendations.append("Implement comprehensive audit trails and enable third-party algorithm auditing")
        
        return recommendations

    def get_generic_recommendations(self, category: str, risk_score: float) -> List[str]:
        """Fallback generic recommendations when no specific evidence is found"""
        base_recs = {
            'bias_fairness': [
                "Conduct bias audit across protected attributes",
                "Implement fairness metrics monitoring",
                "Review training data for representation issues"
            ],
            'privacy_data': [
                "Review data collection and retention policies",
                "Implement privacy-by-design principles",
                "Ensure compliance with data protection regulations"
            ],
            'safety_security': [
                "Perform security risk assessment",
                "Implement robust testing procedures",
                "Design fail-safe mechanisms"
            ],
            'dual_use': [
                "Establish usage guidelines and restrictions",
                "Implement access controls",
                "Monitor for potential misuse"
            ],
            'societal_impact': [
                "Assess broader societal implications",
                "Engage with stakeholders",
                "Consider long-term effects"
            ],
            'transparency': [
                "Improve system interpretability",
                "Document decision-making processes",
                "Enable algorithmic auditing"
            ]
        }
        
        return base_recs.get(category, ["Conduct thorough risk assessment", "Implement mitigation strategies"])

    def analyze_paper_risks(self, text: str, category: str, risk_score: float) -> Dict:
        """Main method to analyze risks and generate dynamic recommendations"""
        # Extract evidence
        evidence = self.find_evidence_sentences(text, category)
        
        # Format evidence for display
        evidence_list = []
        for ev in evidence:
            evidence_list.append({
                'text': ev['sentence'],
                'relevance': ev['relevance_score'],
                'context': ev.get('context', ''),
                'indicators': ev['matched_terms']
            })
        
        # Generate specific recommendations
        recommendations = self.generate_specific_recommendations(evidence, category, risk_score)
        
        return {
            'evidence': evidence_list,
            'recommendations': recommendations,
            'risk_score': risk_score,
            'evidence_count': len(evidence),
            'confidence': min(0.95, 0.7 + (len(evidence) * 0.05))  # Higher confidence with more evidence
        }