"""
Hallucination Detection for LLM outputs
"""

import re
import logging
from typing import List, Dict, Any, Optional
from ..llm.model_loader import GGUFModelLoader

logger = logging.getLogger(__name__)

class HallucinationDetector:
    """
    Detects potential hallucinations in LLM outputs
    """

    def __init__(self, model_loader: Optional[GGUFModelLoader] = None):
        self.model_loader = model_loader

        # Factual consistency patterns
        self.fact_patterns = {
            'temporal': re.compile(r'\b(19|20)\d{2}\b'),  # Years
            'quantitative': re.compile(r'\b\d+(\.\d+)?\b'),  # Numbers
            'entity': re.compile(r'\b[A-Z][a-z]+\b')  # Proper nouns
        }

        # Contradiction patterns
        self.contradiction_patterns = [
            r'impossible.*possible',
            r'never.*always',
            r'no.*yes'
        ]

    def detect_hallucinations(self, text: str, context: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect potential hallucinations in text

        Args:
            text: Generated text to analyze
            context: Supporting context documents

        Returns:
            Hallucination analysis
        """
        score = 0.0
        issues = []

        # Check factual consistency
        fact_score, fact_issues = self._check_factual_consistency(text, context)
        score += fact_score
        issues.extend(fact_issues)

        # Check logical consistency
        logic_score, logic_issues = self._check_logical_consistency(text)
        score += logic_score
        issues.extend(logic_issues)

        # Check confidence markers
        confidence_score, confidence_issues = self._check_confidence_markers(text)
        score += confidence_score
        issues.extend(confidence_issues)

        # Normalize score
        final_score = min(score / 3.0, 1.0)

        return {
            'hallucination_score': final_score,
            'issues': issues,
            'risk_level': self._classify_risk(final_score),
            'recommendations': self._get_recommendations(final_score, issues)
        }

    def _check_factual_consistency(self, text: str, context: List[str] = None) -> tuple:
        """Check factual consistency against context"""
        if not context:
            return 0.3, ['No context provided for factual verification']

        issues = []
        score = 0.0

        # Extract facts from text
        facts = self._extract_facts(text)

        # Check each fact against context
        verified_facts = 0
        for fact in facts:
            if self._verify_fact_against_context(fact, context):
                verified_facts += 1
            else:
                issues.append(f'Unverified fact: {fact}')

        if facts:
            score = verified_facts / len(facts)
        else:
            score = 0.8  # No facts to check

        return score, issues

    def _check_logical_consistency(self, text: str) -> tuple:
        """Check for logical contradictions"""
        issues = []
        score = 1.0

        # Check for contradiction patterns
        for pattern in self.contradiction_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(f'Potential contradiction: {pattern}')
                score -= 0.2

        # Check for nonsensical statements
        nonsensical_patterns = [
            r'\bimpossible\b.*\bpossible\b',
            r'\bnever\b.*\balways\b'
        ]

        for pattern in nonsensical_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(f'Logical inconsistency detected')
                score -= 0.3

        return max(score, 0.0), issues

    def _check_confidence_markers(self, text: str) -> tuple:
        """Check for appropriate confidence markers"""
        issues = []
        score = 0.8

        # Look for uncertainty markers
        uncertainty_markers = ['maybe', 'perhaps', 'possibly', 'might', 'could', 'I think']

        has_uncertainty = any(marker in text.lower() for marker in uncertainty_markers)

        if not has_uncertainty and len(text) > 200:
            issues.append('Long response without uncertainty markers')
            score -= 0.2

        # Check for overconfidence
        overconfidence_markers = ['definitely', 'absolutely', 'certainly', 'obviously']

        overconfidence_count = sum(1 for marker in overconfidence_markers
                                 if marker in text.lower())

        if overconfidence_count > 2:
            issues.append('Excessive overconfidence markers')
            score -= 0.3

        return score, issues

    def _extract_facts(self, text: str) -> List[str]:
        """Extract factual statements from text"""
        facts = []

        # Extract temporal facts
        temporal_facts = self.fact_patterns['temporal'].findall(text)
        facts.extend([f"Year: {fact}" for fact in temporal_facts])

        # Extract quantitative facts
        quant_facts = self.fact_patterns['quantitative'].findall(text)
        facts.extend([f"Number: {fact}" for fact in quant_facts[:5]])  # Limit

        # Extract entity facts
        entities = self.fact_patterns['entity'].findall(text)
        facts.extend([f"Entity: {entity}" for entity in entities[:5]])  # Limit

        return facts

    def _verify_fact_against_context(self, fact: str, context: List[str]) -> bool:
        """Verify a fact against provided context"""
        fact_lower = fact.lower()

        for doc in context:
            if fact_lower in doc.lower():
                return True

        return False

    def _classify_risk(self, score: float) -> str:
        """Classify hallucination risk level"""
        if score < 0.3:
            return 'high'
        elif score < 0.6:
            return 'medium'
        else:
            return 'low'

    def _get_recommendations(self, score: float, issues: List[str]) -> List[str]:
        """Get recommendations based on analysis"""
        recommendations = []

        if score < 0.5:
            recommendations.append('Consider regenerating response with more context')
            recommendations.append('Review source documents for accuracy')

        if any('contradiction' in issue.lower() for issue in issues):
            recommendations.append('Check for logical consistency in the response')

        if any('uncertainty' in issue.lower() for issue in issues):
            recommendations.append('Add appropriate uncertainty markers to response')

        return recommendations
