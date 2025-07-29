"""Explainability utilities for auditable match decisions."""

from typing import List, Dict, Any, Optional
from ..models.matching import (
    MatchExplanation, 
    MatchingLayer, 
    MatchConfidenceLevel,
    MatchResult
)


class ExplainabilityGenerator:
    """Generates human-readable explanations for matching decisions."""
    
    EXPLANATION_TEMPLATES = {
        MatchingLayer.EXACT: "Exact match found: '{input}' matches '{found}' after normalization",
        MatchingLayer.NICKNAME: "Nickname match: '{original}' is a known variant of '{canonical}'",
        MatchingLayer.STRUCTURAL: "Structural match: {details}",
        MatchingLayer.FUZZY: "Fuzzy match with {score:.1%} similarity: {details}",
        MatchingLayer.PHONETIC: "Phonetic match: names sound similar ({algorithm})",
        MatchingLayer.LLM_ENHANCED: "LLM analysis: {reasoning}",
        MatchingLayer.NO_MATCH: "No sufficient match found: {reason}"
    }
    
    CONFIDENCE_THRESHOLDS = {
        MatchingLayer.EXACT: 1.0,
        MatchingLayer.NICKNAME: 0.95,
        MatchingLayer.STRUCTURAL: 0.85,
        MatchingLayer.FUZZY: 0.70,
        MatchingLayer.PHONETIC: 0.65,
        MatchingLayer.LLM_ENHANCED: 0.80,
        MatchingLayer.NO_MATCH: 0.0
    }
    
    def generate_explanation(
        self,
        layer: MatchingLayer,
        confidence_score: float,
        input_name: str,
        found_name: str,
        transformations: List[str],
        additional_context: Optional[Dict[str, Any]] = None
    ) -> MatchExplanation:
        """Generate explanation for matching decision."""
        reasoning = self._generate_reasoning(
            layer, input_name, found_name, additional_context, confidence_score
        )
        evidence = self._extract_evidence(
            layer, input_name, found_name, additional_context
        )
        
        return MatchExplanation(
            layer=layer,
            confidence_score=confidence_score,
            reasoning=reasoning,
            transformations=transformations,
            evidence=evidence
        )
    
    def _generate_reasoning(
        self,
        layer: MatchingLayer,
        input_name: str,
        found_name: str,
        context: Optional[Dict[str, Any]] = None,
        confidence_score: float = 0.0
    ) -> str:
        """Generate human-readable reasoning for a match."""
        template = self.EXPLANATION_TEMPLATES.get(layer, "Match found")
        context = context or {}
        
        # Dictionary mapping layers to their formatting functions
        formatters = {
            MatchingLayer.EXACT: lambda: template.format(input=input_name, found=found_name),
            MatchingLayer.NICKNAME: lambda: template.format(
                original=context.get('original_form', input_name),
                canonical=context.get('canonical_form', found_name)
            ),
            MatchingLayer.STRUCTURAL: lambda: template.format(
                details=self._explain_structural_match(context)
            ),
            MatchingLayer.FUZZY: lambda: template.format(
                score=context.get('similarity_score', confidence_score),
                details=f"using {context.get('algorithm', 'unknown')} algorithm"
            ),
            MatchingLayer.PHONETIC: lambda: template.format(
                algorithm=context.get('algorithm', 'soundex')
            ),
            MatchingLayer.LLM_ENHANCED: lambda: template.format(
                reasoning=context.get('llm_reasoning', 'Contextual analysis performed')
            ),
            MatchingLayer.NO_MATCH: lambda: template.format(
                reason=context.get('reason', 'insufficient similarity')
            )
        }
        
        formatter = formatters.get(layer)
        return formatter() if formatter else f"Match between '{input_name}' and '{found_name}'"
    
    def _explain_structural_match(self, context: Optional[Dict[str, Any]]) -> str:
        """Generate human-readable explanation for structural name matches."""
        if not context:
            return "names have similar structure"
        
        # Mapping of context keys to their explanations
        explanations_map = {
            'first_name_match': "first names match",
            'last_name_match': "last names match", 
            'middle_name_match': "middle names match",
            'initials_match': "initials match",
            'middle_name_swap': "middle name used as first name",
            'reordered': "name components reordered",
            'initials_surname_pattern': "Name to initials pattern detected",
            'name_order_variation': "Name order variation detected",
            # Special cases with dynamic content
            'name_shortening': lambda ctx: f"name shortening detected: {ctx['name_shortening']}",
            # Warnings
            'surname_only_match': "⚠️ WARNING: Only surname matches - analyst review recommended",
            'middle_name_mismatch': lambda ctx: f"⚠️ WARNING: {ctx.get('warning', 'Middle names differ')} - analyst review recommended",
            'single_token_match': lambda ctx: f"⚠️ WARNING: {ctx.get('warning', 'Single name component match')}",
            'surname_mismatch': lambda ctx: f"⚠️ WARNING: {ctx.get('warning', 'Similar first names but different surnames')}"
        }
        
        # Generate explanations in a single comprehension
        explanations = [
            (exp(context) if callable(exp) else exp)
            for key, exp in explanations_map.items()
            if context.get(key)
        ]
        
        return ", ".join(explanations) if explanations else "structural similarity detected"
    
    def _extract_evidence(
        self,
        layer: MatchingLayer,
        input_name: str,
        found_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Extract supporting evidence for the match."""
        evidence = [f"Input: '{input_name}'", f"Found: '{found_name}'"] 
        
        if not context:
            return evidence
        # Evidence extraction mapping with compact lambda functions
        evidence_map = {
            MatchingLayer.NICKNAME: lambda: [f"Source: {context['nickname_source']}"] if 'nickname_source' in context else [],
            MatchingLayer.STRUCTURAL: lambda: [item for item in [
                f"Matching parts: {', '.join(context['matching_components'])}" if 'matching_components' in context else None,
                f"Pattern: {context['pattern']}" if 'pattern' in context else None
            ] if item],
            MatchingLayer.FUZZY: lambda: [item for item in [
                f"Similarity: {context['similarity_score']:.1%}" if 'similarity_score' in context else None,
                f"Edit distance: {context['edit_distance']}" if 'edit_distance' in context else None
            ] if item],
            MatchingLayer.LLM_ENHANCED: lambda: [item for item in [
                f"Cultural factors: {', '.join(context['cultural_factors'])}" if context.get('cultural_factors') else None,
                f"Context clues: {', '.join(context['context_clues'])}" if context.get('context_clues') else None
            ] if item]
        }
        
        if layer in evidence_map:
            evidence.extend(evidence_map[layer]())
        
        if 'article_context' in context:
            evidence.append(f"Context: {context['article_context']}")
        return evidence
    
    def generate_final_decision(
        self,
        explanations: List[MatchExplanation],
        is_match: bool,
        overall_confidence: float
    ) -> str:
        """Generate final decision summary."""
        if not explanations:
            return "No matching attempted"
        
        best_match = max(explanations, key=lambda x: x.confidence_score)
        
        if is_match:
            confidence_level = self._score_to_level(overall_confidence)
            return (
                f"{confidence_level.value.replace('_', ' ').title()} confidence match "
                f"based on {best_match.layer.value.replace('_', ' ')}. "
                f"Overall confidence: {overall_confidence:.1%}"
            )
        else:
            return (
                f"No match recommended. Best attempt was {best_match.layer.value.replace('_', ' ')} "
                f"with {best_match.confidence_score:.1%} confidence, "
                f"below threshold for positive match"
            )
    
    def _score_to_level(self, score: float) -> MatchConfidenceLevel:
        """Convert score to confidence level."""
        # Define thresholds in descending order with corresponding levels
        thresholds = [
            (0.95, MatchConfidenceLevel.EXACT),
            (0.90, MatchConfidenceLevel.VERY_HIGH),
            (0.80, MatchConfidenceLevel.HIGH),
            (0.60, MatchConfidenceLevel.MEDIUM),
            (0.30, MatchConfidenceLevel.LOW),
            (0.0, MatchConfidenceLevel.NO_MATCH)
        ]
        
        return next(level for threshold, level in thresholds if score >= threshold)
    
    def format_explanation_summary(self, explanations: List[MatchExplanation]) -> List[str]:
        """Format explanations into summary list."""
        summary = []
        for exp in explanations:
            summary.append(f"{exp.reasoning} ({exp.confidence_score:.1%} confidence)")
            if exp.transformations:
                summary.append(f"Transformations: {', '.join(exp.transformations[:3])}")
        return summary


# Convenience function
def create_explanation(
    layer: MatchingLayer,
    confidence: float,
    input_name: str,
    found_name: str,
    transformations: List[str] = None,
    **kwargs
) -> MatchExplanation:
    """Create a match explanation."""
    return ExplainabilityGenerator().generate_explanation(
        layer=layer, confidence_score=confidence, input_name=input_name,
        found_name=found_name, transformations=transformations or [],
        additional_context=kwargs
    )