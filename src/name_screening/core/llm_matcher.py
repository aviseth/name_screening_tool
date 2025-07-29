"""LLM-enhanced name matching using Google Gemini."""

import logging
import os
from typing import List, Optional, Dict, Any
import json
from pathlib import Path

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from dotenv import load_dotenv
    # Load environment variables from .env file
    env_path = Path(__file__).parent.parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, skip loading .env file
    pass

from ..models.matching import (
    ExtractedEntity,
    MatchResult,
    MatchExplanation,
    MatchingLayer,
    MatchConfidenceLevel
)
from ..utils.explainability import ExplainabilityGenerator

logger = logging.getLogger(__name__)


class LLMMatcher:
    """Enhanced name matcher using Gemini."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize LLM matcher."""
        self.explainer = ExplainabilityGenerator()
        self.client = None
        
        if genai is None:
            logger.warning("google-generativeai not installed. LLM matching disabled.")
            return
            
        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("No Google API key found. LLM matching disabled.")
            return
            
        try:
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel('gemini-2.0-flash-exp')
            logger.info("LLM matcher initialized with Gemini 2.0 Flash")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if LLM matching is available."""
        return self.client is not None
    
    def enhance_match(
        self,
        input_name: str,
        extracted_entity: ExtractedEntity,
        traditional_result: MatchResult
    ) -> MatchResult:
        """Enhance traditional matching with LLM contextual analysis."""
        if not self.is_available():
            logger.warning("LLM matcher not available, returning traditional result")
            return traditional_result
            
        try:
            llm_analysis = self._analyze_with_llm(input_name, extracted_entity)
            return self._combine_results(
                traditional_result, llm_analysis, input_name, extracted_entity.text
            )
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            enhanced_explanations = traditional_result.explanation.copy()
            enhanced_explanations.append(
                self.explainer.generate_explanation(
                    MatchingLayer.LLM_ENHANCED, 0.0, input_name,
                    extracted_entity.text, [],
                    {"error": f"LLM analysis failed: {str(e)}"}
                )
            )
            
            return MatchResult(
                input_name=traditional_result.input_name,
                article_entity=traditional_result.article_entity,
                match_confidence=traditional_result.match_confidence,
                confidence_level=traditional_result.confidence_level,
                is_match_recommendation=traditional_result.is_match_recommendation,
                explanation=enhanced_explanations,
                final_decision_logic=traditional_result.final_decision_logic
            )
    
    def _analyze_with_llm(self, input_name: str, extracted_entity: ExtractedEntity) -> Dict[str, Any]:
        """Analyze the match using LLM contextual understanding."""
        
        prompt = self._build_analysis_prompt(input_name, extracted_entity)
        
        try:
            response = self.client.generate_content(prompt)
            response_text = response.text.strip()
            
            if '```json' in response_text:
                start = response_text.find('```json') + 7
                end = response_text.find('```', start)
                if end > start:
                    response_text = response_text[start:end].strip()
            elif '```' in response_text:
                start = response_text.find('```') + 3
                end = response_text.find('```', start)
                if end > start:
                    response_text = response_text[start:end].strip()
            
            if response_text.startswith('{') and response_text.endswith('}'):
                return json.loads(response_text)
            else:
                return self._parse_text_response(response_text)
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}")
            logger.warning(f"Response text: {response_text}")
            return self._parse_text_response(response_text)
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
    def _build_analysis_prompt(self, input_name: str, extracted_entity: ExtractedEntity) -> str:
        """Build the prompt for LLM analysis."""
        
        return f"""You are an expert in name matching for financial compliance. 
Analyze whether these two names likely refer to the same person:

INPUT NAME: "{input_name}"
ARTICLE ENTITY: "{extracted_entity.text}"
ARTICLE CONTEXT: "{extracted_entity.context}"

Consider:
1. Different naming conventions and patterns
2. Common nicknames and abbreviations
3. Transliterations and spelling variations  
4. Name order differences (first-last vs last-first)
5. Professional titles and contexts
6. Initials and shortened forms

Provide your analysis as JSON:
{{
  "is_likely_match": true/false,
  "confidence_score": 0.0-1.0,
  "reasoning": "detailed explanation",
  "cultural_factors": ["list", "of", "factors"],
  "name_variations_detected": ["variations", "found"],
  "context_clues": ["relevant", "context", "information"]
}}

Focus on cultural nuances and contextual understanding that rule-based systems might miss."""

    def _parse_text_response(self, response_text: str) -> Dict[str, Any]:
        """Parse text response when JSON parsing fails."""
        
        response_lower = response_text.lower()
        is_match = any(phrase in response_lower for phrase in ["likely match", "same person"])
        
        # Dictionary mapping confidence phrases to scores (order matters for precedence)
        confidence_patterns = {
            "high confidence": 0.9,
            "very likely": 0.85,
            "medium confidence": 0.7,
            "low confidence": 0.3,
            "unlikely": 0.2
        }
        
        confidence = next(
            (score for phrase, score in confidence_patterns.items() if phrase in response_lower),
            0.5  # default confidence
        )
            
        return {
            "is_likely_match": is_match,
            "confidence_score": confidence,
            "reasoning": response_text,
            "cultural_factors": [],
            "name_variations_detected": [],
            "context_clues": []
        }
    
    def _combine_results(
        self,
        traditional_result: MatchResult,
        llm_analysis: Dict[str, Any],
        input_name: str,
        entity_text: str
    ) -> MatchResult:
        """Combine traditional and LLM analysis."""
        
        traditional_conf = traditional_result.match_confidence
        llm_conf = llm_analysis.get("confidence_score", 0.0)
        llm_match = llm_analysis.get("is_likely_match", False)
        
        if llm_conf > 0.8 and traditional_conf < 0.3 and llm_match:
            final_confidence = llm_conf
            is_match = True
            confidence_level = self._get_confidence_level(llm_conf)
        else:
            final_confidence = max(traditional_conf, llm_conf if llm_match else 0)
            is_match = traditional_result.is_match_recommendation or (llm_match and llm_conf > 0.5)
            confidence_level = self._get_confidence_level(final_confidence)
        
        llm_explanation = self.explainer.generate_explanation(
            MatchingLayer.LLM_ENHANCED, llm_conf, input_name, entity_text, [], {
                "llm_reasoning": llm_analysis.get("reasoning", ""),
                "cultural_factors": llm_analysis.get("cultural_factors", []),
                "name_variations": llm_analysis.get("name_variations_detected", []),
                "context_clues": llm_analysis.get("context_clues", []),
                "traditional_confidence": traditional_conf,
                "llm_confidence": llm_conf
            }
        )
        
        enhanced_explanations = traditional_result.explanation.copy()
        enhanced_explanations.append(llm_explanation)
        
        has_warning = any('WARNING' in str(exp.reasoning) for exp in traditional_result.explanation)
        
        llm_reasoning = llm_analysis.get('reasoning', '')
        if isinstance(llm_reasoning, str) and llm_reasoning.startswith('{'):
            try:
                reasoning_data = json.loads(llm_reasoning)
                llm_reasoning = reasoning_data.get('reasoning', llm_reasoning)
            except:
                pass
        
        if len(llm_reasoning) > 150:
            llm_reasoning = llm_reasoning[:150] + '...'
        
        decision_parts = [
            f"Traditional: {traditional_conf:.1%} confidence",
            f"LLM: {llm_conf:.1%} confidence"
        ]
        
        if has_warning:
            decision_parts.append("⚠️ ANALYST REVIEW REQUIRED")
        
        decision_parts.append(f"Decision: {'MATCH' if is_match else 'NO MATCH'}")
        decision_logic = " | ".join(decision_parts)

        return MatchResult(
            input_name=input_name,
            article_entity=entity_text,
            match_confidence=final_confidence,
            confidence_level=confidence_level,
            is_match_recommendation=is_match,
            explanation=enhanced_explanations,
            final_decision_logic=decision_logic
        )
    
    def _get_confidence_level(self, confidence: float) -> MatchConfidenceLevel:
        """Convert numerical confidence to categorical level."""
        thresholds = [
            (0.95, MatchConfidenceLevel.EXACT),
            (0.85, MatchConfidenceLevel.VERY_HIGH),
            (0.7, MatchConfidenceLevel.HIGH),
            (0.5, MatchConfidenceLevel.MEDIUM),
            (0.3, MatchConfidenceLevel.LOW),
            (0.0, MatchConfidenceLevel.NO_MATCH)
        ]
        
        return next(level for threshold, level in thresholds if confidence >= threshold)