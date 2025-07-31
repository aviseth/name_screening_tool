"""LLM-enhanced name matching using Google Gemini."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from dotenv import load_dotenv

    # Load environment variables from .env file
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, skip loading .env file
    pass

from ..models.matching import (
    ExtractedEntity,
    MatchConfidenceLevel,
    MatchingLayer,
    MatchResult,
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
            self.client = genai.GenerativeModel("gemini-2.0-flash-exp")
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
        traditional_result: MatchResult,
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
                    MatchingLayer.LLM_ENHANCED,
                    0.0,
                    input_name,
                    extracted_entity.text,
                    [],
                    {"error": f"LLM analysis failed: {str(e)}"},
                )
            )

            return MatchResult(
                input_name=traditional_result.input_name,
                article_entity=traditional_result.article_entity,
                match_confidence=traditional_result.match_confidence,
                confidence_level=traditional_result.confidence_level,
                is_match_recommendation=traditional_result.is_match_recommendation,
                explanation=enhanced_explanations,
                final_decision_logic=traditional_result.final_decision_logic,
            )

    def _analyze_with_llm(
        self, input_name: str, extracted_entity: ExtractedEntity
    ) -> Dict[str, Any]:
        """Analyze the match using LLM contextual understanding."""

        prompt = self._build_analysis_prompt(input_name, extracted_entity)
        
        # Debug log for James Chen issue
        if "chen" in extracted_entity.text.lower() and "garcia" in input_name.lower():
            logger.info(f"DEBUG: Analyzing {input_name} vs {extracted_entity.text}")
            logger.info(f"DEBUG: First 200 chars of prompt: {prompt[:200]}...")

        try:
            response = self.client.generate_content(prompt)
            response_text = response.text.strip()

            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                if end > start:
                    response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                if end > start:
                    response_text = response_text[start:end].strip()

            if response_text.startswith("{") and response_text.endswith("}"):
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

    def _get_expanded_context(
        self, extracted_entity: ExtractedEntity, window_size: int = 300
    ) -> str:
        """Get expanded context around the entity for better understanding."""
        # If the entity has a source_text attribute, use it for expanded context
        if hasattr(extracted_entity, "source_text") and extracted_entity.source_text:
            text = extracted_entity.source_text
            entity_text = extracted_entity.text

            # Find the entity in the source text
            entity_pos = text.lower().find(entity_text.lower())
            if entity_pos != -1:
                # Get expanded window around the entity
                start = max(0, entity_pos - window_size)
                end = min(len(text), entity_pos + len(entity_text) + window_size)

                # Try to extend to sentence boundaries
                # Look for sentence start
                for i in range(start, max(0, start - 100), -1):
                    if i == 0 or text[i - 1] in ".!?\n":
                        start = i
                        break

                # Look for sentence end
                for i in range(end, min(len(text), end + 100)):
                    if text[i] in ".!?\n":
                        end = i + 1
                        break

                expanded = text[start:end].strip()
                # Add ellipsis if truncated
                if start > 0:
                    expanded = "..." + expanded
                if end < len(text):
                    expanded = expanded + "..."

                return expanded

        # Fallback to standard context if no source text
        return extracted_entity.context

    def _build_analysis_prompt(
        self, input_name: str, extracted_entity: ExtractedEntity
    ) -> str:
        """Build the prompt for LLM analysis."""

        # Get expanded context for better role understanding
        expanded_context = self._get_expanded_context(extracted_entity)

        return f"""You are an expert in name matching and contextual analysis for compliance screening. 

CRITICAL: You must compare these TWO SPECIFIC NAMES:
- INPUT NAME TO SEARCH FOR: "{input_name}"
- ENTITY FOUND IN ARTICLE: "{extracted_entity.text}"

ARE THESE TWO NAMES THE SAME PERSON? This is the primary question.

EXPANDED ARTICLE CONTEXT: "{expanded_context}"

Step 1 - IDENTITY MATCHING:
First, determine if "{input_name}" and "{extracted_entity.text}" could be the same person.
- If the names are completely different (e.g., "John Smith" vs "Maria Garcia"), they are NOT the same person
- Consider variations like nicknames, initials, transliterations, name order
- But fundamentally different names are different people

Step 2 - CONTEXTUAL ROLE (only if names match):
If and only if you determine the names match, then analyze the person's role:
- Is this person the SUBJECT of negative news (accused, charged, investigated)?
- Or are they PERIPHERAL (witness, commenter, colleague, victim, bystander)?
- What is their relationship to the main issue in the article?

Provide your analysis as JSON:
{{
  "is_likely_match": true/false,
  "confidence_score": 0.0-1.0,
  "reasoning": "detailed explanation",
  "cultural_factors": ["list", "of", "factors"],
  "name_variations_detected": ["variations", "found"],
  "context_clues": ["relevant", "context", "information"],
  "contextual_role": {{
    "role_type": "subject|peripheral|unknown",
    "role_description": "specific role (e.g., 'accused of fraud', 'witness', 'CEO commenting')",
    "negative_association": true/false,
    "risk_indicators": ["charged with", "accused of", "investigated for", etc.],
    "role_confidence": 0.0-1.0
  }}
}}

IMPORTANT: Focus on understanding the person's actual involvement. Someone merely commenting on a situation should be marked as 'peripheral' even if the article is about serious crimes."""

    def _parse_text_response(self, response_text: str) -> Dict[str, Any]:
        """Parse text response when JSON parsing fails."""

        response_lower = response_text.lower()
        is_match = any(
            phrase in response_lower for phrase in ["likely match", "same person"]
        )

        # Dictionary mapping confidence phrases to scores (order matters for precedence)
        confidence_patterns = {
            "high confidence": 0.9,
            "very likely": 0.85,
            "medium confidence": 0.7,
            "low confidence": 0.3,
            "unlikely": 0.2,
        }

        confidence = next(
            (
                score
                for phrase, score in confidence_patterns.items()
                if phrase in response_lower
            ),
            0.5,  # default confidence
        )

        # Try to determine role from text response
        role_type = "unknown"
        if any(
            word in response_lower
            for word in ["subject", "accused", "charged", "investigated"]
        ):
            role_type = "subject"
        elif any(
            word in response_lower
            for word in ["peripheral", "witness", "comment", "colleague"]
        ):
            role_type = "peripheral"

        return {
            "is_likely_match": is_match,
            "confidence_score": confidence,
            "reasoning": response_text,
            "cultural_factors": [],
            "name_variations_detected": [],
            "context_clues": [],
            "contextual_role": {
                "role_type": role_type,
                "role_description": "Unable to parse from text response",
                "negative_association": role_type == "subject",
                "risk_indicators": [],
                "role_confidence": 0.3,
            },
        }

    def _combine_results(
        self,
        traditional_result: MatchResult,
        llm_analysis: Dict[str, Any],
        input_name: str,
        entity_text: str,
    ) -> MatchResult:
        """Combine traditional and LLM analysis."""

        traditional_conf = traditional_result.match_confidence
        llm_conf = llm_analysis.get("confidence_score", 0.0)
        llm_match = llm_analysis.get("is_likely_match", False)

        contextual_role = llm_analysis.get("contextual_role", {})
        role_type = contextual_role.get("role_type", "unknown")
        negative_association = contextual_role.get("negative_association", False)
        role_confidence = contextual_role.get("role_confidence", 0.0)

        if llm_match and role_type == "peripheral" and negative_association is False:
            # Peripheral mentions should NOT be considered matches
            llm_conf = 0.0
            llm_match = False
            confidence_adjustment_reason = "Not a match: peripheral mention only (not subject of negative news)"
        elif llm_match and role_type == "subject" and negative_association:
            llm_conf = min(llm_conf * 1.2, 1.0)
            confidence_adjustment_reason = (
                "Increased confidence: subject of negative news"
            )
        else:
            confidence_adjustment_reason = None

        # CRITICAL: If LLM determined it's a peripheral mention, always reject the match
        # regardless of traditional confidence
        if role_type == "peripheral" and not negative_association:
            logger.info(f"Forcing NO MATCH for peripheral mention: {input_name} vs {entity_text}")
            logger.info(f"  Traditional conf: {traditional_conf}, LLM conf: {llm_conf}, LLM match: {llm_match}")
            final_confidence = 0.0
            is_match = False
            confidence_level = MatchConfidenceLevel.NO_MATCH
        elif llm_conf > 0.8 and traditional_conf < 0.3 and llm_match:
            # LLM found a strong match that traditional matcher missed
            logger.info(f"LLM override match: {input_name} vs {entity_text}")
            final_confidence = llm_conf
            is_match = True
            confidence_level = self._get_confidence_level(llm_conf)
        else:
            # Standard combination of traditional and LLM results
            # BUT STILL CHECK FOR PERIPHERAL MENTIONS
            if role_type == "peripheral" and not negative_association:
                logger.info(f"Forcing NO MATCH for peripheral mention (else branch): {input_name} vs {entity_text}")
                final_confidence = 0.0
                is_match = False
                confidence_level = MatchConfidenceLevel.NO_MATCH
            else:
                final_confidence = max(traditional_conf, llm_conf if llm_match else 0)
                is_match = traditional_result.is_match_recommendation or (
                    llm_match and llm_conf > 0.5
                )
                confidence_level = self._get_confidence_level(final_confidence)

        llm_explanation = self.explainer.generate_explanation(
            MatchingLayer.LLM_ENHANCED,
            llm_conf,
            input_name,
            entity_text,
            [],
            {
                "llm_reasoning": llm_analysis.get("reasoning", ""),
                "cultural_factors": llm_analysis.get("cultural_factors", []),
                "name_variations": llm_analysis.get("name_variations_detected", []),
                "context_clues": llm_analysis.get("context_clues", []),
                "traditional_confidence": traditional_conf,
                "llm_confidence": llm_conf,
                "contextual_role": contextual_role,
                "confidence_adjustment": confidence_adjustment_reason,
            },
        )

        enhanced_explanations = traditional_result.explanation.copy()
        enhanced_explanations.append(llm_explanation)

        has_warning = any(
            "WARNING" in str(exp.reasoning) for exp in traditional_result.explanation
        )

        llm_reasoning = llm_analysis.get("reasoning", "")
        if isinstance(llm_reasoning, str) and llm_reasoning.startswith("{"):
            try:
                reasoning_data = json.loads(llm_reasoning)
                llm_reasoning = reasoning_data.get("reasoning", llm_reasoning)
            except:
                pass

        if len(llm_reasoning) > 150:
            llm_reasoning = llm_reasoning[:150] + "..."

        decision_parts = [
            f"Traditional: {traditional_conf:.1%} confidence",
            f"LLM: {llm_conf:.1%} confidence",
        ]

        # Add contextual role to decision logic
        if role_type != "unknown":
            role_desc = contextual_role.get("role_description", role_type)
            decision_parts.append(f"Role: {role_desc}")
            if role_type == "peripheral" and not negative_association:
                decision_parts.append("PERIPHERAL MENTION - NOT A MATCH")
            elif role_type == "subject" and negative_association:
                decision_parts.append("SUBJECT OF CONCERN")

        if has_warning:
            decision_parts.append("⚠️ ANALYST REVIEW REQUIRED")

        decision_parts.append(f"Decision: {'MATCH' if is_match else 'NO MATCH'}")
        decision_logic = " | ".join(decision_parts)

        # Log the final decision for debugging
        if role_type == "peripheral":
            logger.info(f"Peripheral mention detected for {input_name}: is_match={is_match}")
            logger.info(f"  Final values: confidence={final_confidence}, level={confidence_level.value}")
        
        return MatchResult(
            input_name=input_name,
            article_entity=entity_text,
            match_confidence=final_confidence,
            confidence_level=confidence_level,
            is_match_recommendation=is_match,
            explanation=enhanced_explanations,
            final_decision_logic=decision_logic,
            contextual_role=contextual_role,
        )

    def _get_confidence_level(self, confidence: float) -> MatchConfidenceLevel:
        """Convert numerical confidence to categorical level."""
        thresholds = [
            (0.95, MatchConfidenceLevel.EXACT),
            (0.85, MatchConfidenceLevel.VERY_HIGH),
            (0.7, MatchConfidenceLevel.HIGH),
            (0.5, MatchConfidenceLevel.MEDIUM),
            (0.3, MatchConfidenceLevel.LOW),
            (0.0, MatchConfidenceLevel.NO_MATCH),
        ]

        return next(level for threshold, level in thresholds if confidence >= threshold)
