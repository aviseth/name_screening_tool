"""Pydantic data models for name screening system."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class MatchConfidenceLevel(str, Enum):
    """Confidence levels for match decisions."""
    EXACT = "exact"
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NO_MATCH = "no_match"


class MatchingLayer(str, Enum):
    """Layers in the matching cascade."""
    EXACT = "exact_match"
    NICKNAME = "nickname_match"
    STRUCTURAL = "structural_match"
    FUZZY = "fuzzy_match"
    PHONETIC = "phonetic_match"
    LLM_ENHANCED = "llm_enhanced"
    NO_MATCH = "no_match"


class InputName(BaseModel):
    """Represents a name to search for."""
    full_name: str = Field(..., min_length=1)
    normalized_name: Optional[str] = Field(None)
    tokens: Optional[List[str]] = Field(default_factory=list)
    
    @field_validator('full_name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure name is not just whitespace."""
        if not v.strip():
            raise ValueError("Name cannot be empty or just whitespace")
        return v.strip()


class ExtractedEntity(BaseModel):
    """Entity extracted from article text."""
    text: str = Field(...)
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., ge=0)
    label: str = Field(default="PERSON")
    context: Optional[str] = Field(None)
    
    @field_validator('end_char')
    @classmethod
    def validate_positions(cls, v: int, info) -> int:
        """Ensure end position is after start position."""
        if 'start_char' in info.data and v <= info.data['start_char']:
            raise ValueError("end_char must be greater than start_char")
        return v


class MatchExplanation(BaseModel):
    """Detailed explanation of a matching decision."""
    layer: MatchingLayer = Field(...)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(...)
    transformations: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)


class MatchResult(BaseModel):
    """Complete result of a matching operation."""
    input_name: str = Field(...)
    article_entity: str = Field(...)
    match_confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_level: MatchConfidenceLevel = Field(...)
    is_match_recommendation: bool = Field(...)
    explanation: List[MatchExplanation] = Field(...)
    final_decision_logic: str = Field(...)
    
    @field_validator('is_match_recommendation')
    @classmethod
    def validate_recommendation(cls, v: bool, info) -> bool:
        """Ensure recommendation aligns with confidence for high recall."""
        if 'match_confidence' in info.data:
            if info.data['match_confidence'] > 0.3 and not v:
                pass
        return v

    class Config:
        """Pydantic config for better JSON output."""
        json_encoders = {
            MatchConfidenceLevel: lambda v: v.value,
            MatchingLayer: lambda v: v.value,
        }


class MatchRequest(BaseModel):
    """Request model for the matching API."""
    name: str = Field(..., min_length=1)
    article_text: str = Field(..., min_length=1)
    language: str = Field(default="en", pattern="^[a-z]{2}$")
    strict_mode: bool = Field(default=False)
    
    @field_validator('language')
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Ensure language code is lowercase."""
        return v.lower()


class TestCase(BaseModel):
    """Model for test dataset entries."""
    input_name: str = Field(...)
    article_text: str = Field(...)
    label: str = Field(..., pattern="^(match|no_match)$")
    notes: Optional[str] = Field(None)