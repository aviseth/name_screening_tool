"""Core matching logic with confidence cascade for 100% recall."""

import logging
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
import pandas as pd
import textdistance
from pynamematcher import PyNameMatcher

from ..models.matching import (
    InputName,
    ExtractedEntity,
    MatchResult,
    MatchExplanation,
    MatchingLayer,
    MatchConfidenceLevel
)
from ..utils.name_parser import NameParser, normalize_name
from ..utils.explainability import ExplainabilityGenerator
from .llm_matcher import LLMMatcher

logger = logging.getLogger(__name__)


class NameMatcher:
    """Cascading name matcher with multiple strategies."""
    
    MATCH_THRESHOLD = 0.20
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    JARO_WINKLER_THRESHOLD = 0.75
    LEVENSHTEIN_THRESHOLD = 0.7
    
    def __init__(self, nicknames_path: Optional[str] = None, api_key: Optional[str] = None, 
                 allow_first_name_only_match: bool = True):
        """Initialize matcher."""
        self.parser = NameParser()
        self.explainer = ExplainabilityGenerator()
        self.allow_first_name_only_match = allow_first_name_only_match
        
        if nicknames_path is None:
            base_dir = Path(__file__).parent.parent.parent.parent
            nicknames_path = base_dir / "data" / "nicknames.csv"
        
        self.nickname_matcher = self._load_nickname_matcher(nicknames_path)
        self.nickname_map = self._load_nickname_map(nicknames_path)
        
        self.llm_matcher = LLMMatcher(api_key=api_key)
    
    def _load_nickname_matcher(self, path: str) -> Optional[PyNameMatcher]:
        """Load PyNameMatcher with nickname data."""
        if not os.path.exists(path):
            logger.warning(f"Nicknames file not found: {path}")
            return None
        
        for init_method in [lambda: PyNameMatcher(nicknames_file=str(path)), 
                          lambda: PyNameMatcher(str(path))]:
            try:
                return init_method()
            except:
                continue
        return None
    
    def _load_nickname_map(self, path: str) -> Dict[str, Set[str]]:
        """Load nickname mappings for bidirectional lookup."""
        nickname_map = {}
        
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                
                for _, row in df.iterrows():
                    canonical = row['canonical'].lower()
                    nickname = row['nickname'].lower()
                    
                    if nickname not in nickname_map:
                        nickname_map[nickname] = set()
                    nickname_map[nickname].add(canonical)
                    
                    if canonical not in nickname_map:
                        nickname_map[canonical] = set()
                    nickname_map[canonical].add(nickname)
                    
                logger.info(f"Loaded {len(nickname_map)} nickname mappings")
        except Exception as e:
            logger.error(f"Failed to load nickname map: {e}")
        
        return nickname_map
    
    def match(
        self, 
        input_name: str, 
        extracted_entity: ExtractedEntity,
        strict_mode: bool = False,
        use_llm: bool = False
    ) -> MatchResult:
        """Match input name against extracted entity."""
        input_normalized, input_tokens, input_transforms = normalize_name(input_name)
        entity_normalized, entity_tokens, entity_transforms = normalize_name(extracted_entity.text)
        
        explanations: List[MatchExplanation] = []
        # Track high-risk match patterns for security warnings
        critical_warnings = {
            'surname_only_match': False,
            'single_token_match': False,
            'middle_name_mismatch': False,
            'surname_mismatch': False
        }
        
        all_transforms = input_transforms + entity_transforms
        
        # Cascade through matching strategies in order of confidence
        layer_configs = {
            'exact': {
                'method': self._exact_match,
                'args': (input_normalized, entity_normalized, input_name, extracted_entity.text, all_transforms),
                'enabled': True
            },
            'nickname': {
                'method': self._nickname_match,
                'args': (input_tokens, entity_tokens, input_name, extracted_entity.text, all_transforms),
                'enabled': bool(self.nickname_map)
            },
            'structural': {
                'method': self._structural_match,
                'args': (input_tokens, entity_tokens, input_name, extracted_entity.text, all_transforms),
                'enabled': True
            },
            'fuzzy': {
                'method': self._fuzzy_match,
                'args': (input_normalized, entity_normalized, input_name, extracted_entity.text, all_transforms),
                'enabled': True
            },
            'phonetic': {
                'method': self._phonetic_match,
                'args': (input_normalized, entity_normalized, input_name, extracted_entity.text, all_transforms),
                'enabled': True
            }
        }
        
        matching_layers = [(config['method'], config['args']) 
                          for config in layer_configs.values() 
                          if config['enabled']]
        
        # Define critical warning patterns for efficient checking
        warning_patterns = {
            'surname_only_match': "Only surname matches",
            'single_token_match': "Single name component",
            'middle_name_mismatch': "Middle names differ",
            'surname_mismatch': "different surnames"
        }
        
        skip_fuzzy_phonetic = [self._fuzzy_match, self._phonetic_match]
        
        for layer_func, args in matching_layers:
            # Skip low-confidence layers if critical warnings detected
            if layer_func in skip_fuzzy_phonetic and any(critical_warnings.values()):
                continue
                
            result = layer_func(*args)
            if result:
                # Flag security warnings for analyst review
                for warning_key, pattern in warning_patterns.items():
                    if pattern in result.reasoning:
                        critical_warnings[warning_key] = True
                
                explanations.append(result)
                if result.confidence_score >= self.HIGH_CONFIDENCE_THRESHOLD:
                    return self._create_match_result(
                        input_name, extracted_entity.text,
                        explanations, result.confidence_score
                    )
        
        if explanations:
            best_confidence = max(exp.confidence_score for exp in explanations)
        else:
            best_confidence = 0.0
            explanations.append(self.explainer.generate_explanation(
                MatchingLayer.NO_MATCH, 0.0, input_name, extracted_entity.text,
                input_transforms + entity_transforms,
                {'reason': 'No matching strategies succeeded'}
            ))
        
        if strict_mode:
            best_confidence *= 0.9
        
        traditional_result = self._create_match_result(
            input_name, extracted_entity.text,
            explanations, best_confidence
        )
        
        if use_llm:
            if self.llm_matcher.is_available():
                return self.llm_matcher.enhance_match(
                    input_name, extracted_entity, traditional_result
                )
            else:
                enhanced_explanations = traditional_result.explanation.copy()
                enhanced_explanations.append(
                    self.explainer.generate_explanation(
                        MatchingLayer.LLM_ENHANCED, 0.0, input_name,
                        extracted_entity.text, [],
                        {"warning": "LLM enhancement requested but not available"}
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
        return traditional_result
    
    def _exact_match(self, input_normalized: str, entity_normalized: str,
                     input_original: str, entity_original: str,
                     transformations: List[str]) -> Optional[MatchExplanation]:
        """Check for exact match after normalization."""
        if input_normalized == entity_normalized:
            return self.explainer.generate_explanation(
                MatchingLayer.EXACT, 1.0, input_original,
                entity_original, transformations
            )
        return None
    
    def _nickname_match(self, input_tokens: List[str], entity_tokens: List[str],
                        input_original: str, entity_original: str,
                        transformations: List[str]) -> Optional[MatchExplanation]:
        """Check for nickname matches."""
        input_normalized = ' '.join(input_tokens)
        entity_normalized = ' '.join(entity_tokens)
        
        if entity_normalized in self.nickname_map and input_normalized in self.nickname_map[entity_normalized]:
            return self.explainer.generate_explanation(
                MatchingLayer.NICKNAME, 0.95, input_original, entity_original,
                transformations, {
                    'abbreviation': entity_original,
                    'full_name': input_original,
                    'nickname_source': 'nickname database (abbreviation)'
                }
            )
        
        if input_normalized in self.nickname_map and entity_normalized in self.nickname_map[input_normalized]:
            return self.explainer.generate_explanation(
                MatchingLayer.NICKNAME, 0.95, input_original, entity_original,
                transformations, {
                    'abbreviation': input_original,
                    'full_name': entity_original,
                    'nickname_source': 'nickname database (abbreviation)'
                }
            )
        
        matches_found = 0
        total_tokens = max(len(input_tokens), len(entity_tokens))
        
        for i_token in input_tokens:
            for e_token in entity_tokens:
                if i_token == e_token:
                    matches_found += 1
                    continue
                
                if i_token in self.nickname_map and e_token in self.nickname_map[i_token]:
                    matches_found += 1
                    confidence = 0.95 * (matches_found / total_tokens)
                    return self.explainer.generate_explanation(
                        MatchingLayer.NICKNAME, confidence, input_original,
                        entity_original, transformations, {
                            'original_form': i_token,
                            'canonical_form': e_token,
                            'nickname_source': 'nickname database'
                        }
                    )
        return None
    
    def _structural_match(
        self,
        input_tokens: List[str],
        entity_tokens: List[str],
        input_original: str,
        entity_original: str,
        transformations: List[str]
    ) -> Optional[MatchExplanation]:
        """Check for structural matches (initials, reordering, etc.)."""
        context = {}
        confidence = 0.0
        
        # Handle single token vs multi-token scenarios
        if len(entity_tokens) == 1 and len(input_tokens) > 1:
            # Check if the single token exactly matches any token in the input
            exact_match_position = None
            if entity_tokens[0] in input_tokens:
                exact_match_position = input_tokens.index(entity_tokens[0])
            
            # Also check for fuzzy matches to detect similar tokens (Chen vs Cheng)
            fuzzy_match_position = None
            best_fuzzy_score = 0.0
            
            for i, input_token in enumerate(input_tokens):
                fuzzy_score = self._similarity_score(entity_tokens[0], input_token)
                
                if fuzzy_score > best_fuzzy_score and fuzzy_score >= 0.8:  # High similarity threshold
                    best_fuzzy_score = fuzzy_score
                    fuzzy_match_position = i
            
            # Determine which match type we found
            match_position = exact_match_position if exact_match_position is not None else fuzzy_match_position
            is_fuzzy = exact_match_position is None and fuzzy_match_position is not None
            
            if match_position is not None:
                # Score based on name position (first/last names more reliable)
                position_configs = {
                    0: {'context_key': 'first_name_match', 'confidence_boost': 0.3},
                    len(input_tokens) - 1: {'context_key': 'last_name_match', 'confidence_boost': 0.3},
                    'default': {'context_key': 'middle_name_match', 'confidence_boost': 0.2}
                }
                
                config = position_configs.get(match_position, position_configs['default'])
                context[config['context_key']] = True
                confidence += config['confidence_boost']
                    
                context['single_token_match'] = True
                fuzzy_part = f'Single name component fuzzy match ({best_fuzzy_score:.1%} similar)' if is_fuzzy else 'Single name component match'
                context['warning'] = f'{fuzzy_part} - may need review'
        
        # Check initials expansion (J.K. → Joanne Kathleen)
        initials_match, initials_confidence = self._check_initials_expansion(input_tokens, entity_tokens)
        if initials_match:
            context['initials_match'] = True
            context['initials_expansion'] = initials_match
            confidence += initials_confidence
            
        # Check 1b: Full initials match (E.R. Musk case)
        if all(len(token) == 1 for token in entity_tokens[:-1]):
            entity_initials = ''.join(entity_tokens[:-1]).lower()
            input_initials = ''.join(t[0] for t in input_tokens[:-1]).lower()
            if entity_initials == input_initials and input_tokens[-1] == entity_tokens[-1]:
                context['full_initials_match'] = True
                confidence += 0.8
        
        # Check 2: Last name match (if we can identify it)
        if self._check_last_name_match(input_tokens, entity_tokens):
            context['last_name_match'] = True
            
            # Check if it's a distinctive compound surname (e.g., "von der Leyen")
            if len(input_tokens) > 2 and len(entity_tokens) >= 1:
                # Extract the compound surname from input
                compound_parts = []
                for token in input_tokens:
                    if token.lower() in {'von', 'van', 'de', 'der', 'den', 'del', 'della', 'bin', 'al'}:
                        compound_parts.append(token)
                    elif compound_parts:  # Already started collecting compound
                        compound_parts.append(token)
                
                # If we have a multi-part surname match, it's more distinctive
                if len(compound_parts) >= 2 and entity_tokens[-1] == input_tokens[-1]:
                    context['compound_surname_match'] = True
                    confidence += 0.4  # Higher confidence for distinctive compound surnames
                elif confidence > 0:
                    confidence += 0.2
                else:
                    confidence += 0.1
            else:
                # Regular surname match logic
                if confidence > 0:
                    confidence += 0.2
                else:
                    # Just surname match alone is not enough
                    confidence += 0.1
        
        # Check 3: Middle name as first name
        if self._check_middle_name_swap(input_tokens, entity_tokens):
            context['middle_name_swap'] = True
            confidence += 0.3
        
        # Check for possessive forms (e.g., "Warren's" or "Elizabeth's")
        if len(entity_tokens) == 1 and len(input_tokens) >= 1:
            entity_text = entity_tokens[0]
            # Check if entity matches any part of the input name
            for token in input_tokens:
                if entity_text.lower() == token.lower():
                    context['possessive_form'] = True
                    context['pattern'] = f"Possessive or partial reference: '{entity_text}'"
                    confidence += 0.35  # Boost to ensure recall
                    break
        
        # Detect name reordering (Kim Jong-un ↔ Jong un Kim)
        if len(input_tokens) >= 2 and len(entity_tokens) >= 2:
            input_set = set(input_tokens)
            entity_set = set(entity_tokens)
            
            # If all tokens match but in different order, it's likely the same person
            if input_set == entity_set and input_tokens != entity_tokens:
                context['name_order_variation'] = True
                confidence += 0.85
            # Check if it's a subset match with reordering (e.g. "Kim Jong-un" vs "Jong un Kim")
            elif len(input_set & entity_set) >= 2 and len(input_set & entity_set) == min(len(input_set), len(entity_set)):
                # All tokens from the shorter name are in the longer name
                context['name_order_variation'] = True
                confidence += 0.8
        
        if len(input_tokens) == 2 and len(entity_tokens) == 1:
            if (entity_tokens[0] == input_tokens[0] or entity_tokens[0] == input_tokens[1]):
                patronymic_indicators = [
                    any(char in ''.join(input_tokens) for char in 'ðþæøåöäü'),
                    any(token.endswith(('son', 'dóttir', 'sen', 'sson')) for token in input_tokens)
                ]
                
                if any(patronymic_indicators):
                    context['patronymic_partial_match'] = True
                    context['pattern'] = f"Patronymic name: {' '.join(input_tokens)} matches partial '{entity_tokens[0]}'"
                    confidence += 0.75
        
        if len(entity_tokens) >= 2 and len(input_tokens) >= 2:
            if input_tokens[0] == entity_tokens[-1]:
                entity_initials = []
                for token in entity_tokens[:-1]:
                    if len(token) <= 3 and token.isalpha():
                        entity_initials.extend(list(token))
                    else:
                        entity_initials.append(token[0])
                
                input_initials = [t[0] for t in input_tokens[1:]]
                if (len(entity_initials) == len(input_initials) and 
                    all(e.lower() == i.lower() for e, i in zip(entity_initials, input_initials))):
                    context['initials_surname_pattern'] = True
                    context['pattern'] = f"{' '.join(input_tokens)} → initials + surname"
                    confidence += 0.85
        
        for i, input_token in enumerate(input_tokens):
            for j, entity_token in enumerate(entity_tokens):
                if i == 0 and j == 0 and len(input_token) >= 3 and len(entity_token) >= 3 and input_token != entity_token:
                    if input_token.startswith(entity_token) or entity_token.startswith(input_token):
                        shorter = min(input_token, entity_token, key=len)
                        longer = max(input_token, entity_token, key=len)
                        if len(shorter) >= len(longer) * 0.4:
                            context['name_shortening'] = f"{shorter} → {longer}"
                            confidence += 0.7
        
        common_tokens = set(input_tokens) & set(entity_tokens)
        if common_tokens:
            context['matching_components'] = list(common_tokens)
            overlap_ratio = len(common_tokens) / max(len(input_tokens), len(entity_tokens))
            confidence += 0.2 * overlap_ratio
        
        # Flag dangerous surname-only matches for review
        if len(input_tokens) >= 2 and len(entity_tokens) >= 2:
            first_input, first_entity = input_tokens[0].lower(), entity_tokens[0].lower()
            last_match = input_tokens[-1].lower() == entity_tokens[-1].lower()
            
            if last_match and first_input != first_entity:
                fuzzy_score = self._similarity_score(first_input, first_entity)
                
                if fuzzy_score < 0.75:  # Completely different first names
                    context.update({
                        'surname_only_match': True,
                        'warning': 'Only surname matches - likely different person',
                        'first_name_similarity': fuzzy_score
                    })
                    confidence = max(0.2, confidence * 0.3)
        
        # Check for middle name mismatches
        if len(input_tokens) == 3 and len(entity_tokens) == 3:
            first_match = input_tokens[0].lower() == entity_tokens[0].lower()
            last_match = input_tokens[-1].lower() == entity_tokens[-1].lower()
            middle_input, middle_entity = input_tokens[1].lower(), entity_tokens[1].lower()
            
            if first_match and last_match and middle_input != middle_entity:
                middle_similarity = self._similarity_score(middle_input, middle_entity)
                
                if middle_similarity < 0.6:
                    context.update({
                        'middle_name_mismatch': True,
                        'warning': f'Middle names differ: {input_tokens[1]} vs {entity_tokens[1]}',
                        'middle_name_similarity': middle_similarity
                    })

        # Check for surname mismatches (similar first names, different surnames)
        if len(input_tokens) >= 2 and len(entity_tokens) >= 2:
            input_first, input_last = input_tokens[0].lower(), input_tokens[-1].lower()
            entity_first, entity_last = entity_tokens[0].lower(), entity_tokens[-1].lower()
            
            if input_last != entity_last:
                first_similarity = self._similarity_score(input_first, entity_first)
                
                if first_similarity > 0.75:  # Similar first names, different surnames
                    context.update({
                        'surname_mismatch': True,
                        'warning': f'Similar first names but different surnames: {input_tokens[-1]} vs {entity_tokens[-1]} - likely different people',
                        'first_name_similarity': first_similarity
                    })
                    confidence = max(0.2, confidence * 0.4)
        
        # Check for first-name-only matches (configurable behavior)
        if not self.allow_first_name_only_match and len(input_tokens) >= 2 and len(entity_tokens) >= 2:
            # Check if ONLY first names match and last names are different
            first_name_match = input_tokens[0].lower() == entity_tokens[0].lower()
            last_name_mismatch = input_tokens[-1].lower() != entity_tokens[-1].lower()
            
            if first_name_match and last_name_mismatch:
                # This configuration disallows first-name-only matches
                context['first_name_only_match'] = True
                context['match_rejected'] = 'First-name-only matches disabled by configuration'
                # Return None to reject this match entirely
                return None
        
        # Cap confidence for high-risk patterns
        risk_caps = {
            'single_token_match': lambda ctx: 0.35 if (ctx.get('patronymic_partial_match') or 
                                                      ctx.get('possessive_form')) else 0.25,
            'middle_name_mismatch': lambda ctx: 0.3,
            'surname_mismatch': lambda ctx: 0.25
        }
        
        for risk_factor, cap_func in risk_caps.items():
            if context.get(risk_factor):
                confidence = min(confidence, cap_func(context))
                break
        
        # Return match if confidence is sufficient OR if there's a warning
        if confidence > 0.5 or context.get('surname_only_match') or context.get('middle_name_mismatch') or context.get('single_token_match') or context.get('surname_mismatch'):
            return self.explainer.generate_explanation(
                MatchingLayer.STRUCTURAL,
                min(confidence, 0.85),  # Cap structural match confidence
                input_original,
                entity_original,
                transformations,
                context
            )
        
        return None
    
    def _check_initials_match(self, tokens1: List[str], tokens2: List[str]) -> bool:
        """Check if initials match between token sets."""
        initials1 = ''.join(t[0] for t in tokens1 if t)
        initials2 = ''.join(t[0] for t in tokens2 if t)
        return bool(initials1 and (initials1 in initials2 or initials2 in initials1))
    
    def _check_initials_expansion(self, input_tokens: List[str], entity_tokens: List[str]) -> Tuple[str, float]:
        """Check if input contains initials that match full names in entity.
        
        Returns:
            Tuple of (match_description, confidence_boost)
        """
        # Case 1: Concatenated initials like "jk" matching "joanne kathleen"
        if len(input_tokens) >= 1 and len(entity_tokens) >= 2:
            for i, token in enumerate(input_tokens):
                # Check if token looks like concatenated initials (2-4 letters, no vowels except first)
                if (2 <= len(token) <= 4 and 
                    all(c not in 'aeiou' or j == 0 for j, c in enumerate(token))):
                    
                    # Try to match against entity tokens
                    if i == 0 and len(entity_tokens) >= len(token):
                        # Check if initials match first letters of entity tokens
                        entity_initials = ''.join(t[0] for t in entity_tokens[:len(token)])
                        if token.lower() == entity_initials.lower():
                            # Also check if remaining tokens match
                            remaining_input = input_tokens[i+1:]
                            remaining_entity = entity_tokens[len(token):]
                            if remaining_input == remaining_entity:
                                return f"Initials '{token}' expanded to {' '.join(entity_tokens[:len(token)])}", 0.8
        
        # Case 2: Individual initials like "J. K." matching "Joanne Kathleen"
        # This is already handled by the basic initials match
        if self._check_initials_match(input_tokens, entity_tokens):
            return "Basic initials match", 0.4
        
        return "", 0.0
    
    def _check_last_name_match(self, tokens1: List[str], tokens2: List[str]) -> bool:
        """Check if last names match (assuming last token is last name)."""
        return bool(tokens1 and tokens2 and tokens1[-1] == tokens2[-1])
    
    def _check_middle_name_swap(self, tokens1: List[str], tokens2: List[str]) -> bool:
        """Check if middle name is used as first name."""
        return (len(tokens1) >= 2 and len(tokens2) >= 2 and 
                (tokens1[1] == tokens2[0] or tokens1[0] == tokens2[1]))
    
    def _fuzzy_match(
        self,
        input_normalized: str,
        entity_normalized: str,
        input_original: str,
        entity_original: str,
        transformations: List[str]
    ) -> Optional[MatchExplanation]:
        """Perform fuzzy string matching using Jaro-Winkler and Levenshtein."""
        # Check if first-name-only matches are disabled
        if not self.allow_first_name_only_match:
            input_tokens = input_normalized.split()
            entity_tokens = entity_normalized.split()
            
            # If we have full names, check if only first names match
            if len(input_tokens) >= 2 and len(entity_tokens) >= 2:
                first_match = input_tokens[0] == entity_tokens[0]
                last_mismatch = input_tokens[-1] != entity_tokens[-1]
                
                if first_match and last_mismatch:
                    # Skip fuzzy matching when first-name-only is disabled
                    return None
        # First try full string matching
        jaro_score = textdistance.jaro_winkler(input_normalized, entity_normalized)
        levenshtein_score = textdistance.levenshtein.normalized_similarity(
            input_normalized, entity_normalized
        )
        
        best_score = max(jaro_score, levenshtein_score)
        algorithm = "Jaro-Winkler" if jaro_score > levenshtein_score else "Levenshtein"
        
        # Fall back to token-level matching if full string fails
        if best_score < self.JARO_WINKLER_THRESHOLD:
            input_tokens = input_normalized.split()
            entity_tokens = entity_normalized.split()
            
            # Check each entity token against each input token
            for e, e_token in enumerate(entity_tokens):
                for i, i_token in enumerate(input_tokens):
                    token_jaro = textdistance.jaro_winkler(i_token, e_token)
                    token_lev = textdistance.levenshtein.normalized_similarity(i_token, e_token)
                    token_score = max(token_jaro, token_lev)
                    
                    # For single token fuzzy matches, require higher threshold
                    # AND check that it's not just a common surname match
                    if token_score >= 0.85 and len(e_token) > 3:  # Don't match very short tokens
                        # Check if this is likely just a surname match
                        is_likely_surname = (
                            (i == len(input_tokens) - 1 and e == len(entity_tokens) - 1) or  # Both last tokens
                            (len(input_tokens) == 1 or len(entity_tokens) == 1)  # Single token names
                        )
                        
                        # Check if tokens are in different positions (cross-match)
                        is_cross_match = (i != e) and len(input_tokens) > 1 and len(entity_tokens) > 1
                        
                        # For perfect matches (1.0), check if it's a problematic cross-match
                        if token_score == 1.0 and is_cross_match:
                            # This is like "Chen Man" matching "Sarah Chen" - chen at different positions
                            continue  # Skip this match
                        
                        if not is_likely_surname or token_score >= 0.95:
                            context = {
                                'similarity_score': token_score,
                                'algorithm': "Token-based " + ("Jaro-Winkler" if token_jaro > token_lev else "Levenshtein"),
                                'matched_token': f"{i_token} ~ {e_token}",
                                'edit_distance': textdistance.levenshtein(i_token, e_token)
                            }
                            
                            # Lower confidence for surname-only matches
                            confidence_multiplier = 0.5 if is_likely_surname else 0.8
                            
                            return self.explainer.generate_explanation(
                                MatchingLayer.FUZZY,
                                token_score * confidence_multiplier,
                                input_original,
                                entity_original,
                                transformations,
                                context
                            )
        
        # Check threshold for full string match
        if best_score >= self.JARO_WINKLER_THRESHOLD:
            context = {
                'similarity_score': best_score,
                'algorithm': algorithm,
                'edit_distance': textdistance.levenshtein(
                    input_normalized, entity_normalized
                )
            }
            
            return self.explainer.generate_explanation(
                MatchingLayer.FUZZY,
                best_score * 0.9,  # Slightly reduce confidence for fuzzy matches
                input_original,
                entity_original,
                transformations,
                context
            )
        
        return None
    
    def _phonetic_match(
        self,
        input_normalized: str,
        entity_normalized: str,
        input_original: str,
        entity_original: str,
        transformations: List[str]
    ) -> Optional[MatchExplanation]:
        """Perform phonetic matching using MRA algorithm."""
        # Use MRA (Match Rating Approach) for phonetic matching
        mra_score = textdistance.mra.normalized_similarity(
            input_normalized, entity_normalized
        )
        
        if mra_score >= 0.8:  # Phonetic threshold
            context = {
                'algorithm': 'Match Rating Approach (MRA)',
                'phonetic_score': mra_score
            }
            
            return self.explainer.generate_explanation(
                MatchingLayer.PHONETIC,
                mra_score * 0.7,  # Lower confidence for phonetic matches
                input_original,
                entity_original,
                transformations,
                context
            )
        
        return None
    
    def _similarity_score(self, text1: str, text2: str) -> float:
        """Calculate similarity score using Jaro-Winkler and Levenshtein."""
        return max(
            textdistance.jaro_winkler(text1, text2),
            textdistance.levenshtein.normalized_similarity(text1, text2)
        )
    
    def _create_match_result(
        self,
        input_name: str,
        entity_name: str,
        explanations: List[MatchExplanation],
        confidence: float
    ) -> MatchResult:
        """Create final match result with warnings and decision logic."""
        # Determine if this is a match based on threshold
        is_match = confidence >= self.MATCH_THRESHOLD
        
        # Get confidence level
        confidence_level = self.explainer._score_to_level(confidence)
        
        # Generate final decision logic
        final_logic = self.explainer.generate_final_decision(
            explanations, is_match, confidence
        )
        
        # Check for warnings in explanations
        has_warning = any(
            'WARNING' in exp.reasoning for exp in explanations
        )
        
        if has_warning:
            final_logic = "⚠️ ANALYST REVIEW REQUIRED\n" + final_logic
        
        # Format explanation summary
        explanation_summary = self.explainer.format_explanation_summary(explanations)
        
        return MatchResult(
            input_name=input_name,
            article_entity=entity_name,
            match_confidence=confidence,
            confidence_level=confidence_level,
            is_match_recommendation=is_match,
            explanation=explanations,
            final_decision_logic=final_logic
        )