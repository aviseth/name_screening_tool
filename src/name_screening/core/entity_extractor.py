"""Entity extraction using spaCy NLP."""

import logging
from typing import List, Dict, Optional, Set
import re
import spacy
from spacy.language import Language

from ..models.matching import ExtractedEntity

# Configure logging
logger = logging.getLogger(__name__)

LANGUAGE_MODELS = {
    'en': ['en_core_web_lg', 'en_core_web_md', 'en_core_web_sm'],
    'es': ['es_core_news_lg', 'es_core_news_md', 'es_core_news_sm'],
    'de': ['de_core_news_lg', 'de_core_news_md', 'de_core_news_sm'],
    'fr': ['fr_core_news_lg', 'fr_core_news_md', 'fr_core_news_sm'],
    'it': ['it_core_news_lg', 'it_core_news_md', 'it_core_news_sm'],
    'pt': ['pt_core_news_lg', 'pt_core_news_md', 'pt_core_news_sm'],
    'nl': ['nl_core_news_lg', 'nl_core_news_md', 'nl_core_news_sm'],
    'xx': ['xx_ent_wiki_sm'],  # Multilingual fallback
}

CONTEXT_WINDOW = 50  # characters before and after entity


class EntityExtractor:
    """Extracts named entities from text using spaCy."""
    
    def __init__(self):
        self._model_cache: Dict[str, Language] = {}
        self._available_models: Set[str] = self._detect_available_models()
    
    def _detect_available_models(self) -> Set[str]:
        available = set()
        for lang_models in LANGUAGE_MODELS.values():
            for model_name in lang_models:
                try:
                    spacy.info(model_name)
                    available.add(model_name)
                except:
                    pass
        logger.info(f"Available spaCy models: {available}")
        return available
    
    
    def _load_model(self, language: str) -> Optional[Language]:
        if language in self._model_cache:
            return self._model_cache[language]
        
        model_preferences = LANGUAGE_MODELS.get(language, [])
        
        for model_name in model_preferences:
            if model_name in self._available_models:
                try:
                    logger.info(f"Loading spaCy model: {model_name}")
                    model = spacy.load(model_name)
                    self._model_cache[language] = model
                    return model
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
        
        if language != 'xx':
            logger.warning(
                f"No model found for {language}, trying multilingual fallback"
            )
            return self._load_model('xx')
        
        logger.error("No suitable spaCy model found")
        return None
    
    def extract_entities(self, text: str, language: str = 'en') -> List[ExtractedEntity]:
        """Extract person entities using spaCy NER plus custom patterns."""
        nlp = self._load_model(language)
        if not nlp:
            logger.error("Could not load NLP model")
            return []
        
        try:
            doc = nlp(text)
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return []
        
        entities = []
        seen_spans = set()
        
        multi_particles = {'von der', 'van der', 'van den', 'de la', 'de los', 'del la'}
        
        # Process spaCy PERSON entities with overlap detection
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                span = (ent.start_char, ent.end_char)
                if any(s[0] <= span[0] < s[1] or s[0] < span[1] <= s[1] for s in seen_spans):
                    continue
                
                entity_text = ent.text.strip()
                if entity_text.lower() in multi_particles:
                    continue
                
                # Strip common titles from extracted names
                common_titles = {'president', 'ceo', 'senator', 'representative', 'minister',
                                'chancellor', 'chairman', 'director', 'professor', 'dr', 'mr', 'mrs', 'ms'}
                words = entity_text.split()
                if len(words) > 1 and words[0].lower() in common_titles:
                    entity_text = ' '.join(words[1:])
                    ent = type(ent)(entity_text, ent.start_char + len(words[0]) + 1, ent.end_char, ent.label_)
                
                seen_spans.add(span)
                
                context = self._extract_context(text, ent.start_char, ent.end_char)
                
                entity = ExtractedEntity(
                    text=ent.text,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    label=ent.label_,
                    context=context,
                    source_text=text  # Pass full text for expanded context
                )
                entities.append(entity)
        
        additional_entities = self._extract_additional_patterns(text, doc, seen_spans)
        entities.extend(additional_entities)
        
        merged_entities = self._merge_adjacent_proper_names(doc, seen_spans)
        entities.extend(merged_entities)
        
        entities.sort(key=lambda e: e.start_char)
        
        logger.info(f"Extracted {len(entities)} PERSON entities")
        return entities
    
    def _extract_additional_patterns(self, text: str, doc, seen_spans: set) -> List[ExtractedEntity]:
        """Extract names missed by spaCy using regex patterns."""
        additional = []
        
        # Extract potential abbreviations (AOC, JFK, etc.)
        abbrev_pattern = r'\b([A-Z]{2,4})\b'
        all_caps_tokens = [(m.start(), m.end(), m.group(1)) for m in re.finditer(abbrev_pattern, text)]
        
        for i, (start, end, abbrev) in enumerate(all_caps_tokens):
            is_adjacent_to_caps = False
            for j, (other_start, other_end, other_text) in enumerate(all_caps_tokens):
                if i != j:
                    gap = abs(other_start - end) if other_start > end else abs(start - other_end)
                    if gap <= 2:
                        is_adjacent_to_caps = True
                        break
            
            if is_adjacent_to_caps:
                continue
            
            context_start = max(0, start - 50)
            context_end = min(len(text), end + 50)
            context = text[context_start:context_end].lower()
            
            person_indicators = {
                'president', 'ceo', 'founder', 'director', 'minister', 'prince',
                'chairman', 'officer', 'executive', 'leader', 'head', 'chief',
                'mr', 'mrs', 'ms', 'dr', 'prof', 'sir', 'lord', 'lady',
                'announced', 'said', 'stated', 'told', 'spoke', 'testified',
                'accused', 'charged', 'arrested', 'convicted', 'sentenced'
            }
            
            has_person_context = any(indicator in context for indicator in person_indicators)
            
            next_word_idx = end
            if next_word_idx < len(text) - 10:
                next_text = text[next_word_idx:next_word_idx + 20].strip()
                if next_text.startswith(("'s", "announced", "said", "told", "is", "was", "has", "will")):
                    has_person_context = True
            
            if has_person_context:
                span = (start, end)
                if not any(s[0] <= span[0] < s[1] or s[0] < span[1] <= s[1] for s in seen_spans):
                    seen_spans.add(span)
                    additional.append(self._create_entity(abbrev, start, end, text))
        
        # Handle compound surnames with particles (von, de, etc.)
        name_particles = {
            'von', 'van', 'de', 'del', 'della', 'der', 'den', 'das', 'dos', 'du',
            'bin', 'ibn', 'ben', 'bint', 'abu', 'umm', 'al', 'el',
            'mac', 'mc', "o'", 'fitz', 'ap', 'ab', 'ferch',
            'di', 'da', 'le', 'la', 'los', 'las', 'des'
        }
        
        particles_pattern = '|'.join(re.escape(p) for p in name_particles)
        
        multi_particles = {'von der', 'van der', 'van den', 'de la', 'de los', 'del la'}
        
        for particle in multi_particles:
            standalone_pattern = rf'\b{re.escape(particle)}\b(?!\s+[A-Z][a-z]+)'
            for match in re.finditer(standalone_pattern, text, re.IGNORECASE):
                seen_spans.add((match.start(), match.end()))
        
        for particle in multi_particles:
            pattern = rf'\b((?:[A-Z][a-z]+\s+)?{re.escape(particle)}\s+[A-Z][a-z]+)\b'
            
            non_surnames = {
                'was', 'is', 'are', 'were', 'been', 'be', 'has', 'have', 'had',
                'will', 'would', 'could', 'should', 'may', 'might', 'must',
                'said', 'says', 'announced', 'told', 'spoke', 'stated',
                'today', 'yesterday', 'tomorrow', 'now', 'then',
                'here', 'there', 'where', 'when', 'how', 'why'
            }
            
            for match in re.finditer(pattern, text, re.IGNORECASE):
                name = match.group(0)
                words = name.split()
                last_word = words[-1]
                
                if (last_word[0].isupper() and 
                    last_word.lower() not in non_surnames and
                    len(last_word) > 2):
                    span = (match.start(), match.end())
                    if not any(s[0] <= span[0] < s[1] or s[0] < span[1] <= s[1] for s in seen_spans):
                        seen_spans.add(span)
                        additional.append(self._create_entity(name, match.start(), match.end(), text))
        
        compound_pattern = rf'\b((?:[A-Z][a-z]+\s+)?(?:{particles_pattern})(?:\s+[A-Z][a-z]+)+)\b'
        
        for match in re.finditer(compound_pattern, text, re.IGNORECASE):
            name = match.group(0)
            words = name.split()
            if len(words) >= 2 and words[-1][0].isupper():
                span = (match.start(), match.end())
                if not any(s[0] <= span[0] < s[1] or s[0] < span[1] <= s[1] for s in seen_spans):
                    seen_spans.add(span)
                    additional.append(self._create_entity(name, match.start(), match.end(), text))
        
        # Extract hyphenated names (Jean-Claude, Al-Rashid)
        hyphenated_pattern = r'\b([A-Z][a-z]+(?:-[A-Z]?[a-z]+)+)\b'
        for match in re.finditer(hyphenated_pattern, text):
            name = match.group(0)
            span = (match.start(), match.end())
            if not any(s[0] <= span[0] < s[1] or s[0] < span[1] <= s[1] for s in seen_spans):
                seen_spans.add(span)
                additional.append(ExtractedEntity(
                    text=name,
                    start_char=match.start(),
                    end_char=match.end(),
                    label='PERSON',
                    context=self._extract_context(text, match.start(), match.end()),
                    source_text=text
                ))
        
        full_hyphenated_pattern = r'\b([A-Z][a-z]+(?:-[A-Z][a-z]+)+(?:\s+[A-Z][a-z]+)+)\b'
        for match in re.finditer(full_hyphenated_pattern, text):
            name = match.group(0)
            span = (match.start(), match.end())
            if not any(s[0] <= span[0] < s[1] or s[0] < span[1] <= s[1] for s in seen_spans):
                seen_spans.add(span)
                additional.append(self._create_entity(name, match.start(), match.end(), text))
        
        # Extract names with initials (J. Smith, J.K. Rowling)
        initials_patterns = [
            r'\b([A-Z]\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            r'\b([A-Z]\.(?:[A-Z]\.\s*)+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        ]
        
        for pattern in initials_patterns:
            for match in re.finditer(pattern, text):
                name = match.group(0)
                span = (match.start(), match.end())
                if not any(s[0] <= span[0] < s[1] or s[0] < span[1] <= s[1] for s in seen_spans):
                    seen_spans.add(span)
                    additional.append(self._create_entity(name, match.start(), match.end(), text))
        
        # Extract social media handles (@username)
        handle_pattern = r'@([A-Za-z0-9_]+)'
        for match in re.finditer(handle_pattern, text):
            handle = match.group(0)
            span = (match.start(), match.end())
            if not any(s[0] <= span[0] < s[1] or s[0] < span[1] <= s[1] for s in seen_spans):
                seen_spans.add(span)
                additional.append(self._create_entity(handle, match.start(), match.end(), text))
        
        return additional
    
    def _create_entity(self, text: str, start: int, end: int, full_text: str) -> ExtractedEntity:
        """Helper to create ExtractedEntity objects."""
        return ExtractedEntity(
            text=text,
            start_char=start,
            end_char=end,
            label='PERSON',
            context=self._extract_context(full_text, start, end),
            source_text=full_text  # Pass full text for expanded context
        )
    
    def _merge_adjacent_proper_names(self, doc, seen_spans: set) -> List[ExtractedEntity]:
        """Merge adjacent proper nouns into person names."""
        additional = []
        
        i = 0
        while i < len(doc):
            if doc[i].pos_ == 'PROPN':
                start_idx = i
                tokens = [doc[i]]
                i += 1
                
                while i < len(doc) and (
                    doc[i].pos_ == 'PROPN' or 
                    doc[i].text.lower() in {'von', 'van', 'de', 'der', 'den', 'del', 'della', 'bin', 'al'}
                ):
                    if doc[i].text == ',':
                        break
                    tokens.append(doc[i])
                    i += 1
                
                if len(tokens) >= 2:
                    start_char = tokens[0].idx
                    end_char = tokens[-1].idx + len(tokens[-1].text)
                    full_text = doc.text[start_char:end_char]
                    span = (start_char, end_char)
                    
                    if not any(s[0] <= span[0] < s[1] or s[0] < span[1] <= s[1] for s in seen_spans):
                        if not any(t.ent_type_ in ['ORG', 'LOC', 'GPE'] for t in tokens if t.ent_type_):
                            if self._is_likely_person_name(full_text, tokens):
                                seen_spans.add(span)
                                additional.append(ExtractedEntity(
                                    text=full_text,
                                    start_char=start_char,
                                    end_char=end_char,
                                    label='PERSON',
                                    context=self._extract_context(doc.text, start_char, end_char),
                                    source_text=doc.text
                                ))
            else:
                i += 1
        
        # Handle Nordic/cultural names (Björn Eriksson, etc.)
        proper_nouns = [(i, token) for i, token in enumerate(doc) if token.pos_ == 'PROPN']
        
        for i, (idx1, token1) in enumerate(proper_nouns):
            if any(token1.idx >= s[0] and token1.idx < s[1] for s in seen_spans):
                continue
                
            for j in range(i + 1, min(i + 10, len(proper_nouns))):
                idx2, token2 = proper_nouns[j]
                
                if any(token2.idx >= s[0] and token2.idx < s[1] for s in seen_spans):
                    continue
                
                if (any(char in token1.text + token2.text for char in 'ðþæøåöäüßñç') or
                    (token1.text.endswith('dóttir') or token2.text.endswith('dóttir') or
                     token1.text.endswith('son') or token2.text.endswith('son'))):
                    
                    if idx2 - idx1 < 15:
                        combined_text = f"{token1.text} {token2.text}"
                        virtual_span = (token1.idx, token1.idx + len(combined_text))
                        
                        if not any(s[0] <= virtual_span[0] < s[1] or s[0] < virtual_span[1] <= s[1] for s in seen_spans):
                            seen_spans.add(virtual_span)
                            additional.append(ExtractedEntity(
                                text=combined_text,
                                start_char=token1.idx,
                                end_char=token1.idx + len(combined_text),
                                label='PERSON',
                                context=f"...{token1.text}... ...{token2.text}...",
                                source_text=doc.text
                            ))
                            break
        
        return additional
    
    def _is_likely_person_name(self, text: str, tokens) -> bool:
        """Heuristic to determine if proper noun sequence is a person name."""
        if text.isupper() and 2 <= len(tokens) <= 3:
            return True
        
        if any(char in text for char in 'ðþæøåöäüßñçğ'):
            return True
        
        words = text.split()
        if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
            non_name_words = {'CEO', 'CFO', 'CTO', 'Inc', 'Ltd', 'LLC', 'Corp'}
            if not any(w in non_name_words for w in words):
                return True
        
        return False
    
    def _extract_context(self, text: str, start: int, end: int) -> str:
        context_start = max(0, start - CONTEXT_WINDOW)
        context_end = min(len(text), end + CONTEXT_WINDOW)
        
        before = ' '.join(text[context_start:start].split())
        entity = text[start:end]
        after = ' '.join(text[end:context_end].split())
        
        return f"...{before} [[{entity}]] {after}..."
    
    def download_model(self, language: str) -> bool:
        """Download spaCy model for specified language."""
        model_preferences = LANGUAGE_MODELS.get(language, [])
        if not model_preferences:
            logger.error(f"No models defined for language: {language}")
            return False
        
        model_name = model_preferences[0]
        
        try:
            import subprocess
            logger.info(f"Downloading spaCy model: {model_name}")
            subprocess.run(
                ["python", "-m", "spacy", "download", model_name],
                check=True
            )
            self._available_models.add(model_name)
            return True
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            return False


def extract_person_entities(text: str, language: str = 'en') -> List[ExtractedEntity]:
    extractor = EntityExtractor()
    return extractor.extract_entities(text, language)