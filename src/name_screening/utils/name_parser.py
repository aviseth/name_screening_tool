"""Name parsing and normalization utilities with transformation tracking."""

import re
import unicodedata
from typing import List, Tuple, Set

# Cultural name components
NAME_PREFIXES = {
    'de', 'del', 'della', 'di', 'da', 'van', 'von', 'der', 'den', 'te', 'ter',
    'la', 'le', 'du', 'des', 'al', 'el', 'bin', 'ibn', 'abu', 'umm',
    'mc', "o'", 'mac', 'do', 'dos', 'das',
}

NAME_SUFFIXES = {
    'jr', 'sr', 'ii', 'iii', 'iv', 'phd', 'md', 'esq', 'cpa', 'rn',
    'junior', 'senior', 'filho', 'neto', 'sobrinho',
}

TITLES = {
    'mr', 'mrs', 'ms', 'miss', 'dr', 'prof', 'professor', 'rev', 'reverend',
    'sir', 'dame', 'lord', 'lady', 'sheikh', 'sheikha',
}

SHORT_FAMILY_NAMES = {'do', 'ko', 'go', 'no', 'mo', 'so', 'yo', 'ho'}


class NameParser:
    """Name normalization with transformation tracking."""
    
    def __init__(self):
        """Initialize parser."""
        self.transformations: List[str] = []
    
    def parse(self, name: str) -> Tuple[str, List[str], List[str]]:
        """Parse and normalize name through multi-stage pipeline."""
        self.transformations = []
        self._hyphenated_name = False
        self._original_input = name
        original = name
        
        # Processing pipeline as a list of (function, description) tuples
        pipeline = [
            (self._normalize_unicode, "Unicode normalization"),
            (self._lowercase, "Lowercase conversion"),
            (self._remove_titles, "Title removal"),
            (self._handle_punctuation, "Punctuation handling"),
            (self._tokenize, "Tokenization"),
        ]
        
        # Apply sequential text processing pipeline
        result = name
        for processor, description in pipeline[:-1]:  # All except tokenize
            result = processor(result)
        
        # Handle tokenization separately since it returns a list
        tokens = self._tokenize(result)
        tokens = self._remove_affixes(tokens)
        tokens = [t for t in tokens if t]
        normalized = ' '.join(tokens)
        
        if normalized != original.lower():
            self.transformations.append(
                f"Final normalized form: '{original}' → '{normalized}'"
            )
        
        return normalized, tokens, self.transformations.copy()
    
    def _normalize_unicode(self, name: str) -> str:
        """Normalize unicode characters (accents, etc.)."""
        normalized = unicodedata.normalize('NFD', name)
        ascii_name = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
        
        if ascii_name != name:
            self.transformations.append(f"Unicode normalized: '{name}' → '{ascii_name}'")
        return ascii_name
    
    def _lowercase(self, name: str) -> str:
        """Convert to lowercase."""
        lower = name.lower()
        if lower != name:
            self.transformations.append("Converted to lowercase")
        return lower
    
    def _remove_titles(self, name: str) -> str:
        """Remove common titles from the beginning of names."""
        words = name.split()
        if words and words[0] in TITLES:
            self.transformations.append(f"Removed title: '{words[0]}'")
            return ' '.join(words[1:])
        return name
    
    def _handle_punctuation(self, name: str) -> str:
        """Handle punctuation in names."""
        original = name
        
        if name.endswith("'s"):
            name = name[:-2]
        
        name = re.sub(r'([A-Za-z])\.', r'\1', name)
        
        if '-' in self._original_input:
            parts = self._original_input.split('-')
            self._hyphenated_name = (len(parts) == 2 and len(parts[0]) <= 3 and 
                                     parts[0][0].isupper())
            if self._hyphenated_name:
                self.transformations.append("Detected hyphenated name pattern")
            
        name = name.replace('-', ' ')
        name = re.sub(r"[^\w\s']", ' ', name)
        name = ' '.join(name.split())
        
        if name != original:
            self.transformations.append(f"Punctuation handled: '{original}' → '{name}'")
        return name
    
    def _tokenize(self, name: str) -> List[str]:
        """Split name into tokens."""
        tokens = name.split()
        if len(tokens) > 1:
            self.transformations.append(f"Tokenized into {len(tokens)} parts")
        return tokens
    
    def _remove_affixes(self, tokens: List[str]) -> List[str]:
        """Remove common prefixes and suffixes."""
        if not tokens:
            return tokens
        
        filtered = []
        removed_items = []
        
        is_short_family_name_pattern = (
            (len(tokens) == 2 and tokens[0].lower() in SHORT_FAMILY_NAMES and len(tokens[0]) <= 2) or
            (len(tokens) == 3 and tokens[1].lower() in SHORT_FAMILY_NAMES and len(tokens[1]) <= 2)
        )
        if is_short_family_name_pattern:
            self.transformations.append("Detected potential short family name pattern")
        
        # Compact token filtering with inline rules
        for i, token in enumerate(tokens):
            # Check if it's a removable prefix
            if (token in NAME_PREFIXES and len(tokens) > 1 and not (
                token in SHORT_FAMILY_NAMES and 
                (self._hyphenated_name or is_short_family_name_pattern or 
                 (len(tokens) == 3 and i == 1))
            )):
                removed_items.append(f"prefix '{token}'")
            # Check if it's a removable suffix
            elif token in NAME_SUFFIXES and i == len(tokens) - 1:
                removed_items.append(f"suffix '{token}'")
            # Keep the token
            else:
                filtered.append(token)
        
        if removed_items:
            self.transformations.append(f"Removed {', '.join(removed_items)}")
        
        return filtered
    
    def get_initials(self, tokens: List[str]) -> str:
        """Extract initials from tokens."""
        return ''.join(token[0] for token in tokens if token)
    
    def contains_initials(self, tokens: List[str]) -> bool:
        """Check if any token is likely an initial."""
        return any(len(token) == 1 for token in tokens)


def normalize_name(name: str) -> Tuple[str, List[str], List[str]]:
    """Normalize a name."""
    return NameParser().parse(name)