# Name Screening Tool - Comprehensive Technical Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture and Design Decisions](#architecture-and-design-decisions)
3. [File Structure and Components](#file-structure-and-components)
4. [Core Functions and Classes](#core-functions-and-classes)
5. [Data Models and Types](#data-models-and-types)
6. [Matching Algorithms and Logic](#matching-algorithms-and-logic)
7. [LLM Integration](#llm-integration)
8. [Configuration and Data Files](#configuration-and-data-files)
9. [User Interfaces](#user-interfaces)
10. [Testing and Evaluation](#testing-and-evaluation)
11. [Security Features](#security-features)
12. [Performance Considerations](#performance-considerations)

---

## System Overview

The Name Screening Tool is a comprehensive adverse media screening system designed for financial compliance. It addresses the critical challenge of accurately matching names in news articles to individuals of interest while maintaining 100% recall (no false negatives) and maximizing precision through intelligent pattern recognition.

### Core Problem Solved
Financial institutions face a challenging trade-off in adverse media screening:
- **Too strict**: Risk regulatory fines for missing sanctioned individuals
- **Too loose**: Analysts waste hours reviewing false positives

### Solution Approach
The system implements a **confidence cascade** with increasingly fuzzy matches, prioritizing recall over precision while providing detailed explanations for audit trails.

---

## Architecture and Design Decisions

### 1. Confidence Cascade Strategy

**Decision**: Implement multiple matching layers with decreasing confidence levels
**Rationale**: Ensures 100% recall while providing granular confidence scoring

```
1. Exact Match (100% confidence)
2. Nickname Match (95% confidence)
3. Structural Match (85% confidence)
4. Fuzzy Match (70% confidence)
5. Phonetic Match (65% confidence)
6. LLM Enhancement (Variable confidence)
```

**Why this approach over alternatives:**
- Single fuzzy threshold: Would miss cultural naming patterns
- ML black box: Lacks explainability required for compliance
- Simple exact matching: Would miss legitimate variations

### 2. Explainable AI Design

**Decision**: Generate detailed explanations for every matching decision
**Rationale**: Compliance requirements demand audit trails and transparency

**Implementation**: Each match includes:
- Reasoning chain with confidence scores
- Transformation history
- Evidence supporting decision
- Warning flags for analyst review

### 3. Entity-First Processing

**Decision**: Extract entities first, then match against each
**Rationale**: More accurate than searching for exact names in raw text

**Why not string matching:**
- Misses context clues (titles, positions)
- Cannot handle multi-word name variations
- Lacks precision in entity boundaries

### 4. Optional LLM Enhancement

**Decision**: Make LLM integration optional rather than required
**Rationale**: Balance between accuracy and operational concerns

**Traditional Mode Benefits:**
- ~50ms per article processing
- Zero API costs
- Offline capability
- Deterministic results

**LLM Mode Benefits:**
- Cultural context understanding
- Complex abbreviation resolution
- ~2-3 seconds per article

---

## File Structure and Components

### Core Application Files

#### `/src/name_screening/main.py` - CLI Interface (285 lines)
**Purpose**: Command-line interface and application entry point
**Key Functions**:
- `match()`: Main CLI command handler
- `process_match_request()`: Core processing orchestrator
- `display_pretty_results()`: User-friendly output formatting
- `display_json_results()`: Machine-readable output
- `setup_spacy_model()`: Model validation and loading

**Design Decisions**:
- **Typer framework**: Chosen for modern CLI experience with type hints
- **Rich library**: Provides colored output and tables for better UX
- **Dual output modes**: Pretty for humans, JSON for integration
- **Early validation**: Checks file existence and model availability before processing

#### `/src/name_screening/core/matcher.py` - Core Matching Engine (756 lines)
**Purpose**: Implements the confidence cascade matching algorithm
**Key Classes and Methods**:

**Class: `NameMatcher`**
- `__init__()`: Loads nickname databases and initializes components
- `match()`: Main matching orchestrator implementing cascade
- `_exact_match()`: Perfect normalization-based matching
- `_nickname_match()`: Database-driven nickname resolution
- `_structural_match()`: Pattern-based matching (initials, reordering)
- `_fuzzy_match()`: Handles typos and misspellings
- `_phonetic_match()`: Sound-alike matching

**Critical Design Decisions**:
- **Match threshold of 20%**: Ensures 100% recall with warning system
- **Bidirectional nickname mapping**: Handles both directions of nickname relationships
- **Warning system integration**: Flags ambiguous matches for manual review
- **Transformation tracking**: Records all normalization steps for explainability

**Why cascade over single algorithm:**
- Different name variations require different strategies
- Allows fine-tuned confidence scoring
- Enables early exit on high-confidence matches
- Provides multiple evidence sources for decisions

#### `/src/name_screening/core/entity_extractor.py` - NLP Entity Processing (414 lines)
**Purpose**: Extracts person entities from article text using spaCy NER
**Key Functions**:
- `extract_entities()`: Main entity extraction with custom pattern enhancement
- `_extract_additional_patterns()`: Catches abbreviations and patterns spaCy misses
- `_merge_adjacent_proper_names()`: Combines split proper names
- `_load_model()`: Smart model selection with fallbacks

**Design Decisions**:
- **Multi-model support**: Prefers larger models but falls back gracefully
- **Pattern enhancement**: Supplements spaCy with regex for edge cases
- **Title stripping**: Removes positional titles for cleaner matching
- **Context preservation**: Maintains surrounding text for LLM analysis

**Why spaCy over alternatives:**
- Strong multilingual support
- Industrial-strength NER performance
- Customizable pipeline
- Offline capability

#### `/src/name_screening/core/llm_matcher.py` - LLM Enhancement (275 lines)
**Purpose**: Provides contextual understanding via Google Gemini
**Key Functions**:
- `enhance_match()`: Analyzes context for cultural and linguistic patterns
- `_build_prompt()`: Constructs structured prompts for consistent responses
- `_parse_llm_response()`: Extracts structured data from LLM output

**Design Decisions**:
- **Google Gemini 2.0 Flash**: Chosen for speed and cost efficiency
- **Graceful degradation**: System functions without LLM if unavailable
- **Structured prompts**: Ensures consistent JSON responses
- **Context window optimization**: Provides relevant context without token waste

#### `/src/name_screening/utils/name_parser.py` - Name Normalization (169 lines)
**Purpose**: Handles global naming conventions and normalization
**Key Functions**:
- `parse()`: Main normalization pipeline
- `normalize_name()`: Public API for name normalization
- `_handle_cultural_patterns()`: Special handling for various naming systems

**Cultural Considerations**:
- **Name particles**: Handles 'von', 'de', 'al', 'bin', etc.
- **Hyphenated names**: Preserves structure while enabling matching
- **Unicode normalization**: Handles accented characters consistently
- **Title removal**: Strips prefixes and suffixes appropriately

**Why comprehensive normalization:**
- Global financial institutions handle diverse names
- Regulatory requirements for consistency
- Reduces false negatives from formatting differences

#### `/src/name_screening/utils/explainability.py` - Decision Documentation (232 lines)
**Purpose**: Generates human-readable explanations for audit compliance
**Key Functions**:
- `generate_explanation()`: Creates structured explanations
- `_generate_reasoning()`: Builds human-readable reasoning chains
- `_determine_confidence_level()`: Maps scores to categorical levels

**Compliance Design**:
- **Audit trails**: Every decision includes reasoning
- **Regulatory language**: Uses compliance-appropriate terminology
- **Evidence preservation**: Maintains transformation history
- **Warning integration**: Highlights security concerns

### Data Models

#### `/src/name_screening/models/matching.py` - Type System (117 lines)
**Purpose**: Pydantic models ensuring type safety and validation
**Key Models**:
- `MatchResult`: Complete matching outcome with explanations
- `ExtractedEntity`: NLP-extracted person entities
- `MatchExplanation`: Detailed reasoning for decisions
- `InputName`: Validated input name structure

**Why Pydantic over native types:**
- Runtime validation prevents errors
- Automatic JSON serialization
- Clear API contracts
- IDE support and documentation

### User Interfaces

#### `/web_app.py` - Web Interface (762 lines)
**Purpose**: Flask-based web UI for interactive screening
**Key Features**:
- Form-based input with validation
- Real-time processing feedback
- Visual confidence indicators
- Color-coded results with warnings

**Design Decisions**:
- **Flask over FastAPI**: Simpler deployment for smaller teams
- **Server-side rendering**: Reduces client complexity
- **Progressive enhancement**: Works without JavaScript
- **Responsive design**: Supports mobile and desktop

#### `/templates/index.html` - Frontend Interface
**Purpose**: User interface for web-based screening
**Features**:
- Clean, professional design
- Real-time validation feedback
- Accessibility considerations
- Mobile-responsive layout

### Testing and Evaluation

#### `/evaluate.py` - Core Test Suite (566 lines)
**Purpose**: Comprehensive testing of matching accuracy
**Test Categories**:
- Exact matches (baseline verification)
- Nickname variants (cultural awareness)
- Structural patterns (initial handling)
- Fuzzy matching (typo tolerance)
- Edge cases (boundary conditions)

#### `/evaluate_extended.py` - Extended Test Suite (666 lines)
**Purpose**: Additional test cases for advanced scenarios
**Focus Areas**:
- International names with accents
- Complex cultural patterns
- Multi-word entity handling
- False positive prevention

**Why separate test files:**
- Core tests run quickly for development
- Extended tests catch edge cases
- Different execution environments
- Modular test organization

### Configuration and Data

#### `/data/nicknames.csv` - Nickname Database
**Purpose**: Cultural and linguistic nickname mappings
**Structure**: `canonical,nickname,language,source`
**Coverage**:
- Common English nicknames (William → Bill)
- Cultural variants (José → Pepe)
- Business aliases (Ma Yun → Jack Ma)
- Professional abbreviations

**Maintenance Strategy**:
- Manual curation for accuracy
- Source attribution for verification
- Language tagging for context
- Expandable via LLM analysis

#### `/pyproject.toml` - Project Configuration
**Purpose**: Poetry-based dependency management
**Key Dependencies**:
- **spaCy**: NLP and entity recognition
- **Typer**: Modern CLI framework
- **Pydantic**: Data validation and serialization
- **Flask**: Web interface
- **Google Generative AI**: LLM integration

#### `/setup.py` - Model Installation (76 lines)
**Purpose**: Automated spaCy model downloading
**Function**: Downloads required NLP models for various languages

---

## Core Functions and Classes

### NameMatcher Class - Core Matching Logic

```python
class NameMatcher:
    """Cascading name matcher with multiple strategies."""
    
    MATCH_THRESHOLD = 0.20  # Ensures 100% recall
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    JARO_WINKLER_THRESHOLD = 0.75
    LEVENSHTEIN_THRESHOLD = 0.7
```

**Critical Methods**:

#### `match()` - Primary Matching Orchestrator
**Purpose**: Implements the confidence cascade algorithm
**Input**: Input name, extracted entity, configuration flags
**Output**: Complete MatchResult with explanations
**Logic Flow**:
1. Normalize both input and entity names
2. Initialize warning tracking
3. Cascade through matching layers in confidence order
4. Aggregate results and generate final decision
5. Apply security warnings based on match patterns

**Why this design:**
- Single entry point simplifies testing
- Clear separation of concerns
- Consistent warning application
- Comprehensive result generation

#### `_exact_match()` - Perfect Matching
**Purpose**: Handles cases where names match after normalization
**Algorithm**: String comparison post-normalization
**Confidence**: 100% when successful
**Edge Cases**: Unicode variants, title differences, whitespace

#### `_nickname_match()` - Cultural Nickname Resolution
**Purpose**: Resolves known nickname relationships
**Data Source**: CSV database with bidirectional mapping
**Algorithm**:
1. Check if any input token is nickname of entity token
2. Check reverse relationship
3. Handle partial nickname matches
**Confidence**: 95% for database matches

#### `_structural_match()` - Pattern-Based Matching
**Purpose**: Handles initials, reordering, and partial matches
**Patterns Supported**:
- Initial expansion (J. Biden → Joe Biden)
- Name reordering (Smith, John → John Smith)
- Last name + initial (Smith, J. → John Smith)
- Middle name handling

**Algorithm Complexity**: O(n²) for token comparison
**Confidence Calculation**: Based on match comprehensiveness

#### `_fuzzy_match()` - Typo and Variation Handling
**Purpose**: Catches misspellings and OCR errors
**Algorithms Used**:
- Jaro-Winkler distance (75% threshold)
- Levenshtein distance (70% threshold)
**Confidence**: Proportional to similarity score

#### `_phonetic_match()` - Sound-Alike Matching
**Purpose**: Handles phonetically similar names
**Algorithm**: Soundex-based comparison
**Use Cases**: Different transliterations, homophones
**Confidence**: 65% base score

### EntityExtractor Class - NLP Processing

#### `extract_entities()` - Main Entity Extraction
**Purpose**: Identifies person entities in article text
**Process**:
1. Load appropriate spaCy model for language
2. Process text through NLP pipeline
3. Filter for PERSON entities
4. Apply custom pattern enhancement
5. Merge adjacent proper names
6. Add context information

**Why multi-stage process:**
- spaCy catches most standard names
- Custom patterns catch abbreviations
- Merging handles split entities
- Context enables LLM analysis

#### `_extract_additional_patterns()` - Pattern Enhancement
**Purpose**: Catches entities spaCy misses
**Patterns**:
- All-caps abbreviations (AOC, JFK)
- Context-based proper names
- Non-standard formatting

#### `_merge_adjacent_proper_names()` - Entity Consolidation
**Purpose**: Combines split proper name entities
**Algorithm**: Analyzes POS tags and capitalization patterns
**Benefit**: Handles "John Smith" when spaCy extracts separately

### LLMMatcher Class - AI Enhancement

#### `enhance_match()` - Contextual Analysis
**Purpose**: Provides cultural and linguistic context
**Input**: Name pair, context, existing confidence
**Output**: Enhanced explanation with cultural insights

**Prompt Design Philosophy**:
- Structured output for parsing
- Cultural context emphasis
- Confidence calibration
- Conservative guidance

#### `_build_prompt()` - Prompt Construction
**Purpose**: Creates consistent, effective prompts
**Components**:
- System role definition
- Cultural context examples
- Output format specification
- Confidence guidelines

**Why structured prompts:**
- Ensures consistent responses
- Enables reliable parsing
- Improves accuracy
- Reduces hallucination

---

## Data Models and Types

### Core Data Structures

#### `MatchResult` - Complete Matching Outcome
```python
class MatchResult(BaseModel):
    input_name: str
    article_entity: str
    match_confidence: float
    confidence_level: MatchConfidenceLevel
    is_match_recommendation: bool
    explanation: List[MatchExplanation]
    final_decision_logic: str
```

**Design Rationale**:
- Self-contained for API responses
- Includes all information for audit
- Structured for downstream processing
- Human-readable decision logic

#### `ExtractedEntity` - NLP Entity Representation
```python
class ExtractedEntity(BaseModel):
    text: str
    start_char: int
    end_char: int
    label: str = "PERSON"
    context: Optional[str] = None
```

**Purpose**: Bridges NLP output to matching input
**Context Inclusion**: Enables LLM analysis
**Position Tracking**: Allows verification in original text

#### `MatchExplanation` - Decision Documentation
```python
class MatchExplanation(BaseModel):
    layer: MatchingLayer
    confidence_score: float
    reasoning: str
    transformations: List[str]
    evidence: List[str]
```

**Compliance Focus**: Meets audit trail requirements
**Layered Approach**: Shows cascade decision process
**Evidence Preservation**: Maintains supporting data

### Enumeration Types

#### `MatchConfidenceLevel` - Categorical Confidence
```python
class MatchConfidenceLevel(str, Enum):
    EXACT = "exact"
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NO_MATCH = "no_match"
```

**Why categorical over numeric:**
- Easier business interpretation
- Consistent thresholds
- Regulatory language alignment
- Clear escalation paths

#### `MatchingLayer` - Algorithm Classification
```python
class MatchingLayer(str, Enum):
    EXACT = "exact_match"
    NICKNAME = "nickname_match"
    STRUCTURAL = "structural_match"
    FUZZY = "fuzzy_match"
    PHONETIC = "phonetic_match"
    LLM_ENHANCED = "llm_enhanced"
    NO_MATCH = "no_match"
```

**Purpose**: Tracks which algorithm produced result
**Benefit**: Enables algorithm-specific confidence tuning

---

## Matching Algorithms and Logic

### Algorithm Selection Rationale

#### Exact Matching - Foundation Layer
**When Used**: After normalization, names are identical
**Strengths**: 100% precision, fast execution
**Limitations**: Misses legitimate variations
**Implementation**: String comparison with comprehensive normalization

#### Nickname Resolution - Cultural Awareness
**When Used**: Database contains known relationship
**Data Source**: Curated CSV with cultural variants
**Algorithm**: Bidirectional hash lookup
**Performance**: O(1) lookup after preprocessing

**Why database over ML:**
- Guarantees accuracy for known cases
- Culturally sensitive curation
- Explainable decisions
- Easy maintenance and updates

#### Structural Matching - Pattern Recognition
**When Used**: Names follow recognizable patterns
**Patterns**:
- Initial expansion/contraction
- Name component reordering
- Partial name matching

**Algorithm Details**:
```python
def _check_initial_pattern(self, tokens1, tokens2):
    """Check if one name is initials of another."""
    for i, token1 in enumerate(tokens1):
        if len(token1) == 1:  # Potential initial
            # Find corresponding position in other name
            if i < len(tokens2) and tokens2[i].startswith(token1):
                return True
    return False
```

#### Fuzzy Matching - Error Tolerance
**Algorithms Used**:
- **Jaro-Winkler**: Emphasizes prefix similarity
- **Levenshtein**: Character-level edit distance

**Threshold Selection**:
- 75% for Jaro-Winkler (optimized for names)
- 70% for Levenshtein (allows more variation)

**Why multiple algorithms:**
- Different error types require different approaches
- Consensus scoring improves accuracy
- Fallback options for edge cases

#### Phonetic Matching - Sound Similarity
**Algorithm**: Soundex with enhancements
**Use Cases**: Transliteration variants, homophones
**Limitations**: English-centric algorithm
**Future Enhancement**: Language-specific phonetic algorithms

### Confidence Calculation Methodology

#### Base Confidence Assignment
Each algorithm has a maximum confidence:
- Exact: 100%
- Nickname: 95%
- Structural: 85%
- Fuzzy: 70%
- Phonetic: 65%

#### Confidence Adjustment Factors
1. **Match Completeness**: Partial matches reduce confidence
2. **Transformation Count**: More changes reduce confidence
3. **Warning Conditions**: Trigger mandatory review
4. **Supporting Evidence**: Multiple algorithms increase confidence

#### Final Decision Logic
```python
if max_confidence >= 0.8:
    return "High confidence match"
elif max_confidence >= 0.5:
    return "Medium confidence match"
elif max_confidence >= 0.2:
    return "Low confidence match - review required"
else:
    return "No match"
```

**20% Threshold Rationale**:
- Ensures 100% recall requirement
- Allows for unusual but valid variations
- Triggers manual review for edge cases
- Balances automation with human oversight

---

## LLM Integration

### Integration Philosophy

#### Optional Enhancement Strategy
**Core Principle**: System must function without LLM
**Benefits**:
- Operational reliability
- Cost control
- Speed for bulk processing
- Offline capability

#### When LLM Adds Value
1. **Cultural Context**: Non-Western naming patterns
2. **Abbreviations**: Professional/political nicknames
3. **Transliteration**: Multiple valid romanizations
4. **Ambiguous Cases**: Require cultural knowledge

### Technical Implementation

#### Google Gemini Selection
**Model**: Gemini 2.0 Flash
**Rationale**:
- Speed: ~2-3 seconds vs GPT-4's 10+ seconds
- Cost: Competitive pricing for bulk processing
- Accuracy: Strong performance on name tasks
- Integration: Simple API with good reliability

#### Prompt Engineering Strategy

**System Role Definition**:
```
You are a cultural naming expert for financial compliance. 
Analyze name pairs considering global naming conventions, 
cultural context, and professional abbreviations.
```

**Output Format Specification**:
```json
{
  "is_likely_match": boolean,
  "confidence": 0.0-1.0,
  "reasoning": "detailed explanation",
  "cultural_context": "relevant background"
}
```

**Why Structured Output**:
- Reliable parsing
- Consistent format
- Error detection
- Integration simplicity

#### Error Handling and Fallback

**Network Issues**: Graceful degradation to traditional matching
**API Limits**: Automatic throttling with queuing
**Invalid Responses**: Fallback to rule-based confidence
**Cost Controls**: Optional per-request budget limits

### LLM Enhancement Examples

#### Cultural Context Understanding
**Input**: "Xi" vs "Xi Jinping"
**Traditional Matching**: Low confidence (partial match)
**LLM Enhancement**: High confidence with cultural context
**Reasoning**: "In Chinese political context, 'Xi' commonly refers to Xi Jinping when discussing leadership"

#### Professional Abbreviation Resolution
**Input**: "MBS" vs "Mohammed bin Salman"
**Traditional Matching**: No match (no database entry)
**LLM Enhancement**: High confidence match
**Reasoning**: "MBS is widely recognized abbreviation for Crown Prince Mohammed bin Salman in international media"

#### Transliteration Variants
**Input**: "Zelensky" vs "Zelenskyy"
**Traditional Matching**: Medium confidence (fuzzy match)
**LLM Enhancement**: High confidence with explanation
**Reasoning**: "Both are valid transliterations of Ukrainian surname Зеленський"

---

## Configuration and Data Files

### Nickname Database Design

#### File Structure: `/data/nicknames.csv`
```csv
canonical,nickname,language,source
William,Bill,en,common
José,Pepe,es,cultural
Ma Yun,Jack Ma,zh,business
```

#### Field Definitions
- **canonical**: Official/formal name variant
- **nickname**: Informal/alternative variant
- **language**: Primary language context
- **source**: Origin category (common/cultural/business)

#### Maintenance Strategy
1. **Manual Curation**: Ensures accuracy for critical cases
2. **Source Attribution**: Enables verification and updates
3. **Language Tagging**: Supports multilingual awareness
4. **Category Classification**: Helps confidence calibration

#### Expansion Possibilities
- **LLM-Generated Candidates**: Automated discovery of new patterns
- **Regional Variants**: Country-specific nickname patterns
- **Historical Names**: Evolution of naming conventions
- **Professional Aliases**: Industry-specific abbreviations

### Environment Configuration

#### API Key Management
```bash
# .env file
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_API_KEY=alternative_key_name  # Fallback
```

#### Security Considerations
- Environment variable isolation
- No hardcoded credentials
- Optional key configuration
- Graceful degradation without keys

#### Model Configuration
```python
LANGUAGE_MODELS = {
    'en': ['en_core_web_lg', 'en_core_web_md', 'en_core_web_sm'],
    'es': ['es_core_news_lg', 'es_core_news_md', 'es_core_news_sm'],
    # ... additional languages
}
```

**Model Selection Logic**:
1. Prefer larger models for accuracy
2. Fall back to smaller models for availability
3. Use multilingual model as last resort
4. Fail gracefully if none available

---

## User Interfaces

### Command Line Interface (CLI)

#### Design Philosophy
**Target Users**: Technical analysts, automation systems
**Priorities**: Scriptability, integration, detailed output

#### Key Features
- **Type-safe arguments**: Typer framework with validation
- **Multiple output formats**: Pretty display and JSON
- **Rich formatting**: Colors, tables, and progress indicators
- **Error handling**: Clear messages with actionable guidance

#### Usage Patterns
```bash
# Basic matching
name_screening match --name "John Smith" --article-file article.txt

# JSON output for automation
name_screening match --name "John Smith" --article-text "..." --output json

# High-security mode
name_screening match --name "John Smith" --article-file article.txt --no-first-name-only

# LLM enhancement
name_screening match --name "John Smith" --article-file article.txt --use-llm
```

#### Output Design

**Pretty Mode Features**:
- Overall status indicators (✓/✗)
- Color-coded confidence levels
- Warning highlights for manual review
- Detailed explanation panels
- Summary table for multiple entities

**JSON Mode Features**:
- Machine-readable structure
- Complete data preservation
- Integration-friendly format
- Consistent schema

### Web Interface

#### Design Philosophy
**Target Users**: Business analysts, occasional users
**Priorities**: Usability, visual feedback, accessibility

#### User Experience Flow
1. **Input Form**: Name and article text with validation
2. **Processing Feedback**: Real-time status updates
3. **Results Display**: Visual confidence indicators
4. **Detailed View**: Expandable explanations
5. **Export Options**: Copy/download results

#### Technical Implementation
- **Flask Backend**: Simple, reliable web framework
- **Server-Side Rendering**: Reduces client complexity
- **Progressive Enhancement**: Works without JavaScript
- **Responsive Design**: Mobile and desktop compatibility

#### Security Features
- **Input Validation**: Prevents injection attacks
- **Content Sanitization**: Safe handling of user input
- **Rate Limiting**: Prevents abuse
- **Error Boundaries**: Graceful failure handling

---

## Testing and Evaluation

### Test Suite Architecture

#### Core Test Categories

**Exact Matches**: Baseline verification
- Perfect name matches after normalization
- Unicode variant handling
- Title and punctuation removal

**Nickname Variants**: Cultural awareness
- Common English nicknames (William → Bill)
- Cultural variants (José → Pepe)
- Business aliases (Ma Yun → Jack Ma)

**Structural Patterns**: Initial and reordering handling
- Initial expansion (J. Biden → Joe Biden)
- Name reordering (Smith, John → John Smith)
- Partial name matching

**Fuzzy Matching**: Typo tolerance
- Common misspellings
- OCR errors
- Transliteration variants

**Edge Cases**: Boundary conditions
- Very short names
- Hyphenated names
- Multiple middle names
- Name particles (von, de, al)

#### Test Execution Strategy

**Development Testing**: Fast core test suite
```bash
poetry run python evaluate.py
```

**Comprehensive Testing**: Extended test suite
```bash
poetry run python evaluate.py --extended
```

**LLM Testing**: Enhanced matching validation
```bash
poetry run python evaluate.py --use-llm
```

#### Success Metrics

**Primary KPI**: 100% Recall (no false negatives)
**Secondary KPIs**:
- Precision optimization within recall constraint
- Warning system accuracy
- Processing speed benchmarks
- LLM enhancement value

### Quality Assurance Process

#### Test Case Development
1. **Real-World Scenarios**: Based on actual compliance cases
2. **Edge Case Discovery**: Systematic boundary testing
3. **Cultural Validation**: Input from regional experts
4. **Regulatory Alignment**: Compliance requirement verification

#### Regression Testing
- **Automated Execution**: CI/CD integration
- **Performance Monitoring**: Speed and accuracy tracking
- **Model Update Validation**: spaCy model compatibility
- **API Integration Testing**: LLM service reliability

---

## Security Features

### High-Security Mode Configuration

#### First-Name-Only Match Control
**Problem**: "John Smith" matching "John Doe" may be too permissive
**Solution**: Configurable restriction of first-name-only matches

**Implementation**:
```python
def _check_first_name_only_restriction(self, input_tokens, entity_tokens):
    """Check if match violates first-name-only restrictions."""
    if not self.allow_first_name_only_match:
        # Only first names match, different or missing surnames
        return self._is_first_name_only_match(input_tokens, entity_tokens)
    return False
```

**Use Cases**:
- Financial crime investigations
- High-security environments
- Regions with common first names
- Regulatory requirements for strict matching

#### Warning System Integration

**Warning Categories**:
1. **Surname-Only Matches**: May indicate maiden name changes
2. **Single Token Matches**: Insufficient information for confidence
3. **Middle Name Mismatches**: Potential false positive
4. **First-Name-Only Matches**: When surnames differ

**Warning Triggers**:
```python
critical_warnings = {
    'surname_only_match': False,
    'single_token_match': False,
    'middle_name_mismatch': False,
    'surname_mismatch': False
}
```

**Analyst Workflow Integration**:
- Visual indicators in web interface
- CLI warning symbols (⚠️)
- Automatic escalation flags
- Audit trail preservation

### Data Protection and Privacy

#### Input Data Handling
- **No Persistent Storage**: Article text not retained
- **Memory Cleanup**: Explicit cleanup of sensitive data
- **API Key Protection**: Environment variable isolation
- **Logging Controls**: Configurable detail levels

#### Audit Trail Security
- **Immutable Records**: MatchResult objects are read-only
- **Complete Documentation**: Every decision fully explained
- **Transformation History**: All normalization steps recorded
- **Compliance Formatting**: Regulatory-appropriate language

---

## Performance Considerations

### Processing Speed Optimization

#### spaCy Model Selection Strategy
**Large Models**: Best accuracy, slower processing
**Medium Models**: Balanced performance
**Small Models**: Fastest, acceptable accuracy

**Automatic Fallback Chain**:
1. Try large model (if available)
2. Fall back to medium model
3. Use small model as minimum
4. Multilingual model as last resort

#### Caching and Optimization

**Model Caching**: Loaded models retained in memory
**Entity Caching**: Avoid re-processing identical text
**Nickname Lookup**: Pre-built hash maps for O(1) access
**Regex Compilation**: Compiled patterns reused

#### Memory Management
- **Model Sharing**: Single instance per language
- **Garbage Collection**: Explicit cleanup of large objects
- **Streaming Processing**: For bulk analysis scenarios
- **Resource Monitoring**: Memory usage tracking

### Scalability Architecture

#### Horizontal Scaling Options
1. **Load Balancer**: Distribute requests across instances
2. **Queue Processing**: Async batch handling
3. **Microservice Split**: Separate entity extraction and matching
4. **Database Integration**: Persistent result storage

#### Performance Benchmarks

**Traditional Mode**:
- ~50ms per article (spaCy processing)
- ~5ms additional per entity (matching)
- Memory: ~200MB per spaCy model
- CPU: Single-threaded per request

**LLM Enhanced Mode**:
- ~2-3 seconds per article (including API call)
- Network dependent latency
- Additional memory: ~50MB
- Cost: ~$0.001-0.005 per article

#### Optimization Strategies

**Batch Processing**: Group multiple articles
**Result Caching**: Store previous analyses
**Model Preloading**: Warm start optimization
**Connection Pooling**: Reuse LLM API connections

---

---

## Quick Reference

### File Summary
| File | Lines | Purpose |
|------|-------|---------|
| `main.py` | 285 | CLI interface and application entry point |
| `core/matcher.py` | 756 | Core matching engine with confidence cascade |
| `core/entity_extractor.py` | 414 | spaCy-based entity extraction |
| `core/llm_matcher.py` | 275 | Optional LLM enhancement via Gemini |
| `utils/name_parser.py` | 169 | Name normalization and parsing |
| `utils/explainability.py` | 232 | Decision documentation for compliance |
| `models/matching.py` | 117 | Pydantic data models and validation |
| `web_app.py` | 762 | Flask web interface |
| `evaluate.py` | 566 | Core test suite |
| `evaluate_extended.py` | 666 | Extended test scenarios |

### Key Functions by Component

#### Core Matching (`matcher.py`)
- `match()`: Main matching orchestrator
- `_exact_match()`: Perfect normalization-based matching
- `_nickname_match()`: Cultural nickname resolution
- `_structural_match()`: Pattern-based matching (initials, reordering)
- `_fuzzy_match()`: Typo and misspelling handling
- `_phonetic_match()`: Sound-alike matching

#### Entity Extraction (`entity_extractor.py`)
- `extract_entities()`: Main NLP entity extraction
- `_extract_additional_patterns()`: Custom pattern enhancement
- `_merge_adjacent_proper_names()`: Entity consolidation
- `_load_model()`: Smart spaCy model selection

#### LLM Enhancement (`llm_matcher.py`)
- `enhance_match()`: Contextual analysis via Gemini
- `_build_prompt()`: Structured prompt construction
- `_parse_llm_response()`: Response parsing and validation

#### Name Processing (`name_parser.py`)
- `parse()`: Main normalization pipeline
- `normalize_name()`: Public normalization API
- `_handle_cultural_patterns()`: Global naming conventions

#### Explainability (`explainability.py`)
- `generate_explanation()`: Audit trail creation
- `_generate_reasoning()`: Human-readable explanations
- `_determine_confidence_level()`: Categorical confidence mapping

### Configuration Files
- `pyproject.toml`: Poetry dependency management
- `data/nicknames.csv`: Cultural nickname database
- `.env`: API keys and environment variables
- `setup.py`: spaCy model installation

### Command Line Usage
```bash
# Basic matching
poetry run name_screening match --name "John Smith" --article-file article.txt

# JSON output for automation
poetry run name_screening match --name "John Smith" --article-text "..." --output json

# High-security mode (disable first-name-only matches)
poetry run name_screening match --name "John Smith" --article-file article.txt --no-first-name-only

# LLM enhancement
poetry run name_screening match --name "John Smith" --article-file article.txt --use-llm

# Web interface
poetry run python web_app.py
```

### Testing Commands
```bash
# Quick test
poetry run name_screening test

# Core evaluation
poetry run python evaluate.py

# Extended evaluation
poetry run python evaluate_extended.py

# LLM evaluation
poetry run python evaluate.py --use-llm
```

---

## Conclusion

The Name Screening Tool represents a comprehensive solution to the adverse media screening challenge in financial compliance. Its architecture prioritizes recall over precision while providing the explainability and configurability required for regulatory environments.

### Key Architectural Strengths

1. **Confidence Cascade**: Multiple algorithms ensure comprehensive coverage
2. **Cultural Awareness**: Global naming patterns and conventions
3. **Explainable AI**: Complete audit trails for compliance
4. **Flexible Integration**: CLI, web, and API interfaces
5. **Optional Enhancement**: LLM integration without dependency
6. **Security Features**: Configurable restrictions and warning systems

### Design Philosophy Success

The system successfully addresses the core problem:
- **100% Recall**: No false negatives through conservative thresholds
- **Precision Optimization**: Layered matching strategies minimize false positives
- **Regulatory Compliance**: Comprehensive documentation and audit trails
- **Operational Flexibility**: Multiple deployment and integration options

This technical documentation provides the foundation for understanding, maintaining, and extending the Name Screening Tool to meet evolving compliance and business requirements.