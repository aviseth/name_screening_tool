# Name Screening Tool - Technical Report

I developed this tool to address a critical pain point in financial compliance: adverse media screening produces excessive false positives, wasting analyst time. My solution achieves 100% recall (zero false negatives) while maximizing precision through intelligent pattern recognition and cultural awareness.

## Problem Analysis

Financial institutions face a challenging trade-off in adverse media screening:

- **Too strict**: Risk regulatory fines for missing sanctioned individuals
- **Too loose**: Analysts waste hours reviewing false positives

I do understand that in this case, you would much rather have Too loose over Too strict, and thus the need to optimize for recall.

Current solutions fail because they rely on simplistic fuzzy matching without understanding:

- Different global naming conventions and patterns
- Common abbreviations and nicknames
- Contextual entity extraction
- Name particles and their significance

## Technical Approach

I implemented a confidence cascade with increasingly fuzzy matches:
1. **Exact Match** (100% confidence)
2. **Nickname Match** (95% confidence)  
3. **Structural Match** (85% confidence)
4. **Fuzzy Match** (70% confidence)
5. **Phonetic Match** (65% confidence)
6. **LLM Enhancement** (Optional) - Contextual analysis using Google Gemini

The system returns positive matches at 30% confidence to ensure 100% recall. This can of course be changed according to the need. 

### Matching Examples

#### Exact Match (100% confidence)

After normalization, names match exactly:
- Input: "John Smith" → Article: "john smith" ✓
- Input: "José García" → Article: "Jose Garcia" ✓ (unicode normalized)
- Input: "Dr. John Smith Jr." → Article: "John Smith" ✓ (titles/suffixes removed)

#### Nickname Match (95% confidence)

Database-driven nickname resolution:

- Input: "William Clinton" → Article: "Bill Clinton" ✓
- Input: "Robert Smith" → Article: "Bob Smith" ✓
- Input: "Elizabeth Warren" → Article: "Liz Warren" ✓
- Input: "MBS" → Article: "Mohammed bin Salman" ✓ (abbreviation mapping)

This nickname repository can be added to manually, or an LLM can be used to increase its knowledge. 

#### Structural Match (85% confidence)

Pattern-based matching for initials, reordering, partial names:

- Input: “J.E. Root → Article: “Joseph Edwards Root” ✓ (initials expansion)
- Input: "J. Biden" → Article: "Joe Biden" ✓ (initial match)
- Input: "Ng Ka Ming" → Article: "K.M. Ng" ✓ (name to initials pattern)
- Input: "Kim Jong-un" → Article: "Jong-un Kim" ✓ (name reordering)

#### Fuzzy Match (70% confidence)

Handles typos and misspellings:

- Input: "Zelenskyy" → Article: "Zelensky" ✓ (spelling variation)
- Input: "Bolsonaro" → Article: "Bolsanaro" ✓ (typo)
- Input: "Macron" → Article: "Macrone" ✓ (OCR error)

#### Phonetic Match (65% confidence)

Catches names that sound similar:

- Input: "Sean" → Article: "Shawn" ✓
- Input: "Catherine" → Article: "Kathryn" ✓
- Input: "Mohammed" → Article: "Muhammad" ✓

This is the least important layer. I doubt a use case exists for this because this matching won’t be done based on names “heard” but names on record.

#### LLM Enhancement (Variable confidence)

Uses Google Gemini for contextual understanding:

- Input: "Xi" → Article: "Chinese President Xi Jinping announced..." ✓ (understands cultural context)
- Input: "MBS" → Article: "Crown Prince Mohammed bin Salman's reforms..." ✓ (recognizes common abbreviations)
- Input: "Zelensky" → Article: "Ukrainian leader Volodymyr Zelenskyy..." ✓ (handles transliteration variants)

### Key Innovations

#### 1. Pattern-Based Entity Extraction

Instead of hardcoding celebrity names, I use contextual pattern recognition:

```python
# Detect abbreviations from context
"CEO MBS announced..." → Extracts "MBS" as person
"Company MBS filed..." → Ignores "MBS"
```

#### 2. Surname-Only Match Detection

One of the biggest challenges in name matching is avoiding false positives when only surnames match. I implemented a sophisticated warning system:

**The Problem**: "James Chen" matching "Sarah Chen" - clearly different people despite the surname match.

**My Solution**:

- Detect when first names are completely different but last names match
- Drastically reduce confidence (cap at 20%)
- Add prominent warning: "⚠️ ANALYST REVIEW REQUIRED"
- Still return as a potential match (maintaining 100% recall) but flag for human review

This approach maintains our regulatory requirement of zero false negatives while dramatically reducing analyst workload by clearly marking questionable matches.

**Enhanced Fuzzy Detection**: The system uses intelligent fuzzy matching to distinguish between:

- True mismatches: "James" vs "Sarah" (0% similarity) → triggers warning
- Typos/variations: "Susan" vs "Suzen" (80%+ similarity) → no warning
- Nicknames: "Michael" vs "Mike" → handled by nickname layer, no warning

**Security Rationale**: This decision was made with financial crime scenarios in mind. Criminals might:

- Use spouse's accounts with different first names but same surname
- Have legally changed their first name while keeping family name
- Operate under family member identities

Therefore, surname-only matches are flagged but not discarded, ensuring investigators can catch sophisticated evasion attempts.

**Configurable Behavior**: Organizations can disable first-name-only matches entirely by setting `allow_first_name_only_match=False` in the matcher configuration. This is useful for:

- High-security environments where surname changes are tracked separately
- Jurisdictions where first names are more culturally significant
- Reducing false positives in ethnically homogeneous populations

**Middle Name Mismatch Detection**: Added sophisticated detection for cases where first and last names match but middle names differ completely:

- Example: "Daniel Ramesh Brotman" vs "Daniel Holderg Brotman" → triggers warning
- Checks if middle names are < 60% similar (not just typos)
- Reduces confidence to 30% and adds "⚠️ WARNING: Middle names differ" message
- Ensures analysts review potential identity confusion cases

This prevents false matches when different people share first and last names but have distinct middle names - a common scenario in many cultures (including where last names or first names tend to be cities or towns of family origin).

#### 3. Global Name Pattern Support

I handle diverse naming patterns from around the world:

**Family Name Variations**:

- Preserve short family names that might be mistaken for particles
- Handle hyphenated given names
- Support different name ordering conventions (family-first vs given-first)

**Compound Name Elements**:

- Validate compound particles that require surnames
- Handle nobility prefixes and titles correctly

**Patronymic Patterns**:

- Preserve patronymic indicators (bin, ibn, etc.)
- Handle multi-generational name chains

#### 4. Smart Matching Strategies

**Initials Matching**:

- "J.K. Smith" matches "John Kevin Smith"
- "K.M. Ng" matches "Ng Ka Ming" (family-name-first order)

**Dynamic Name Shortenings**:

- Automatically detects common truncations (Sam→Samuel, Chris→Christopher)
- No hardcoding needed for obvious shortenings
- Requires at least 40% character overlap

**True Nickname Resolution**:

- Minimal database with only non-derivable nicknames (Bill↔William, Bob↔Robert)
- Cultural nicknames preserved (Pepe↔José, Paco↔Francisco)
- No redundant entries for programmatically detectable patterns

### Technology Stack Decisions

#### Poetry vs pip

I chose **Poetry** over pip because:

- **Lock files**: Guarantees reproducible builds across environments
- **Dependency resolution**: Handles complex dependency conflicts automatically
- **Virtual environment**: Built-in venv management
- **Why not pip**: pip-tools exists but Poetry provides an integrated solution

#### spaCy vs NLTK/Transformers

I chose **spaCy** for entity extraction because:

- **Speed**: 10x faster than BERT-based models
- **Multi-language**: Pre-trained models for 20+ languages
- **Why not NLTK**: Academic focus, slower, less accurate NER
- **Why not Transformers**: Overkill for this task, 100x slower

#### Pydantic vs dataclasses/attrs

I chose **Pydantic** for data validation because:

- **Runtime validation**: Catches bad data before it breaks things
- **JSON serialization**: Built-in, fast, and correct
- **Type coercion**: Intelligently handles type conversion
- **Why not dataclasses**: No runtime validation
- **Why not attrs**: Less ecosystem support, more verbose

#### Typer vs argparse/click

I chose **Typer** for CLI because:

- **Type hints**: Generates CLI from function signatures
- **Modern**: Built on Click but more Pythonic
- **Documentation**: Auto-generates help text
- **Why not argparse**: Verbose, repetitive boilerplate
- **Why not Click**: Typer is Click with better ergonomics

#### textdistance vs fuzzywuzzy/jellyfish

I chose **textdistance** for fuzzy matching because:

- **Algorithm variety**: 30+ algorithms in one library
- **Pure Python**: No C dependencies, easier deployment
- **Maintained**: Active development
- **Why not fuzzywuzzy**: GPL license issues, fewer algorithms
- **Why not jellyfish**: Limited to phonetic algorithms only

#### Google Gemini vs OpenAI

I chose **Google Gemini 2.0 Flash** for LLM enhancement because:

- **Cost**: More affordable than GPT-4 for high-volume use
- **Context window**: Large enough for article analysis
- **Multimodal**: Future-proof for image/document analysis
- **Why not OpenAI**: Higher cost, rate limits
- **Most importantly**, it is the only one that gives limited usage for free.

#### Hybrid Architecture: spaCy + LLM vs LLM-Only

**Decision**: I chose the hybrid **spaCy + LLM** approach over a pure LLM solution for both cost efficiency and reliability.

**Cost Analysis**:

- **Hybrid approach**: ~$0.001-0.005 per article (LLM only processes 2-3 extracted entities)
- **LLM-only approach**: ~$0.01-0.05 per article (10x higher due to processing entire article text)
- **Token efficiency**: 150 tokens vs 1000+ tokens per article

**Technical Benefits**:

- **Reliability**: spaCy provides deterministic entity extraction, LLM adds contextual nuance. Easy to develop solutions when you know how your code is going to perform.
- **Performance**: Sub-second spaCy processing + targeted LLM calls vs. 10-15 second full-text LLM analysis
- **Explainability**: Clear separation between rule-based and AI-based decisions for regulatory compliance
- **Fallback**: System works without LLM if API is unavailable


**Visual Warning System**:

- Bright orange banner for surname-only matches
- Warning icon (⚠️) in result headers
- Red highlighting for items requiring review

These visual cues help analysts prioritize their review efforts on the most questionable matches.

### Module Structure

```text
src/
├── core/
│   ├── entity_extractor.py  # Enhanced spaCy extraction
│   └── matcher.py           # Cascading match logic
├── utils/
│   ├── name_parser.py       # Cultural-aware normalization
│   └── explainability.py    # Audit trail generation
└── models/
    └── matching.py          # Type-safe data models
```

## Explainability

The system provides detailed explanations for every matching decision, crucial for regulatory compliance and user trust.

### How It Works

The system processes names through multiple layers, each documenting its decision process:

1. **Exact Match** (100% confidence) - Direct string comparison
2. **Nickname Match** (95% confidence) - Known nickname patterns  
3. **Structural Match** (85% confidence) - Name reordering and variations
4. **Fuzzy Match** (variable confidence) - Similarity algorithms
5. **LLM Enhancement** (contextual confidence) - AI-powered analysis

### Example: "William Johnson" vs "Bill Johnson"

**Input:** "William Johnson"  
**Article:** "Local businessman Bill Johnson announced a new venture today."  
**Result:** MATCH (95% confidence)

The system first attempts an exact match, which fails. It then detects that "Bill" is a known nickname for "William". The transformation "William → Bill" is applied, and since "Johnson" matches exactly, the system concludes with 95% confidence.

### LLM Enhancement Example

**Input:** "AOC"  
**Article:** "Rep. Alexandria Ocasio-Cortez announced new legislation today."  
**Result:** MATCH (95% confidence)

Traditional layers fail to match "AOC" with "Alexandria Ocasio-Cortez". However, the LLM layer recognizes "AOC" as a widely recognized initialism. The context clue "Rep." aligns with her role as a US Representative, providing additional confidence.

### Compliance Features

**Warning System:** Automatically flags matches needing human review. For example: "Single token match - potential false positive. Recommend: ANALYST REVIEW"

**Configurable Strictness:** In strict mode, explanations include why lower confidence matches were rejected and what additional evidence would be needed.

### Output Formats

**CLI Format:**

```
✅ Match Found - CLEAR
Input Name: William Johnson
Article Entity: Bill Johnson
Match Confidence: 95.0%
Recommendation: MATCH

Explanation:
• Nickname match: 'William' is a known variant of 'Bill' (95.0% confidence)
```

**JSON Format:**

```json
{
  "explanation": [
    {
      "layer": "nickname_match",
      "confidence_score": 0.95,
      "reasoning": "Nickname match: 'William' is a known variant of 'Bill'",
      "transformations": ["William -> Bill"],
      "evidence": ["Nickname database match", "Surname exact match"]
    }
  ],
  "final_decision_logic": "Match found with VERY_HIGH confidence"
}
```

This explainability ensures every decision can be understood, verified, and trusted by both humans and automated systems.

