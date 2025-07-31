# Name Screening Tool

A command-line tool for adverse media screening that determines if a name in a news article matches an individual of interest. Achieves 100% recall (no false negatives) while maximizing precision through intelligent pattern recognition and cultural awareness.

Combines traditional rule-based matching with optional LLM enhancement for contextual understanding.

## Key Features

- **Visual Status Indicators**: Clear ✓/✗ symbols show if matches are clean or require review
- **100% Recall**: Never misses a true match (no false negatives)
- **Explainable Decisions**: Every match includes detailed reasoning
- **Global Name Support**: Handles diverse naming patterns and conventions worldwide
- **LLM Enhancement**: Optional contextual analysis using Google Gemini (tested on 2.0 Flash)
- **Warning System**: Flags surname-only matches and other ambiguous cases

## Quick Start

```bash
# Install dependencies
poetry install

# Download required spaCy models
poetry run python setup.py

# Run the web UI
poetry run python web_app.py
# Then open http://localhost:port

# Or use the CLI
poetry run name_screening match --name "John Smith" --article-file article.txt
```

## Installation

### Prerequisites

- Python 3.9 or higher
- Poetry (for dependency management)

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd name_screening
```

2. Install dependencies:

```bash
poetry install
```

3. Download spaCy models:

```bash
poetry run python setup.py
```

This downloads English and multilingual models. For additional languages:

```bash
poetry run python -m spacy download es_core_news_sm  # Spanish
poetry run python -m spacy download de_core_news_sm  # German
poetry run python -m spacy download fr_core_news_sm  # French
```

## Usage

### Web Interface

```bash
poetry run python web_app.py
```

Open <http://localhost:5000> (or check the terminal output for another port, if 5000 is busy) for a user-friendly interface with:

- Form-based input
- Visual confidence indicators
- Color-coded results
- Explanations
- LLM enhancement toggle

### Command Line Interface

```bash
# Basic usage
poetry run name_screening match --name "Jane Doe" --article-file article.txt

# With direct text input
poetry run name_screening match --name "Jane Doe" --article-text "Jane Doe was arrested..."

# Spanish article
poetry run name_screening match --name "José García" --article-file spanish.txt --lang es

# JSON output for integration
poetry run name_screening match --name "John Smith" --article-file article.txt --output json

# Strict mode (higher confidence required)
poetry run name_screening match --name "John Smith" --article-file article.txt --strict

# LLM-enhanced matching (requires API in .env)
poetry run name_screening match --name "John Smith" --article-file article.txt --use-llm

# High-security mode (disallow first-name-only matches)
poetry run name_screening match --name "John Smith" --article-file article.txt --no-first-name-only
```

### CLI Options

```
Options:
  -n, --name TEXT              Name to search for [required]
  -f, --article-file PATH      Path to article text file
  -t, --article-text TEXT      Article text directly
  -l, --lang TEXT              Two-letter language code (default: en)
  -s, --strict                 Use strict matching (higher confidence)
      --use-llm                Enhance matching with LLM contextual analysis
      --no-first-name-only     Disable first-name-only matches (high-security mode)
  -o, --output TEXT            Output format: pretty or json (default: pretty)
```

## LLM vs spaCy-Only Mode

### spaCy-Only Mode (Default)

- **Speed**: ~50ms per article
- **Cost**: Free (no API calls)
- **Capabilities**: Rule-based matching (exact, nickname, fuzzy, phonetic)
- **Use case**: High-volume screening, offline environments

### LLM-Enhanced Mode

- **Speed**: ~2-3 seconds per article
- **Cost**: ~$0.001-0.005 per article
- **Capabilities**: Contextual understanding, cultural patterns, abbreviations
- **Use case**: Ambiguous cases, cultural names, professional abbreviations

**Setup for LLM mode**:

1. Get Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Copy `.env.example` to `.env` and add your API key:
3.Edit .env and add:

   ```bash
   cp .env.example .env
   ```

 ```
 GOOGLE_API_KEY=your_actual_api_key_here
 ```

## Security Features

### First-Name-Only Match Control

By default, the system allows matches where only first names match (e.g., "John Smith" matching "John Doe"). This can be disabled for high-security environments:

**CLI**: Use `--no-first-name-only` flag

```bash
poetry run name_screening match --name "John Smith" --article-file article.txt --no-first-name-only
```

**Web UI**: Uncheck "Allow first-name-only matches" checkbox

**When to disable**:

- Financial crime investigations where surname changes are tracked separately
- High-security environments requiring stricter identity verification
- Reducing false positives in regions with common first names

## Testing

```bash
# Run sample test cases (remove the flag --use-llm if you want to test a spacy-only implementation)
poetry run python evaluate.py --use-llm

# Run sample test cases without accented names (remove the flag --use-llm if you want to test a spacy-only implementation)
poetry run python evaluate.py --use-llm
```

## Output Examples

**JSON Output**

```json
{
  "input_name": "William Clinton",
  "article_entity": "Bill Clinton",
  "match_confidence": 0.95,
  "is_match_recommendation": true,
  "explanation": ["Nickname match: 'William' vs. 'Bill'"],
  "final_decision_logic": "Very high confidence match"
}
```

## Configuration

### LLM Enhancement Setup

Google Gemini 2.0 Flash provides contextual understanding beyond rule-based patterns:

- Cultural naming conventions
- Context-dependent abbreviations  
- Transliteration variants
- Professional title recognition

### Nicknames Database

Add custom nickname mappings to `data/nicknames.csv`:

```csv
canonical,nickname,language,source
William,Bill,en,common
José,Pepe,es,common
```

## Troubleshooting

### Missing spaCy Models

```bash
poetry run python -m spacy download en_core_web_sm
```

### Memory Issues

- Use smaller spaCy models (_sm instead of_lg)
- Process articles in batches
- Increase Python memory limit if needed

## Documentation

- **Comprehensive Technical Documentation**: See [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) - Complete guide covering all functions, files, and design decisions
- **Technical Implementation**: See [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)
- **Evaluation Results**: See the second half of the Implementation Notes
