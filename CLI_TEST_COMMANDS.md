# CLI Test Commands for Name Screening Tool

This document contains comprehensive CLI test commands to validate all matching logic without LLM enhancement. Copy and paste these commands to test different scenarios.

## Basic Functionality Tests

```bash
# Test 1: Exact match
poetry run python -m src.name_screening.main match --name "Sarah Chen" --article-text "Local businesswoman Sarah Chen announced the expansion of her technology startup."

# Test 2: Nickname resolution
poetry run python -m src.name_screening.main match --name "William Clinton" --article-text "Former President Bill Clinton spoke at the conference yesterday."

# Test 3: Unicode normalization
poetry run python -m src.name_screening.main match --name "José García" --article-text "Jose Garcia, the renowned chef, opened his new restaurant."
```

## Critical Warning Tests

```bash
# Test 4: Surname-only match (should warn)
poetry run python -m src.name_screening.main match --name "James Chen" --article-text "Local businesswoman Sarah Chen announced her startup expansion. Chen founded the company in 2019."

# Test 5: Surname mismatch (should warn and not match)
poetry run python -m src.name_screening.main match --name "Seraah Dean" --article-text "Local businesswoman Sarah Chen announced the expansion of her technology startup."

# Test 6: Single token fuzzy match (should warn)
poetry run python -m src.name_screening.main match --name "Sarah Cheng" --article-text "Local businesswoman Sarah Chen announced her expansion. Chen, who founded the company, plans to hire employees."

# Test 7: Middle name mismatch (should warn)
poetry run python -m src.name_screening.main match --name "Daniel Ramesh Brotman" --article-text "Daniel Holderg Brotman, also known as DH Brotman, announced his retirement from the board."
```

## Complex Cultural Names

```bash
# Test 8: Hyphenated Korean name
poetry run python -m src.name_screening.main match --name "Kim Jong-un" --article-text "North Korean leader Jong-un Kim made an announcement yesterday."

# Test 9: Arabic patronymic
poetry run python -m src.name_screening.main match --name "Mohammed bin Salman" --article-text "Crown Prince Mohammed bin Salman Al Saud attended the summit."

# Test 10: European compound name
poetry run python -m src.name_screening.main match --name "Ursula von der Leyen" --article-text "European Commission President von der Leyen spoke about climate policy."

# Test 11: Name order variation
poetry run python -m src.name_screening.main match --name "Li Wei" --article-text "Wei Li, the Chinese ambassador, delivered a speech at the UN."
```

## Initials and Abbreviations

```bash
# Test 12: Initials expansion
poetry run python -m src.name_screening.main match --name "J.K. Rowling" --article-text "Author Joanne Kathleen Rowling signed copies of her latest book."

# Test 13: Single initial
poetry run python -m src.name_screening.main match --name "J. Biden" --article-text "President Joe Biden announced new climate initiatives."

# Test 14: Name to initials pattern
poetry run python -m src.name_screening.main match --name "Ng Ka Ming" --article-text "K.M. Ng, the Hong Kong businessman, invested in the startup."
```

## Fuzzy Matching Edge Cases

```bash
# Test 15: Spelling variations
poetry run python -m src.name_screening.main match --name "Zelenskyy" --article-text "Ukrainian President Volodymyr Zelensky addressed the nation."

# Test 16: Transliteration variants
poetry run python -m src.name_screening.main match --name "Mohammed" --article-text "Muhammad Ali was considered the greatest boxer of all time."

# Test 17: OCR-style errors
poetry run python -m src.name_screening.main match --name "Macron" --article-text "French President Emmanuel Macrone visited Germany last week."
```

## Name Shortening Tests

```bash
# Test 18: Valid first name shortening
poetry run python -m src.name_screening.main match --name "Samuel Johnson" --article-text "Sam Johnson, the renowned author, published his dictionary."

# Test 19: Invalid surname shortening (should not match)
poetry run python -m src.name_screening.main match --name "Sarah Cheng" --article-text "Sarah Chen announced her company's expansion plans."

# Test 20: Identical tokens (not shortening)
poetry run python -m src.name_screening.main match --name "Chen Chen" --article-text "Chen Chen, the artist, displayed his work at the gallery."
```

## Complex Multi-Entity Tests

```bash
# Test 21: Multiple entities with warnings
poetry run python -m src.name_screening.main match --name "John Smith" --article-text "John Smith, CEO of TechCorp, met with Jonathan Smith from rival company. Smith Industries also announced new products."

# Test 22: Family members (different people, same surname)
poetry run python -m src.name_screening.main match --name "Michael Jordan" --article-text "Basketball legend Michael Jordan's son Marcus Jordan opened a new sneaker store."

# Test 23: Professional titles and particles
poetry run python -m src.name_screening.main match --name "Dr. Angela Merkel" --article-text "Former German Chancellor Angela Merkel received an honorary doctorate."
```

## True Negative Tests

```bash
# Test 24: Completely different names
poetry run python -m src.name_screening.main match --name "Albert Einstein" --article-text "Local business owner Jane Doe announced her retirement plans."

# Test 25: Similar but different
poetry run python -m src.name_screening.main match --name "Michael Jordan" --article-text "Michael Johnson won the 400-meter race at the Olympics."

# Test 26: Partial name overlap only
poetry run python -m src.name_screening.main match --name "Sarah Wilson" --article-text "Dr. Robert Wilson and nurse Sarah Thompson worked together."
```

## Advanced Configuration Tests

```bash
# Test 27: Strict mode
poetry run python -m src.name_screening.main match --name "William Clinton" --article-text "Bill Clinton spoke at the conference." --strict

# Test 28: Disable first-name-only matches
poetry run python -m src.name_screening.main match --name "John Smith" --article-text "John Wilson and Sarah Smith attended the meeting." --no-first-name-only

# Test 29: Multiple similar entities
poetry run python -m src.name_screening.main match --name "Chris Johnson" --article-text "Christopher Johnson, CEO, met with Christine Johnson, the board member, and Chris Johnston from legal."
```

## Performance and Edge Cases

```bash
# Test 30: Very long article with multiple entities
poetry run python -m src.name_screening.main match --name "Elizabeth Warren" --article-text "Senator Elizabeth Warren spoke extensively about economic policy. Warren, who previously taught at Harvard, has been a vocal advocate for financial reform. The senator, known for her detailed policy proposals, met with Treasury officials. Elizabeth's office confirmed the meeting. Warren's staff released a statement afterward."

# Test 31: Names with special characters (Unicode normalization test)
poetry run python -m src.name_screening.main match --name "Bjork Gudmundsdottir" --article-text "Icelandic singer Björk released her new album. Guðmundsdóttir has been performing for over three decades."

# Test 32: Names with numbers and mixed case
poetry run python -m src.name_screening.main match --name "Elon Musk" --article-text "ELON MUSK, CEO of Tesla and SpaceX, announced new plans. The billionaire entrepreneur's latest venture involves AI development."
```


## Testing Notes

1. **Run from project root**: Ensure you're in `/name_screening/`
2. **Check confidence levels**: Pay attention to confidence percentages and decision levels
3. **Verify warnings**: Look for ⚠️ symbols and warning messages
4. **Status indicators**: Check for ✓ (clean) vs ✗ (warnings) overall status
5. **No LLM**: All tests run without LLM enhancement to test core logic. Add the flag ``` --use-llm ``` to enable LLM-enhancement.

## Quick Test Script

```
poetry run python evaluate.py
poetry run python evaluate_extended.py
```
