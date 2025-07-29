#!/usr/bin/env python3
"""
Extended Name Screening Tool Evaluation - No Accented Names

This script runs a more comprehensive evaluation without accented names
to determine if the recall issues are specific to that case.
"""

import sys
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from name_screening.core.entity_extractor import EntityExtractor
from name_screening.core.matcher import NameMatcher


class TestCase:
    """Test case definition."""

    def __init__(
        self,
        name: str,
        input_name: str,
        article_text: str,
        expected_match: bool,
        category: str,
        notes: str = "",
    ):
        self.name = name
        self.input_name = input_name
        self.article_text = article_text
        self.expected_match = expected_match
        self.category = category
        self.notes = notes


def create_extended_test_suite() -> List[TestCase]:
    """Create comprehensive test suite without accented names."""
    return [
        # Basic exact matches
        TestCase(
            "Exact Match 1",
            "John Smith",
            "Reporter John Smith wrote the article.",
            True,
            "Basic",
            "Simple exact match",
        ),
        TestCase(
            "Exact Match 2",
            "Maria Garcia",
            "Dr. Maria Garcia presented her research.",
            True,
            "Basic",
            "Exact match with title",
        ),
        TestCase(
            "Case Insensitive",
            "ROBERT JONES",
            "robert jones won the competition.",
            True,
            "Basic",
            "Case normalization",
        ),
        # Unicode normalization (non-accented)
        TestCase(
            "French Accents",
            "François Hollande",
            "Former president Francois Hollande spoke.",
            True,
            "Unicode",
            "French accent normalization",
        ),
        TestCase(
            "Spanish Tildes",
            "José Pérez",
            "Jose Perez announced the merger.",
            True,
            "Unicode",
            "Spanish character normalization",
        ),
        TestCase(
            "German Umlauts",
            "Jürgen Müller",
            "CEO Jurgen Muller resigned today.",
            True,
            "Unicode",
            "German umlaut normalization",
        ),
        # Name variations and patterns
        TestCase(
            "First Name Shortening",
            "Christopher Williams",
            "Chris Williams scored the winning goal.",
            True,
            "Pattern",
            "Chris -> Christopher",
        ),
        TestCase(
            "Nickname Standard",
            "Robert Johnson",
            "Bob Johnson was elected mayor.",
            True,
            "Pattern",
            "Bob -> Robert",
        ),
        TestCase(
            "Multiple Nicknames",
            "Elizabeth Taylor",
            "Actress Liz Taylor won an Oscar.",
            True,
            "Pattern",
            "Liz -> Elizabeth",
        ),
        TestCase(
            "Initials Simple",
            "T. Williams",
            "Author Tom Williams signed books.",
            True,
            "Pattern",
            "Initial expansion",
        ),
        TestCase(
            "Multiple Initials",
            "R.J. Smith",
            "Director Robert James Smith premiered his film.",
            True,
            "Pattern",
            "Multiple initials",
        ),
        TestCase(
            "Initials Reversed",
            "David R. Johnson",
            "D.R. Johnson published the paper.",
            True,
            "Pattern",
            "Full name to initials",
        ),
        # Cultural name patterns (non-accented)
        TestCase(
            "Name Order Variation",
            "Zhang Wei",
            "Ambassador Wei Zhang addressed the UN.",
            True,
            "Cultural",
            "Name order reordering",
        ),
        TestCase(
            "Korean Hyphenated",
            "Park Geun-hye",
            "President Geun-hye Park announced reforms.",
            True,
            "Cultural",
            "Korean name with hyphen",
        ),
        TestCase(
            "Japanese Name Order",
            "Tanaka Ichiro",
            "Ichiro Tanaka won the Nobel Prize.",
            True,
            "Cultural",
            "Japanese name reordering",
        ),
        TestCase(
            "Patronymic Pattern",
            "Ahmed bin Hassan",
            "Prince Ahmed bin Hassan Al-Rashid attended.",
            True,
            "Cultural",
            "Patronymic naming convention",
        ),
        TestCase(
            "Dutch Particles",
            "Willem van Oranje",
            "Historical figure van Oranje was mentioned.",
            True,
            "Cultural",
            "Dutch naming particles",
        ),
        TestCase(
            "Irish O'Names",
            "Patrick O'Brien",
            "Author Pat O'Brien released his memoir.",
            True,
            "Cultural",
            "Irish names with apostrophe",
        ),
        # Complex matching scenarios
        TestCase(
            "Possessive Reference",
            "Margaret Thatcher",
            "Thatcher's policies were controversial.",
            True,
            "Complex",
            "Possessive form",
        ),
        TestCase(
            "Split Reference",
            "Nelson Mandela",
            "Nelson's legacy lives on. Mandela inspired millions.",
            True,
            "Complex",
            "Name split across sentences",
        ),
        TestCase(
            "Title Changes",
            "President Obama",
            "Barack Obama, former president, spoke.",
            True,
            "Complex",
            "Title variations",
        ),
        TestCase(
            "Compound Surnames",
            "Hillary Rodham Clinton",
            "Secretary Clinton addressed the assembly.",
            True,
            "Complex",
            "Compound surname partial",
        ),
        TestCase(
            "Hyphenated Surnames",
            "Marie-Claire Dubois",
            "Journalist Dubois broke the story.",
            True,
            "Complex",
            "Hyphenated name partial",
        ),
        # Fuzzy matching cases
        TestCase(
            "Typo Simple",
            "Stephen Hawking",
            "Physicist Steven Hawking passed away.",
            True,
            "Fuzzy",
            "Common spelling variation",
        ),
        TestCase(
            "Transliteration Var",
            "Muammar Gaddafi",
            "Leader Moammar Qaddafi was overthrown.",
            True,
            "Fuzzy",
            "Transliteration differences",
        ),
        TestCase(
            "Missing Letter",
            "Arnold Schwarzenegger",
            "Actor Arnold Schwarzeneger announced.",
            True,
            "Fuzzy",
            "Missing letter typo",
        ),
        TestCase(
            "Extra Letter",
            "Madonna",
            "Singer Maddonna performed last night.",
            True,
            "Fuzzy",
            "Extra letter typo",
        ),
        # Security/false positive tests
        TestCase(
            "Different Surname 1",
            "John Williams",
            "John Anderson won the race.",
            False,
            "Security",
            "Same first name, different person",
        ),
        TestCase(
            "Different Surname 2",
            "Sarah Johnson",
            "Sarah Mitchell was appointed CEO.",
            False,
            "Security",
            "Common first name confusion",
        ),
        TestCase(
            "Partial Company Name",
            "James Ford",
            "Ford Motor Company announced layoffs.",
            False,
            "Security",
            "Person name in company",
        ),
        TestCase(
            "Similar Sound Diff Person",
            "Claire Smith",
            "Clare Jones testified in court.",
            False,
            "Security",
            "Similar sounding, different person",
        ),
        TestCase(
            "Middle Name Different",
            "John David Smith",
            "John Michael Smith was arrested.",
            False,
            "Security",
            "Different middle names",
        ),
        TestCase(
            "Jr Sr Confusion",
            "Martin Luther King Jr",
            "Martin Luther King Sr. preached.",
            False,
            "Security",
            "Generation suffix difference",
        ),
        # Edge cases
        TestCase(
            "Single Name Entity",
            "Cher",
            "Singer Cher announced her tour.",
            True,
            "Edge",
            "Mononym",
        ),
        TestCase(
            "Numbers in Context",
            "George Bush 41",
            "President George H.W. Bush served.",
            True,
            "Edge",
            "Numerical designation",
        ),
        TestCase(
            "ALL CAPS Article",
            "Tim Cook",
            "APPLE CEO TIM COOK ANNOUNCED NEW PRODUCTS.",
            True,
            "Edge",
            "All caps article text",
        ),
        TestCase(
            "Mixed Case Weird",
            "Bill Gates",
            "bILl gAtEs donated billions to charity.",
            True,
            "Edge",
            "Weird mixed case",
        ),
        TestCase(
            "Emoji Context",
            "Elon Musk",
            "🚀 Elon Musk launched another rocket 🚀",
            True,
            "Edge",
            "Emoji in text",
        ),
        TestCase(
            "Special Chars",
            "will.i.am",
            "Musician will.i.am produced the album.",
            True,
            "Edge",
            "Dots in name",
        ),
        # More cultural variations
        TestCase(
            "Full Name Shortening",
            "Alexander Boris Johnson",
            "Prime Minister Johnson visited Europe.",
            True,
            "Cultural",
            "Name shortening pattern",
        ),
        TestCase(
            "Russian Patronymic",
            "Vladimir Vladimirovich Putin",
            "President Putin addressed the nation.",
            True,
            "Cultural",
            "Russian patronymic",
        ),
        TestCase(
            "Brazilian Full",
            "Luiz Inácio Lula da Silva",
            "Former president Lula was acquitted.",
            True,
            "Cultural",
            "Brazilian nickname usage",
        ),
        TestCase(
            "Thai Nickname",
            "Prayut Chan-o-cha",
            "General Prayut announced elections.",
            True,
            "Cultural",
            "Thai name patterns",
        ),
        # Professional contexts
        TestCase(
            "Doctor Title",
            "Dr. Jane Smith",
            "Surgeon Jane Smith performed the operation.",
            True,
            "Professional",
            "Medical title",
        ),
        TestCase(
            "Professor Title",
            "Prof. Alan Turing",
            "Alan Turing developed the machine.",
            True,
            "Professional",
            "Academic title",
        ),
        TestCase(
            "Military Rank",
            "General James Mattis",
            "Jim Mattis served as Secretary.",
            True,
            "Professional",
            "Military title and nickname",
        ),
        TestCase(
            "Rev Title",
            "Reverend Jesse Jackson",
            "Jesse Jackson led the march.",
            True,
            "Professional",
            "Religious title",
        ),
        # Negative cases - should NOT match
        TestCase(
            "Completely Different",
            "Barack Obama",
            "Angela Merkel visited France.",
            False,
            "Negative",
            "No connection at all",
        ),
        TestCase(
            "Same Last Name Only",
            "William Johnson",
            "Sarah Johnson won the award.",
            False,
            "Negative",
            "Only surname matches",
        ),
        TestCase(
            "Company Not Person",
            "Morgan Freeman",
            "Morgan Stanley reported earnings.",
            False,
            "Negative",
            "Company name confusion",
        ),
        TestCase(
            "Location Not Person",
            "Jordan Peterson",
            "The Jordan River flooded.",
            False,
            "Negative",
            "Location name confusion",
        ),
        TestCase(
            "Partial in Phrase",
            "Mark Stone",
            "The building's cornerstone marked history.",
            False,
            "Negative",
            "Name words in different context",
        ),
    ]


def run_evaluation(
    test_cases: List[TestCase], extractor: EntityExtractor, matcher: NameMatcher, use_llm: bool = False
) -> Dict:
    """Run evaluation on test cases."""
    results = []

    for test_case in test_cases:
        # Extract entities
        entities = extractor.extract_entities(test_case.article_text)

        if not entities:
            prediction = False
            confidence = 0.0
            best_entity = None
        else:
            # Find best match
            best_confidence = 0.0
            best_match = None
            best_entity = None

            for entity in entities:
                result = matcher.match(test_case.input_name, entity, strict_mode=False, use_llm=use_llm)
                if result.match_confidence > best_confidence:
                    best_confidence = result.match_confidence
                    best_match = result
                    best_entity = entity.text

            prediction = best_match.is_match_recommendation if best_match else False
            confidence = best_confidence

        # Determine correctness
        correct = prediction == test_case.expected_match

        result = {
            "name": test_case.name,
            "input_name": test_case.input_name,
            "category": test_case.category,
            "expected": test_case.expected_match,
            "predicted": prediction,
            "correct": correct,
            "confidence": confidence,
            "best_entity": best_entity,
            "notes": test_case.notes,
        }

        results.append(result)

    return results


def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate evaluation metrics."""
    total = len(results)
    correct = sum(1 for r in results if r["correct"])

    # By category
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0}
        categories[cat]["total"] += 1
        if r["correct"]:
            categories[cat]["correct"] += 1

    # Confusion matrix
    tp = sum(1 for r in results if r["expected"] and r["predicted"])
    fp = sum(1 for r in results if not r["expected"] and r["predicted"])
    tn = sum(1 for r in results if not r["expected"] and not r["predicted"])
    fn = sum(1 for r in results if r["expected"] and not r["predicted"])

    # Metrics
    accuracy = correct / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "categories": categories,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }


def print_results(results: List[Dict], metrics: Dict):
    """Print evaluation results."""
    print("\n" + "=" * 80)
    print("EXTENDED NAME SCREENING EVALUATION (NO ACCENTED NAMES)")
    print("=" * 80)

    print("\nOVERALL PERFORMANCE:")
    print(f"Total test cases: {metrics['total']}")
    print(f"Accuracy: {metrics['accuracy']:.1%}")
    print(f"Precision: {metrics['precision']:.1%}")
    print(f"Recall: {metrics['recall']:.1%}")
    print(f"F1 Score: {metrics['f1_score']:.1%}")

    print("\nCATEGORY BREAKDOWN:")
    for cat, stats in sorted(metrics["categories"].items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"{cat}: {acc:.1%} ({stats['total']} tests)")

    print("\nCONFUSION MATRIX:")
    print(f"True Positives:  {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"True Negatives:  {metrics['true_negatives']}")
    print(f"False Negatives: {metrics['false_negatives']}")

    # Show failures only
    failures = [r for r in results if not r["correct"]]
    if failures:
        print(f"\n⚠️  FAILURES ({len(failures)}):")
        for failure in failures:
            status = "Should match" if failure["expected"] else "Should NOT match"
            got = "Matched" if failure["predicted"] else "No match"
            print(
                f"   - {failure['name']}: {status} but {got} ({failure['confidence']:.1%})"
            )

    # Show borderline cases
    borderline = [r for r in results if 0.20 <= r["confidence"] <= 0.35]
    if borderline:
        print(f"\n🔍 BORDERLINE CASES ({len(borderline)}):")
        for b in borderline:
            print(f"   - {b['name']}: {b['confidence']:.1%} confidence")

    print("=" * 80)

    # Key findings
    if metrics["recall"] < 1.0:
        print(f"❌ RECALL WARNING: Missing {metrics['false_negatives']} matches!")
        print("   False negatives:")
        for r in results:
            if r["expected"] and not r["predicted"]:
                print(
                    f"   - {r['name']}: '{r['input_name']}' vs '{r['best_entity'] or 'NO ENTITY'}'"
                )
    else:
        print("✅ PERFECT RECALL: All expected matches found!")

    if metrics["precision"] < 0.8:
        print(f"\n⚠️  PRECISION NOTE: {metrics['false_positives']} false positives")

    print("=" * 80)


def main():
    """Run extended evaluation."""
    import argparse
    parser = argparse.ArgumentParser(description="Extended Name Screening Evaluation")
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM enhancement")
    args = parser.parse_args()
    
    print("Extended Name Screening Evaluation - No Accented Names")
    print("=" * 60)
    print(f"LLM Enhancement: {'ENABLED' if args.use_llm else 'DISABLED'}")
    print("Initializing system...")

    # Initialize components
    extractor = EntityExtractor()
    matcher = NameMatcher()

    print("✅ System ready. Creating extended test suite...")

    # Create test cases
    test_cases = create_extended_test_suite()
    print(f"✅ Created {len(test_cases)} test cases (no accented names)")

    print("\n🔄 Running evaluation...")

    # Run evaluation
    results = run_evaluation(test_cases, extractor, matcher, use_llm=args.use_llm)

    # Calculate metrics
    metrics = calculate_metrics(results)

    # Print results
    print_results(results, metrics)


if __name__ == "__main__":
    main()
