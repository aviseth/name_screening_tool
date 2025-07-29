#!/usr/bin/env python3
"""
Name Screening Tool Evaluation

This script runs a comprehensive evaluation of the name screening system,
testing critical security features and generating performance metrics.
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


def create_test_suite() -> List[TestCase]:
    """Create comprehensive test suite."""
    return [
        # Exact matches (should match)
        TestCase(
            "Exact Match",
            "Sarah Chen",
            "Local businesswoman Sarah Chen announced expansion.",
            True,
            "Basic",
            "Perfect match",
        ),
        TestCase(
            "Unicode Match",
            "José García",
            "Jose Garcia opened his restaurant.",
            True,
            "Basic",
            "Unicode normalization",
        ),
        # Critical security tests (should NOT match)
        TestCase(
            "Surname Mismatch",
            "Seraah Dean",
            "Local businesswoman Sarah Chen announced expansion.",
            False,
            "Security",
            "Similar first names, different surnames",
        ),
        TestCase(
            "Invalid Surname Shortening",
            "Sarah Cheng",
            "Sarah Chen announced her expansion plans.",
            False,
            "Security",
            "Surname shortening not allowed",
        ),
        # Valid pattern matches (should match)
        TestCase(
            "Valid Name Shortening",
            "Samuel Johnson",
            "Sam Johnson published his dictionary.",
            True,
            "Pattern",
            "First name shortening allowed",
        ),
        TestCase(
            "Name Order Variation",
            "Li Wei",
            "Wei Li delivered a speech at the UN.",
            True,
            "Pattern",
            "Cultural name order",
        ),
        TestCase(
            "Fuzzy Match",
            "Zelenskyy",
            "Ukrainian President Volodymyr Zelensky addressed the nation.",
            True,
            "Pattern",
            "Spelling variation",
        ),
        TestCase(
            "Initial Match",
            "J. Biden",
            "President Joe Biden announced new initiatives.",
            True,
            "Pattern",
            "Initial expansion",
        ),
        # Edge cases (various expectations)
        TestCase(
            "Family Member",
            "Michael Jordan",
            "Michael Jordan's son Marcus Jordan opened a store.",
            True,
            "Edge",
            "Should match Michael, not Marcus",
        ),
        TestCase(
            "Single Token",
            "James Chen",
            "Sarah Chen announced expansion. Chen founded the company.",
            False,
            "Security",
            "Single token should be capped/rejected",
        ),
        TestCase(
            "Middle Name Mismatch",
            "Daniel Ramesh Brotman",
            "Daniel Holderg Brotman announced retirement.",
            False,
            "Security",
            "Different middle names",
        ),
        # True negatives (should NOT match)
        TestCase(
            "True Negative",
            "Albert Einstein",
            "Local business owner Jane Doe announced retirement.",
            False,
            "Negative",
            "Completely different names",
        ),
        TestCase(
            "Similar Names",
            "Michael Jordan",
            "Michael Johnson won the 400-meter race.",
            False,
            "Negative",
            "Similar but different people",
        ),
        # Complex edge cases
        TestCase(
            "All Caps Entity",
            "Elon Musk",
            "ELON MUSK, CEO of Tesla, announced new plans.",
            True,
            "Edge",
            "All caps entity extraction",
        ),
        TestCase(
            "Hyphenated Name Order",
            "Kim Jong-un",
            "North Korean leader Jong-un Kim made an announcement.",
            True,
            "Edge",
            "Hyphenated name reordering",
        ),
        TestCase(
            "Compound Name Particles",
            "Ursula von der Leyen",
            "European Commission President von der Leyen spoke today.",
            True,
            "Edge",
            "Compound name with particles",
        ),
        TestCase(
            "Multiple Initials",
            "J.K. Rowling",
            "Author Joanne Kathleen Rowling signed books.",
            True,
            "Edge",
            "Concatenated initials expansion",
        ),
        TestCase(
            "Professional Title",
            "Dr. Angela Merkel",
            "Former Chancellor Angela Merkel received an award.",
            True,
            "Edge",
            "Title removal in input",
        ),
        TestCase(
            "Transliteration",
            "Mohammed bin Salman",
            "Crown Prince Mohammad bin Salman Al Saud attended.",
            True,
            "Edge",
            "Name transliteration variants",
        ),
        TestCase(
            "Nickname Complex",
            "William Jefferson Clinton",
            "Former President Bill Clinton spoke at the event.",
            True,
            "Edge",
            "Nickname with middle name",
        ),
        TestCase(
            "Name Format Variation",
            "Xi Jinping",
            "President Jinping Xi addressed the conference.",
            True,
            "Edge",
            "Name order variation",
        ),
        TestCase(
            "Possessive Form",
            "Elizabeth Warren",
            "Warren's policies were discussed. Elizabeth's team responded.",
            True,
            "Edge",
            "Possessive forms and partial matches",
        ),
        TestCase(
            "Similar Sound Different Spelling",
            "Catherine",
            "Actress Katherine Hepburn won multiple awards.",
            True,
            "Edge",
            "Phonetic matching",
        ),
        # Security-critical false positive tests
        TestCase(
            "Common Surname Different Person",
            "John Smith",
            "Sarah Smith, the scientist, won the Nobel Prize.",
            False,
            "Security",
            "Common surname, different person",
        ),
        TestCase(
            "Partial Name in Compound",
            "Robert Johnson",
            "Johnson & Johnson announced new products.",
            False,
            "Security",
            "Name in company name",
        ),
        TestCase(
            "Similar First Name Only",
            "Christina Lopez",
            "Christine Johnson was elected mayor.",
            False,
            "Security",
            "Similar first name, different surname",
        ),
        TestCase(
            "Abbreviation Not Person",
            "Sam",
            "The SAM missile system was deployed.",
            False,
            "Security",
            "Abbreviation not a person",
        ),
        TestCase(
            "Title Without Name",
            "President Johnson",
            "The president announced new policies.",
            False,
            "Security",
            "Title without actual name",
        ),
        # Unicode and special character tests
        TestCase(
            "Accented Characters",
            "Bjork Gudmundsdottir",
            "Icelandic singer Björk released her new album.",
            True,
            "Unicode",
            "Special accented characters",
        ),
        TestCase(
            "Mixed Script Characters",
            "Ma Yun",
            "Alibaba founder Jack Ma (馬雲) stepped down.",
            True,
            "Unicode",
            "Mixed scripts",
        ),
        TestCase(
            "Alternative Script",
            "Muhammad Ali",
            "محمد علي (Muhammad Ali) was the greatest.",
            True,
            "Unicode",
            "Alternative script handling",
        ),
        # Compound and complex structures
        TestCase(
            "Double Barreled Surname",
            "Helena Bonham Carter",
            "Actress Bonham Carter won the award.",
            True,
            "Complex",
            "Double-barreled surname",
        ),
        TestCase(
            "Multiple Middle Names",
            "George Herbert Walker Bush",
            "President George H.W. Bush served one term.",
            True,
            "Complex",
            "Multiple middle names with initials",
        ),
        TestCase(
            "Name with Apostrophe",
            "Shaquille O'Neal",
            "Basketball star Shaq O'Neal retired.",
            True,
            "Complex",
            "Apostrophe in surname",
        ),
        TestCase(
            "Mononym",
            "Madonna",
            "Singer Madonna performed at the concert.",
            True,
            "Complex",
            "Single name entity",
        ),
        # Cross-entity confusion tests
        TestCase(
            "Similar Names Different Context",
            "Michael Jackson",
            "Michael Jackson, the beer expert, not the singer.",
            True,
            "Context",
            "Same name, different person context",
        ),
        TestCase(
            "Partial Match Multiple Entities",
            "Chris Johnson",
            "Christopher Johnson met with Christine Johnson.",
            True,
            "Context",
            "Multiple similar entities",
        ),
        TestCase(
            "Organization vs Person",
            "Morgan Stanley",
            "Investment bank Morgan Stanley announced results.",
            False,
            "Context",
            "Organization not person",
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
    security_tests = [r for r in results if r["category"] == "Security"]
    security_correct = sum(1 for r in security_tests if r["correct"])

    pattern_tests = [r for r in results if r["category"] == "Pattern"]
    pattern_correct = sum(1 for r in pattern_tests if r["correct"])

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
        "security_accuracy": security_correct / len(security_tests)
        if security_tests
        else 0,
        "pattern_accuracy": pattern_correct / len(pattern_tests)
        if pattern_tests
        else 0,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "security_tests": len(security_tests),
        "pattern_tests": len(pattern_tests),
    }


def print_results(results: List[Dict], metrics: Dict):
    """Print evaluation results."""
    print("\n" + "=" * 80)
    print("NAME SCREENING TOOL - COMPREHENSIVE EVALUATION")
    print("=" * 80)

    print("\nOVERALL PERFORMANCE:")
    print(f"Total test cases: {metrics['total']}")
    print(f"Accuracy: {metrics['accuracy']:.1%}")
    print(f"Precision: {metrics['precision']:.1%}")
    print(f"Recall: {metrics['recall']:.1%}")
    print(f"F1 Score: {metrics['f1_score']:.1%}")

    print("\nCATEGORY BREAKDOWN:")
    print(
        f"Security Tests: {metrics['security_accuracy']:.1%} ({metrics['security_tests']} tests)"
    )
    print(
        f"Pattern Tests: {metrics['pattern_accuracy']:.1%} ({metrics['pattern_tests']} tests)"
    )

    print("\nCONFUSION MATRIX:")
    print(f"True Positives:  {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"True Negatives:  {metrics['true_negatives']}")
    print(f"False Negatives: {metrics['false_negatives']}")

    # Show detailed results
    print("\nDETAILED RESULTS:")
    print("-" * 80)

    for result in results:
        status = "✅" if result["correct"] else "❌"
        match_status = "MATCH" if result["predicted"] else "NO MATCH"
        expected_status = "MATCH" if result["expected"] else "NO MATCH"

        print(f"{status} {result['name']}")
        print(f"   Input: {result['input_name']}")
        print(
            f"   Expected: {expected_status} | Got: {match_status} | Confidence: {result['confidence']:.1%}"
        )
        if result["best_entity"]:
            print(f"   Best Entity: {result['best_entity']}")
        print(f"   Notes: {result['notes']}")
        print()

    # Critical failures
    failures = [r for r in results if not r["correct"]]
    if failures:
        print(f"⚠️  FAILURES ({len(failures)}):")
        for failure in failures:
            print(f"   - {failure['name']}: {failure['notes']}")

    print("=" * 80)

    # Overall assessment
    if metrics["security_accuracy"] == 1.0:
        print("🔒 SECURITY: PASSED - All critical security tests passed")
    else:
        print("🚨 SECURITY: FAILED - Some security tests failed")

    if metrics["accuracy"] >= 0.8:
        print(f"✅ OVERALL: GOOD - {metrics['accuracy']:.1%} accuracy")
    else:
        print(f"⚠️  OVERALL: NEEDS IMPROVEMENT - {metrics['accuracy']:.1%} accuracy")

    print("=" * 80)


def main():
    """Run comprehensive evaluation."""
    import argparse
    parser = argparse.ArgumentParser(description="Name Screening Tool Evaluation")
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM enhancement")
    args = parser.parse_args()
    
    print("Name Screening Tool - Comprehensive Evaluation")
    print("=" * 60)
    print(f"LLM Enhancement: {'ENABLED' if args.use_llm else 'DISABLED'}")
    print("Initializing system...")

    # Initialize components
    extractor = EntityExtractor()
    matcher = NameMatcher()

    print("✅ System ready. Creating test suite...")

    # Create test cases
    test_cases = create_test_suite()
    print(f"✅ Created {len(test_cases)} test cases")

    print("\n🔄 Running evaluation...")

    # Run evaluation
    results = run_evaluation(test_cases, extractor, matcher, use_llm=args.use_llm)

    # Calculate metrics
    metrics = calculate_metrics(results)

    # Print results
    print_results(results, metrics)

    print("\nFor additional testing, see CLI_TEST_COMMANDS.md (32 scenarios)")


if __name__ == "__main__":
    main()
