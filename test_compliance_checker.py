"""
Test Compliance Checker Integration
Full end-to-end testing with real scenarios.
"""

from cuad_integration.simple_compliance_checker import SimpleComplianceChecker

# Sample Contract
CONTRACT = """
SUPPLY AGREEMENT

This Supply Agreement ("Agreement") is entered into as of January 1, 2024,
by and between ABC Corporation, a Delaware corporation ("Buyer"), and 
XYZ Supplies Inc., a California corporation ("Supplier").

1. DELIVERY OBLIGATIONS
   Supplier shall deliver all ordered goods to Buyer within ten (10) business 
   days from the date of receipt of Buyer's purchase order. Time is of the 
   essence for all deliveries under this Agreement.

2. MINIMUM PURCHASE COMMITMENT
   Buyer agrees to purchase a minimum of one thousand (1,000) units per 
   calendar quarter during the term of this Agreement.

3. PAYMENT TERMS
   Buyer shall pay Supplier within thirty (30) days of invoice date.

4. TERMINATION
   Either party may terminate this Agreement for convenience upon providing 
   ninety (90) days' prior written notice to the other party.
"""


def test_scenario(checker, query, expected_status):
    """Test a single compliance scenario."""
    print(f"\n{'='*80}")
    print(f"TEST SCENARIO: {query}")
    print(f"{'='*80}")
    
    result = checker.check_compliance(CONTRACT, query)
    
    print("\n" + "="*80)
    print("RESULT:")
    print("="*80)
    print(f"Status: {result.status.value.upper()}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"\nExplanation:")
    print(result.explanation)
    
    # Check if result matches expected
    passed = result.status.value == expected_status
    print(f"\n{'✅ PASSED' if passed else '❌ FAILED'}")
    print(f"Expected: {expected_status}, Got: {result.status.value}")
    print("="*80)
    
    return passed


def main():
    """Run all compliance tests."""
    
    print("\n" + "="*80)
    print("COMPLIANCE CHECKER - INTEGRATION TESTS")
    print("="*80 + "\n")
    
    # Initialize checker
    checker = SimpleComplianceChecker("Rakib/roberta-base-on-cuad")
    
    # Test scenarios
    scenarios = [
        {
            "query": "Supplier delivered goods 15 days after order",
            "expected": "breach",
            "description": "Delivery late (15 > 10 days)"
        },
        {
            "query": "Supplier delivered goods 8 days after order",
            "expected": "compliant",
            "description": "Delivery on time (8 <= 10 days)"
        },
        {
            "query": "Buyer paid on day 35",
            "expected": "breach",
            "description": "Payment late (35 > 30 days)"
        },
        {
            "query": "Buyer paid on day 25",
            "expected": "compliant",
            "description": "Payment on time (25 <= 30 days)"
        },
        {
            "query": "Buyer terminated with 60 days notice",
            "expected": "breach",
            "description": "Termination notice insufficient (60 < 90 days)"
        },
        {
            "query": "Buyer terminated with 90 days notice",
            "expected": "compliant",
            "description": "Termination notice sufficient (90 >= 90 days)"
        },
        {
            "query": "Buyer purchased 800 units",
            "expected": "breach",
            "description": "Purchase below minimum (800 < 1000 units)"
        },
        {
            "query": "Buyer purchased 1200 units",
            "expected": "compliant",
            "description": "Purchase meets minimum (1200 >= 1000 units)"
        },
    ]
    
    # Run tests
    results = []
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n\n{'#'*80}")
        print(f"SCENARIO {i}/{len(scenarios)}: {scenario['description']}")
        print(f"{'#'*80}")
        
        passed = test_scenario(
            checker,
            scenario['query'],
            scenario['expected']
        )
        
        results.append({
            "scenario": scenario['description'],
            "passed": passed
        })
    
    # Summary
    print("\n\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80 + "\n")
    
    passed_count = sum(1 for r in results if r['passed'])
    total_count = len(results)
    pass_rate = (passed_count / total_count) * 100
    
    for i, result in enumerate(results, 1):
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"{i}. {status} - {result['scenario']}")
    
    print(f"\n{'='*80}")
    print(f"PASS RATE: {passed_count}/{total_count} ({pass_rate:.0f}%)")
    print(f"{'='*80}\n")
    
    if pass_rate == 100:
        print("🎉 ALL TESTS PASSED! 🎉")
    elif pass_rate >= 75:
        print("✅ Most tests passed!")
    else:
        print("⚠️  Some tests failed - review results above")


if __name__ == "__main__":
    main()
