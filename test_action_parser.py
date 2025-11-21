"""
Test Action Parser with Different Query Formats
"""

from cuad_integration.action_parser import ActionParser

def test_parser():
    """Test the action parser with various inputs."""
    
    parser = ActionParser()
    
    print("="*80)
    print("TESTING ACTION PARSER")
    print("="*80 + "\n")
    
    # Test positive statements (what the parser expects)
    print("📝 POSITIVE STATEMENTS (Declarative):\n")
    positive_tests = [
        "Supplier delivered goods 15 days after order",
        "Buyer paid on day 35",
        "Buyer terminated with 60 days notice",
        "Buyer purchased 800 units",
        "Supplier sold products to CompetitorX",
        "Tenant paid within 30 days",
        "Company terminated the agreement with 90 days notice",
        "Vendor purchased 1000 units",
    ]
    
    for query in positive_tests:
        print(f"Query: {query}")
        result = parser.parse(query)
        if result:
            print(f"  ✓ Parsed: {result.action_type.value}")
            print(f"    Actor: {result.actor}")
            print(f"    Parameters: {result.parameters}")
            print(f"    FOPL: {result.to_fopl_atom()}")
        else:
            print(f"  ✗ Failed to parse")
        print()
    
    # Test questions (need conversion)
    print("\n" + "="*80)
    print("❓ QUESTION FORMATS (Interrogative - need conversion):\n")
    question_tests = [
        ("Did Supplier breach by delivering 15 days late?",
         "Supplier delivered goods 15 days after order"),
        ("Can Buyer terminate with 60 days notice?",
         "Buyer terminated with 60 days notice"),
        ("Did Buyer comply by purchasing 800 units?",
         "Buyer purchased 800 units"),
        ("Did Buyer pay on time by paying on day 35?",
         "Buyer paid on day 35"),
    ]
    
    for question, declarative in question_tests:
        print(f"Question: {question}")
        print(f"Converted: {declarative}")
        result = parser.parse(declarative)
        if result:
            print(f"  ✓ Parsed: {result.action_type.value}")
            print(f"    Actor: {result.actor}")
            print(f"    Parameters: {result.parameters}")
        else:
            print(f"  ✗ Failed to parse")
        print()
    
    print("="*80)
    print("\n📊 SUMMARY:")
    print("  • Action parser works with DECLARATIVE statements")
    print("  • Questions need to be converted to statements first")
    print("  • Next step: Add question-to-statement converter")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_parser()
