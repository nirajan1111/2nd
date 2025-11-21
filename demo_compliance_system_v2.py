"""
COMPLETE END-TO-END DEMO v2
Contract Compliance Checking System with REAL Compliance Checking

This demo shows the full workflow with working compliance checker!
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
   essence for all deliveries under this Agreement. Any delivery beyond the 
   ten-day period shall be considered a material breach of this Agreement.

2. MINIMUM PURCHASE COMMITMENT
   Buyer agrees to purchase a minimum of one thousand (1,000) units per 
   calendar quarter during the term of this Agreement. Failure to meet this 
   minimum shall not constitute a breach but may result in price adjustments.

3. PAYMENT TERMS
   Buyer shall pay Supplier within thirty (30) days of invoice date. Late 
   payments shall accrue interest at the rate of 1.5% per month.

4. TERMINATION
   Either party may terminate this Agreement for convenience upon providing 
   ninety (90) days' prior written notice to the other party. Either party 
   may terminate immediately upon written notice if the other party commits 
   a material breach and fails to cure within thirty (30) days of notice.

5. GOVERNING LAW
   This Agreement shall be governed by and construed in accordance with the 
   laws of the State of Delaware.
"""

# Test Scenarios
SCENARIOS = [
    {
        "question": "Did Supplier breach by delivering goods 15 days after the order?",
        "query": "Supplier delivered goods 15 days after order",
    },
    {
        "question": "Can Buyer terminate the agreement with 60 days notice?",
        "query": "Buyer terminated with 60 days notice",
    },
    {
        "question": "Did Buyer comply by purchasing 800 units in Q1?",
        "query": "Buyer purchased 800 units",
    },
    {
        "question": "Did Buyer pay on time by paying on day 35?",
        "query": "Buyer paid on day 35",
    },
]


def print_header(text, char="="):
    """Print formatted header."""
    print(f"\n{char * 80}")
    print(f"{text}")
    print(f"{char * 80}\n")


def main():
    print_header("🏛️  CONTRACT COMPLIANCE CHECKER - FULL DEMO v2", "=")
    
    print("This demo shows the complete end-to-end workflow:")
    print("  1. Extract clauses from contract (CUAD)")
    print("  2. Parse user compliance questions")
    print("  3. Check compliance with rule-based reasoning")
    print("  4. Provide detailed explanations")
    print()
    
    # Initialize
    print("⚙️  Initializing system...")
    checker = SimpleComplianceChecker("Rakib/roberta-base-on-cuad")
    
    # Contract summary
    print_header("📄 CONTRACT SUMMARY")
    print(f"  • Length: {len(CONTRACT)} characters")
    print(f"  • Parties: ABC Corporation (Buyer) & XYZ Supplies Inc. (Supplier)")
    print(f"  • Key Terms:")
    print(f"    - Delivery: within 10 business days")
    print(f"    - Payment: within 30 days")
    print(f"    - Termination: 90 days notice")
    print(f"    - Minimum Purchase: 1,000 units per quarter")
    
    # Test each scenario
    print_header("🔍 COMPLIANCE CHECKING SCENARIOS")
    
    results = []
    for i, scenario in enumerate(SCENARIOS, 1):
        print(f"\n{'━' * 80}")
        print(f"SCENARIO {i}/{len(SCENARIOS)}")
        print(f"{'━' * 80}")
        print(f"\n❓ Question: {scenario['question']}\n")
        
        # Check compliance
        result = checker.check_compliance(CONTRACT, scenario['query'])
        results.append(result)
        
        # Display result
        print(f"\n{'─' * 80}")
        print("📊 RESULT:")
        print(f"{'─' * 80}\n")
        
        # Status with emoji
        status_emoji = {
            "breach": "❌",
            "compliant": "✅",
            "uncertain": "❓",
            "insufficient_information": "ℹ️"
        }
        emoji = status_emoji.get(result.status.value, "")
        
        print(f"{emoji} Status: {result.status.value.upper()}")
        print(f"🎯 Confidence: {result.confidence:.0%}\n")
        print("📋 Explanation:")
        print(result.explanation)
        
        if result.breach_details:
            print(f"\n📌 Details:")
            for key, value in result.breach_details.items():
                print(f"   {key.capitalize()}: {value}")
    
    # Summary
    print_header("✅ DEMO COMPLETE", "=")
    
    print("System Capabilities Demonstrated:")
    print("  ✓ Extract clauses from contracts (CUAD)")
    print("  ✓ Parse natural language compliance questions")
    print("  ✓ Check compliance with rule-based reasoning")
    print("  ✓ Provide clear breach/compliance explanations")
    print("  ✓ Calculate confidence scores")
    
    # Results summary
    breach_count = sum(1 for r in results if r.status.value == "breach")
    compliant_count = sum(1 for r in results if r.status.value == "compliant")
    
    print(f"\nResults Summary:")
    print(f"  • Breaches detected: {breach_count}")
    print(f"  • Compliant actions: {compliant_count}")
    print(f"  • Average confidence: {sum(r.confidence for r in results) / len(results):.0%}")
    
    print("\nNext Steps:")
    print("  → Expand test suite to 20+ scenarios")
    print("  → Build REST API with FastAPI")
    print("  → Create web interface with Streamlit")
    print("  → Add PDF upload support")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
