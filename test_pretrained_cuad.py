"""
Test Pre-trained CUAD Model

Quick test to verify Rakib/roberta-base-on-cuad works for clause extraction.
"""

from cuad_integration.clause_extractor import CUADClauseExtractor

# Sample contract
SAMPLE_CONTRACT = """
SUPPLY AGREEMENT

This Supply Agreement ("Agreement") is entered into as of January 1, 2024,
by and between ABC Corporation, a Delaware corporation ("Buyer"), and 
XYZ Supplies Inc., a California corporation ("Supplier").

1. TERM: This Agreement shall commence on January 1, 2024 and shall continue 
for a period of two (2) years, unless earlier terminated in accordance with 
Section 4 below.

2. DELIVERY: Supplier shall deliver the goods to Buyer within ten (10) business 
days of receiving a purchase order from Buyer. Time is of the essence for all 
deliveries under this Agreement.

3. MINIMUM COMMITMENT: Buyer agrees to purchase a minimum of 1,000 units per 
quarter during the term of this Agreement.

4. TERMINATION: Either party may terminate this Agreement for convenience upon 
providing ninety (90) days' written notice to the other party.

5. GOVERNING LAW: This Agreement shall be governed by and construed in accordance 
with the laws of the State of Delaware, without regard to its conflict of laws 
principles.

6. EXCLUSIVITY: Supplier grants Buyer exclusive rights to purchase and resell the 
goods in the United States territory.

7. LIABILITY: In no event shall either party's total liability under this Agreement 
exceed the total amount paid by Buyer to Supplier during the twelve (12) months 
preceding the claim. This cap on liability shall not apply to breaches of 
confidentiality or intellectual property rights.

8. WARRANTY: Supplier warrants that all goods delivered shall be free from defects 
in materials and workmanship for a period of one (1) year from the date of delivery.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first 
written above.

ABC Corporation                    XYZ Supplies Inc.
By: _________________              By: _________________
Name: John Smith                   Name: Jane Doe
Title: CEO                         Title: President
"""


def main():
    print("="*70)
    print("Testing Pre-trained CUAD Model: Rakib/roberta-base-on-cuad")
    print("="*70 + "\n")
    
    # Initialize extractor with pre-trained model
    print("Step 1: Loading pre-trained model...")
    extractor = CUADClauseExtractor(
        model_path="Rakib/roberta-base-on-cuad"  # Pre-trained model
    )
    
    # Extract clauses
    print("\nStep 2: Extracting clauses from sample contract...")
    print(f"Contract length: {len(SAMPLE_CONTRACT)} characters\n")
    
    results = extractor.extract_and_format(SAMPLE_CONTRACT)
    
    # Display results
    print("\n" + "="*70)
    print("EXTRACTION RESULTS")
    print("="*70 + "\n")
    
    found_count = 0
    for category, clause_text in results.items():
        if clause_text and clause_text != "Not found":
            found_count += 1
            print(f"✓ {category}:")
            print(f"  {clause_text[:200]}..." if len(clause_text) > 200 else f"  {clause_text}")
            print()
        else:
            print(f"✗ {category}: Not found")
    
    print("="*70)
    print(f"Summary: Found {found_count} out of {len(results)} categories")
    print("="*70)
    
    # Test specific categories
    print("\n\nDetailed Results:\n")
    
    key_categories = [
        "Parties",
        "Governing Law", 
        "Expiration Date",
        "Minimum Commitment",
        "Termination For Convenience"
    ]
    
    for cat in key_categories:
        print(f"\n{cat}:")
        print("-" * 50)
        if cat in results and results[cat] != "Not found":
            print(f"✓ FOUND: {results[cat]}")
        else:
            print("✗ NOT FOUND")


if __name__ == "__main__":
    main()
