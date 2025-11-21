"""
Test Full Pipeline: Contract → CUAD Extraction → T5 FOPL Generation

This tests the integration of:
1. CUAD clause extraction (Rakib/roberta-base-on-cuad)
2. T5 FOPL generation (your trained model)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from cuad_integration.clause_extractor import CUADClauseExtractor
from inference.pipeline import LegalReasoningPipeline


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
preceding the claim.

8. WARRANTY: Supplier warrants that all goods delivered shall be free from defects 
in materials and workmanship for a period of one (1) year from the date of delivery.
"""


def main():
    print("="*80)
    print("FULL PIPELINE TEST: Contract → CUAD → T5 → FOPL")
    print("="*80 + "\n")
    
    # Step 1: Load CUAD extractor
    print("Step 1: Loading CUAD clause extractor...")
    cuad_extractor = CUADClauseExtractor(
        model_path="Rakib/roberta-base-on-cuad"
    )
    print("✓ CUAD model loaded\n")
    
    # Step 2: Load T5 FOPL generator
    print("Step 2: Loading T5 FOPL generation model...")
    try:
        fopl_pipeline = LegalReasoningPipeline(
            model_path="checkpoints/best_model"
        )
        print("✓ T5 FOPL model loaded\n")
    except Exception as e:
        print(f"⚠️  Could not load T5 model: {e}")
        print("   Using mock FOPL generation for demo\n")
        fopl_pipeline = None
    
    # Step 3: Extract clauses using CUAD
    print("Step 3: Extracting clauses from contract...")
    print(f"Contract length: {len(SAMPLE_CONTRACT)} characters\n")
    
    extracted_clauses = cuad_extractor.extract_and_format(SAMPLE_CONTRACT)
    
    print("✓ Clauses extracted:\n")
    for category, text in extracted_clauses.items():
        if text and text != "Not found":
            print(f"  • {category}")
    
    # Step 4: Generate FOPL for each extracted clause
    print(f"\nStep 4: Generating FOPL formulas for extracted clauses...")
    print("="*80 + "\n")
    
    fopl_results = {}
    
    # Focus on key categories for FOPL generation
    key_categories = [
        "Minimum Commitment",
        "Termination For Convenience",
        "Governing Law",
        "Expiration Date",
        "Exclusivity"
    ]
    
    for category in key_categories:
        clause_text = extracted_clauses.get(category)
        
        if not clause_text or clause_text == "Not found":
            print(f"✗ {category}: No clause found\n")
            continue
        
        print(f"Category: {category}")
        print(f"Clause: {clause_text[:150]}..." if len(clause_text) > 150 else f"Clause: {clause_text}")
        
        # Generate FOPL
        if fopl_pipeline:
            try:
                fopl_formula = fopl_pipeline.generate(clause_text)
                print(f"FOPL: {fopl_formula}")
                fopl_results[category] = {
                    'clause': clause_text,
                    'fopl': fopl_formula
                }
            except Exception as e:
                print(f"⚠️  FOPL generation error: {e}")
                fopl_results[category] = {
                    'clause': clause_text,
                    'fopl': f"[Error: {str(e)}]"
                }
        else:
            # Mock FOPL for demo
            mock_fopl = generate_mock_fopl(category, clause_text)
            print(f"FOPL (mock): {mock_fopl}")
            fopl_results[category] = {
                'clause': clause_text,
                'fopl': mock_fopl
            }
        
        print("-" * 80 + "\n")
    
    # Step 5: Summary
    print("="*80)
    print("PIPELINE TEST SUMMARY")
    print("="*80)
    print(f"✓ Clauses extracted: {len([c for c in extracted_clauses.values() if c != 'Not found'])}")
    print(f"✓ FOPL formulas generated: {len(fopl_results)}")
    print("\nFOPL Results:")
    for category, data in fopl_results.items():
        print(f"\n{category}:")
        print(f"  FOPL: {data['fopl']}")
    
    return fopl_results


def generate_mock_fopl(category: str, clause_text: str) -> str:
    """Generate mock FOPL for demo purposes when T5 model not available."""
    
    # Simple pattern-based mock FOPL generation
    if "Minimum Commitment" in category:
        if "1,000 units" in clause_text and "quarter" in clause_text:
            return "∀q (Quarter(q) → Purchase(Buyer, Units, n) ∧ n ≥ 1000)"
    
    elif "Termination" in category:
        if "90 days" in clause_text or "ninety (90) days" in clause_text:
            return "∀p (Party(p) → CanTerminate(p, Agreement, 90))"
    
    elif "Governing Law" in category:
        if "Delaware" in clause_text:
            return "GoverningLaw(Agreement, Delaware)"
    
    elif "Expiration Date" in category:
        if "two (2) years" in clause_text:
            return "Duration(Agreement, StartDate, 2_years)"
    
    elif "Exclusivity" in category:
        if "exclusive rights" in clause_text:
            return "ExclusiveRights(Supplier, Buyer, Territory(US))"
    
    return f"[FOPL representation of {category}]"


if __name__ == "__main__":
    main()
