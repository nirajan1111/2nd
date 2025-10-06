import torch
import json
from typing import Dict, List, Tuple
from dataclasses import asdict

import sys
sys.path.append('..')
from models.neural_parser import NeuralLegalParser
from models.symbolic_reasoner import SymbolicReasoner, ReasoningResult


class LegalReasoningPipeline:
    """
    End-to-end pipeline for legal reasoning:
    1. Parse legal clause to FOPL (Neural)
    2. Reason over FOPL with facts (Symbolic)
    3. Generate explanation
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the pipeline
        
        Args:
            model_path: Path to trained model checkpoint
        """
        # Initialize neural parser
        if model_path:
            self.parser = NeuralLegalParser()
            self.parser.load_pretrained(model_path)
            print(f"✅ Loaded model from {model_path}")
        else:
            self.parser = NeuralLegalParser(model_name='t5-base')
            print("⚠️  Using untrained model")
        
        # Initialize symbolic reasoner
        self.reasoner = SymbolicReasoner()
        
        print("✅ Pipeline initialized")
    
    def process(self, clause_text: str, context: Dict, 
                compliance_case: Dict) -> Dict:
        """
        Process a legal clause end-to-end
        
        Args:
            clause_text: Natural language legal clause
            context: Entity mappings (e.g., {"Tenant": "PartyA"})
            compliance_case: Facts to check (e.g., {"PartyA": {"PayRentDate": 4}})
            
        Returns:
            Complete reasoning result with explanation
        """
        # Step 1: Parse clause to FOPL
        print("\n" + "="*60)
        print("STEP 1: Neural Parsing (Legal Text → FOPL)")
        print("="*60)
        print(f"Input: {clause_text}")
        print(f"Context: {context}")
        
        fopl_rule = self.parser.parse(clause_text, context)
        print(f"\nGenerated FOPL: {fopl_rule}")
        
        # Step 2: Symbolic reasoning
        print("\n" + "="*60)
        print("STEP 2: Symbolic Reasoning")
        print("="*60)
        print(f"Compliance Case: {compliance_case}")
        
        reasoning_result = self.reasoner.evaluate_compliance(
            fopl_rule, compliance_case, context
        )
        
        print(f"\nOutcome: {reasoning_result.outcome}")
        print(f"Explanation: {reasoning_result.explanation}")
        
        # Step 3: Generate detailed explanation
        print("\n" + "="*60)
        print("STEP 3: Proof Trace")
        print("="*60)
        for i, step in enumerate(reasoning_result.proof_trace, 1):
            print(f"{i}. {step}")
        
        # Compile results
        result = {
            'input': {
                'clause_text': clause_text,
                'context': context,
                'compliance_case': compliance_case
            },
            'fopl_rule': fopl_rule,
            'reasoning': {
                'outcome': reasoning_result.outcome,
                'explanation': reasoning_result.explanation,
                'proof_trace': reasoning_result.proof_trace,
                'satisfied_rules': reasoning_result.satisfied_rules,
                'violated_rules': reasoning_result.violated_rules
            }
        }
        
        return result
    
    def batch_process(self, cases: List[Dict]) -> List[Dict]:
        """
        Process multiple cases in batch
        
        Args:
            cases: List of dictionaries with 'clause_text', 'context', 'compliance_case'
            
        Returns:
            List of reasoning results
        """
        results = []
        
        for i, case in enumerate(cases):
            print(f"\n{'#'*60}")
            print(f"Processing Case {i+1}/{len(cases)}")
            print(f"{'#'*60}")
            
            result = self.process(
                case['clause_text'],
                case['context'],
                case['compliance_case']
            )
            results.append(result)
        
        return results
    
    def explain_reasoning(self, result: Dict) -> str:
        """
        Generate human-readable explanation of the reasoning process
        
        Args:
            result: Output from process() method
            
        Returns:
            Formatted explanation string
        """
        explanation = []
        explanation.append("="*60)
        explanation.append("LEGAL REASONING EXPLANATION")
        explanation.append("="*60)
        
        # Input clause
        explanation.append("\n1️⃣ INPUT CLAUSE:")
        explanation.append(f"   \"{result['input']['clause_text']}\"")
        
        # Context
        explanation.append("\n2️⃣ ENTITIES:")
        for entity, party in result['input']['context'].items():
            explanation.append(f"   • {entity} = {party}")
        
        # Generated logic
        explanation.append("\n3️⃣ FORMAL LOGIC (FOPL):")
        explanation.append(f"   {result['fopl_rule']}")
        
        # Compliance case
        explanation.append("\n4️⃣ ACTUAL SITUATION:")
        for party, facts in result['input']['compliance_case'].items():
            for fact, value in facts.items():
                explanation.append(f"   • {party}: {fact} = {value}")
        
        # Reasoning steps
        explanation.append("\n5️⃣ REASONING PROCESS:")
        for i, step in enumerate(result['reasoning']['proof_trace'], 1):
            explanation.append(f"   {i}. {step}")
        
        # Final decision
        explanation.append("\n6️⃣ DECISION:")
        outcome = "✅ COMPLIANT" if result['reasoning']['outcome'] else "❌ NON-COMPLIANT"
        explanation.append(f"   {outcome}")
        explanation.append(f"   {result['reasoning']['explanation']}")
        
        explanation.append("\n" + "="*60)
        
        return "\n".join(explanation)
    
    def save_results(self, results: List[Dict], output_path: str):
        """Save reasoning results to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Results saved to {output_path}")


def main():
    """Demo of the complete pipeline"""
    
    # Initialize pipeline (using untrained model for demo)
    pipeline = LegalReasoningPipeline()
    
    # Test cases
    test_cases = [
        {
            'clause_text': "The tenant must pay rent by the 5th of each month.",
            'context': {"Tenant": "PartyA", "Landlord": "PartyB"},
            'compliance_case': {"PartyA": {"PayRentDate": 4}}
        },
        {
            'clause_text': "The tenant must pay rent by the 5th of each month.",
            'context': {"Tenant": "PartyA", "Landlord": "PartyB"},
            'compliance_case': {"PartyA": {"PayRentDate": 10}}
        },
        {
            'clause_text': "Either party may terminate with 30 days written notice.",
            'context': {"Party": "PartyA"},
            'compliance_case': {"PartyA": {"NoticeGiven": 45}}
        }
    ]
    
    # Process all cases
    results = pipeline.batch_process(test_cases)
    
    # Generate explanations
    print("\n\n" + "#"*60)
    print("DETAILED EXPLANATIONS")
    print("#"*60)
    
    for i, result in enumerate(results, 1):
        print(f"\n\nCASE {i}:")
        print(pipeline.explain_reasoning(result))
    
    # Save results
    pipeline.save_results(results, 'reasoning_results.json')
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    total = len(results)
    compliant = sum(1 for r in results if r['reasoning']['outcome'])
    non_compliant = total - compliant
    
    print(f"Total cases processed: {total}")
    print(f"Compliant: {compliant} ({compliant/total*100:.1f}%)")
    print(f"Non-compliant: {non_compliant} ({non_compliant/total*100:.1f}%)")
    print("="*60)


if __name__ == "__main__":
    main()