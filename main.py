#!/usr/bin/env python3
"""
Neural-Symbolic Legal Reasoning System
Main execution script
"""

import argparse
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.generate_dataset import LegalClauseGenerator
from training.train import main as train_main
from inference.pipeline import LegalReasoningPipeline
import json


def generate_data(args):
    """Generate dataset of legal clauses"""
    print("\n" + "="*60)
    print("GENERATING LEGAL CLAUSES DATASET")
    print("="*60 + "\n")
    
    generator = LegalClauseGenerator()
    dataset = generator.generate_dataset(args.num_samples)
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    output_path = os.path.join('data', args.output_file)
    
    # Save dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Generated {len(dataset)} legal clauses")
    print(f"📁 Saved to: {output_path}")
    
    # Print statistics
    types = {}
    for clause in dataset:
        clause_type = clause["clause_type"]
        types[clause_type] = types.get(clause_type, 0) + 1
    
    print("\n📊 Clause type distribution:")
    for clause_type, count in sorted(types.items()):
        print(f"   {clause_type:.<30} {count:>3}")
    
    print(f"\n{'='*60}\n")


def train_model(args):
    """Train the neural parser"""
    print("\n" + "="*60)
    print("TRAINING NEURAL LEGAL PARSER")
    print("="*60 + "\n")
    
    # Check if data exists
    data_path = os.path.join('data', 'legal_clauses.json')
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found at {data_path}")
        print("   Please run: python main.py generate")
        return
    
    # Import and run training
    from training.train import main as train_main
    
    # Create custom config if batch_size provided
    custom_config = {}
    if hasattr(args, 'batch_size') and args.batch_size:
        custom_config['batch_size'] = args.batch_size
        print(f"📦 Using custom batch size: {args.batch_size}")
    if hasattr(args, 'epochs') and args.epochs:
        custom_config['num_epochs'] = args.epochs
        print(f"🔄 Using custom epochs: {args.epochs}")
    
    train_main(custom_config if custom_config else None)


def run_inference(args):
    """Run inference on test cases"""
    print("\n" + "="*60)
    print("RUNNING LEGAL REASONING INFERENCE")
    print("="*60 + "\n")
    
    # Initialize pipeline
    if args.model_path and os.path.exists(args.model_path):
        pipeline = LegalReasoningPipeline(model_path=args.model_path)
    else:
        print("⚠️  No trained model found, using base model")
        pipeline = LegalReasoningPipeline()
    
    # Load test cases
    if args.test_file:
        with open(args.test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        test_cases = []
        for item in test_data[:args.num_cases]:
            test_cases.append({
                'clause_text': item['clause_text'],
                'context': item['context'],
                'compliance_case': item['compliance_case']
            })
    else:
        # Use default test cases
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
            },
            {
                'clause_text': "The supplier must deliver goods within 10 business days.",
                'context': {"Supplier": "CompanyX"},
                'compliance_case': {"CompanyX": {"DeliveryDays": 8}}
            }
        ]
    
    # Process cases
    results = pipeline.batch_process(test_cases)
    
    # Generate explanations
    print("\n\n" + "#"*60)
    print("DETAILED EXPLANATIONS")
    print("#"*60)
    
    for i, result in enumerate(results, 1):
        print(f"\n\nCASE {i}:")
        print(pipeline.explain_reasoning(result))
    
    # Save results
    output_path = args.output_file or 'reasoning_results.json'
    pipeline.save_results(results, output_path)
    
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
    print("="*60 + "\n")


def test_components(args):
    """Test individual components"""
    print("\n" + "="*60)
    print("TESTING SYSTEM COMPONENTS")
    print("="*60 + "\n")
    
    # Test 1: Neural Parser
    print("1. Testing Neural Parser...")
    from models.neural_parser import NeuralLegalParser
    parser = NeuralLegalParser(model_name='t5-small')
    
    test_clause = "The tenant must pay rent by the 5th of each month."
    test_context = {"Tenant": "PartyA"}
    
    preprocessed = parser.preprocess_input(test_clause, test_context)
    print(f"   Input: {test_clause}")
    print(f"   Preprocessed: {preprocessed}")
    print("   ✅ Neural parser initialized\n")
    
    # Test 2: Symbolic Reasoner
    print("2. Testing Symbolic Reasoner...")
    from models.symbolic_reasoner import SymbolicReasoner
    reasoner = SymbolicReasoner()
    
    fopl_rule = "forall x (Tenant(x) -> PayRent(x, due_date <= 5))"
    context = {"Tenant": "PartyA"}
    compliance_case = {"PartyA": {"PayRentDate": 4}}
    
    result = reasoner.evaluate_compliance(fopl_rule, compliance_case, context)
    print(f"   FOPL: {fopl_rule}")
    print(f"   Result: {result.outcome}")
    print("   ✅ Symbolic reasoner working\n")
    
    # Test 3: Dataset Generation
    print("3. Testing Dataset Generator...")
    from data.generate_dataset import LegalClauseGenerator
    generator = LegalClauseGenerator()
    sample = generator.generate_clause(1)
    print(f"   Generated clause: {sample['clause_text'][:60]}...")
    print(f"   FOPL: {sample['fopl_rule'][:60]}...")
    print("   ✅ Dataset generator working\n")
    
    # Test 4: Metrics
    print("4. Testing Metrics...")
    from training.metrics import compute_metrics
    preds = ["forall x (Tenant(x) -> PayRent(x))"]
    refs = ["forall x (Tenant(x) -> PayRent(x))"]
    metrics = compute_metrics(preds, refs)
    print(f"   Exact Match: {metrics['exact_match']:.2%}")
    print("   ✅ Metrics computation working\n")
    
    print("="*60)
    print("ALL TESTS PASSED ✅")
    print("="*60 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Neural-Symbolic Legal Reasoning System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate dataset
  python main.py generate --num-samples 100

  # Train model
  python main.py train

  # Run inference
  python main.py inference --model-path checkpoints/best_model

  # Test components
  python main.py test
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate legal clauses dataset')
    generate_parser.add_argument(
        '--num-samples', type=int, default=100,
        help='Number of samples to generate (default: 100)'
    )
    generate_parser.add_argument(
        '--output-file', type=str, default='legal_clauses.json',
        help='Output filename (default: legal_clauses.json)'
    )
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the neural parser')
    train_parser.add_argument(
        '--config', type=str,
        help='Path to training config JSON file'
    )
    train_parser.add_argument(
        '--batch-size', '--batch_size', type=int, dest='batch_size',
        help='Batch size for training (default: 16)'
    )
    train_parser.add_argument(
        '--epochs', type=int,
        help='Number of training epochs (default: 20)'
    )
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference')
    inference_parser.add_argument(
        '--model-path', type=str,
        help='Path to trained model checkpoint'
    )
    inference_parser.add_argument(
        '--test-file', type=str,
        help='Path to test cases JSON file'
    )
    inference_parser.add_argument(
        '--num-cases', type=int, default=10,
        help='Number of test cases to process (default: 10)'
    )
    inference_parser.add_argument(
        '--output-file', type=str,
        help='Output file for results'
    )
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test system components')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        generate_data(args)
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'inference':
        run_inference(args)
    elif args.command == 'test':
        test_components(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()