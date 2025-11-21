"""
Test and Inference Script for T5 FOPL Model
Evaluates trained model and provides inference examples
"""

import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Dict
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.neural_parser import FOPLValidator


class T5FOPLInference:
    """Inference wrapper for trained T5 FOPL model"""
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: Path to trained model checkpoint
        """
        print(f"📦 Loading model from {model_path}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded on {self.device}")
        
        self.validator = FOPLValidator()
    
    def predict(self, clause_text: str, context: Dict = None, 
                num_beams: int = 4, max_length: int = 512) -> Dict:
        """
        Generate FOPL from legal clause
        
        Returns:
            Dict with prediction, confidence, and validation results
        """
        # Prepare input
        input_text = f"translate to english logic: {clause_text}"
        if context:
            context_str = " ".join([f"{k}={v}" for k, v in context.items()])
            input_text += f" context: {context_str}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode
        predicted_fopl = self.tokenizer.decode(
            outputs.sequences[0],
            skip_special_tokens=True
        ).strip()
        
        # Validate
        is_valid, validation_msg = self.validator.validate(predicted_fopl)
        predicates = self.validator.extract_predicates(predicted_fopl)
        
        # Compute confidence (average log probability)
        if hasattr(outputs, 'sequences_scores'):
            confidence = torch.exp(outputs.sequences_scores[0]).item()
        else:
            confidence = 0.0
        
        return {
            'input': clause_text,
            'context': context,
            'predicted_fopl': predicted_fopl,
            'is_valid': is_valid,
            'validation_message': validation_msg,
            'predicates': predicates,
            'confidence': confidence
        }
    
    def batch_predict(self, clauses: List[str], contexts: List[Dict] = None,
                      batch_size: int = 8) -> List[Dict]:
        """Batch prediction for multiple clauses"""
        if contexts is None:
            contexts = [None] * len(clauses)
        
        results = []
        for i in range(0, len(clauses), batch_size):
            batch_clauses = clauses[i:i+batch_size]
            batch_contexts = contexts[i:i+batch_size]
            
            for clause, context in zip(batch_clauses, batch_contexts):
                result = self.predict(clause, context)
                results.append(result)
        
        return results


def evaluate_on_test_set(model_path: str, test_data_path: str, 
                         output_path: str = None):
    """
    Evaluate trained model on test set
    
    Args:
        model_path: Path to trained model
        test_data_path: Path to test data JSON
        output_path: Path to save results (optional)
    """
    print("=" * 80)
    print("  T5 FOPL MODEL EVALUATION")
    print("=" * 80)
    
    # Load model
    inference = T5FOPLInference(model_path)
    
    # Load test data
    print(f"\n📚 Loading test data from {test_data_path}...")
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(test_data)} test examples")
    
    # Evaluate
    print(f"\n🔬 Evaluating...")
    results = []
    exact_matches = 0
    valid_syntax = 0
    
    for i, item in enumerate(test_data):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(test_data)}")
        
        # Predict
        prediction = inference.predict(
            item['clause_text'],
            item.get('context')
        )
        
        # Compare with ground truth
        ground_truth = item['fopl_rule']
        is_exact_match = prediction['predicted_fopl'] == ground_truth
        
        if is_exact_match:
            exact_matches += 1
        
        if prediction['is_valid']:
            valid_syntax += 1
        
        results.append({
            'clause_id': item['id'],
            'clause_type': item.get('clause_type', 'unknown'),
            'input': item['clause_text'],
            'ground_truth': ground_truth,
            'prediction': prediction['predicted_fopl'],
            'exact_match': is_exact_match,
            'valid_syntax': prediction['is_valid'],
            'confidence': prediction['confidence']
        })
    
    # Compute metrics
    total = len(test_data)
    exact_match_acc = exact_matches / total
    syntax_accuracy = valid_syntax / total
    
    print(f"\n{'='*80}")
    print("  EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Total Examples: {total}")
    print(f"Exact Match Accuracy: {exact_match_acc:.2%} ({exact_matches}/{total})")
    print(f"Syntax Validity: {syntax_accuracy:.2%} ({valid_syntax}/{total})")
    
    # Compute per-clause-type metrics
    print(f"\n📊 Per-Clause-Type Results:")
    clause_types = {}
    for result in results:
        ctype = result['clause_type']
        if ctype not in clause_types:
            clause_types[ctype] = {'total': 0, 'exact': 0, 'valid': 0}
        
        clause_types[ctype]['total'] += 1
        if result['exact_match']:
            clause_types[ctype]['exact'] += 1
        if result['valid_syntax']:
            clause_types[ctype]['valid'] += 1
    
    for ctype, stats in sorted(clause_types.items()):
        exact_pct = stats['exact'] / stats['total'] * 100
        valid_pct = stats['valid'] / stats['total'] * 100
        print(f"  {ctype:20s}: Exact={exact_pct:5.1f}%  Valid={valid_pct:5.1f}%  (n={stats['total']})")
    
    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump({
                'metrics': {
                    'total_examples': total,
                    'exact_match_accuracy': exact_match_acc,
                    'syntax_accuracy': syntax_accuracy
                },
                'per_clause_type': clause_types,
                'predictions': results
            }, f, indent=2)
        print(f"\n✅ Results saved to {output_path}")
    
    return results, exact_match_acc, syntax_accuracy


def interactive_demo(model_path: str):
    """Interactive demo for testing model"""
    print("=" * 80)
    print("  T5 FOPL INTERACTIVE DEMO")
    print("=" * 80)
    print("Enter legal clauses to convert to FOPL (type 'quit' to exit)")
    print()
    
    # Load model
    inference = T5FOPLInference(model_path)
    
    while True:
        print("-" * 80)
        clause = input("📝 Enter legal clause: ").strip()
        
        if clause.lower() in ['quit', 'exit', 'q']:
            break
        
        if not clause:
            continue
        
        # Optional context
        context_str = input("📋 Enter context (e.g., 'Tenant=PartyA' or press Enter to skip): ").strip()
        context = None
        if context_str:
            context = {}
            for pair in context_str.split():
                if '=' in pair:
                    k, v = pair.split('=', 1)
                    context[k.strip()] = v.strip()
        
        # Predict
        result = inference.predict(clause, context)
        
        print(f"\n{'='*80}")
        print("  RESULT")
        print(f"{'='*80}")
        print(f"📥 Input: {result['input']}")
        if result['context']:
            print(f"📋 Context: {result['context']}")
        print(f"📤 FOPL: {result['predicted_fopl']}")
        print(f"✓ Valid: {result['is_valid']} - {result['validation_message']}")
        print(f"📊 Confidence: {result['confidence']:.2%}")
        print(f"🔍 Predicates: {', '.join(result['predicates'])}")
        print()


def run_examples(model_path: str):
    """Run example predictions"""
    print("=" * 80)
    print("  T5 FOPL EXAMPLE PREDICTIONS")
    print("=" * 80)
    
    # Load model
    inference = T5FOPLInference(model_path)
    
    # Example clauses
    examples = [
        {
            'clause': "The tenant must pay rent by the 5th of each month.",
            'context': {"Tenant": "PartyA", "Landlord": "PartyB"}
        },
        {
            'clause': "Supplier shall deliver goods within 10 business days.",
            'context': {"Supplier": "CompanyX"}
        },
        {
            'clause': "Either party may terminate with 30 days written notice.",
            'context': {"Party": "PartyA"}
        },
        {
            'clause': "Employee must maintain confidentiality for 2 years after termination.",
            'context': {"Employee": "John"}
        },
        {
            'clause': "Buyer agrees to purchase minimum 1000 units per order.",
            'context': {"Buyer": "CompanyZ"}
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{'='*80}")
        print(f"  EXAMPLE {i}")
        print(f"{'='*80}")
        
        result = inference.predict(example['clause'], example['context'])
        
        print(f"📥 Input: {result['input']}")
        print(f"📋 Context: {result['context']}")
        print(f"📤 FOPL: {result['predicted_fopl']}")
        print(f"✓ Valid: {result['is_valid']} - {result['validation_message']}")
        print(f"📊 Confidence: {result['confidence']:.2%}")
        print(f"🔍 Predicates: {', '.join(result['predicates'])}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test T5 FOPL Model")
    parser.add_argument('--model_path', type=str, default='checkpoints/t5_fopl/final',
                       help='Path to trained model')
    parser.add_argument('--mode', type=str, choices=['eval', 'demo', 'examples'],
                       default='examples',
                       help='Run mode: eval (test set), demo (interactive), examples')
    parser.add_argument('--test_data', type=str, default='data/legal_clauses.json',
                       help='Path to test data (for eval mode)')
    parser.add_argument('--output', type=str, default='results/t5_fopl_evaluation.json',
                       help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    if args.mode == 'eval':
        evaluate_on_test_set(args.model_path, args.test_data, args.output)
    elif args.mode == 'demo':
        interactive_demo(args.model_path)
    else:  # examples
        run_examples(args.model_path)
