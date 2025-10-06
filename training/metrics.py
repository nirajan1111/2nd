import re
from typing import List, Dict
from collections import Counter
import numpy as np


def compute_metrics(predictions: List[str], references: List[str]) -> Dict:
    """
    Compute evaluation metrics for FOPL generation
    
    Args:
        predictions: List of predicted FOPL strings
        references: List of reference FOPL strings
        
    Returns:
        Dictionary with metrics
    """
    metrics = {}
    
    # Exact match accuracy
    exact_matches = sum(
        normalize_fopl(pred) == normalize_fopl(ref)
        for pred, ref in zip(predictions, references)
    )
    metrics['exact_match'] = exact_matches / len(predictions)
    
    # Token-level accuracy
    token_accuracy = compute_token_accuracy(predictions, references)
    metrics['token_accuracy'] = token_accuracy
    
    # BLEU score
    bleu = compute_bleu(predictions, references)
    metrics['bleu'] = bleu
    
    # Predicate accuracy (specific to FOPL)
    pred_acc = compute_predicate_accuracy(predictions, references)
    metrics['predicate_accuracy'] = pred_acc
    
    # Syntax validity
    syntax_valid = sum(is_valid_fopl(pred) for pred in predictions)
    metrics['syntax_validity'] = syntax_valid / len(predictions)
    
    return metrics


def normalize_fopl(fopl_string: str) -> str:
    """Normalize FOPL string for comparison"""
    # Remove extra whitespace
    fopl_string = re.sub(r'\s+', ' ', fopl_string.strip())
    # Standardize operators
    fopl_string = fopl_string.replace(' & ', '&').replace(' | ', '|')
    fopl_string = fopl_string.replace(' -> ', '->')
    return fopl_string


def compute_token_accuracy(predictions: List[str], references: List[str]) -> float:
    """Compute token-level accuracy"""
    total_tokens = 0
    correct_tokens = 0
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        
        # Align tokens (simple approach)
        max_len = max(len(pred_tokens), len(ref_tokens))
        for i in range(max_len):
            total_tokens += 1
            if i < len(pred_tokens) and i < len(ref_tokens):
                if pred_tokens[i] == ref_tokens[i]:
                    correct_tokens += 1
    
    return correct_tokens / total_tokens if total_tokens > 0 else 0.0


def compute_bleu(predictions: List[str], references: List[str], 
                 max_n: int = 4) -> float:
    """
    Compute BLEU score (simplified implementation)
    """
    from collections import defaultdict
    
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        
        # Compute n-gram precision for n=1 to max_n
        precisions = []
        for n in range(1, max_n + 1):
            pred_ngrams = get_ngrams(pred_tokens, n)
            ref_ngrams = get_ngrams(ref_tokens, n)
            
            if len(pred_ngrams) == 0:
                precisions.append(0.0)
                continue
            
            # Count matches
            matches = 0
            for ngram in pred_ngrams:
                if ngram in ref_ngrams:
                    matches += min(pred_ngrams[ngram], ref_ngrams[ngram])
            
            precision = matches / sum(pred_ngrams.values())
            precisions.append(precision)
        
        # Compute geometric mean
        if all(p > 0 for p in precisions):
            score = np.exp(np.mean([np.log(p) for p in precisions]))
            
            # Brevity penalty
            bp = 1.0
            if len(pred_tokens) < len(ref_tokens):
                bp = np.exp(1 - len(ref_tokens) / len(pred_tokens))
            
            score *= bp
        else:
            score = 0.0
        
        scores.append(score)
    
    return np.mean(scores)


def get_ngrams(tokens: List[str], n: int) -> Dict[tuple, int]:
    """Extract n-grams from token list"""
    ngrams = Counter()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams[ngram] += 1
    return ngrams


def compute_predicate_accuracy(predictions: List[str], 
                               references: List[str]) -> float:
    """Compute accuracy of predicate extraction"""
    total = 0
    correct = 0
    
    for pred, ref in zip(predictions, references):
        pred_predicates = extract_predicates(pred)
        ref_predicates = extract_predicates(ref)
        
        if len(ref_predicates) > 0:
            total += len(ref_predicates)
            for p in pred_predicates:
                if p in ref_predicates:
                    correct += 1
    
    return correct / total if total > 0 else 0.0


def extract_predicates(fopl_string: str) -> List[str]:
    """Extract predicates from FOPL string"""
    # Pattern: CapitalizedWord followed by parentheses
    pattern = r'[A-Z][a-zA-Z]*\([^)]*\)'
    predicates = re.findall(pattern, fopl_string)
    return list(set(predicates))


def is_valid_fopl(fopl_string: str) -> bool:
    """Check if FOPL string has valid syntax"""
    try:
        # Check balanced parentheses
        if fopl_string.count('(') != fopl_string.count(')'):
            return False
        
        # Check for quantifiers
        if not any(q in fopl_string for q in ['forall', 'exists']):
            return False
        
        # Check for predicates (at least one capitalized word with parentheses)
        if not re.search(r'[A-Z][a-zA-Z]*\(', fopl_string):
            return False
        
        return True
    except:
        return False


def compute_structural_similarity(pred: str, ref: str) -> float:
    """
    Compute structural similarity between predicted and reference FOPL
    Based on matching of quantifiers, operators, and predicate structure
    """
    score = 0.0
    total = 0.0
    
    # Check quantifier match
    total += 1
    pred_quant = 'forall' if 'forall' in pred else 'exists' if 'exists' in pred else None
    ref_quant = 'forall' if 'forall' in ref else 'exists' if 'exists' in ref else None
    if pred_quant == ref_quant:
        score += 1
    
    # Check operator presence
    operators = ['->', '&', '|', '~']
    for op in operators:
        total += 1
        if (op in pred) == (op in ref):
            score += 1
    
    # Check number of predicates
    total += 1
    pred_preds = len(extract_predicates(pred))
    ref_preds = len(extract_predicates(ref))
    if pred_preds == ref_preds:
        score += 1
    elif pred_preds > 0 and ref_preds > 0:
        score += min(pred_preds, ref_preds) / max(pred_preds, ref_preds)
    
    return score / total if total > 0 else 0.0


def print_metrics_report(metrics: Dict, title: str = "Evaluation Metrics"):
    """Print formatted metrics report"""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            if 'accuracy' in metric_name or 'match' in metric_name:
                print(f"{metric_name:.<40} {value:>7.2%}")
            else:
                print(f"{metric_name:.<40} {value:>7.4f}")
        else:
            print(f"{metric_name:.<40} {value}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test metrics
    predictions = [
        "forall x (Tenant(x) -> PayRent(x, due_date <= 5))",
        "forall x (Buyer(x) -> PayAmount(x, 1000, event=signing))",
        "forall x, y (Tenant(x) & Landlord(y) & ~PayRent(x) -> RightToTerminate(y, x))"
    ]
    
    references = [
        "forall x (Tenant(x) -> PayRent(x, due_date <= 5))",
        "forall x (Buyer(x) -> PayAmount(x, 5000, event=signing))",
        "forall x, y (Tenant(x) & Landlord(y) & ~PayRent(x) -> RightToTerminate(y, x))"
    ]
    
    metrics = compute_metrics(predictions, references)
    print_metrics_report(metrics, "Test Metrics")
    
    # Test individual functions
    print("\nTesting predicate extraction:")
    for pred in predictions:
        predicates = extract_predicates(pred)
        print(f"  {pred}")
        print(f"  → {predicates}\n")
    
    print("Testing FOPL validity:")
    test_cases = [
        "forall x (Tenant(x) -> PayRent(x))",
        "invalid fopl without quantifier",
        "forall x Tenant(x",  # Unbalanced
    ]
    for test in test_cases:
        valid = is_valid_fopl(test)
        print(f"  {test}")
        print(f"  → Valid: {valid}\n")