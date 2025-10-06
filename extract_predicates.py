"""
Extract all unique predicates from FOPL dataset to add to tokenizer vocabulary
"""
import json
import re
from pathlib import Path


def extract_predicates(data_path: str):
    """Extract all unique predicate names from FOPL rules"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    predicates = set()
    
    for item in data:
        fopl = item['fopl_rule']
        # Find all predicate names (capitalized word followed by opening paren)
        matches = re.findall(r'([A-Z][a-zA-Z]+)\s*\(', fopl)
        predicates.update(matches)
    
    return sorted(list(predicates))


def main():
    data_path = 'data/legal_clauses.json'
    
    predicates = extract_predicates(data_path)
    
    print(f"Found {len(predicates)} unique predicates")
    print("\nPredicates:")
    print("=" * 80)
    
    for i, pred in enumerate(predicates, 1):
        print(f"{i:3d}. {pred}")
    
    # Save to file
    output_path = 'data/predicates.txt'
    with open(output_path, 'w') as f:
        f.write('\n'.join(predicates))
    
    print(f"\n✅ Saved to {output_path}")
    
    # Also create a Python list format
    output_py = 'data/predicates.py'
    with open(output_py, 'w') as f:
        f.write("# Auto-generated list of FOPL predicates\n")
        f.write("PREDICATES = [\n")
        for pred in predicates:
            f.write(f"    '{pred}',\n")
        f.write("]\n")
    
    print(f"✅ Saved to {output_py}")


if __name__ == '__main__':
    main()
