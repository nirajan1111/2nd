"""
Train/Test Split Utilities for Legal Reasoning Dataset
Handles splitting legal clauses dataset into train/validation/test sets
"""

import json
import random
import os
from typing import List, Dict, Tuple
import argparse
from collections import Counter


class DatasetSplitter:
    """
    Splits legal clauses dataset into train/validation/test sets
    with stratification by clause type to ensure balanced representation
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the dataset splitter
        
        Args:
            random_seed: Random seed for reproducible splits
        """
        self.random_seed = random_seed
        random.seed(random_seed)
    
    def load_dataset(self, data_path: str) -> List[Dict]:
        """
        Load dataset from JSON file
        
        Args:
            data_path: Path to the legal clauses JSON file
            
        Returns:
            List of legal clause dictionaries
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print(f"✅ Loaded {len(dataset)} samples from {data_path}")
        return dataset
    
    def analyze_dataset(self, dataset: List[Dict]) -> Dict:
        """
        Analyze dataset distribution by clause types and other features
        
        Args:
            dataset: List of legal clause dictionaries
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'total_samples': len(dataset),
            'clause_types': Counter(),
            'avg_text_length': 0,
            'avg_fopl_length': 0,
            'unique_predicates': set(),
            'variables_used': Counter()
        }
        
        total_text_len = 0
        total_fopl_len = 0
        
        for item in dataset:
            # Clause type distribution
            analysis['clause_types'][item.get('clause_type', 'unknown')] += 1
            
            # Text lengths
            text_len = len(item.get('clause_text', ''))
            fopl_len = len(item.get('fopl_rule', ''))
            total_text_len += text_len
            total_fopl_len += fopl_len
            
            # Predicates
            predicates = item.get('predicates_used', [])
            analysis['unique_predicates'].update(predicates)
            
            # Variables
            variables = item.get('variables', {})
            for var, role in variables.items():
                analysis['variables_used'][role] += 1
        
        analysis['avg_text_length'] = total_text_len / len(dataset)
        analysis['avg_fopl_length'] = total_fopl_len / len(dataset)
        analysis['unique_predicates'] = list(analysis['unique_predicates'])
        
        return analysis
    
    def stratified_split(self, dataset: List[Dict], 
                        train_ratio: float = 0.8,
                        val_ratio: float = 0.1,
                        test_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split dataset with stratification by clause type
        
        Args:
            dataset: List of legal clause dictionaries
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set  
            test_ratio: Proportion for test set
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        # Group by clause type
        type_groups = {}
        for item in dataset:
            clause_type = item.get('clause_type', 'unknown')
            if clause_type not in type_groups:
                type_groups[clause_type] = []
            type_groups[clause_type].append(item)
        
        print(f"📊 Found {len(type_groups)} clause types:")
        for clause_type, items in type_groups.items():
            print(f"   {clause_type}: {len(items)} samples")
        
        train_data = []
        val_data = []
        test_data = []
        
        # Split each type proportionally
        for clause_type, items in type_groups.items():
            # Shuffle items within type
            random.shuffle(items)
            
            n_items = len(items)
            n_train = int(n_items * train_ratio)
            n_val = int(n_items * val_ratio)
            n_test = n_items - n_train - n_val
            
            # Split
            train_data.extend(items[:n_train])
            val_data.extend(items[n_train:n_train + n_val])
            test_data.extend(items[n_train + n_val:])
        
        # Final shuffle
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)
        
        print(f"\n✅ Split completed:")
        print(f"   Training: {len(train_data)} samples ({len(train_data)/len(dataset):.1%})")
        print(f"   Validation: {len(val_data)} samples ({len(val_data)/len(dataset):.1%})")
        print(f"   Test: {len(test_data)} samples ({len(test_data)/len(dataset):.1%})")
        
        return train_data, val_data, test_data
    
    def save_splits(self, train_data: List[Dict], val_data: List[Dict], 
                   test_data: List[Dict], output_dir: str = 'data/splits'):
        """
        Save train/val/test splits to separate JSON files
        
        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            output_dir: Directory to save split files
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each split
        splits = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        for split_name, data in splits.items():
            output_path = os.path.join(output_dir, f'{split_name}.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"📁 Saved {len(data)} {split_name} samples to {output_path}")
    
    def create_combined_file(self, dataset: List[Dict], output_path: str):
        """
        Create a combined file with split indicators
        
        Args:
            dataset: Complete dataset with split labels
            output_path: Path to save combined file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"📁 Saved combined dataset with splits to {output_path}")
    
    def add_split_labels(self, train_data: List[Dict], val_data: List[Dict], 
                        test_data: List[Dict]) -> List[Dict]:
        """
        Add split labels to dataset items
        
        Returns:
            Combined dataset with 'split' field added
        """
        # Add split labels
        for item in train_data:
            item['split'] = 'train'
        for item in val_data:
            item['split'] = 'val'
        for item in test_data:
            item['split'] = 'test'
        
        # Combine all
        combined = train_data + val_data + test_data
        
        # Sort by ID to maintain order
        combined.sort(key=lambda x: x.get('id', ''))
        
        return combined
    
    def validate_splits(self, train_data: List[Dict], val_data: List[Dict], 
                       test_data: List[Dict]) -> Dict:
        """
        Validate that splits maintain clause type distribution
        
        Returns:
            Dictionary with validation results
        """
        def get_distribution(data):
            types = Counter(item.get('clause_type', 'unknown') for item in data)
            total = len(data)
            return {t: count/total for t, count in types.items()}
        
        train_dist = get_distribution(train_data)
        val_dist = get_distribution(val_data)
        test_dist = get_distribution(test_data)
        
        validation = {
            'train_distribution': train_dist,
            'val_distribution': val_dist,
            'test_distribution': test_dist,
            'distribution_similarity': {}
        }
        
        # Check similarity between distributions
        all_types = set(train_dist.keys()) | set(val_dist.keys()) | set(test_dist.keys())
        
        for clause_type in all_types:
            train_pct = train_dist.get(clause_type, 0)
            val_pct = val_dist.get(clause_type, 0)
            test_pct = test_dist.get(clause_type, 0)
            
            # Calculate coefficient of variation
            percentages = [train_pct, val_pct, test_pct]
            mean_pct = sum(percentages) / len(percentages)
            if mean_pct > 0:
                cv = (sum((p - mean_pct)**2 for p in percentages) / len(percentages))**0.5 / mean_pct
                validation['distribution_similarity'][clause_type] = cv
        
        return validation
    
    def print_analysis(self, analysis: Dict):
        """Print dataset analysis in a formatted way"""
        print("\n" + "="*60)
        print("DATASET ANALYSIS")
        print("="*60)
        
        print(f"Total samples: {analysis['total_samples']}")
        print(f"Average text length: {analysis['avg_text_length']:.1f} characters")
        print(f"Average FOPL length: {analysis['avg_fopl_length']:.1f} characters")
        print(f"Unique predicates: {len(analysis['unique_predicates'])}")
        
        print(f"\nClause type distribution:")
        for clause_type, count in analysis['clause_types'].most_common():
            percentage = count / analysis['total_samples'] * 100
            print(f"   {clause_type:.<25} {count:>3} ({percentage:>5.1f}%)")
        
        print(f"\nVariable roles used:")
        for role, count in analysis['variables_used'].most_common():
            print(f"   {role:.<25} {count:>3}")
        
        print("="*60 + "\n")
    
    def print_validation(self, validation: Dict):
        """Print split validation results"""
        print("\n" + "="*60)
        print("SPLIT VALIDATION")
        print("="*60)
        
        print("Distribution similarity (lower is better):")
        for clause_type, cv in validation['distribution_similarity'].items():
            status = "✅" if cv < 0.2 else "⚠️" if cv < 0.5 else "❌"
            print(f"   {clause_type:.<25} {cv:>6.3f} {status}")
        
        print("="*60 + "\n")


def main():
    """Main function for train/test splitting"""
    parser = argparse.ArgumentParser(description='Split legal clauses dataset')
    parser.add_argument(
        '--input', '-i', type=str, default='data/legal_clauses.json',
        help='Input dataset file (default: data/legal_clauses.json)'
    )
    parser.add_argument(
        '--output-dir', '-o', type=str, default='data/splits',
        help='Output directory for split files (default: data/splits)'
    )
    parser.add_argument(
        '--train-ratio', type=float, default=0.8,
        help='Training set ratio (default: 0.8)'
    )
    parser.add_argument(
        '--val-ratio', type=float, default=0.1,
        help='Validation set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--test-ratio', type=float, default=0.1,
        help='Test set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducible splits (default: 42)'
    )
    parser.add_argument(
        '--analyze-only', action='store_true',
        help='Only analyze dataset without splitting'
    )
    parser.add_argument(
        '--combined-output', type=str,
        help='Save combined dataset with split labels to this file'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        print("   Please generate the dataset first:")
        print("   python data/generate_dataset.py")
        return
    
    # Initialize splitter
    splitter = DatasetSplitter(random_seed=args.seed)
    
    # Load and analyze dataset
    dataset = splitter.load_dataset(args.input)
    analysis = splitter.analyze_dataset(dataset)
    splitter.print_analysis(analysis)
    
    if args.analyze_only:
        print("Analysis complete. Use --analyze-only flag to skip splitting.")
        return
    
    # Perform split
    print(f"Splitting with ratios: {args.train_ratio}/{args.val_ratio}/{args.test_ratio}")
    train_data, val_data, test_data = splitter.stratified_split(
        dataset, args.train_ratio, args.val_ratio, args.test_ratio
    )
    
    # Validate splits
    validation = splitter.validate_splits(train_data, val_data, test_data)
    splitter.print_validation(validation)
    
    # Save splits
    splitter.save_splits(train_data, val_data, test_data, args.output_dir)
    
    # Save combined file if requested
    if args.combined_output:
        combined_dataset = splitter.add_split_labels(train_data, val_data, test_data)
        splitter.create_combined_file(combined_dataset, args.combined_output)
    
    print(f"\n✅ Dataset splitting completed successfully!")
    print(f"📂 Split files saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
