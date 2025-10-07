"""
CUAD Data Processor
Converts CUAD_v1.json to training format for RoBERTa clause extraction model.

Usage:
    python process_cuad_data.py --input ../CUAD_v1/CUAD_v1.json --output ../cuad_processed/
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import random
import argparse
from tqdm import tqdm


class CUADProcessor:
    """Process CUAD dataset for clause extraction training."""
    
    CUAD_CATEGORIES = [
        "Document Name", "Parties", "Agreement Date", "Effective Date",
        "Expiration Date", "Renewal Term", "Notice Period To Terminate Renewal",
        "Governing Law", "Most Favored Nation", "Non-Compete", "Exclusivity",
        "No-Solicit Of Employees", "Non-Disparagement", "Termination For Convenience",
        "Rofr/Rofo/Rofn", "Change Of Control", "Anti-Assignment",
        "Revenue/Profit Sharing", "Price Restrictions", "Minimum Commitment",
        "Volume Restriction", "IP Ownership Assignment", "Joint IP Ownership",
        "License Grant", "Non-Transferable License", "Affiliate License-Licensee",
        "Affiliate License-Licensor", "Unlimited/All-You Can-Eat-License",
        "Irrevocable Or Perpetual License", "Source Code Escrow",
        "Post-Termination Services", "Audit Rights", "Uncapped Liability",
        "Cap On Liability", "Liquidated Damages", "Warranty Duration",
        "Insurance", "Covenant Not To Sue", "Third Party Beneficiary"
    ]
    
    # Map CUAD categories to our clause types for FOPL generation
    CATEGORY_TO_CLAUSE_TYPE = {
        "Minimum Commitment": "obligation",
        "Termination For Convenience": "termination",
        "Governing Law": "jurisdiction",
        "Expiration Date": "temporal",
        "Renewal Term": "temporal",
        "Notice Period To Terminate Renewal": "temporal",
        "Exclusivity": "exclusive_rights",
        "Non-Compete": "restriction",
        "Anti-Assignment": "restriction",
        "Revenue/Profit Sharing": "payment",
        "Price Restrictions": "payment",
        "Volume Restriction": "volume",
        "License Grant": "license",
        "Audit Rights": "audit",
        "Cap On Liability": "liability",
        "Warranty Duration": "warranty",
        "Insurance": "insurance",
    }
    
    def __init__(self, input_path: str, output_dir: str):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = defaultdict(int)
        
    def load_cuad_json(self) -> Dict:
        """Load CUAD_v1.json file."""
        print(f"Loading CUAD data from {self.input_path}...")
        with open(self.input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✓ Loaded CUAD dataset version: {data.get('version', 'unknown')}")
        return data
    
    def process_contract(self, contract: Dict) -> List[Dict]:
        """Process a single contract and extract QA pairs."""
        contract_title = contract['title']
        examples = []
        
        for paragraph in contract['paragraphs']:
            context = paragraph['context']
            
            for qa in paragraph['qas']:
                question = qa['question']
                qa_id = qa['id']
                is_impossible = qa.get('is_impossible', False)
                
                # Extract category from question
                category = self._extract_category(question)
                
                # Get answers
                if is_impossible or len(qa.get('answers', [])) == 0:
                    # No answer for this category
                    answer_texts = []
                    answer_starts = []
                else:
                    answer_texts = [ans['text'] for ans in qa['answers']]
                    answer_starts = [ans['answer_start'] for ans in qa['answers']]
                
                example = {
                    'id': qa_id,
                    'contract_title': contract_title,
                    'context': context,
                    'question': question,
                    'category': category,
                    'clause_type': self.CATEGORY_TO_CLAUSE_TYPE.get(category, 'other'),
                    'answers': {
                        'text': answer_texts,
                        'answer_start': answer_starts
                    },
                    'is_impossible': is_impossible
                }
                
                examples.append(example)
                self.stats['total_examples'] += 1
                self.stats[f'category_{category}'] += 1
                
                if is_impossible:
                    self.stats['impossible_examples'] += 1
                else:
                    self.stats['answerable_examples'] += 1
        
        return examples
    
    def _extract_category(self, question: str) -> str:
        """Extract category name from question text."""
        # Questions are formatted like: "Highlight the parts (if any) of this contract 
        # related to "Category Name" that should be reviewed by a lawyer."
        
        for category in self.CUAD_CATEGORIES:
            if category.lower() in question.lower():
                return category
        
        return "Unknown"
    
    def split_data(self, examples: List[Dict], 
                   train_ratio: float = 0.8,
                   val_ratio: float = 0.1,
                   test_ratio: float = 0.1) -> Tuple[List, List, List]:
        """Split data into train/val/test sets."""
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        # Shuffle examples
        random.shuffle(examples)
        
        total = len(examples)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        train_data = examples[:train_size]
        val_data = examples[train_size:train_size + val_size]
        test_data = examples[train_size + val_size:]
        
        print(f"\nData split:")
        print(f"  Train: {len(train_data)} examples ({len(train_data)/total*100:.1f}%)")
        print(f"  Val:   {len(val_data)} examples ({len(val_data)/total*100:.1f}%)")
        print(f"  Test:  {len(test_data)} examples ({len(test_data)/total*100:.1f}%)")
        
        return train_data, val_data, test_data
    
    def convert_to_squad_format(self, examples: List[Dict]) -> Dict:
        """Convert examples to SQuAD 2.0 format for HuggingFace."""
        
        # Group examples by contract
        contracts = defaultdict(list)
        for example in examples:
            contracts[example['contract_title']].append(example)
        
        data = []
        for contract_title, contract_examples in contracts.items():
            # Group by context (paragraph)
            paragraphs_dict = defaultdict(list)
            for ex in contract_examples:
                paragraphs_dict[ex['context']].append(ex)
            
            paragraphs = []
            for context, qas_list in paragraphs_dict.items():
                qas = []
                for ex in qas_list:
                    qa = {
                        'id': ex['id'],
                        'question': ex['question'],
                        'answers': ex['answers'],
                        'is_impossible': ex['is_impossible']
                    }
                    qas.append(qa)
                
                paragraphs.append({
                    'context': context,
                    'qas': qas
                })
            
            data.append({
                'title': contract_title,
                'paragraphs': paragraphs
            })
        
        return {'data': data, 'version': 'cuad_v1_processed'}
    
    def save_dataset(self, dataset: Dict, split_name: str):
        """Save dataset to JSON file."""
        output_path = self.output_dir / f"{split_name}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved {split_name} set to {output_path}")
    
    def save_statistics(self):
        """Save processing statistics."""
        stats_path = self.output_dir / "processing_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(dict(self.stats), f, indent=2)
        print(f"✓ Saved statistics to {stats_path}")
    
    def create_metadata_csv(self, examples: List[Dict]):
        """Create metadata CSV for contract analysis."""
        import csv
        
        # Group by contract
        contracts_data = defaultdict(lambda: {
            'categories': set(),
            'num_clauses': 0,
            'has_answers': 0
        })
        
        for ex in examples:
            contract = ex['contract_title']
            contracts_data[contract]['categories'].add(ex['category'])
            contracts_data[contract]['num_clauses'] += 1
            if not ex['is_impossible']:
                contracts_data[contract]['has_answers'] += 1
        
        # Write CSV
        metadata_path = self.output_dir / "contracts_metadata.csv"
        with open(metadata_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'contract_title', 
                'num_categories', 
                'num_clauses', 
                'num_answered',
                'coverage_percent'
            ])
            
            for contract, data in sorted(contracts_data.items()):
                coverage = (data['has_answers'] / data['num_clauses'] * 100) \
                          if data['num_clauses'] > 0 else 0
                writer.writerow([
                    contract,
                    len(data['categories']),
                    data['num_clauses'],
                    data['has_answers'],
                    f"{coverage:.1f}%"
                ])
        
        print(f"✓ Saved contract metadata to {metadata_path}")
    
    def process(self):
        """Main processing pipeline."""
        print("\n" + "="*70)
        print("CUAD Dataset Processor")
        print("="*70 + "\n")
        
        # Load data
        cuad_data = self.load_cuad_json()
        
        # Process all contracts
        print("\nProcessing contracts...")
        all_examples = []
        for contract in tqdm(cuad_data['data'], desc="Contracts"):
            examples = self.process_contract(contract)
            all_examples.extend(examples)
        
        print(f"\n✓ Processed {len(cuad_data['data'])} contracts")
        print(f"✓ Total examples: {len(all_examples)}")
        
        # Print statistics
        print("\nDataset Statistics:")
        print(f"  Total examples: {self.stats['total_examples']}")
        print(f"  Answerable: {self.stats['answerable_examples']}")
        print(f"  Impossible: {self.stats['impossible_examples']}")
        
        # Show top categories
        category_counts = {k: v for k, v in self.stats.items() 
                          if k.startswith('category_')}
        top_categories = sorted(category_counts.items(), 
                               key=lambda x: x[1], reverse=True)[:10]
        print("\n  Top 10 Categories:")
        for cat, count in top_categories:
            cat_name = cat.replace('category_', '')
            print(f"    {cat_name}: {count}")
        
        # Split data
        train_examples, val_examples, test_examples = self.split_data(all_examples)
        
        # Convert to SQuAD format
        print("\nConverting to SQuAD format...")
        train_dataset = self.convert_to_squad_format(train_examples)
        val_dataset = self.convert_to_squad_format(val_examples)
        test_dataset = self.convert_to_squad_format(test_examples)
        
        # Save datasets
        print("\nSaving datasets...")
        self.save_dataset(train_dataset, 'train')
        self.save_dataset(val_dataset, 'val')
        self.save_dataset(test_dataset, 'test')
        
        # Save statistics
        self.save_statistics()
        
        # Create metadata CSV
        self.create_metadata_csv(all_examples)
        
        print("\n" + "="*70)
        print("✓ Processing complete!")
        print(f"✓ Output directory: {self.output_dir}")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Process CUAD dataset for clause extraction training'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to CUAD_v1.json file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for data splitting'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Process data
    processor = CUADProcessor(args.input, args.output)
    processor.process()


if __name__ == '__main__':
    main()
