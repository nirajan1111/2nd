import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer
from typing import List, Dict


class LegalReasoningDataset(Dataset):
    """
    PyTorch Dataset for Legal Clause to FOPL translation
    """
    
    def __init__(self, data_path: str, tokenizer: T5Tokenizer, 
                 max_length: int = 512, split: str = 'train'):
        """
        Args:
            data_path: Path to JSON file with legal clauses
            tokenizer: T5 tokenizer
            max_length: Maximum sequence length
            split: 'train', 'val', or 'test'
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Split data (80% train, 10% val, 10% test)
        total = len(self.data)
        if split == 'train':
            self.data = self.data[:int(0.8 * total)]
        elif split == 'val':
            self.data = self.data[int(0.8 * total):int(0.9 * total)]
        else:  # test
            self.data = self.data[int(0.9 * total):]
        
        print(f"✅ Loaded {len(self.data)} samples for {split} set")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Prepare input: "translate legal to logic: <clause> context: <entities>"
        input_text = self._prepare_input(item)
        
        # Prepare target: FOPL rule
        target_text = item['fopl_rule']
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare labels (replace padding token id with -100)
        labels = target_encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels,
            'clause_text': item['clause_text'],
            'fopl_rule': item['fopl_rule'],
            'clause_id': item['id']
        }
    
    def _prepare_input(self, item: Dict) -> str:
        """Prepare input text with task prefix and context"""
        input_text = f"translate legal to logic: {item['clause_text']}"
        
        if item.get('context'):
            context_str = " ".join([f"{k}={v}" for k, v in item['context'].items()])
            input_text += f" context: {context_str}"
        
        return input_text


def create_dataloaders(data_path: str, tokenizer: T5Tokenizer,
                      batch_size: int = 8, max_length: int = 512,
                      num_workers: int = 4):
    """
    Create train, validation, and test dataloaders
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = LegalReasoningDataset(
        data_path, tokenizer, max_length, split='train'
    )
    val_dataset = LegalReasoningDataset(
        data_path, tokenizer, max_length, split='val'
    )
    test_dataset = LegalReasoningDataset(
        data_path, tokenizer, max_length, split='test'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


class CollateFunction:
    """Custom collate function for batching with dynamic padding"""
    
    def __init__(self, tokenizer: T5Tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """Collate batch with dynamic padding"""
        
        # Extract fields
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # Stack tensors
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        labels = torch.stack(labels)
        
        # Collect metadata
        clause_texts = [item['clause_text'] for item in batch]
        fopl_rules = [item['fopl_rule'] for item in batch]
        clause_ids = [item['clause_id'] for item in batch]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': labels,
            'clause_texts': clause_texts,
            'fopl_rules': fopl_rules,
            'clause_ids': clause_ids
        }


if __name__ == "__main__":
    from transformers import T5Tokenizer
    
    # Test dataset
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    
    # Add special tokens
    special_tokens = {
        'additional_special_tokens': [
            'forall', 'exists', '&', '|', '~', '->', '<->'
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # Create dataset
    dataset = LegalReasoningDataset(
        'data/legal_clauses.json',
        tokenizer,
        max_length=256,
        split='train'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test one sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Input shape: {sample['input_ids'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    print(f"Clause: {sample['clause_text']}")
    print(f"FOPL: {sample['fopl_rule']}")
    
    # Test dataloader
    train_loader, val_loader, test_loader = create_dataloaders(
        'data/legal_clauses.json',
        tokenizer,
        batch_size=4
    )
    
    print(f"\nDataloader splits:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Test batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Input IDs: {batch['input_ids'].shape}")
    print(f"  Attention mask: {batch['attention_mask'].shape}")
    print(f"  Labels: {batch['labels'].shape}")