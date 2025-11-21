"""
T5 Training Script for Legal Text → FOPL Conversion
Fine-tunes google/t5-v1_1-base on legal clause to FOPL predicate translation
"""

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import load_metric
import numpy as np
from typing import Dict, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LegalFOPLDataset(Dataset):
    """Dataset for Legal Text → FOPL translation"""
    
    def __init__(self, json_path: str, tokenizer: T5Tokenizer, max_length: int = 512):
        """
        Args:
            json_path: Path to legal_clauses.json
            tokenizer: T5 tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        logger.info(f"Loading dataset from {json_path}")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} examples")
        
        # Prepare examples
        self.examples = []
        for item in self.data:
            # Input: "translate to english logic: <clause_text> context: <context>"
            clause_text = item['clause_text']
            context = item.get('context', {})
            context_str = " ".join([f"{k}={v}" for k, v in context.items()])
            
            input_text = f"translate to english logic: {clause_text}"
            if context_str:
                input_text += f" context: {context_str}"
            
            # Output: FOPL rule
            target_text = item['fopl_rule']
            
            self.examples.append({
                'input_text': input_text,
                'target_text': target_text,
                'clause_id': item['id'],
                'clause_type': item.get('clause_type', 'unknown')
            })
        
        logger.info(f"Prepared {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize input
        input_encoding = self.tokenizer(
            example['input_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            example['target_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = target_encoding['input_ids'].squeeze()
        # Replace padding token id with -100 so it's ignored by loss
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels
        }


def compute_metrics(eval_preds):
    """Compute BLEU and ROUGE metrics"""
    predictions, labels = eval_preds
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in labels (used for padding)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Clean up text
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    # Compute ROUGE scores
    rouge = load_metric('rouge')
    rouge_output = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        rouge_types=['rouge1', 'rouge2', 'rougeL']
    )
    
    # Compute BLEU score
    bleu = load_metric('sacrebleu')
    bleu_output = bleu.compute(
        predictions=decoded_preds,
        references=[[label] for label in decoded_labels]
    )
    
    # Compute exact match accuracy
    exact_match = sum([pred == label for pred, label in zip(decoded_preds, decoded_labels)]) / len(decoded_preds)
    
    return {
        'rouge1': rouge_output['rouge1'].mid.fmeasure,
        'rouge2': rouge_output['rouge2'].mid.fmeasure,
        'rougeL': rouge_output['rougeL'].mid.fmeasure,
        'bleu': bleu_output['score'],
        'exact_match': exact_match
    }


def train_t5_fopl_model(
    data_path: str = "data/legal_clauses.json",
    model_name: str = "google/t5-v1_1-base",
    output_dir: str = "checkpoints/t5_fopl",
    num_train_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    warmup_steps: int = 500,
    save_steps: int = 500,
    eval_steps: int = 500,
    max_length: int = 512,
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 42
):
    """
    Train T5 model on Legal → FOPL task
    
    Args:
        data_path: Path to legal_clauses.json
        model_name: HuggingFace model name (google/t5-v1_1-base)
        output_dir: Directory to save checkpoints
        num_train_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        max_length: Maximum sequence length
        train_split: Training split ratio
        val_split: Validation split ratio
        seed: Random seed
    """
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    logger.info("=" * 80)
    logger.info("  T5 FOPL TRAINING")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Data: {data_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Epochs: {num_train_epochs}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Learning Rate: {learning_rate}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer and model
    logger.info(f"\n📦 Loading tokenizer and model...")
    global tokenizer  # Needed for compute_metrics
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Add FOPL tokens to vocabulary
    logger.info("Adding FOPL tokens to vocabulary...")
    fopl_operators = [
        'forall', 'exists', '&', '|', '~', '->', '<->', '<=', '>=', '!=',
        '(', ')', ',', '[', ']'
    ]
    
    # Load predicates
    predicates_path = os.path.join('data', 'predicates.txt')
    if os.path.exists(predicates_path):
        with open(predicates_path, 'r') as f:
            predicates = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(predicates)} predicates from predicates.txt")
    else:
        predicates = [
            'Tenant', 'Landlord', 'Buyer', 'Seller', 'Employee', 
            'Contractor', 'Supplier', 'Client', 'Party', 'Contract',
            'PayRent', 'Terminate', 'Maintain', 'Deliver', 'DeliverByTime',
            'Liable', 'Indemnify', 'Warranty', 'NonCompeteApplies',
            'DirectCompetitor', 'SameIndustry', 'Goods'
        ]
        logger.info(f"Using {len(predicates)} basic predicates")
    
    new_tokens = fopl_operators + predicates
    num_added = tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"✅ Added {num_added} tokens, vocabulary size: {len(tokenizer)}")
    
    # Load full dataset
    logger.info(f"\n📚 Loading dataset...")
    full_dataset = LegalFOPLDataset(data_path, tokenizer, max_length)
    
    # Split dataset
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    logger.info(f"Dataset split:")
    logger.info(f"  Train: {train_size} ({train_split*100:.0f}%)")
    logger.info(f"  Val: {val_size} ({val_split*100:.0f}%)")
    logger.info(f"  Test: {test_size} ({(1-train_split-val_split)*100:.0f}%)")
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        save_steps=save_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        save_total_limit=3,  # Keep only 3 best checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
        report_to=["tensorboard"],
        fp16=torch.cuda.is_available(),  # Mixed precision if GPU available
        dataloader_num_workers=4,
        seed=seed,
    )
    
    # Initialize Trainer
    logger.info(f"\n🚀 Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    logger.info(f"\n{'='*80}")
    logger.info("  STARTING TRAINING")
    logger.info(f"{'='*80}\n")
    
    train_result = trainer.train()
    
    # Save final model
    logger.info(f"\n💾 Saving final model to {output_dir}/final")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    
    # Evaluate on test set
    logger.info(f"\n📊 Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    
    logger.info(f"\n{'='*80}")
    logger.info("  TRAINING COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Final Train Loss: {train_result.training_loss:.4f}")
    logger.info(f"Test Loss: {test_results['eval_loss']:.4f}")
    logger.info(f"Test ROUGE-1: {test_results.get('eval_rouge1', 0):.4f}")
    logger.info(f"Test ROUGE-L: {test_results.get('eval_rougeL', 0):.4f}")
    logger.info(f"Test BLEU: {test_results.get('eval_bleu', 0):.4f}")
    logger.info(f"Test Exact Match: {test_results.get('eval_exact_match', 0):.4f}")
    
    # Save training metrics
    metrics_path = f"{output_dir}/training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'train_loss': float(train_result.training_loss),
            'test_metrics': {k: float(v) for k, v in test_results.items()}
        }, f, indent=2)
    
    logger.info(f"\n✅ Training metrics saved to {metrics_path}")
    logger.info(f"✅ Best model saved to {output_dir}/final")
    
    return trainer, test_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train T5 for Legal → FOPL")
    parser.add_argument('--data_path', type=str, default='data/legal_clauses.json',
                       help='Path to training data')
    parser.add_argument('--model_name', type=str, default='google/t5-v1_1-base',
                       help='HuggingFace model name')
    parser.add_argument('--output_dir', type=str, default='checkpoints/t5_fopl',
                       help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Start training
    trainer, results = train_t5_fopl_model(
        data_path=args.data_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length
    )
    
    print("\n🎉 Training completed successfully!")
    print(f"📁 Model saved to: {args.output_dir}/final")
    print(f"📊 View training logs: tensorboard --logdir {args.output_dir}/logs")
