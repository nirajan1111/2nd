"""
T5 Training Script for Legal Text → FOPL Conversion
Clean, production-ready training pipeline without Kaggle-specific code
"""

import json
import os
import sys
import torch
from torch.utils.data import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import evaluate
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config, print_config, T5FOPLConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# DATASET
# ==============================================================================

class LegalFOPLDataset(Dataset):
    """Dataset for Legal Text → FOPL translation"""
    
    def __init__(self, data: List[Dict], tokenizer: T5Tokenizer, max_length: int = 512):
        """
        Args:
            data: List of examples with clause_text, context, fopl_rule
            tokenizer: T5 tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        for item in data:
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
                'target_text': target_text
            })
    
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


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_and_split_data(data_path: str, 
                        train_ratio: float = 0.8, 
                        val_ratio: float = 0.1) -> Tuple[List, List, List]:
    """
    Load and split dataset into train/val/test
    
    Args:
        data_path: Path to legal_clauses.json
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
    
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    logger.info(f"Loading dataset from {data_path}")
    
    with open(data_path, 'r') as f:
        full_data = json.load(f)
    
    # Shuffle data
    import random
    random.seed(42)
    random.shuffle(full_data)
    
    # Split
    n = len(full_data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_data = full_data[:train_end]
    val_data = full_data[train_end:val_end]
    test_data = full_data[val_end:]
    
    logger.info(f"Dataset split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    return train_data, val_data, test_data


# ==============================================================================
# MODEL SETUP
# ==============================================================================

def setup_model_and_tokenizer(config: T5FOPLConfig, predicates_path: Optional[str] = None):
    """
    Load model and tokenizer, add FOPL vocabulary
    
    Args:
        config: T5 configuration
        predicates_path: Path to predicates.txt (optional)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {config.model_name}")
    
    tokenizer = T5Tokenizer.from_pretrained(config.model_name)
    model = T5ForConditionalGeneration.from_pretrained(config.model_name)
    
    # Add FOPL tokens to vocabulary
    logger.info("Adding FOPL tokens to vocabulary...")
    
    fopl_operators = [
        'forall', 'exists', '&', '|', '~', '->', '<->', 
        '<=', '>=', '!=', '(', ')', ',', '[', ']'
    ]
    
    new_tokens = fopl_operators.copy()
    
    # Load predicates if available
    if predicates_path and os.path.exists(predicates_path):
        with open(predicates_path, 'r') as f:
            predicates = [line.strip() for line in f if line.strip()]
        new_tokens.extend(predicates)
        logger.info(f"Loaded {len(predicates)} predicates")
    
    # Add tokens
    num_added = tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    logger.info(f"Added {num_added} tokens | Vocab size: {len(tokenizer)}")
    
    # Enable gradient checkpointing if configured
    if config.gradient_checkpointing:
        model.config.use_cache = False
        logger.info("Gradient checkpointing enabled")
    
    return model, tokenizer


# ==============================================================================
# METRICS
# ==============================================================================

def get_compute_metrics_fn(tokenizer):
    """
    Create compute_metrics function for Trainer
    
    Args:
        tokenizer: T5 tokenizer
    
    Returns:
        compute_metrics function
    """
    # Load metrics
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('sacrebleu')
    
    def compute_metrics(eval_preds):
        """Compute BLEU, ROUGE, and Exact Match metrics"""
        predictions, labels = eval_preds
        
        # Decode predictions
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Decode labels (replace -100 with pad token)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Clean whitespace
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        # ROUGE
        rouge_output = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            rouge_types=['rouge1', 'rouge2', 'rougeL']
        )
        
        # BLEU
        bleu_output = bleu.compute(
            predictions=decoded_preds,
            references=[[label] for label in decoded_labels]
        )
        
        # Exact match
        exact_match = sum([pred == label for pred, label in zip(decoded_preds, decoded_labels)]) / len(decoded_preds)
        
        return {
            'rouge1': rouge_output['rouge1'],
            'rouge2': rouge_output['rouge2'],
            'rougeL': rouge_output['rougeL'],
            'bleu': bleu_output['score'],
            'exact_match': exact_match
        }
    
    return compute_metrics


# ==============================================================================
# TRAINING
# ==============================================================================

def train_t5_fopl(
    config: Optional[T5FOPLConfig] = None,
    data_path: Optional[str] = None,
    predicates_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    custom_config: Optional[Dict] = None
):
    """
    Main training function
    
    Args:
        config: T5 configuration object
        data_path: Path to legal_clauses.json
        predicates_path: Path to predicates.txt
        output_dir: Output directory for checkpoints
        custom_config: Dictionary to override config
    """
    # Load configuration
    if config is None:
        from config import Config
        cfg = Config()
        if custom_config:
            cfg.update(**custom_config)
        cfg.update_for_gpu()
        config = cfg.t5_fopl
        
        # Set paths if not provided
        if data_path is None:
            data_path = str(cfg.paths.legal_clauses)
        if predicates_path is None:
            predicates_path = str(cfg.paths.predicates)
        if output_dir is None:
            output_dir = str(cfg.paths.t5_fopl_checkpoint)
    
    # Print configuration
    logger.info("\n" + "="*60)
    logger.info("T5 FOPL TRAINING")
    logger.info("="*60)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Data: {data_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Epochs: {config.num_epochs}")
    logger.info(f"Batch Size: {config.batch_size}")
    logger.info(f"Learning Rate: {config.learning_rate}")
    logger.info("="*60 + "\n")
    
    # Check if data exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load and split data
    train_data, val_data, test_data = load_and_split_data(data_path)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config, predicates_path)
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = LegalFOPLDataset(train_data, tokenizer, config.max_input_length)
    val_dataset = LegalFOPLDataset(val_data, tokenizer, config.max_input_length)
    test_dataset = LegalFOPLDataset(test_data, tokenizer, config.max_input_length)
    
    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logging_dir=f"{output_dir}/logs",
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=False,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        max_grad_norm=config.max_grad_norm,
        report_to=["tensorboard"],
        seed=42,
    )
    
    # Initialize Trainer
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=get_compute_metrics_fn(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)]
    )
    
    # Train
    logger.info("\n" + "="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60 + "\n")
    
    train_result = trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    final_model_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    
    # Save metrics
    metrics = {
        'train_loss': float(train_result.training_loss),
        'test_metrics': {k: float(v) for k, v in test_results.items()}
    }
    
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Final Model: {final_model_dir}")
    logger.info(f"Train Loss: {train_result.training_loss:.4f}")
    logger.info(f"Test Loss: {test_results['eval_loss']:.4f}")
    logger.info(f"Test ROUGE-1: {test_results.get('eval_rouge1', 0):.4f}")
    logger.info(f"Test BLEU: {test_results.get('eval_bleu', 0):.2f}")
    logger.info(f"Test Exact Match: {test_results.get('eval_exact_match', 0):.2%}")
    logger.info("="*60 + "\n")
    
    return trainer, test_results


# ==============================================================================
# MAIN
# ==============================================================================

def main(custom_config: Optional[Dict] = None):
    """Main entry point"""
    try:
        trainer, results = train_t5_fopl(custom_config=custom_config)
        return trainer, results
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train T5 for FOPL generation")
    parser.add_argument("--data", type=str, help="Path to legal_clauses.json")
    parser.add_argument("--predicates", type=str, help="Path to predicates.txt")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    
    args = parser.parse_args()
    
    # Build custom config
    custom_config = {}
    if args.epochs:
        custom_config['num_epochs'] = args.epochs
    if args.batch_size:
        custom_config['batch_size'] = args.batch_size
    if args.lr:
        custom_config['learning_rate'] = args.lr
    
    # Train
    train_t5_fopl(
        data_path=args.data,
        predicates_path=args.predicates,
        output_dir=args.output,
        custom_config=custom_config if custom_config else None
    )
