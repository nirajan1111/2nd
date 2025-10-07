"""
Train RoBERTa-large for CUAD Clause Extraction

This script fine-tunes RoBERTa on CUAD dataset for extractive QA.
Target F1 score: 85-90%

Usage:
    python train_cuad_extractor.py --data_dir ../cuad_processed/ --output_dir ../cuad_models/roberta_extractor/
"""

import os
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

# Disable HuggingFace Hub's chat template checking (causes 404 errors)
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from datasets import load_dataset
import numpy as np


@dataclass
class CUADTrainingConfig:
    """Configuration for CUAD training."""
    model_name: str = "roberta-large"
    max_length: int = 512
    doc_stride: int = 128
    batch_size: int = 8
    num_epochs: int = 5
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    use_mps: bool = False  # Auto-detect MPS/CPU
    warmup_ratio: float = 0.1
    fp16: bool = False  # Disable fp16 for CPU/MPS
    gradient_accumulation_steps: int = 2


class CUADTrainer:
    """Trainer for CUAD clause extraction model."""
    
    def __init__(self, config: CUADTrainingConfig, data_dir: str, output_dir: str):
        self.config = config
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect device
        if torch.cuda.is_available():
            self.device = "cuda"
            self.config.fp16 = True
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.config.fp16 = False  # MPS doesn't support fp16
            self.config.use_mps = True
        else:
            self.device = "cpu"
            self.config.fp16 = False
        
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        print(f"Loading {config.model_name} tokenizer and model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(config.model_name)
        
        print(f"✓ Model loaded: {config.model_name}")
        print(f"✓ Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def load_datasets(self):
        """Load train, val, test datasets."""
        print("\nLoading datasets...")
        
        datasets = load_dataset(
            'json',
            data_files={
                'train': str(self.data_dir / 'train.json'),
                'validation': str(self.data_dir / 'val.json'),
                'test': str(self.data_dir / 'test.json')
            },
            field='data'
        )
        
        print(f"✓ Train: {len(datasets['train'])} contracts")
        print(f"✓ Val: {len(datasets['validation'])} contracts")
        print(f"✓ Test: {len(datasets['test'])} contracts")
        
        return datasets
    
    def preprocess_function(self, examples):
        """Preprocess examples for extractive QA."""
        
        questions = [q.strip() for q in examples["question"]]
        contexts = examples["context"]
        
        # Tokenize inputs
        tokenized_examples = self.tokenizer(
            questions,
            contexts,
            truncation="only_second",
            max_length=self.config.max_length,
            stride=self.config.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )
        
        # Map back to original examples
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")
        
        # Initialize start and end positions
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        
        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)
            
            # Get sample index
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            
            # If no answers, set CLS index
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Get start and end character positions
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                
                # Find token start and end positions
                token_start_index = 0
                while token_start_index < len(offsets) and \
                      offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                token_start_index -= 1
                
                token_end_index = len(offsets) - 1
                while token_end_index >= 0 and \
                      offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                token_end_index += 1
                
                # Verify answer is within context
                sequence_ids = tokenized_examples.sequence_ids(i)
                context_start = sequence_ids.index(1) if 1 in sequence_ids else 0
                context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1) \
                             if 1 in sequence_ids else len(sequence_ids)
                
                # If answer not in context, use CLS
                if not (context_start <= token_start_index < context_end and \
                       context_start <= token_end_index < context_end):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    tokenized_examples["start_positions"].append(token_start_index)
                    tokenized_examples["end_positions"].append(token_end_index)
        
        return tokenized_examples
    
    def compute_metrics(self, pred):
        """Compute F1 and Exact Match metrics."""
        predictions, labels = pred
        start_predictions, end_predictions = predictions
        
        # Get predicted spans
        start_preds = np.argmax(start_predictions, axis=1)
        end_preds = np.argmax(end_predictions, axis=1)
        
        start_labels = labels[0]
        end_labels = labels[1]
        
        # Compute exact match
        exact_match = np.mean(
            (start_preds == start_labels) & (end_preds == end_labels)
        )
        
        # Compute F1 (simplified - just for monitoring)
        # In practice, you'd need to compute token-level F1
        f1_score = exact_match  # Placeholder
        
        return {
            'exact_match': exact_match,
            'f1': f1_score
        }
    
    def train(self):
        """Train the model."""
        print("\n" + "="*70)
        print("Starting CUAD Extractor Training")
        print("="*70 + "\n")
        
        # Load datasets
        datasets = self.load_datasets()
        
        # Flatten datasets (convert SQuAD format to flat examples)
        print("\nFlattening datasets...")
        
        def flatten_squad(example):
            """Flatten SQuAD format to individual QA pairs."""
            flattened = []
            for paragraph in example['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    flattened.append({
                        'id': qa['id'],
                        'context': context,
                        'question': qa['question'],
                        'answers': qa['answers'],
                        'is_impossible': qa.get('is_impossible', False)
                    })
            return flattened
        
        train_examples = []
        for example in datasets['train']:
            train_examples.extend(flatten_squad(example))
        
        val_examples = []
        for example in datasets['validation']:
            val_examples.extend(flatten_squad(example))
        
        print(f"✓ Train examples: {len(train_examples)}")
        print(f"✓ Val examples: {len(val_examples)}")
        
        # Convert to HuggingFace Dataset format
        from datasets import Dataset as HFDataset
        train_dataset = HFDataset.from_list(train_examples)
        val_dataset = HFDataset.from_list(val_examples)
        
        # Preprocess datasets
        print("\nTokenizing datasets...")
        print("⚠️  This may take 5-10 minutes on GPU, please be patient...")
        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            batch_size=100,  # Process in larger batches
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train"
        )
        
        val_dataset = val_dataset.map(
            self.preprocess_function,
            batched=True,
            batch_size=100,  # Process in larger batches
            remove_columns=val_dataset.column_names,
            desc="Tokenizing validation"
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            fp16=self.config.fp16,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            logging_dir=str(self.output_dir / 'logs'),
            logging_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model='exact_match',
            report_to="none",  # Disable wandb/tensorboard for now
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=default_data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Train
        print("\nStarting training...")
        train_result = trainer.train()
        
        # Save model
        print("\nSaving model...")
        trainer.save_model(str(self.output_dir / "best_model"))
        self.tokenizer.save_pretrained(str(self.output_dir / "best_model"))
        
        # Save training results
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
        print("\n" + "="*70)
        print("✓ Training complete!")
        print(f"✓ Model saved to: {self.output_dir / 'best_model'}")
        print(f"✓ Val Exact Match: {metrics.get('eval_exact_match', 0):.4f}")
        print(f"✓ Val F1: {metrics.get('eval_f1', 0):.4f}")
        print("="*70 + "\n")
        
        return trainer, metrics


def main():
    parser = argparse.ArgumentParser(
        description='Train RoBERTa for CUAD clause extraction'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing train.json, val.json, test.json'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for model checkpoints'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='roberta-large',
        help='Pretrained model name'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Training batch size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=3e-5,
        help='Learning rate'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = CUADTrainingConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    # Train
    trainer = CUADTrainer(config, args.data_dir, args.output_dir)
    trainer.train()


if __name__ == '__main__':
    main()
