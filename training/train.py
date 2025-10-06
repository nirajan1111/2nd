import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import T5Tokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import json
from typing import Dict
import numpy as np

import sys
sys.path.append('..')
from models.neural_parser import NeuralLegalParser
from training.dataset import create_dataloaders
from training.metrics import compute_metrics


class Trainer:
    """
    Trainer for Neural Legal Parser
    """
    
    def __init__(self, model: NeuralLegalParser, train_loader: DataLoader,
                 val_loader: DataLoader, config: Dict):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Setup learning rate scheduler
        total_steps = len(train_loader) * config['num_epochs']
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.get('warmup_steps', 100),
            num_training_steps=total_steps
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        print(f"✅ Trainer initialized on {self.device}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Training batches: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}")
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('max_grad_norm', 1.0)
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log periodically
            if (batch_idx + 1) % self.config.get('log_interval', 10) == 0:
                avg_loss = total_loss / (batch_idx + 1)
                self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self) -> Dict:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs['loss'].item()
                
                # Generate predictions for metrics
                generated = self.model.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.config.get('max_length', 512),
                    num_beams=4
                )
                
                # Decode predictions and references
                preds = self.model.tokenizer.batch_decode(
                    generated, skip_special_tokens=True
                )
                
                # Replace -100 with pad_token_id before decoding
                labels_for_decode = labels.clone()
                labels_for_decode[labels_for_decode == -100] = self.model.tokenizer.pad_token_id
                refs = self.model.tokenizer.batch_decode(
                    labels_for_decode, skip_special_tokens=True
                )
                
                predictions.extend(preds)
                references.extend(refs)
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Debug: Print sample predictions vs references
        if len(predictions) > 0:
            print(f"\n📊 Sample Predictions (First 3):")
            for i in range(min(3, len(predictions))):
                print(f"\n  Example {i+1}:")
                print(f"    Predicted: {predictions[i][:100]}...")
                print(f"    Reference: {references[i][:100]}...")
        
        # Compute additional metrics
        metrics = compute_metrics(predictions, references)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("Starting training...")
        print("="*60 + "\n")
        
        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_metrics = self.validate()
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics.get('exact_match', 0))
            
            # Print epoch summary
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config['num_epochs']} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Exact Match: {val_metrics.get('exact_match', 0):.2%}")
            print(f"  Val BLEU: {val_metrics.get('bleu', 0):.4f}")
            print(f"{'='*60}\n")
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_model')
                print(f"✅ New best model saved (val_loss: {self.best_val_loss:.4f})")
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_interval', 5) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}')
        
        print("\n" + "="*60)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*60 + "\n")
        
        # Save training history
        self.save_history()
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        save_dir = os.path.join(self.config['output_dir'], name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_dir)
        
        # Save training state
        state = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'config': self.config
        }
        torch.save(state, os.path.join(save_dir, 'training_state.pt'))
    
    def save_history(self):
        """Save training history"""
        history_path = os.path.join(self.config['output_dir'], 'history.json')
        with open(history_path, 'w') as f:
            # Convert numpy types to native Python types
            history_serializable = {}
            for key, values in self.history.items():
                history_serializable[key] = [
                    float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for v in values
                ]
            json.dump(history_serializable, f, indent=2)
        print(f"✅ Training history saved to {history_path}")


def main(custom_config=None):
    """Main training script"""
    
    # Training configuration
    config = {
        'model_name': 't5-base',
        'data_path': 'data/legal_clauses.json',
        'output_dir': 'checkpoints',
        'batch_size': 16,  # Change this value (default: 16)
        'num_epochs': 20,
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'warmup_steps': 100,
        'max_grad_norm': 1.0,
        'max_length': 512,
        'log_interval': 10,
        'save_interval': 5,
        'num_workers': 4
    }
    
    # Override with custom config if provided
    if custom_config:
        config.update(custom_config)
        print(f"📝 Custom configuration applied: {custom_config}")
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Save config
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize model
    print("Initializing model...")
    model = NeuralLegalParser(model_name=config['model_name'])
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=config['data_path'],
        tokenizer=model.tokenizer,
        batch_size=16,
        max_length=config['max_length'],
        num_workers=config['num_workers']
    )
    
    # Initialize trainer
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # Start training
    trainer.train()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, config)
    print(f"\n{'='*60}")
    print("Test Set Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Exact Match: {test_metrics.get('exact_match', 0):.2%}")
    print(f"  BLEU Score: {test_metrics.get('bleu', 0):.4f}")
    print(f"{'='*60}\n")
    
    # Save test results
    with open(os.path.join(config['output_dir'], 'test_results.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)


def evaluate_model(model: NeuralLegalParser, test_loader: DataLoader, 
                   config: Dict) -> Dict:
    """Evaluate model on test set"""
    model.eval()
    device = next(model.parameters()).device
    
    total_loss = 0
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs['loss'].item()
            
            # Generate predictions
            generated = model.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=config['max_length'],
                num_beams=4
            )
            
            # Decode
            preds = model.tokenizer.batch_decode(generated, skip_special_tokens=True)
            
            # Replace -100 with pad_token_id before decoding
            labels_for_decode = labels.clone()
            labels_for_decode[labels_for_decode == -100] = model.tokenizer.pad_token_id
            refs = model.tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
            
            predictions.extend(preds)
            references.extend(refs)
    
    avg_loss = total_loss / len(test_loader)
    metrics = compute_metrics(predictions, references)
    metrics['loss'] = avg_loss
    
    return metrics


if __name__ == "__main__":
    main()