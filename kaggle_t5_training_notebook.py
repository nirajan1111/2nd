"""
Kaggle Notebook for T5 FOPL Training
Copy this entire code into a Kaggle notebook to train on GPU
"""

# ============================================================================
# CELL 1: Setup and Imports
# ============================================================================

print("="*80)
print("  T5 LEGAL → FOPL TRAINING ON KAGGLE")
print("="*80)
print()

# Install required packages (if not already installed)
import sys
!{sys.executable} -m pip install -q transformers datasets sacrebleu rouge-score accelerate evaluate

import json
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import evaluate  # Modern replacement for load_metric

print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()


# ============================================================================
# CELL 2: Dataset Class
# ============================================================================

class LegalFOPLDataset(Dataset):
    """Dataset for Legal Text → FOPL translation"""
    
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        for item in data:
            # Input format
            clause_text = item['clause_text']
            context = item.get('context', {})
            context_str = " ".join([f"{k}={v}" for k, v in context.items()])
            
            input_text = f"translate to english logic: {clause_text}"
            if context_str:
                input_text += f" context: {context_str}"
            
            # Output format
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


# Debug function to check dataset
def check_dataset_sample(dataset, tokenizer, num_samples=3):
    """Check if dataset is properly formatted"""
    print(f"\n🔍 Checking {num_samples} dataset samples...")
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        print(f"\n--- Sample {i+1} ---")
        print(f"Input IDs shape: {sample['input_ids'].shape}")
        print(f"Labels shape: {sample['labels'].shape}")
        print(f"Non-padding labels: {(sample['labels'] != -100).sum().item()}")
        
        # Decode to verify
        input_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        label_ids = sample['labels'].clone()
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        label_text = tokenizer.decode(label_ids, skip_special_tokens=True)
        
        print(f"Input: {input_text[:100]}...")
        print(f"Label: {label_text[:100]}...")
    print()

print("✅ Dataset class defined")


# ============================================================================
# CELL 3: Load Data from Kaggle Input
# ============================================================================

# IMPORTANT: Replace 'yourusername/legal-fopl-training-data' with your actual dataset path
# After uploading dataset, it will be at: /kaggle/input/your-dataset-name/

DATA_PATH = '/kaggle/input/legal-fopl-training-data/legal_clauses.json'  # UPDATE THIS
PREDICATES_PATH = '/kaggle/input/legal-fopl-training-data/predicates.txt'  # UPDATE THIS

# Alternative: If you uploaded via notebook, use:
# DATA_PATH = '/kaggle/working/legal_clauses.json'

print("📚 Loading training data...")
print(f"Data path: {DATA_PATH}")

try:
    with open(DATA_PATH, 'r') as f:
        full_data = json.load(f)
    
    print(f"✅ Loaded {len(full_data)} examples")
    
    # Split data
    np.random.seed(42)
    indices = np.random.permutation(len(full_data))
    train_size = int(0.8 * len(full_data))
    val_size = int(0.1 * len(full_data))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    train_data = [full_data[i] for i in train_indices]
    val_data = [full_data[i] for i in val_indices]
    test_data = [full_data[i] for i in test_indices]
    
    print(f"✅ Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    
except FileNotFoundError:
    print("❌ ERROR: Data file not found!")
    print("Please:")
    print("1. Upload legal_clauses.json to Kaggle Datasets")
    print("2. Add dataset to this notebook (+ Add Data)")
    print("3. Update DATA_PATH variable above")
    raise

print()


# ============================================================================
# CELL 4: Load Model and Tokenizer
# ============================================================================

MODEL_NAME = "google/t5-v1_1-base"  # or "t5-small" for faster training
OUTPUT_DIR = "/kaggle/working/t5_fopl_model"

print(f"📦 Loading model: {MODEL_NAME}")

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Enable gradient checkpointing to save memory during training
model.config.use_cache = False  # Required for gradient checkpointing

print("✅ Model and tokenizer loaded")

# Add FOPL tokens to vocabulary
print("Adding FOPL tokens...")
fopl_operators = [
    'forall', 'exists', '&', '|', '~', '->', '<->', '<=', '>=', '!=',
    '(', ')', ',', '[', ']'
]

# Load predicates if available
try:
    with open(PREDICATES_PATH, 'r') as f:
        predicates = [line.strip() for line in f if line.strip()]
    print(f"✅ Loaded {len(predicates)} predicates")
except:
    predicates = [
        'Tenant', 'Landlord', 'Buyer', 'Seller', 'Employee', 
        'Contractor', 'Supplier', 'Client', 'Party', 'Contract',
        'PayRent', 'Terminate', 'Maintain', 'Deliver', 'DeliverByTime',
        'Liable', 'Indemnify', 'Warranty', 'NonCompeteApplies',
        'DirectCompetitor', 'SameIndustry', 'Goods'
    ]
    print(f"⚠️  Using {len(predicates)} default predicates")

new_tokens = fopl_operators + predicates
num_added = tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))

print(f"✅ Added {num_added} tokens | Vocab size: {len(tokenizer)}")
print()


# ============================================================================
# CELL 5: Prepare Datasets
# ============================================================================

print("🔄 Preparing datasets...")

train_dataset = LegalFOPLDataset(train_data, tokenizer)
val_dataset = LegalFOPLDataset(val_data, tokenizer)
test_dataset = LegalFOPLDataset(test_data, tokenizer)

print(f"✅ Train dataset: {len(train_dataset)} examples")
print(f"✅ Val dataset: {len(val_dataset)} examples")
print(f"✅ Test dataset: {len(test_dataset)} examples")
print()


# ============================================================================
# CELL 6: Define Metrics
# ============================================================================

def compute_metrics(eval_preds):
    """DISABLED - Skip metrics during training to prevent OOM"""
    # Don't compute any metrics during intermediate evaluations
    # This prevents the generation step that causes OOM
    # We'll compute metrics manually at the end
    return {}

# Optional: Full metrics function for final evaluation only
def compute_full_metrics(eval_preds):
    """Compute all metrics - use only for final evaluation"""
    predictions, labels = eval_preds
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    # ROUGE - using evaluate library
    rouge = evaluate.load('rouge')
    rouge_output = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        rouge_types=['rouge1', 'rouge2', 'rougeL']
    )
    
    # BLEU - using evaluate library
    bleu = evaluate.load('sacrebleu')
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

print("✅ Metrics function defined")
print()


# ============================================================================
# CELL 7: Training Configuration
# ============================================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=10,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,  # Smaller eval batch to save memory
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,
    save_steps=1000,
    eval_steps=1000,
    eval_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,  # Mixed precision for faster training
    report_to=["tensorboard"],
    seed=42,
    # Memory optimization for evaluation
    eval_accumulation_steps=4,  # Accumulate eval batches before moving to CPU
    gradient_checkpointing=True,  # Trade compute for memory
    prediction_loss_only=True,  # ✅ CRITICAL: Only compute loss, don't generate predictions
)

print("✅ Training configuration:")
print(f"  • Epochs: {training_args.num_train_epochs}")
print(f"  • Batch size: {training_args.per_device_train_batch_size}")
print(f"  • Learning rate: {training_args.learning_rate}")
print(f"  • FP16: {training_args.fp16}")
print()


# ============================================================================
# CELL 8: Initialize Trainer
# ============================================================================

# Custom generation config to reduce memory during evaluation
from transformers import GenerationConfig

generation_config = GenerationConfig(
    max_length=128,  # Shorter max length
    num_beams=1,     # Greedy decoding instead of beam search
    do_sample=False,
    # early_stopping only works with beam search, so we remove it
)

# Set generation config on model
model.generation_config = generation_config

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=None,  # ✅ No metrics during training - prevents generation
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("✅ Trainer initialized with memory-efficient settings")
print(f"  • Eval batch size: {training_args.per_device_eval_batch_size}")
print(f"  • Eval accumulation steps: {training_args.eval_accumulation_steps}")
print(f"  • Gradient checkpointing: {training_args.gradient_checkpointing}")
print(f"  • Prediction loss only: {training_args.prediction_loss_only}")
print(f"  ⚠️  No metrics during training - will evaluate manually at end")
print()


# ============================================================================
# CELL 9: START TRAINING
# ============================================================================

print("="*80)
print("  STARTING TRAINING")
print("="*80)
print()

train_result = trainer.train()

print()
print("="*80)
print("  TRAINING COMPLETE!")
print("="*80)
print(f"Final Train Loss: {train_result.training_loss:.4f}")
print()


# ============================================================================
# CELL 10: Evaluate on Test Set (Manual - Memory Safe)
# ============================================================================

print("📊 Evaluating on test set...")
print("Computing loss only (no generation to save memory)...")

# Just compute loss
test_results = trainer.evaluate(test_dataset)

print()
print("="*80)
print("  TEST SET RESULTS")
print("="*80)
print(f"Test Loss: {test_results['eval_loss']:.4f}")

# Manual evaluation on small sample for metrics
print()
print("📊 Computing metrics on sample (to save memory)...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()

# Evaluate on first 20 test examples only
sample_size = min(20, len(test_data))
predictions = []
references = []

for i in range(sample_size):
    example = test_data[i]
    
    # Prepare input
    clause_text = example['clause_text']
    context = example.get('context', {})
    context_str = " ".join([f"{k}={v}" for k, v in context.items()])
    input_text = f"translate to english logic: {clause_text}"
    if context_str:
        input_text += f" context: {context_str}"
    
    # Generate
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=1,  # Greedy
            do_sample=False
        )
    
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    ref = example['fopl_rule'].strip()
    
    predictions.append(pred)
    references.append(ref)
    
    # Clear cache every few iterations
    if i % 5 == 0:
        torch.cuda.empty_cache()

# Compute metrics
exact_match = sum([p == r for p, r in zip(predictions, references)]) / len(predictions)

print(f"Sample Exact Match ({sample_size} examples): {exact_match:.2%}")
print()

# Show some examples
print("Sample Predictions:")
for i in range(min(3, len(predictions))):
    print(f"\n  Example {i+1}:")
    print(f"  Expected: {references[i][:80]}")
    print(f"  Predicted: {predictions[i][:80]}")
    print(f"  Match: {'✅' if predictions[i] == references[i] else '❌'}")

print()


# ============================================================================
# CELL 11: Save Final Model
# ============================================================================

print("💾 Saving final model...")

final_model_dir = f"{OUTPUT_DIR}/final"
trainer.save_model(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

print(f"✅ Model saved to {final_model_dir}")

# Save metrics
with open(f"{OUTPUT_DIR}/training_metrics.json", 'w') as f:
    json.dump({
        'train_loss': float(train_result.training_loss),
        'test_metrics': {k: float(v) for k, v in test_results.items()}
    }, f, indent=2)

print("✅ Metrics saved")
print()


# ============================================================================
# CELL 12: Test Model with Examples
# ============================================================================

print("="*80)
print("  TESTING TRAINED MODEL")
print("="*80)
print()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

test_clauses = [
    {
        'clause': "The tenant must pay rent by the 5th of each month.",
        'context': {"Tenant": "PartyA"}
    },
    {
        'clause': "Supplier shall deliver goods within 10 business days.",
        'context': {"Supplier": "CompanyX"}
    },
    {
        'clause': "Either party may terminate with 30 days written notice.",
        'context': {"Party": "PartyA"}
    }
]

for i, example in enumerate(test_clauses, 1):
    print(f"\n{'='*80}")
    print(f"  EXAMPLE {i}")
    print(f"{'='*80}")
    
    # Prepare input
    clause = example['clause']
    context = example['context']
    context_str = " ".join([f"{k}={v}" for k, v in context.items()])
    input_text = f"translate to english logic: {clause} context: {context_str}"
    
    print(f"📥 Input: {clause}")
    print(f"📋 Context: {context}")
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate (memory-efficient settings)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,      # Reduced from 512
            num_beams=1,         # Greedy decoding (reduced from 4)
            do_sample=False
            # No early_stopping with greedy decoding
        )
    
    # Decode
    fopl = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"📤 FOPL: {fopl}")
    
    # Clear cache after each generation
    torch.cuda.empty_cache()

print()
print("="*80)
print("  ✅ ALL DONE!")
print("="*80)
print()
print("📥 Download model from Output tab:")
print(f"   {final_model_dir}/")
print()
print("📊 View training logs:")
print("   Click on TensorBoard tab (if available)")
print()
