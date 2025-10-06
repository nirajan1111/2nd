# Training Configuration Guide

## 🎯 How to Set Batch Size and Other Training Parameters

### Method 1: Command Line Arguments (RECOMMENDED)

```bash
# Set batch size
python main.py train --batch-size 32

# Set batch size and epochs
python main.py train --batch-size 32 --epochs 50

# Just epochs
python main.py train --epochs 30
```

### Method 2: Edit Configuration in Code

Edit `training/train.py`, line ~263:

```python
config = {
    'batch_size': 16,    # ← Change this
    'num_epochs': 20,    # ← Change this
    'learning_rate': 5e-5,
    # ... other params
}
```

### Method 3: Create a Config JSON File

Create `config.json`:
```json
{
    "batch_size": 32,
    "num_epochs": 50,
    "learning_rate": 3e-5,
    "max_length": 512
}
```

Then run:
```bash
python main.py train --config config.json
```

## 📊 Batch Size Recommendations

### Based on Your Hardware:

| Hardware | Recommended Batch Size | Memory Usage |
|----------|------------------------|--------------|
| **CPU (your case)** | 4-8 | Low |
| **GPU 8GB** | 16-32 | Medium |
| **GPU 16GB+** | 32-64 | High |

### Current Setup:
- **Default**: 16 (optimized for CPU)
- **Model**: T5-base (222M parameters)
- **Device**: CPU (MPS not fully supported)

## ⚙️ All Available Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 16 | Number of samples per batch |
| `num_epochs` | 20 | Training epochs |
| `learning_rate` | 5e-5 | Learning rate for optimizer |
| `weight_decay` | 0.01 | Weight decay (L2 regularization) |
| `warmup_steps` | 100 | Warmup steps for scheduler |
| `max_grad_norm` | 1.0 | Gradient clipping threshold |
| `max_length` | 512 | Maximum sequence length |
| `log_interval` | 10 | Log every N batches |
| `save_interval` | 5 | Save checkpoint every N epochs |
| `num_workers` | 4 | Data loading workers |

## 🚀 Performance Tips

### To Speed Up Training:
```bash
# Increase batch size (if you have memory)
python main.py train --batch-size 32

# Reduce epochs for quick experiments
python main.py train --epochs 5

# Use smaller model (edit code)
# Change 'model_name': 't5-small' in training/train.py
```

### To Improve Model Quality:
```bash
# Train longer
python main.py train --epochs 50

# Use smaller batch size with lower learning rate
python main.py train --batch-size 4
# (Then edit learning_rate to 3e-5 in code)
```

### If Running Out of Memory:
```bash
# Reduce batch size
python main.py train --batch-size 4

# Or reduce max_length in code (line ~274)
# 'max_length': 256
```

## 📈 Training Progress Monitoring

Watch for:
- **Loss decreasing**: ✅ Good! Model is learning
- **Val Loss < Train Loss**: ✅ Good! No overfitting
- **Val Loss > Train Loss (large gap)**: ⚠️ Overfitting
- **BLEU Score increasing**: ✅ Model generating better outputs

### Expected Timeline:
- **Epochs 1-5**: BLEU ~0-5%, Loss decreasing rapidly
- **Epochs 6-10**: BLEU ~5-15%, Loss stabilizing
- **Epochs 11-15**: BLEU ~15-30%, Refinement phase
- **Epochs 16-20**: BLEU ~30-50%+, Convergence

## 🔧 Quick Examples

```bash
# Fast experiment (small batch, few epochs)
python main.py train --batch-size 4 --epochs 5

# Standard training (current default)
python main.py train --batch-size 16 --epochs 20

# High-quality training (if you have time/resources)
python main.py train --batch-size 8 --epochs 50

# Resume from checkpoint (edit code to load checkpoint)
# See training/train.py for checkpoint loading
```

## 💾 Checkpoints

Models are saved to:
- `checkpoints/best_model/` - Best validation loss
- `checkpoints/checkpoint_epoch_N/` - Every 5 epochs

## 🎯 Current Setup (Default)

```
Batch Size: 16
Epochs: 20
Learning Rate: 5e-5
Device: CPU
Model: T5-base (222M params)
Training Time: ~10-12 min/epoch
Total Time: ~3-4 hours
```

---

**Need help?** Check the error messages or adjust parameters based on your hardware capabilities!
