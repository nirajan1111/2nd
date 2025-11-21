"""
Pre-training Checklist - Verify everything is ready for T5 training
"""

import sys
import os
import json

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 8:
        print("  ✅ Python version OK")
        return True
    else:
        print("  ❌ Python 3.8+ required")
        return False

def check_dependencies():
    """Check required packages"""
    required = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'datasets': 'Datasets',
        'numpy': 'NumPy'
    }
    
    missing = []
    for package, name in required.items():
        try:
            __import__(package)
            print(f"  ✅ {name} installed")
        except ImportError:
            print(f"  ❌ {name} NOT installed")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    return True

def check_data():
    """Check training data"""
    data_path = 'data/legal_clauses.json'
    
    if not os.path.exists(data_path):
        print(f"  ❌ {data_path} not found")
        return False
    
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        num_examples = len(data)
        print(f"  ✅ Training data found: {num_examples} examples")
        
        if num_examples < 100:
            print(f"  ⚠️  Warning: Only {num_examples} examples (recommend 1000+)")
        
        # Check data format
        if data and all(key in data[0] for key in ['clause_text', 'fopl_rule']):
            print("  ✅ Data format valid")
        else:
            print("  ❌ Data format invalid")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error loading data: {e}")
        return False

def check_predicates():
    """Check predicates file"""
    predicates_path = 'data/predicates.txt'
    
    if os.path.exists(predicates_path):
        with open(predicates_path, 'r') as f:
            predicates = [line.strip() for line in f if line.strip()]
        print(f"  ✅ Predicates file found: {len(predicates)} predicates")
        return True
    else:
        print(f"  ⚠️  {predicates_path} not found (will use default predicates)")
        return True  # Not critical

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  ℹ️  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("  ⚠️  No GPU detected (training will be slower on CPU)")
            return True  # Not critical, just slower
    except ImportError:
        print("  ❌ PyTorch not installed")
        return False

def check_disk_space():
    """Check available disk space"""
    try:
        import shutil
        stats = shutil.disk_usage('.')
        free_gb = stats.free / (1024**3)
        print(f"  ℹ️  Free disk space: {free_gb:.1f} GB")
        
        if free_gb < 5:
            print("  ⚠️  Warning: Low disk space (recommend 10+ GB)")
        else:
            print("  ✅ Sufficient disk space")
        
        return True
    except Exception as e:
        print(f"  ⚠️  Could not check disk space: {e}")
        return True  # Not critical

def estimate_training_time():
    """Estimate training time"""
    try:
        import torch
        
        # Load data size
        with open('data/legal_clauses.json', 'r') as f:
            num_examples = len(json.load(f))
        
        if torch.cuda.is_available():
            time_estimate = "2-3 hours (GPU)"
        else:
            time_estimate = "8-12 hours (CPU)"
        
        print(f"\n📊 Training Estimates:")
        print(f"  • Dataset size: {num_examples} examples")
        print(f"  • Estimated time (10 epochs): {time_estimate}")
        print(f"  • Model size: ~220M parameters (t5-v1_1-base)")
        print(f"  • Checkpoint size: ~850 MB per checkpoint")
        
    except Exception as e:
        print(f"  ⚠️  Could not estimate: {e}")

def main():
    print("=" * 80)
    print("  T5 FOPL TRAINING - PRE-FLIGHT CHECKLIST")
    print("=" * 80)
    print()
    
    checks = []
    
    print("1️⃣  Checking Python version...")
    checks.append(check_python_version())
    print()
    
    print("2️⃣  Checking dependencies...")
    checks.append(check_dependencies())
    print()
    
    print("3️⃣  Checking training data...")
    checks.append(check_data())
    print()
    
    print("4️⃣  Checking predicates file...")
    checks.append(check_predicates())
    print()
    
    print("5️⃣  Checking GPU...")
    checks.append(check_gpu())
    print()
    
    print("6️⃣  Checking disk space...")
    checks.append(check_disk_space())
    print()
    
    estimate_training_time()
    
    print()
    print("=" * 80)
    
    if all(checks):
        print("  ✅ ALL CHECKS PASSED - READY TO TRAIN!")
        print("=" * 80)
        print()
        print("🚀 Start training with:")
        print("   ./start_t5_training.sh")
        print()
        print("Or directly:")
        print("   python training/train_t5_fopl.py --model_name google/t5-v1_1-base")
        print()
        return 0
    else:
        print("  ❌ SOME CHECKS FAILED - PLEASE FIX ISSUES ABOVE")
        print("=" * 80)
        print()
        print("Install missing dependencies:")
        print("   pip install transformers datasets sacrebleu rouge-score tensorboard")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
