# Bug Fix Report: FOPL Generation Issue

## Date: October 6, 2025

## 🔴 Problem Identified

The trained model was generating **malformed FOPL output**:
```
❌ BAD:  x închirier x PayRent x due date = 5
✅ GOOD: forall x (Tenant(x) -> PayRent(x, due_date <= 5))
```

Despite 100% accuracy metrics during training, the model learned to output corrupted FOPL without proper quantifiers, parentheses, or logical operators.

## 🔍 Root Cause Analysis

### Issue 1: Special Tokens vs. Regular Tokens

**Problem**: FOPL tokens (`forall`, `->`, `&`, `(`, `)`, etc.) were added as **special tokens** using `add_special_tokens()`.

**Impact**: When `skip_special_tokens=True` is used during decoding, these critical FOPL structure tokens are REMOVED, resulting in:
- `forall` → removed
- `(` `)` → removed  
- `&` → removed
- `->` → removed

### Issue 2: Training Code Used skip_special_tokens=True

**Location**: `training/train.py` lines 151-160

**Problem**: During validation and testing, the code decoded predictions with:
```python
preds = tokenizer.batch_decode(generated, skip_special_tokens=True)  # BUG!
```

This meant:
1. Training labels were decoded without FOPL structure
2. Model learned to output the corrupted format
3. Metrics compared corrupted predictions to corrupted references (both wrong but "matching")
4. 100% accuracy was achieved on the WRONG output format!

## 🔧 Fixes Applied

### Fix 1: Change Special Tokens to Regular Vocabulary

**File**: `models/neural_parser.py` lines 19-32

**Before**:
```python
special_tokens = {
    'additional_special_tokens': ['forall', 'exists', '&', ...]
}
self.tokenizer.add_special_tokens(special_tokens)
```

**After**:
```python
new_tokens = ['forall', 'exists', '&', '|', '~', '->', '<->', ...]
num_added = self.tokenizer.add_tokens(new_tokens)  # Regular tokens, not special!
```

### Fix 2: Update Training Decoding

**File**: `training/train.py` lines 151-160, 361-366

**Changed**:
- `skip_special_tokens=True` → `skip_special_tokens=False`
- Added cleanup for actual special tokens (`</s>`, `<pad>`)

**Before**:
```python
preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
```

**After**:
```python
preds = tokenizer.batch_decode(generated, skip_special_tokens=False)
preds = [p.replace('</s>', '').replace('<pad>', '').strip() for p in preds]
```

### Fix 3: Update Inference Decoding

**File**: `models/neural_parser.py` lines 111, 159

**Same changes**: Use `skip_special_tokens=False` and clean up actual special tokens.

## ✅ Expected Results After Fix

### Training
- Model will now see correct FOPL structure in labels
- Metrics will validate against proper FOPL format
- Sample predictions during training will show `forall x (...)`

### Inference
- Generated FOPL will have proper structure:
  ```
  forall x (Tenant(x) -> PayRent(x, due_date <= 5))
  ```

## 🚀 Next Steps

### 1. Generate Fresh Dataset
```bash
python main.py generate --num-samples 2000
```

### 2. Train with Fixed Code
```bash
python main.py train --epochs 5 --batch-size 8
```

### 3. Verify Output
During training, check the sample predictions printed after each epoch. They should now show:
```
Predicted: forall x (Tenant(x) -> PayRent(x, due_date <= 5))
Reference: forall x (Tenant(x) -> PayRent(x, due_date <= 5))
```

### 4. Deploy to Kaggle
Upload the fixed code to Kaggle and retrain there with GPU acceleration.

## 📝 Lessons Learned

1. **Never add structural syntax as special tokens** - use regular vocabulary tokens instead
2. **Always print actual predictions during training** - don't trust metrics alone
3. **Be careful with skip_special_tokens parameter** - understand what it removes
4. **Test tokenizer encode/decode cycles** before training

## 🔗 Related Files

- `models/neural_parser.py` - Tokenizer initialization
- `training/train.py` - Training loop and validation
- `training/dataset.py` - Data loading (correct, no changes needed)
