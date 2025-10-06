# Kaggle Training Checklist - FIXED CODE

## ✅ Bug Fixes Applied

The model was generating malformed FOPL because of tokenizer issues. All fixes have been applied to the local code.

## 🚀 Upload to Kaggle

### Files to Upload/Update:

1. **models/neural_parser.py** ⚠️ CRITICAL FIX
   - Changed `add_special_tokens()` to `add_tokens()` 
   - Added comparison operators: `<=`, `>=`, `!=`

2. **training/train.py** ⚠️ CRITICAL FIX
   - Changed all `skip_special_tokens=True` → `False`
   - Added cleaning for actual special tokens (`</s>`, `<pad>`)

3. **data/legal_clauses.json** (optional - regenerate on Kaggle)
   - Can use existing 3000 samples or generate fresh

## 📝 Kaggle Notebook Setup

### Step 1: Clone and Update Code

```python
# In Kaggle notebook
!git clone https://github.com/nirajan1111/2nd.git
%cd 2nd

# Or manually upload the fixed files
```

### Step 2: Verify Tokenizer Fix

```python
from models.neural_parser import NeuralLegalParser

# Test the fixed tokenizer
parser = NeuralLegalParser(model_name="t5-base")
test_fopl = "forall x (Tenant(x) -> PayRent(x, due_date <= 5))"
tokens = parser.tokenizer.encode(test_fopl)
decoded = parser.tokenizer.decode(tokens, skip_special_tokens=False)
print(f"Original: {test_fopl}")
print(f"Decoded:  {decoded}")

# Should output: forall x ( Tenant ( x ) -> PayRent ( x , due_date <= 5 ) )
# NOT: x Tenant x PayRent x due_date = 5
```

### Step 3: Generate Dataset (if needed)

```python
!python main.py generate --num-samples 3000
```

### Step 4: Train with Fixed Code

```python
!python main.py train --epochs 5 --batch-size 8
```

### Step 5: Verify Output During Training

**IMPORTANT**: Watch the sample predictions printed after each epoch!

Expected output:
```
Sample Predictions (First 3):
  Example 1:
    Predicted: forall x (Tenant(x) -> PayRent(x, due_date <= 5))
    Reference: forall x (Tenant(x) -> PayRent(x, due_date <= 5))
```

If you see this instead, **STOP - bug not fixed**:
```
Predicted: x Tenant x PayRent x due_date = 5  ❌ WRONG!
```

### Step 6: Test Inference

```python
!python main.py inference --model-path checkpoints/best_model
```

Expected FOPL output should have:
- ✅ `forall` quantifier
- ✅ Parentheses `( )`
- ✅ Logical operators `->`, `&`, `~`
- ✅ Comparison operators `<=`, `>=`, `=`

## 🎯 Success Criteria

| Metric | Expected |
|--------|----------|
| Val Loss | < 0.1 |
| Exact Match | > 95% |
| BLEU Score | > 0.95 |
| **FOPL Structure** | ✅ **Must have forall, (), ->** |

## ⚠️ Common Issues

### Issue: Still seeing malformed FOPL
**Solution**: Make sure you uploaded the FIXED neural_parser.py and train.py files

### Issue: tokenizer.add_tokens() not working
**Solution**: You're loading an old checkpoint. Start fresh with t5-base

### Issue: Model not improving
**Solution**: Check learning rate, increase epochs, verify data quality

## 📥 Download Trained Model

After successful training:

```python
# In Kaggle, create output directory
!mkdir -p /kaggle/working/final_model
!cp -r checkpoints/best_model/* /kaggle/working/final_model/

# These files will appear in Kaggle Output section for download
```

## 🔗 Quick Test Script

Add this cell to verify everything works:

```python
from models.neural_parser import NeuralLegalParser

# Load trained model
parser = NeuralLegalParser()
parser.load_pretrained('checkpoints/best_model')

# Test
test_clause = "The tenant must pay rent by the 5th of each month."
test_context = {'Tenant': 'PartyA', 'Landlord': 'PartyB'}
fopl = parser.parse(test_clause, test_context)

print(f"Input: {test_clause}")
print(f"FOPL:  {fopl}")

# Expected: forall x (Tenant(x) -> PayRent(x, due_date <= 5))
# Check for these markers:
assert 'forall' in fopl, "❌ Missing forall!"
assert '(' in fopl and ')' in fopl, "❌ Missing parentheses!"
assert '->' in fopl, "❌ Missing implication operator!"
print("✅ All checks passed!")
```

---

**Once training completes successfully on Kaggle, download the checkpoint and test it locally!**
