# Neural-Symbolic Legal Reasoning System

A hybrid AI system that combines neural networks (T5) with symbolic reasoning for legal contract analysis and compliance checking.

## 🎯 Overview

This system implements a **neural-symbolic architecture** for legal reasoning:

1. **Neural Parser (T5)**: Translates natural language legal clauses into First-Order Predicate Logic (FOPL)
2. **Symbolic Reasoner**: Performs logical inference to check compliance and detect contradictions
3. **Explanation Engine**: Generates human-readable proof traces

## 🏗️ Architecture

```
Legal Clause (Natural Language)
         ↓
    Neural Parser (T5)
         ↓
    FOPL Representation
         ↓
    Symbolic Reasoner
         ↓
    Compliance Decision + Proof Trace
```

### Components

| Component | Description | Technology |
|-----------|-------------|------------|
| **Neural Parser** | Clause → FOPL translation | T5 Transformer |
| **FOPL Generator** | Structured logic representation | Custom Parser |
| **Symbolic Reasoner** | Logic-based inference engine | Prolog-inspired |
| **Explanation Generator** | Proof trace generation | Rule-based |

## 📁 Project Structure

```
legal_reasoning_system/
│
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── main.py                   # Main execution script
│
├── data/
│   ├── generate_dataset.py  # Dataset generator
│   └── legal_clauses.json   # Generated dataset (100 clauses)
│
├── models/
│   ├── neural_parser.py     # T5-based neural parser
│   ├── fopl_generator.py    # FOPL generation utilities
│   └── symbolic_reasoner.py # Symbolic reasoning engine
│
├── training/
│   ├── train.py             # Training script
│   ├── dataset.py           # PyTorch dataset class
│   └── metrics.py           # Evaluation metrics
│
├── inference/
│   ├── pipeline.py          # End-to-end inference pipeline
│   └── explainer.py         # Explanation generation
│
└── checkpoints/             # Saved model checkpoints
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd legal_reasoning_system

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Dataset

Generate 100 legal clause examples with FOPL annotations:

```bash
python main.py generate --num-samples 100
```

This creates `data/legal_clauses.json` with diverse legal clauses covering:
- Payment obligations
- Termination clauses
- Maintenance requirements
- Liability clauses
- Confidentiality agreements
- And more...

### 3. Train the Model

Train the T5-based neural parser:

```bash
python main.py train
```

Training configuration:
- Model: T5-base
- Epochs: 20
- Batch size: 8
- Learning rate: 5e-5
- Optimizer: AdamW with warmup

### 4. Run Inference

Test the trained model on legal clauses:

```bash
# With trained model
python main.py inference --model-path checkpoints/best_model

# Without trained model (uses base T5)
python main.py inference
```

### 5. Test Components

Verify all system components:

```bash
python main.py test
```

## 📊 Example Usage

### Input Legal Clause

```
"The tenant must pay rent by the 5th of each month."
```

### Generated FOPL

```
forall x (Tenant(x) -> PayRent(x, due_date <= 5))
```

### Compliance Check

**Case 1: Paid on day 4**
```python
compliance_case = {"PartyA": {"PayRentDate": 4}}
# Result: ✅ COMPLIANT
```

**Case 2: Paid on day 10**
```python
compliance_case = {"PartyA": {"PayRentDate": 10}}
# Result: ❌ NON-COMPLIANT
```

### Proof Trace

```
1. Checking predicate: Tenant
2. ✓ Entity Tenant exists in context
3. Evaluating antecedent
4. Checking predicate: PayRent
5. Constraint: 4 <= 5 = True
6. → Implication result: True
```

## 🎓 Dataset Examples

The generated dataset includes 100 examples like:

```json
{
  "id": "clause_001",
  "clause_text": "The tenant must pay rent by the 5th of each month.",
  "context": {
    "Tenant": "PartyA",
    "Landlord": "PartyB"
  },
  "fopl_rule": "forall x (Tenant(x) -> PayRent(x, due_date <= 5))",
  "predicates_used": ["Tenant(x)", "PayRent(x, date)"],
  "variables": {"x": "Tenant"},
  "clause_type": "payment",
  "compliance_case": {"PartyA": {"PayRentDate": 4}},
  "expected_outcome": true
}
```

## 📈 Evaluation Metrics

The system tracks multiple metrics:

- **Exact Match**: Percentage of perfectly matching FOPL translations
- **Token Accuracy**: Token-level correctness
- **BLEU Score**: N-gram overlap measure
- **Predicate Accuracy**: Correct predicate extraction
- **Syntax Validity**: Percentage of syntactically valid FOPL
- **Compliance Accuracy**: Correct reasoning outcomes

## 🔧 Advanced Usage

### Custom Training Configuration

Create a `config.json` file:

```json
{
  "model_name": "t5-large",
  "batch_size": 16,
  "num_epochs": 30,
  "learning_rate": 3e-5,
  "max_length": 512
}
```

Train with custom config:

```bash
python main.py train --config config.json
```

### Custom Test Cases

Create `test_cases.json`:

```json
[
  {
    "clause_text": "Your custom legal clause",
    "context": {"Entity": "Party"},
    "compliance_case": {"Party": {"SomeFact": "value"}}
  }
]
```

Run inference:

```bash
python main.py inference --test-file test_cases.json
```

## 🧪 Testing

Run unit tests for all components:

```bash
# Test neural parser
python models/neural_parser.py

# Test symbolic reasoner
python models/symbolic_reasoner.py

# Test dataset
python training/dataset.py

# Test metrics
python training/metrics.py

# Test complete pipeline
python inference/pipeline.py
```

## 📝 Key Features

✅ **Neural-Symbolic Hybrid**: Combines deep learning with formal logic  
✅ **Interpretable**: Generates human-readable proof traces  
✅ **Extensible**: Easy to add new clause types and reasoning rules  
✅ **Accurate**: Validates both syntax and semantics  
✅ **Scalable**: Batch processing support  
✅ **Well-documented**: Comprehensive inline documentation  

## 🔍 FOPL Grammar

The system supports standard First-Order Predicate Logic:

- **Quantifiers**: `forall`, `exists`
- **Logical Operators**: `&` (AND), `|` (OR), `~` (NOT), `->` (IMPLIES)
- **Predicates**: `Predicate(args)`
- **Constraints**: `<=`, `>=`, `<`, `>`, `=`

Example:
```
forall x, y (Tenant(x) & Landlord(y) & ~PayRent(x) -> RightToTerminate(y, x))
```

## 🚧 Future Enhancements

- [ ] Switch to Mamba model for improved efficiency
- [ ] Add DeepStochLog for differentiable probabilistic reasoning
- [ ] Support for temporal logic (LTL/CTL)
- [ ] Multi-clause contract analysis
- [ ] Contradiction detection across clauses
- [ ] Interactive web interface
- [ ] Support for more languages

## 📚 References

- **T5**: Text-to-Text Transfer Transformer
- **DeepStochLog**: Neural Probabilistic Logic Programming
- **First-Order Logic**: Classical predicate logic
- **Legal AI**: Automated legal reasoning systems

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - feel free to use this project for research or commercial purposes.

## 🙋 Support

For questions or issues, please open an issue on GitHub.

---

**Built with ❤️ for advancing AI in legal tech**