#!/bin/bash

# Neural-Symbolic Legal Reasoning System Setup Script

echo "=================================================="
echo "Setting up Legal Reasoning System"
echo "=================================================="
echo ""

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p data
mkdir -p models
mkdir -p training
mkdir -p inference
mkdir -p utils
mkdir -p checkpoints
mkdir -p outputs

# Create __init__.py files
echo "📝 Creating __init__.py files..."

cat > models/__init__.py << 'EOF'
"""Neural and symbolic models for legal reasoning"""
from .neural_parser import NeuralLegalParser, FOPLValidator
from .symbolic_reasoner import SymbolicReasoner, ReasoningResult

__all__ = ['NeuralLegalParser', 'FOPLValidator', 'SymbolicReasoner', 'ReasoningResult']
EOF

cat > training/__init__.py << 'EOF'
"""Training utilities and dataset classes"""
from .dataset import LegalReasoningDataset, create_dataloaders
from .metrics import compute_metrics, print_metrics_report
from .train import Trainer

__all__ = ['LegalReasoningDataset', 'create_dataloaders', 'compute_metrics', 'print_metrics_report', 'Trainer']
EOF

cat > inference/__init__.py << 'EOF'
"""Inference pipeline and explanation generation"""
from .pipeline import LegalReasoningPipeline

__all__ = ['LegalReasoningPipeline']
EOF

cat > utils/__init__.py << 'EOF'
"""Utility functions"""
__all__ = []
EOF

cat > data/__init__.py << 'EOF'
"""Data generation and processing"""
__all__ = []
EOF

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "=================================================="
echo "Quick Start Guide"
echo "=================================================="
echo ""
echo "1. Generate dataset:"
echo "   python main.py generate --num-samples 100"
echo ""
echo "2. Test components:"
echo "   python main.py test"
echo ""
echo "3. Train model:"
echo "   python main.py train"
echo ""
echo "4. Run inference:"
echo "   python main.py inference"
echo ""
echo "=================================================="