#!/bin/bash
# Quick start script for T5 FOPL training

echo "🚀 T5 FOPL Training - Quick Start"
echo "=================================="
echo ""

# Check if in correct directory
if [ ! -f "data/legal_clauses.json" ]; then
    echo "❌ Error: data/legal_clauses.json not found"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p checkpoints/t5_fopl
mkdir -p results

# Install required packages
echo ""
echo "📦 Installing required packages..."
pip install transformers datasets sacrebleu rouge-score tensorboard

# Show training options
echo ""
echo "=================================="
echo "  TRAINING OPTIONS"
echo "=================================="
echo ""
echo "1. Quick Training (t5-small, 5 epochs) - ~30 minutes"
echo "2. Standard Training (google/t5-v1_1-base, 10 epochs) - ~2-3 hours"
echo "3. Custom Training (specify parameters)"
echo ""
read -p "Select option (1-3): " option

case $option in
    1)
        echo ""
        echo "🏃 Starting QUICK training..."
        python training/train_t5_fopl.py \
            --model_name t5-small \
            --epochs 5 \
            --batch_size 16 \
            --lr 5e-5 \
            --output_dir checkpoints/t5_fopl_quick
        ;;
    2)
        echo ""
        echo "🚀 Starting STANDARD training..."
        python training/train_t5_fopl.py \
            --model_name google/t5-v1_1-base \
            --epochs 10 \
            --batch_size 8 \
            --lr 5e-5 \
            --output_dir checkpoints/t5_fopl
        ;;
    3)
        echo ""
        read -p "Model name (default: google/t5-v1_1-base): " model_name
        model_name=${model_name:-google/t5-v1_1-base}
        
        read -p "Epochs (default: 10): " epochs
        epochs=${epochs:-10}
        
        read -p "Batch size (default: 8): " batch_size
        batch_size=${batch_size:-8}
        
        read -p "Learning rate (default: 5e-5): " lr
        lr=${lr:-5e-5}
        
        read -p "Output directory (default: checkpoints/t5_fopl): " output_dir
        output_dir=${output_dir:-checkpoints/t5_fopl}
        
        echo ""
        echo "🔧 Starting CUSTOM training..."
        python training/train_t5_fopl.py \
            --model_name "$model_name" \
            --epochs "$epochs" \
            --batch_size "$batch_size" \
            --lr "$lr" \
            --output_dir "$output_dir"
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

echo ""
echo "=================================="
echo "  TRAINING COMPLETE!"
echo "=================================="
echo ""
echo "📊 View training logs with:"
echo "   tensorboard --logdir checkpoints/t5_fopl/logs"
echo ""
echo "🧪 Test the model with:"
echo "   python training/test_t5_fopl.py --mode examples"
echo ""
echo "📈 Evaluate on test set:"
echo "   python training/test_t5_fopl.py --mode eval"
echo ""
echo "🎮 Interactive demo:"
echo "   python training/test_t5_fopl.py --mode demo"
echo ""
