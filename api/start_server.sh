#!/bin/bash
# Simple script to start the FastAPI server

echo "🚀 Starting Contract Compliance Checker API..."
echo "📍 Working directory: $(pwd)"
echo ""

# Change to project directory
cd "$(dirname "$0")/.." || exit

# Start the server
/opt/miniconda3/bin/conda run -n major python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
