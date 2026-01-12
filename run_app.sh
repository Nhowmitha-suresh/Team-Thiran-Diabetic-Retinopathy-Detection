#!/bin/bash
# Retinal AI - Diabetic Retinopathy Detection Application
# Run Script

echo "=========================================="
echo "Retinal AI - Starting Application"
echo "=========================================="
echo ""

# Navigate to project directory
cd "/home/adithya/Documents/Innovation lab/Team-Thiran-Diabetic-Retinopathy-Detection"

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if model exists
if [ ! -f "classifier.pt" ]; then
    echo "⚠️  Model file not found. Creating dummy model..."
    python create_model.py
fi

# Run the application
echo "🚀 Launching Retinal AI GUI..."
echo ""
python blindness.py
