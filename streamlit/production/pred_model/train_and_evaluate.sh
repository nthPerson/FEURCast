#!/bin/bash

# FUREcast Model Training Quick Start Script
# This script trains the GBR model and generates evaluation reports

echo "=================================="
echo "FUREcast Model Training Pipeline"
echo "=================================="
echo ""

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if data file exists
DATA_FILE="../../../data/rich_features_SPLG_history_full.csv"
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ Error: Data file not found at $DATA_FILE"
    echo "Please ensure the feature-engineered dataset exists."
    exit 1
fi

echo "✓ Data file found"
echo ""

# Check Python dependencies
echo "Checking Python dependencies..."
python3 -c "import numpy, pandas, sklearn, matplotlib, seaborn, joblib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Missing dependencies. Installing..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
fi

echo "✓ Dependencies satisfied"
echo ""

# Parse arguments
TUNE_FLAG=""
if [ "$1" == "--tune" ]; then
    TUNE_FLAG="--tune"
    echo "🔍 Running with hyperparameter tuning (this will take longer)"
else
    TUNE_FLAG="--quick"
    echo "⚡ Running with recommended hyperparameters (quick mode)"
fi
echo ""

# Train model
echo "=================================="
echo "Step 1: Training Model"
echo "=================================="
python3 scripts/train_gbr_model.py $TUNE_FLAG
if [ $? -ne 0 ]; then
    echo "❌ Training failed"
    exit 1
fi

echo ""
echo "=================================="
echo "Step 2: Evaluating Model"
echo "=================================="
python3 scripts/evaluate_model.py
if [ $? -ne 0 ]; then
    echo "❌ Evaluation failed"
    exit 1
fi

echo ""
echo "=================================="
echo "✅ Training Complete!"
echo "=================================="
echo ""
echo "Results:"
echo "  📊 Model:      models/gbr_model.pkl"
echo "  📈 Metrics:    models/metrics.json"
echo "  🖼️  Plots:      plots/*.png"
echo "  📝 Logs:       logs/*.txt"
echo ""
echo "Next steps:"
echo "  1. Review metrics:  cat models/metrics.json"
echo "  2. View plots:      open plots/"
echo "  3. Test prediction: python3 predict.py"
echo "  4. Run Streamlit:   cd .. && streamlit run app.py"
echo ""
