#!/bin/bash
# Setup script for FUREcast GBR Demo

echo "🚀 Setting up FUREcast GBR Demo..."
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Please run this script from the gbr_ui_test directory"
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✅ Dependencies installed"
echo ""

# Check for .env file in parent directory
if [ ! -f "../.env" ]; then
    echo "⚠️  No .env file found in workspace root"
    echo "📝 Creating .env file from template..."
    cp .env.example ../.env
    echo "✅ Created ../.env"
    echo ""
    echo "⚠️  IMPORTANT: Edit ../.env and add your OpenAI API key!"
    echo "   The file is located at: $(cd .. && pwd)/.env"
    echo ""
else
    echo "✅ Found existing .env file"
fi

# Check if OPENAI_API_KEY is set
if grep -q "your_openai_api_key_here" "../.env" 2>/dev/null; then
    echo "⚠️  WARNING: OpenAI API key not configured!"
    echo "   Edit ../.env and replace 'your_openai_api_key_here' with your actual API key"
    echo ""
fi

echo "✨ Setup complete!"
echo ""
echo "To run the demo:"
echo "  streamlit run app.py"
echo ""
