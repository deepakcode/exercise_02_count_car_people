#!/bin/bash

set -e

REQ_FILE="requirements.txt"

echo "======================================="
echo "🚀 Setting up fresh virtual environment"
echo "======================================="

# Remove existing venv if present
if [ -d ".venv" ]; then
    echo "🗑️  Removing old .venv"
    rm -rf .venv
fi

# Create new venv with Python 3.12
echo "📦 Creating new .venv with Python 3.12"
python3.12 -m venv .venv

# Activate it
echo "⚡ Activating .venv"
source .venv/bin/activate

# Verify Python version
PY_VER=$(python --version 2>&1)
echo "🐍 Using $PY_VER"
if [[ "$PY_VER" != *"3.12"* ]]; then
    echo "❌ Error: This project requires Python 3.12"
    deactivate
    exit 1
fi

# Upgrade pip/setuptools/wheel
python -m pip install --upgrade pip setuptools wheel

# If requirements.txt is missing, bootstrap a default one
if [ ! -f "$REQ_FILE" ]; then
    echo "---------------------------------------"
    echo "⚠️  $REQ_FILE not found — creating a default one"
    cat > $REQ_FILE <<EOL
# Core ML stack (compatible with macOS Intel + Python 3.12)
numpy==1.26.4
torch==2.2.2
torchvision==0.17.2
transformers==4.44.2

# Image processing
opencv-python==4.9.0.80
pillow==10.4.0
matplotlib==3.9.2

# Data handling
pandas==2.2.2
EOL
    echo "✅ Created $REQ_FILE with locked versions"
fi

# Install from requirements.txt
echo "---------------------------------------"
echo "📖 Installing from $REQ_FILE"
python -m pip install -r $REQ_FILE

echo "======================================="
echo "🎉 Setup complete! Environment is ready"
echo "📖 Dependencies locked in $REQ_FILE"
echo "======================================="
