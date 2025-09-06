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
echo "---------------------------------------"
echo "⚡ Upgrading pip, setuptools, wheel"
python -m pip install --upgrade pip setuptools wheel

# Install numpy<2 first
echo "---------------------------------------"
echo "📦 Installing numpy<2"
pip install --force-reinstall "numpy<2"

# Install OpenCV compatible with numpy<2
echo "---------------------------------------"
echo "📦 Installing OpenCV 4.9.0.80"
pip install --force-reinstall opencv-python==4.9.0.80

# Install PyTorch CPU version compatible with numpy<2
echo "---------------------------------------"
echo "📦 Installing PyTorch 2.2.2 + torchvision 0.17.2 (CPU)"
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# Install YOLO ultralytics
echo "---------------------------------------"
echo "📦 Installing ultralytics 8.0.171"
pip install ultralytics==8.0.171

# Install timm (DETR dependency)
echo "---------------------------------------"
echo "📦 Installing timm"
pip install timm

# Install remaining packages but force numpy<2 to prevent upgrade
echo "---------------------------------------"
echo "📦 Installing transformers, pandas, matplotlib"
pip install --force-reinstall transformers==4.44.2 pandas==2.2.2 matplotlib==3.9.2 "numpy<2"

# Freeze exact versions into requirements.txt
echo "---------------------------------------"
echo "📌 Freezing working versions into $REQ_FILE"
pip freeze > $REQ_FILE

echo "======================================="
echo "🎉 Setup complete! Environment is ready"
echo "📖 Dependencies locked in $REQ_FILE"
echo "======================================="
