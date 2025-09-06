#!/bin/bash

set -e

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

# Function: check and install
check_and_install () {
    PACKAGE=$1
    MODULE=$2
    EXTRA=$3

    echo "---------------------------------------"
    echo "🔍 Checking $PACKAGE ..."
    if python -c "import $MODULE" 2>/dev/null; then
        echo "✅ $PACKAGE already installed"
    else
        echo "⬇️ Installing $PACKAGE ..."
        python -m pip install $PACKAGE $EXTRA
    fi
}

# Core dependencies
check_and_install torch torch "--index-url https://download.pytorch.org/whl/cpu"
check_and_install torchvision torchvision "--index-url https://download.pytorch.org/whl/cpu"
check_and_install transformers transformers
check_and_install opencv-python cv2
check_and_install pandas pandas
check_and_install pillow PIL
check_and_install matplotlib matplotlib

# YOLOv5 requirements
echo "---------------------------------------"
echo "🔍 Checking YOLOv5 dependencies ..."
if python -c "import yolov5" 2>/dev/null; then
    echo "✅ YOLOv5 already present"
else
    echo "⬇️ Installing YOLOv5 requirements ..."
    python -m pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
fi

echo "======================================="
echo "🎉 Setup complete! Environment is ready"
echo "======================================="
