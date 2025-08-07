#!/bin/bash
# Robust setup script using uv for fast dependency installation

set -e  # Exit on error

echo "🚀 Setting up environment with uv..."

# Get the script directory (works even if script is sourced)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check for multiple possible uv installation paths
check_uv() {
    if command -v uv &> /dev/null; then
        return 0
    fi
    
    # Check common installation paths
    for path in "$HOME/.local/bin" "$HOME/.cargo/bin" "/usr/local/bin"; do
        if [ -f "$path/uv" ]; then
            export PATH="$path:$PATH"
            return 0
        fi
    done
    
    return 1
}

# Install uv if not present
if ! check_uv; then
    echo "📦 Installing uv..."
    
    # Check for internet connectivity
    if ! curl -s --head https://astral.sh &> /dev/null; then
        echo "❌ Error: Cannot reach astral.sh. Please check your internet connection."
        echo "💡 Falling back to standard pip installation:"
        echo "   Run: pip install -r requirements.txt"
        exit 1
    fi
    
    # Install uv with error handling
    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        echo "❌ Error: Failed to install uv"
        echo "💡 Falling back to standard pip installation:"
        echo "   Run: pip install -r requirements.txt"
        exit 1
    fi
    
    # Re-check after installation
    if ! check_uv; then
        echo "❌ Error: uv installed but not found in PATH"
        exit 1
    fi
fi

# Use .venv as default, allow override with environment variable
VENV_NAME="${VENV_NAME:-.venv}"

# Check if venv already exists
if [ -d "$VENV_NAME" ]; then
    echo "⚠️  Virtual environment '$VENV_NAME' already exists."
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "📌 Using existing virtual environment."
    else
        echo "🗑️  Removing existing virtual environment..."
        rm -rf "$VENV_NAME"
        echo "🔧 Creating fresh virtual environment at $VENV_NAME..."
        uv venv "$VENV_NAME"
    fi
else
    echo "🔧 Creating virtual environment at $VENV_NAME..."
    uv venv "$VENV_NAME"
fi

# Install requirements with uv
echo "📥 Installing requirements with uv (this will be fast!)..."
if ! uv pip install -r requirements.txt; then
    echo "❌ Error: Failed to install requirements"
    exit 1
fi

# Prepare activation command
ACTIVATE_CMD="source $SCRIPT_DIR/$VENV_NAME/bin/activate"

# Verify installation using the venv's Python
echo "🔍 Verifying installation..."
PYTHON_CMD="$SCRIPT_DIR/$VENV_NAME/bin/python"

if ! $PYTHON_CMD -c "import torch; print(f'✅ PyTorch {torch.__version__}')" 2>/dev/null; then
    echo "⚠️  PyTorch not installed or import failed"
fi

if ! $PYTHON_CMD -c "import transformers; print(f'✅ Transformers {transformers.__version__}')" 2>/dev/null; then
    echo "⚠️  Transformers not installed or import failed"
fi

if ! $PYTHON_CMD -c "import trl; print(f'✅ TRL {trl.__version__}')" 2>/dev/null; then
    echo "⚠️  TRL not installed or import failed"
fi

if ! $PYTHON_CMD -c "import wandb; print(f'✅ WandB {wandb.__version__}')" 2>/dev/null; then
    echo "⚠️  WandB not installed or import failed"
fi

echo ""
echo "🎉 Setup complete! Virtual environment is ready at $VENV_NAME"
echo "📝 To activate, run: $ACTIVATE_CMD"

# If script is being sourced, activate the environment
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "🔄 Auto-activating environment (script was sourced)..."
    source "$SCRIPT_DIR/$VENV_NAME/bin/activate"
fi