#!/usr/bin/env bash
# ==============================================================================
# Barca Vision Labeler - macOS & Linux Launcher
# ==============================================================================

set -e

# Change directory to the repository root where this script resides
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================================"
echo " ⚽ Launching Barca Vision Labeler Studio..."
echo "========================================================"

# Determine Python 3 executable
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Error: Python 3 is not installed or not found in PATH."
    echo "Please install Python 3 (e.g. brew install python) and try again."
    exit 1
fi

# Check required packages
if ! $PYTHON_CMD -c "import flask, numpy, pandas" &>/dev/null; then
    echo "📦 Installing required dependencies (flask, numpy, pandas)..."
    $PYTHON_CMD -m pip install -r labeler/requirements.txt
fi

echo "🚀 Starting server..."
$PYTHON_CMD labeler/run_labeler.py
