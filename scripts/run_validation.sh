#!/bin/bash
# Comprehensive validation script for PSO-LSTM codebase

set -e

echo "=================================================="
echo "PSO-LSTM Codebase Validation Suite"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ PASS${NC}: $2"
    else
        echo -e "${RED}✗ FAIL${NC}: $2"
    fi
}

# A. Check environment setup
echo "A. Environment Checks"
echo "--------------------"

# Check .env exists
if [ -f .env ]; then
    print_status 0 ".env file exists"
else
    print_status 1 ".env file missing"
fi

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "  Python version: $python_version"

# Check if poetry is available
if command -v poetry &> /dev/null; then
    print_status 0 "Poetry installed"
else
    print_status 1 "Poetry not installed"
fi

echo ""

# B. Static Analysis
echo "B. Static Analysis"
echo "------------------"

# Check if ruff is available
if command -v ruff &> /dev/null || poetry run ruff --version &> /dev/null; then
    echo "Running Ruff linter..."
    if poetry run ruff check src/ --quiet; then
        print_status 0 "Ruff linting"
    else
        print_status 1 "Ruff linting (see errors above)"
    fi
else
    print_status 1 "Ruff not installed"
fi

# Check if black is available
if command -v black &> /dev/null || poetry run black --version &> /dev/null; then
    echo "Running Black formatter check..."
    if poetry run black --check src/ --quiet; then
        print_status 0 "Black formatting"
    else
        print_status 1 "Black formatting (run 'poetry run black src/' to fix)"
    fi
else
    print_status 1 "Black not installed"
fi

# Check if mypy is available
if command -v mypy &> /dev/null || poetry run mypy --version &> /dev/null; then
    echo "Running Mypy type checker..."
    if poetry run mypy src/ --ignore-missing-imports --no-error-summary 2>&1 | grep -q "Success"; then
        print_status 0 "Mypy type checking"
    else
        print_status 1 "Mypy type checking (see errors above)"
    fi
else
    print_status 1 "Mypy not installed"
fi

echo ""

# C. Unit Tests
echo "C. Unit Tests"
echo "-------------"

if command -v pytest &> /dev/null || poetry run pytest --version &> /dev/null; then
    echo "Running pytest..."
    if poetry run pytest tests/ -v --tb=short; then
        print_status 0 "Unit tests"
    else
        print_status 1 "Unit tests (see failures above)"
    fi
    
    echo ""
    echo "Running pytest with coverage..."
    poetry run pytest tests/ --cov=src --cov-report=term-missing --cov-report=html
else
    print_status 1 "Pytest not installed"
fi

echo ""

# D. Import Validation
echo "D. Import Validation"
echo "--------------------"

echo "Checking main.py imports..."
if python -c "import sys; sys.path.insert(0, '.'); from main import load_config" 2>/dev/null; then
    print_status 0 "main.py imports"
else
    print_status 1 "main.py imports (see errors above)"
fi

echo "Checking all src modules..."
all_imports_ok=true
for module in src/data/*.py src/features/*.py src/models/*.py src/training/*.py src/optimization/*.py src/evaluation/*.py src/utils/*.py; do
    if [ -f "$module" ] && [ "$(basename $module)" != "__init__.py" ]; then
        module_name=$(echo $module | sed 's/\.py$//' | sed 's/\//./g')
        if python -c "import $module_name" 2>/dev/null; then
            echo "  ✓ $module_name"
        else
            echo "  ✗ $module_name"
            all_imports_ok=false
        fi
    fi
done

if $all_imports_ok; then
    print_status 0 "All module imports"
else
    print_status 1 "Some module imports failed"
fi

echo ""

# E. Configuration Validation
echo "E. Configuration Validation"
echo "---------------------------"

echo "Checking Hydra configs..."
if python -c "
from hydra import initialize, compose
with initialize(version_base=None, config_path='configs'):
    cfg = compose(config_name='config')
    print('Config loaded successfully')
" 2>/dev/null; then
    print_status 0 "Hydra configuration"
else
    print_status 1 "Hydra configuration"
fi

echo ""

# F. Directory Structure
echo "F. Directory Structure"
echo "----------------------"

required_dirs=("data/raw" "data/interim" "data/processed" "logs" "models/checkpoints" "models/scalers" "reports/results" "tests" "src/data" "src/features" "src/models" "src/training" "src/optimization" "src/evaluation" "src/utils")

all_dirs_ok=true
for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✓ $dir"
    else
        echo "  ✗ $dir (missing)"
        all_dirs_ok=false
    fi
done

if $all_dirs_ok; then
    print_status 0 "Directory structure"
else
    print_status 1 "Some directories missing"
fi

echo ""

# G. Git Status
echo "G. Git Status"
echo "-------------"

if git rev-parse --git-dir > /dev/null 2>&1; then
    print_status 0 "Git repository initialized"
    
    # Check .gitignore
    if [ -f .gitignore ]; then
        if grep -q ".env" .gitignore && grep -q "logs/" .gitignore && grep -q "data/processed" .gitignore; then
            print_status 0 ".gitignore configured correctly"
        else
            print_status 1 ".gitignore missing required entries"
        fi
    else
        print_status 1 ".gitignore missing"
    fi
else
    print_status 1 "Not a git repository"
fi

echo ""
echo "=================================================="
echo "Validation Complete"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Fix any failing tests"
echo "  2. Run 'poetry run black src/' to format code"
echo "  3. Run 'poetry run ruff check src/ --fix' to auto-fix linting issues"
echo "  4. Review VALIDATION_REPORT.md for detailed findings"
echo ""
