#!/bin/bash
# Verification script to check the reorganized structure

echo "=========================================="
echo "Verifying Reorganized Code Structure"
echo "=========================================="
echo ""

ERRORS=0

# Check 1: Verify no Python files in scripts/
echo "✓ Checking scripts/ directory..."
PY_FILES=$(ls scripts/*.py 2>/dev/null | wc -l)
if [ $PY_FILES -eq 0 ]; then
    echo "  ✓ No Python files in scripts/ (correct)"
else
    echo "  ✗ Found $PY_FILES Python files in scripts/ (should be 0)"
    ERRORS=$((ERRORS + 1))
fi

# Check 2: Verify shell scripts are executable
echo ""
echo "✓ Checking shell script permissions..."
NON_EXEC=$(find scripts/ -name "*.sh" ! -perm -u+x | wc -l)
if [ $NON_EXEC -eq 0 ]; then
    echo "  ✓ All shell scripts are executable"
else
    echo "  ✗ Found $NON_EXEC non-executable shell scripts"
    find scripts/ -name "*.sh" ! -perm -u+x
    ERRORS=$((ERRORS + 1))
fi

# Check 3: Verify pipelines directory has Python files
echo ""
echo "✓ Checking pipelines/ directory..."
PY_COUNT=$(ls pipelines/*.py 2>/dev/null | wc -l)
if [ $PY_COUNT -ge 10 ]; then
    echo "  ✓ Found $PY_COUNT Python files in pipelines/"
else
    echo "  ⚠ Only found $PY_COUNT Python files in pipelines/ (expected 10+)"
fi

# Check 4: Verify removed files are gone
echo ""
echo "✓ Checking removed files..."
REMOVED_FILES=(
    "src/models/xgboost/xgboost_train.py"
    "src/models/xgboost/xgboost_val.py"
    "src/models/xgboost/xgboost_test.py"
    "src/models/xgboost/helpers.py"
    "src/models/xgboost/consts.py"
    "src/models/lstm/lstm_baseline"
)

for file in "${REMOVED_FILES[@]}"; do
    if [ -e "$file" ]; then
        echo "  ✗ File still exists: $file"
        ERRORS=$((ERRORS + 1))
    fi
done
echo "  ✓ All removed files are gone"

# Check 5: Verify new pipeline files exist
echo ""
echo "✓ Checking new pipeline files..."
NEW_FILES=(
    "pipelines/train_xgboost.py"
    "pipelines/validate_xgboost.py"
    "pipelines/test_xgboost.py"
    "pipelines/backtest.py"
    "pipelines/evaluate.py"
    "pipelines/run_pso.py"
    "pipelines/train_rl_agent.py"
    "pipelines/evaluate_rl_agent.py"
)

for file in "${NEW_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "  ✗ Missing file: $file"
        ERRORS=$((ERRORS + 1))
    fi
done
echo "  ✓ All new pipeline files exist"

# Check 6: Verify new shell scripts exist
echo ""
echo "✓ Checking new shell scripts..."
NEW_SCRIPTS=(
    "scripts/run_data_ingestion.sh"
    "scripts/run_feature_engineering.sh"
    "scripts/run_xgboost_pipeline.sh"
    "scripts/run_lstm_pipeline.sh"
)

for script in "${NEW_SCRIPTS[@]}"; do
    if [ ! -f "$script" ]; then
        echo "  ✗ Missing script: $script"
        ERRORS=$((ERRORS + 1))
    fi
done
echo "  ✓ All new shell scripts exist"

# Check 7: Test Python syntax (not full imports due to dependencies)
echo ""
echo "✓ Testing Python syntax..."

python3 -m py_compile src/models/xgboost/xgboost_model.py 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  ✓ XGBoost model syntax is valid"
else
    echo "  ✗ XGBoost model has syntax errors"
    ERRORS=$((ERRORS + 1))
fi

python3 -m py_compile src/models/lstm/lstm_model.py 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  ✓ LSTM model syntax is valid"
else
    echo "  ✗ LSTM model has syntax errors"
    ERRORS=$((ERRORS + 1))
fi

python3 -m py_compile src/models/baselines.py 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  ✓ Baselines syntax is valid"
else
    echo "  ✗ Baselines has syntax errors"
    ERRORS=$((ERRORS + 1))
fi

python3 -m py_compile pipelines/train_xgboost.py 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  ✓ XGBoost training pipeline syntax is valid"
else
    echo "  ✗ XGBoost training pipeline has syntax errors"
    ERRORS=$((ERRORS + 1))
fi

echo "  ℹ Note: Full import testing requires dependencies (numpy, torch, etc.)"

# Summary
echo ""
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo "✓ All checks passed! Structure is correct."
    echo "=========================================="
    exit 0
else
    echo "✗ Found $ERRORS error(s)"
    echo "=========================================="
    exit 1
fi
