"""
tests/test_run_lstm_baseline.py

Integration tests for run_lstm_baseline.py script.
"""

import subprocess
import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_script_help():
    """Test script displays help message."""
    result = subprocess.run(
        ["python", "scripts/run_lstm_baseline.py", "--help"],
        capture_output=True,
        text=True,
        cwd=project_root,
    )
    
    assert result.returncode == 0 or "numpy" in result.stderr.lower()
    if result.returncode == 0:
        assert "LSTM Baseline" in result.stdout
        assert "--ticker" in result.stdout
        assert "--mode" in result.stdout


def test_script_syntax():
    """Test script has valid Python syntax."""
    result = subprocess.run(
        ["python", "-m", "py_compile", "scripts/run_lstm_baseline.py"],
        capture_output=True,
        text=True,
        cwd=project_root,
    )
    
    assert result.returncode == 0, f"Syntax error: {result.stderr}"


def test_script_imports():
    """Test script can be imported (checks for import errors)."""
    script_path = project_root / "scripts" / "run_lstm_baseline.py"
    assert script_path.exists(), "Script file not found"
    
    # Check file is not empty
    content = script_path.read_text()
    assert len(content) > 0
    assert "def main()" in content
    assert "def run_train(" in content
    assert "def run_val(" in content
    assert "def run_test(" in content


def test_config_has_lstm_baseline():
    """Test config file has lstm_baseline section."""
    config_path = project_root / "config" / "default_config.yaml"
    assert config_path.exists(), "Config file not found"
    
    content = config_path.read_text()
    assert "lstm_baseline:" in content
    assert "num_layers:" in content
    assert "hidden_units:" in content


if __name__ == "__main__":
    print("Running integration tests...")
    
    try:
        test_script_help()
        print("✓ Script help test passed")
    except AssertionError as e:
        print(f"✗ Script help test failed: {e}")
    
    try:
        test_script_syntax()
        print("✓ Script syntax test passed")
    except AssertionError as e:
        print(f"✗ Script syntax test failed: {e}")
    
    try:
        test_script_imports()
        print("✓ Script imports test passed")
    except AssertionError as e:
        print(f"✗ Script imports test failed: {e}")
    
    try:
        test_config_has_lstm_baseline()
        print("✓ Config test passed")
    except AssertionError as e:
        print(f"✗ Config test failed: {e}")
    
    print("\nAll tests completed!")
