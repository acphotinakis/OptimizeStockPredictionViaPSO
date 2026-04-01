"""
Unit tests for utility modules.
"""
import pytest
import torch
import numpy as np
import random
import logging
import tempfile
from pathlib import Path
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.utils.paths import ProjectPaths
from src.utils.config_validator import validate_config, validate_environment


def test_set_seed_reproducibility():
    """Test that set_seed produces reproducible results."""
    set_seed(42)
    
    # Generate random numbers
    py_rand1 = random.random()
    np_rand1 = np.random.rand()
    torch_rand1 = torch.rand(1).item()
    
    # Reset seed
    set_seed(42)
    
    # Generate again
    py_rand2 = random.random()
    np_rand2 = np.random.rand()
    torch_rand2 = torch.rand(1).item()
    
    # Should be identical
    assert py_rand1 == py_rand2
    assert np_rand1 == np_rand2
    assert torch_rand1 == torch_rand2


def test_set_seed_cuda_available():
    """Test CUDA seed setting if available."""
    set_seed(42)
    
    if torch.cuda.is_available():
        # Should not raise error
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False


def test_setup_logger_creates_file():
    """Test that logger creates log file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = setup_logger("test_logger", log_dir=tmpdir)
        
        # Check log file was created
        log_files = list(Path(tmpdir).glob("test_logger_*.log"))
        assert len(log_files) == 1
        
        # Test logging
        logger.info("Test message")
        
        # Verify message was written
        with open(log_files[0], "r") as f:
            content = f.read()
            assert "Test message" in content


def test_setup_logger_dual_handlers():
    """Test that logger has both file and console handlers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = setup_logger("test_logger", log_dir=tmpdir)
        
        # Should have 2 handlers (file + console)
        assert len(logger.handlers) == 2


def test_setup_logger_no_duplicate_handlers():
    """Test that re-initializing logger doesn't add duplicate handlers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger1 = setup_logger("test_logger", log_dir=tmpdir)
        logger2 = setup_logger("test_logger", log_dir=tmpdir)
        
        # Should still only have 2 handlers
        assert len(logger2.handlers) == 2


def test_project_paths_initialization(mock_config):
    """Test ProjectPaths initialization."""
    paths = ProjectPaths(mock_config)
    
    assert paths.raw_data == Path("data/raw")
    assert paths.interim_data == Path("data/interim")
    assert paths.processed_data == Path("data/processed")


def test_project_paths_ensure_directories(mock_config):
    """Test directory creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Modify config to use temp directory
        mock_config.paths.data_storage.raw = str(Path(tmpdir) / "raw")
        mock_config.paths.data_storage.interim = str(Path(tmpdir) / "interim")
        mock_config.paths.data_storage.processed = str(Path(tmpdir) / "processed")
        mock_config.paths.model_save_path = str(Path(tmpdir) / "models")
        mock_config.paths.results_dir = str(Path(tmpdir) / "results")
        
        paths = ProjectPaths(mock_config)
        paths.ensure_directories()
        
        # Check all directories were created
        assert paths.raw_data.exists()
        assert paths.interim_data.exists()
        assert paths.processed_data.exists()
        assert paths.checkpoints.exists()
        assert paths.results.exists()


def test_project_paths_get_raw_file_path(mock_config):
    """Test raw file path construction."""
    paths = ProjectPaths(mock_config)
    
    file_path = paths.get_raw_file_path("AAPL")
    
    assert file_path == Path("data/raw/AAPL_1Min.parquet")


def test_validate_config_pass(mock_config):
    """Test configuration validation passes on valid config."""
    is_valid = validate_config(mock_config)
    
    assert is_valid is True


def test_validate_config_missing_key(mock_config):
    """Test configuration validation fails on missing key."""
    # Remove a required key
    del mock_config.data.tickers
    
    is_valid = validate_config(mock_config)
    
    assert is_valid is False


def test_validate_config_invalid_value(mock_config):
    """Test configuration validation fails on invalid value."""
    # Set invalid value
    mock_config.data.validation.max_missing_pct = 1.5  # Should be between 0 and 1
    
    is_valid = validate_config(mock_config)
    
    assert is_valid is False


def test_validate_environment(monkeypatch):
    """Test environment variable validation."""
    # Set required env vars
    monkeypatch.setenv("ALPACA_API_KEY", "test_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")
    
    is_valid = validate_environment()
    
    assert is_valid is True


def test_validate_environment_missing_vars(monkeypatch):
    """Test environment validation fails when vars missing."""
    # Unset env vars
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    
    is_valid = validate_environment()
    
    assert is_valid is False
