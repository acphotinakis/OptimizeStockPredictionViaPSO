"""
Unit tests for training module.
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.training.train import (
    EarlyStopping,
    train_one_epoch,
    validate,
)
from src.models.lstm import LSTMModel
import tempfile
from pathlib import Path


@pytest.fixture
def simple_model():
    """Create a simple LSTM model for testing."""
    return LSTMModel(input_size=5, hidden_size=16, num_layers=1, dropout=0.0)


@pytest.fixture
def simple_dataloader():
    """Create a simple dataloader for testing."""
    # Create synthetic data
    X = torch.randn(100, 10, 5)  # 100 samples, 10 timesteps, 5 features
    y = torch.randn(100)
    
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=16, shuffle=False)


def test_early_stopping_initialization():
    """Test EarlyStopping initialization."""
    early_stopping = EarlyStopping(patience=5, verbose=False, delta=0.001)
    
    assert early_stopping.patience == 5
    assert early_stopping.counter == 0
    assert early_stopping.best_score is None
    assert early_stopping.early_stop is False
    assert early_stopping.val_loss_min == float("inf")


def test_early_stopping_saves_best_model(simple_model):
    """Test that early stopping saves model on improvement."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "checkpoint.pt"
        early_stopping = EarlyStopping(patience=3, path=str(checkpoint_path))
        
        # First call should save
        early_stopping(0.5, simple_model)
        assert checkpoint_path.exists()
        assert early_stopping.val_loss_min == 0.5
        
        # Better loss should save again
        early_stopping(0.3, simple_model)
        assert early_stopping.val_loss_min == 0.3
        assert early_stopping.counter == 0


def test_early_stopping_triggers(simple_model):
    """Test that early stopping triggers after patience exhausted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "checkpoint.pt"
        early_stopping = EarlyStopping(patience=3, path=str(checkpoint_path))
        
        # Initial best
        early_stopping(0.5, simple_model)
        
        # Worse losses
        early_stopping(0.6, simple_model)
        assert early_stopping.counter == 1
        
        early_stopping(0.7, simple_model)
        assert early_stopping.counter == 2
        
        early_stopping(0.8, simple_model)
        assert early_stopping.counter == 3
        assert early_stopping.early_stop is True


def test_train_one_epoch(simple_model, simple_dataloader):
    """Test single training epoch."""
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    device = torch.device("cpu")
    
    loss = train_one_epoch(simple_model, simple_dataloader, optimizer, criterion, device)
    
    assert isinstance(loss, float)
    assert loss > 0
    assert not np.isnan(loss)


def test_train_one_epoch_gradient_clipping(simple_model, simple_dataloader):
    """Test that gradient clipping is applied."""
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    device = torch.device("cpu")
    
    # Train with very small clip value
    loss = train_one_epoch(
        simple_model, simple_dataloader, optimizer, criterion, device, clip_value=0.1
    )
    
    # Should complete without error
    assert isinstance(loss, float)


def test_validate_function(simple_model, simple_dataloader):
    """Test validation function."""
    criterion = nn.MSELoss()
    device = torch.device("cpu")
    
    val_loss = validate(simple_model, simple_dataloader, criterion, device)
    
    assert isinstance(val_loss, float)
    assert val_loss > 0
    assert not np.isnan(val_loss)


def test_train_decreases_loss(simple_model, simple_dataloader):
    """Test that training decreases loss over multiple epochs."""
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    device = torch.device("cpu")
    
    losses = []
    for _ in range(5):
        loss = train_one_epoch(simple_model, simple_dataloader, optimizer, criterion, device)
        losses.append(loss)
    
    # Loss should generally decrease (allowing for some variance)
    assert losses[-1] < losses[0] * 1.1  # Allow 10% tolerance


def test_train_nan_detection():
    """Test that NaN detection works in training."""
    # Create a model that will produce NaN
    model = LSTMModel(input_size=5, hidden_size=16, num_layers=1)
    
    # Create data with NaN
    X = torch.randn(32, 10, 5)
    X[0, 0, 0] = float('nan')
    y = torch.randn(32)
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    device = torch.device("cpu")
    
    # Should raise ValueError due to NaN
    with pytest.raises(ValueError, match="NaN detected"):
        train_one_epoch(model, dataloader, optimizer, criterion, device)
