"""
Unit tests for LSTM model module.
"""
import pytest
import torch
from src.models.lstm import LSTMModel


def test_model_initialization():
    """Test LSTM model initialization."""
    model = LSTMModel(
        input_size=10,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        bidirectional=False,
    )
    
    assert model.hidden_size == 64
    assert model.num_layers == 2
    assert model.bidirectional is False
    assert isinstance(model.lstm, torch.nn.LSTM)
    assert isinstance(model.fc, torch.nn.Linear)


def test_model_forward_pass_shape():
    """Test that forward pass produces expected output shape."""
    batch_size = 16
    seq_len = 30
    input_size = 10
    
    model = LSTMModel(
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        bidirectional=False,
    )
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Forward pass
    output = model(x)
    
    # Output should be (batch_size, 1) for single-step prediction
    assert output.shape == (batch_size, 1)


def test_model_forward_pass_no_nan():
    """Test that forward pass doesn't produce NaN values."""
    model = LSTMModel(input_size=10, hidden_size=32, num_layers=1)
    x = torch.randn(8, 20, 10)
    
    output = model(x)
    
    assert not torch.isnan(output).any()


def test_model_bidirectional():
    """Test bidirectional LSTM."""
    model = LSTMModel(
        input_size=10,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        bidirectional=True,
    )
    
    x = torch.randn(8, 20, 10)
    output = model(x)
    
    # Should still output (batch, 1)
    assert output.shape == (8, 1)
    
    # FC layer input should be 2 * hidden_size for bidirectional
    assert model.fc.in_features == 128


def test_model_single_layer_no_dropout():
    """Test that single-layer LSTM has no dropout."""
    model = LSTMModel(
        input_size=10,
        hidden_size=64,
        num_layers=1,
        dropout=0.5,  # Should be ignored for single layer
    )
    
    x = torch.randn(8, 20, 10)
    output = model(x)
    
    assert output.shape == (8, 1)


def test_model_gradient_flow():
    """Test that gradients flow correctly through the model."""
    model = LSTMModel(input_size=10, hidden_size=32, num_layers=2)
    x = torch.randn(4, 15, 10, requires_grad=True)
    
    output = model(x)
    loss = output.mean()
    loss.backward()
    
    # Check that gradients exist for all parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


def test_model_device_compatibility():
    """Test model can be moved to different devices."""
    model = LSTMModel(input_size=10, hidden_size=32, num_layers=1)
    
    # Test CPU
    model_cpu = model.to("cpu")
    x_cpu = torch.randn(4, 10, 10)
    output_cpu = model_cpu(x_cpu)
    assert output_cpu.device.type == "cpu"
    
    # Test CUDA if available
    if torch.cuda.is_available():
        model_cuda = model.to("cuda")
        x_cuda = torch.randn(4, 10, 10).to("cuda")
        output_cuda = model_cuda(x_cuda)
        assert output_cuda.device.type == "cuda"
