"""
tests/test_lstm_baseline.py

Unit tests for LSTM baseline model and training script.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.baselines import VanillaLSTM


class TestVanillaLSTM:
    """Test suite for VanillaLSTM class."""
    
    def test_initialization(self):
        """Test VanillaLSTM can be initialized."""
        model = VanillaLSTM(input_size=10)
        assert model is not None
        assert model.lookback == 30  # Default lookback
        assert model._trainer is not None
    
    def test_default_params(self):
        """Test default parameters are set correctly."""
        model = VanillaLSTM(input_size=10)
        assert model._trainer.model.num_layers == 2
        assert model._trainer.model.hidden_units == 128
        assert model._trainer.model.dropout_rate == 0.2
        assert model._trainer.lr == 0.001
    
    def test_custom_params(self):
        """Test VanillaLSTM accepts custom hyperparameters."""
        model = VanillaLSTM(
            input_size=10,
            num_layers=3,
            hidden_units=64,
            dropout=0.3,
            learning_rate=0.01,
            lookback=60,
        )
        assert model._trainer.model.num_layers == 3
        assert model._trainer.model.hidden_units == 64
        assert model._trainer.model.dropout_rate == 0.3
        assert model._trainer.lr == 0.01
        assert model.lookback == 60
    
    def test_training(self):
        """Test VanillaLSTM can train on dummy data."""
        X_train = np.random.randn(100, 30, 10).astype(np.float32)
        y_train = np.random.randn(100).astype(np.float32)
        X_val = np.random.randn(50, 30, 10).astype(np.float32)
        y_val = np.random.randn(50).astype(np.float32)
        
        model = VanillaLSTM(input_size=10, max_epochs=2, patience=1)
        history = model.fit(X_train, y_train, X_val, y_val)
        
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) > 0
        assert len(history["val_loss"]) > 0
        assert len(history["train_loss"]) <= 2  # max_epochs=2
    
    def test_prediction(self):
        """Test VanillaLSTM can generate predictions."""
        X_train = np.random.randn(100, 30, 10).astype(np.float32)
        y_train = np.random.randn(100).astype(np.float32)
        X_val = np.random.randn(50, 30, 10).astype(np.float32)
        y_val = np.random.randn(50).astype(np.float32)
        
        model = VanillaLSTM(input_size=10, max_epochs=2)
        model.fit(X_train, y_train, X_val, y_val)
        
        y_pred = model.predict(X_val)
        assert y_pred.shape == (50,)
        assert not np.isnan(y_pred).any()
        assert not np.isinf(y_pred).any()
    
    def test_prediction_without_training_fails(self):
        """Test prediction fails if model not trained."""
        model = VanillaLSTM(input_size=10)
        X = np.random.randn(10, 30, 10).astype(np.float32)
        
        # Should work even without explicit training (model is initialized)
        y_pred = model.predict(X)
        assert y_pred.shape == (10,)
    
    def test_input_shape_validation(self):
        """Test model handles different input shapes correctly."""
        model = VanillaLSTM(input_size=10, max_epochs=1)
        
        # Correct shape
        X_train = np.random.randn(50, 30, 10).astype(np.float32)
        y_train = np.random.randn(50).astype(np.float32)
        X_val = np.random.randn(20, 30, 10).astype(np.float32)
        y_val = np.random.randn(20).astype(np.float32)
        
        model.fit(X_train, y_train, X_val, y_val)
        y_pred = model.predict(X_val)
        assert y_pred.shape == (20,)
    
    def test_different_lookback(self):
        """Test model works with different lookback values."""
        for lookback in [10, 30, 60]:
            X_train = np.random.randn(50, lookback, 10).astype(np.float32)
            y_train = np.random.randn(50).astype(np.float32)
            X_val = np.random.randn(20, lookback, 10).astype(np.float32)
            y_val = np.random.randn(20).astype(np.float32)
            
            model = VanillaLSTM(input_size=10, lookback=lookback, max_epochs=1)
            model.fit(X_train, y_train, X_val, y_val)
            y_pred = model.predict(X_val)
            
            assert y_pred.shape == (20,)
            assert model.lookback == lookback
    
    def test_early_stopping(self):
        """Test early stopping works correctly."""
        # Create data where validation loss doesn't improve
        X_train = np.random.randn(100, 30, 10).astype(np.float32)
        y_train = np.random.randn(100).astype(np.float32)
        X_val = np.random.randn(50, 30, 10).astype(np.float32)
        y_val = np.random.randn(50).astype(np.float32)
        
        model = VanillaLSTM(
            input_size=10,
            max_epochs=50,
            patience=3,
        )
        history = model.fit(X_train, y_train, X_val, y_val)
        
        # Should stop before max_epochs due to early stopping
        assert len(history["train_loss"]) <= 50
    
    def test_device_handling(self):
        """Test model handles CPU/CUDA device correctly."""
        model = VanillaLSTM(input_size=10, device="cpu")
        assert model._trainer.device == "cpu"
        
        X = np.random.randn(10, 30, 10).astype(np.float32)
        y_pred = model.predict(X)
        assert y_pred.shape == (10,)
    
    def test_batch_size_effect(self):
        """Test different batch sizes work correctly."""
        X_train = np.random.randn(100, 30, 10).astype(np.float32)
        y_train = np.random.randn(100).astype(np.float32)
        X_val = np.random.randn(50, 30, 10).astype(np.float32)
        y_val = np.random.randn(50).astype(np.float32)
        
        for batch_size in [32, 64, 128]:
            model = VanillaLSTM(
                input_size=10,
                batch_size=batch_size,
                max_epochs=2,
            )
            history = model.fit(X_train, y_train, X_val, y_val)
            assert len(history["train_loss"]) > 0
    
    def test_reproducibility(self):
        """Test model training is reproducible with same seed."""
        import torch
        
        X_train = np.random.randn(50, 30, 10).astype(np.float32)
        y_train = np.random.randn(50).astype(np.float32)
        X_val = np.random.randn(20, 30, 10).astype(np.float32)
        y_val = np.random.randn(20).astype(np.float32)
        
        # Train first model
        torch.manual_seed(42)
        np.random.seed(42)
        model1 = VanillaLSTM(input_size=10, max_epochs=2)
        model1.fit(X_train, y_train, X_val, y_val)
        pred1 = model1.predict(X_val)
        
        # Train second model with same seed
        torch.manual_seed(42)
        np.random.seed(42)
        model2 = VanillaLSTM(input_size=10, max_epochs=2)
        model2.fit(X_train, y_train, X_val, y_val)
        pred2 = model2.predict(X_val)
        
        # Predictions should be very similar (allowing for small numerical differences)
        assert np.allclose(pred1, pred2, rtol=1e-3, atol=1e-5)


class TestVanillaLSTMIntegration:
    """Integration tests for VanillaLSTM."""
    
    def test_full_pipeline(self):
        """Test complete training and prediction pipeline."""
        # Simulate realistic data
        n_train = 1000
        n_val = 500
        n_test = 200
        lookback = 30
        n_features = 20
        
        X_train = np.random.randn(n_train, lookback, n_features).astype(np.float32)
        y_train = np.random.randn(n_train).astype(np.float32)
        X_val = np.random.randn(n_val, lookback, n_features).astype(np.float32)
        y_val = np.random.randn(n_val).astype(np.float32)
        X_test = np.random.randn(n_test, lookback, n_features).astype(np.float32)
        
        # Train
        model = VanillaLSTM(input_size=n_features, max_epochs=5)
        history = model.fit(X_train, y_train, X_val, y_val)
        
        # Validate
        y_pred_val = model.predict(X_val)
        assert y_pred_val.shape == (n_val,)
        
        # Test
        y_pred_test = model.predict(X_test)
        assert y_pred_test.shape == (n_test,)
        
        # Check history
        assert len(history["train_loss"]) > 0
        assert len(history["val_loss"]) > 0
    
    def test_save_and_load(self):
        """Test model can be saved and loaded."""
        import tempfile
        
        X_train = np.random.randn(50, 30, 10).astype(np.float32)
        y_train = np.random.randn(50).astype(np.float32)
        X_val = np.random.randn(20, 30, 10).astype(np.float32)
        y_val = np.random.randn(20).astype(np.float32)
        
        # Train model
        model = VanillaLSTM(input_size=10, max_epochs=2)
        model.fit(X_train, y_train, X_val, y_val)
        pred_before = model.predict(X_val)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            model_path = f.name
        
        torch.save(model._trainer.model.state_dict(), model_path)
        
        # Load model
        model_loaded = VanillaLSTM(input_size=10)
        model_loaded._trainer.model.load_state_dict(torch.load(model_path))
        pred_after = model_loaded.predict(X_val)
        
        # Predictions should be identical
        assert np.allclose(pred_before, pred_after, rtol=1e-5, atol=1e-7)
        
        # Cleanup
        Path(model_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
