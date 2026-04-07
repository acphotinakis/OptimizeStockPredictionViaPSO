"""
tests/test_lstm.py

Unit tests for LSTM model and trainer.
"""

import numpy as np
import pytest
import torch

from src.models.lstm.lstm_model import LSTMModel, LSTMTrainer
from src.models.baselines import PersistenceModel, VanillaLSTM, XGBoostBaseline


class TestLSTMModel:
    """Test LSTM model architecture."""

    def test_model_initialization(self):
        """Test LSTM model can be initialized."""
        model = LSTMModel(
            input_size=50,
            num_layers=2,
            hidden_units=128,
            dropout=0.2,
        )
        
        assert model.input_size == 50
        assert model.num_layers == 2
        assert model.hidden_units == 128
        assert model.dropout_rate == 0.2

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        model = LSTMModel(
            input_size=50,
            num_layers=2,
            hidden_units=128,
            dropout=0.2,
        )
        
        # Create dummy input [batch=16, seq=30, features=50]
        x = torch.randn(16, 30, 50)
        output = model(x)
        
        # Output should be [batch=16, 1]
        assert output.shape == (16, 1)

    def test_predict_numpy(self):
        """Test prediction with numpy arrays."""
        model = LSTMModel(
            input_size=50,
            num_layers=2,
            hidden_units=128,
            dropout=0.2,
        )
        
        X = np.random.randn(10, 30, 50).astype(np.float32)
        predictions = model.predict(X, device="cpu")
        
        assert predictions.shape == (10,)
        assert isinstance(predictions, np.ndarray)


class TestLSTMTrainer:
    """Test LSTM trainer."""

    def test_trainer_initialization(self):
        """Test trainer can be initialized."""
        model = LSTMModel(input_size=50, num_layers=2, hidden_units=128, dropout=0.2)
        trainer = LSTMTrainer(
            model=model,
            lr=0.001,
            max_epochs=10,
            patience=3,
            batch_size=32,
        )
        
        assert trainer.lr == 0.001
        assert trainer.max_epochs == 10
        assert trainer.patience == 3

    def test_training_loop(self):
        """Test that training loop runs without errors."""
        model = LSTMModel(input_size=10, num_layers=1, hidden_units=32, dropout=0.1)
        trainer = LSTMTrainer(
            model=model,
            lr=0.01,
            max_epochs=3,
            patience=2,
            batch_size=16,
            device="cpu",
        )
        
        # Create dummy data
        X_train = np.random.randn(100, 20, 10).astype(np.float32)
        y_train = np.random.randn(100).astype(np.float32)
        X_val = np.random.randn(20, 20, 10).astype(np.float32)
        y_val = np.random.randn(20).astype(np.float32)
        
        history = trainer.fit(X_train, y_train, X_val, y_val)
        
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) > 0

    def test_prediction(self):
        """Test prediction after training."""
        model = LSTMModel(input_size=10, num_layers=1, hidden_units=32, dropout=0.1)
        trainer = LSTMTrainer(model=model, lr=0.01, max_epochs=2, device="cpu")
        
        X_train = np.random.randn(50, 20, 10).astype(np.float32)
        y_train = np.random.randn(50).astype(np.float32)
        X_val = np.random.randn(10, 20, 10).astype(np.float32)
        y_val = np.random.randn(10).astype(np.float32)
        
        trainer.fit(X_train, y_train, X_val, y_val)
        
        X_test = np.random.randn(5, 20, 10).astype(np.float32)
        predictions = trainer.predict(X_test)
        
        assert predictions.shape == (5,)


class TestPersistenceModel:
    """Test persistence baseline."""

    def test_persistence_prediction(self):
        """Test persistence model returns previous values."""
        model = PersistenceModel()
        
        y_prev = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        predictions = model.predict(None, y_prev)
        
        np.testing.assert_array_equal(predictions, y_prev)

    def test_persistence_from_returns(self):
        """Test persistence prediction from return series."""
        y = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        predictions = PersistenceModel.predict_from_returns(y)
        
        expected = np.array([0.0, 0.01, -0.02, 0.03, -0.01])
        np.testing.assert_array_equal(predictions, expected)


class TestVanillaLSTM:
    """Test vanilla LSTM baseline."""

    def test_vanilla_initialization(self):
        """Test vanilla LSTM uses default hyperparameters."""
        model = VanillaLSTM(input_size=50)
        
        # Check default params are used
        assert model.lookback == 30
        assert model._trainer.model.num_layers == 2
        assert model._trainer.model.hidden_units == 128

    def test_vanilla_override_params(self):
        """Test vanilla LSTM can override default params."""
        model = VanillaLSTM(input_size=50, num_layers=3, hidden_units=256)
        
        assert model._trainer.model.num_layers == 3
        assert model._trainer.model.hidden_units == 256


class TestXGBoostBaseline:
    """Test XGBoost baseline."""

    def test_xgboost_initialization(self):
        """Test XGBoost model can be initialized."""
        model = XGBoostBaseline(lookback=30)
        
        assert model.lookback == 30
        assert model._model is None  # Not fitted yet

    def test_xgboost_flatten(self):
        """Test input flattening."""
        model = XGBoostBaseline(lookback=30)
        
        X = np.random.randn(100, 30, 50).astype(np.float32)
        X_flat = model._flatten(X)
        
        assert X_flat.shape == (100, 30 * 50)

    def test_xgboost_fit_predict(self):
        """Test XGBoost can fit and predict."""
        model = XGBoostBaseline(lookback=20, n_estimators=10)
        
        X_train = np.random.randn(100, 20, 10).astype(np.float32)
        y_train = np.random.randn(100).astype(np.float32)
        
        model.fit(X_train, y_train)
        
        X_test = np.random.randn(20, 20, 10).astype(np.float32)
        predictions = model.predict(X_test)
        
        assert predictions.shape == (20,)
        assert model._model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
