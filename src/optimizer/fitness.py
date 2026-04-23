"""
src/optimizer/fitness.py

Spec-compliant fitness function for IPSO-LSTM optimizer.

SPECIFICATION (per IPSO_LSTM_AUDIT.md):
    F(x) = γ × MSE + (1 - γ) × MSW

Where:
    - γ = 0.9 (gamma weight)
    - MSE = Mean Squared Error on validation predictions
    - MSW = Mean Squared Weights (L2 regularization of model parameters)

Lower fitness is better.

References:
    - Deng & Peng 2025: MSE+MSW composite fitness
    - IPSO_LSTM_AUDIT.md Section 4: Fitness Function specification
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

# Specification-required parameters
GAMMA = 0.9  # Weight for MSE component
MSW_WEIGHT = 1.0 - GAMMA  # Weight for MSW component (0.1)


def compute_msw(model: torch.nn.Module) -> float:
    """Compute Mean Squared Weight (MSW) for LSTM model.
    
    MSW = mean of squared weights across all trainable parameters.
    
    This provides L2 regularization penalty to prevent overfitting
    by penalizing models with excessively large weights.
    
    Args:
        model: PyTorch LSTM model with trainable parameters
    
    Returns:
        MSW value (float, non-negative)
    
    References:
        Deng & Peng 2025: MSW regularization for improved generalization
    """
    total_sq_weight = 0.0
    n_params = 0
    
    for param in model.parameters():
        if param.requires_grad:
            total_sq_weight += torch.sum(param ** 2).item()
            n_params += param.numel()
    
    if n_params == 0:
        return 0.0
    
    msw = total_sq_weight / n_params
    return float(msw)


class SpecCompliantFitness:
    """Specification-compliant fitness function: γ×MSE + (1-γ)×MSW
    
    CRITICAL: This replaces the previous financial fitness function.
    
    Per IPSO_LSTM_AUDIT.md Section 4:
        F(x) = 0.9 × MSE + 0.1 × MSW
    
    Where:
        - MSE: Mean Squared Error on validation predictions
        - MSW: Mean Squared Weights (L2 regularization)
        - γ = 0.9 (fixed per specification)
    
    This fitness function does NOT use:
        - RMSE (uses MSE instead)
        - Sharpe ratio
        - Drawdown penalty
        - Financial trading metrics
    
    Lower fitness is better.
    """

    def __init__(self, gamma: float = GAMMA) -> None:
        """Initialize spec-compliant fitness function.
        
        Args:
            gamma: Weight for MSE component (default: 0.9)
        """
        self.gamma = gamma
        self.msw_weight = 1.0 - gamma
        
    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model: Optional[torch.nn.Module] = None,
    ) -> float:
        """Evaluate fitness: γ×MSE + (1-γ)×MSW
        
        Args:
            y_true: True validation targets (N,)
            y_pred: Predicted validation targets (N,)
            model: Trained PyTorch model (required for MSW computation)
        
        Returns:
            Fitness value (lower is better)
        
        Raises:
            ValueError: If model is None (MSW cannot be computed)
        """
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        
        # Compute MSE on validation predictions
        mse = float(np.mean((y_true - y_pred) ** 2))
        
        # Compute MSW from model weights
        if model is None:
            raise ValueError(
                "Model required for MSW computation. "
                "Fitness function must receive trained model."
            )
        
        msw = compute_msw(model)
        
        # Composite fitness per specification
        fitness = self.gamma * mse + self.msw_weight * msw
        
        return float(fitness)

    def reset(self) -> None:
        """Reset state (no-op for stateless fitness function)."""
        pass


# Backward compatibility alias
CompositeFitness = SpecCompliantFitness
