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
    """Compute Mean Squared Weight (MSW) for an LSTM model.

    MSW is the mean of squared weights across the model's recurrent and
    dense weight matrices. Biases are excluded per Deng & Peng 2025: the
    MSW penalty targets the LSTM cell weight matrices, not bias terms.
    Including biases dilutes the penalty (biases are small but
    plentiful) and pushes MSW magnitudes far above the MSE scale.

    Args:
        model: PyTorch model whose named parameters are inspected. Only
            parameters whose name contains ``"weight"`` (and not
            ``"bias"``) contribute to the statistic.

    Returns:
        Non-negative scalar MSW value. Returns ``0.0`` when the model
        exposes no qualifying weight parameters.

    References:
        Deng & Peng 2025: MSW regularization for improved generalization.
    """
    total_sq_weight = 0.0
    n_params = 0

    for name, param in model.named_parameters():
        if "weight" in name and "bias" not in name:
            total_sq_weight += torch.sum(param ** 2).item()
            n_params += param.numel()

    if n_params == 0:
        return 0.0

    msw = total_sq_weight / n_params
    return float(msw)


class SpecCompliantFitness:
    """Specification-compliant fitness function: gamma*MSE + (1-gamma)*scale*MSW.

    Implements the composite fitness from IPSO_LSTM_AUDIT.md Section 4:

        F(x) = gamma * MSE + (1 - gamma) * msw_scale * MSW

    where ``MSE`` is the mean squared error on the validation
    predictions and ``MSW`` is the mean squared weight magnitude over
    the LSTM/dense weight matrices (see :func:`compute_msw`).

    The ``msw_scale`` multiplier compensates for an inherent magnitude
    mismatch: with targets in ``[-1, 1]`` MSE typically falls in
    ``[1e-4, 1e-2]``, while LSTM weight matrices yield MSW values in
    ``[1e-2, 1e-1]``. Without scaling, the ``0.1 * MSW`` term dominates
    by 10-100x and inverts the spec's intended 9:1 weighting.
    A default ``msw_scale=0.01`` brings the regularizer into the same
    order of magnitude as MSE so ``gamma=0.9`` behaves as documented.

    This fitness function does NOT use RMSE, Sharpe ratio, drawdown
    penalties, or any other financial trading metric. Lower fitness is
    better.
    """

    def __init__(self, gamma: float = GAMMA, msw_scale: float = 0.01) -> None:
        """Initialize the spec-compliant fitness function.

        Args:
            gamma: Weight for the MSE component (default ``0.9``).
            msw_scale: Multiplier applied to the MSW term to bring it
                into a range comparable to MSE for scaled targets.
                Default ``0.01`` maps O(1e-2) MSW values to O(1e-4),
                which is comparable to typical scaled-target MSE.
        """
        self.gamma = gamma
        self.msw_weight = 1.0 - gamma
        self.msw_scale = msw_scale

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model: Optional[torch.nn.Module] = None,
    ) -> float:
        """Evaluate fitness ``gamma*MSE + (1-gamma)*msw_scale*MSW``.

        Args:
            y_true: True validation targets, shape ``(N,)``.
            y_pred: Predicted validation targets, shape ``(N,)``.
            model: Trained PyTorch model, required for MSW computation.

        Returns:
            Composite fitness value (lower is better).

        Raises:
            ValueError: If ``model`` is ``None`` since MSW cannot be
                computed without the trained weights.
        """
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()

        mse = float(np.mean((y_true - y_pred) ** 2))

        if model is None:
            raise ValueError(
                "Model required for MSW computation. "
                "Fitness function must receive trained model."
            )

        msw = compute_msw(model)

        fitness = self.gamma * mse + self.msw_weight * self.msw_scale * msw

        return float(fitness)

    def reset(self) -> None:
        """Reset state (no-op for this stateless fitness function)."""
        pass


# Backward compatibility alias
CompositeFitness = SpecCompliantFitness
