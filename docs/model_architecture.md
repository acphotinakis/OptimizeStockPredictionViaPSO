# Model Architecture
## LSTM for Many-to-One Stock Price Prediction

**Document Version:** 1.0 | March 2026

---

## Table of Contents

1. [Model Overview](#1-model-overview)
2. [LSTM Cell Mathematics](#2-lstm-cell-mathematics)
3. [Network Architecture](#3-network-architecture)
4. [Input/Output Shapes](#4-inputoutput-shapes)
5. [Many-to-One Design](#5-many-to-one-design)
6. [Training Loop Design](#6-training-loop-design)
7. [Regularization Strategy](#7-regularization-strategy)
8. [Baseline Models](#8-baseline-models)
9. [PyTorch Implementation Reference](#9-pytorch-implementation-reference)

---

## 1. Model Overview

The LSTM model is a stacked recurrent neural network configured in a **Many-to-One** prediction paradigm: it receives a sequence of $T$ timesteps of $F$ features and outputs a single scalar prediction of the next-period log return.

**Design philosophy:** The architecture is kept deliberately flexible, with all structural hyperparameters (number of layers, hidden units, dropout, learning rate, lookback window) determined by the IPSO optimizer rather than set manually.

| Property | Value |
|---|---|
| Architecture | Stacked LSTM (1–4 layers) |
| Input | Sequence of $T$ timesteps × $F$ features |
| Output | Single scalar (predicted log return at $t+1$) |
| Prediction task | Regression (log return) |
| Loss function | Mean Squared Error (MSE) |
| Optimizer | Adam |

---

## 2. LSTM Cell Mathematics

### 2.1 Cell State and Gate Equations

At each timestep $t$, the LSTM cell processes input $\mathbf{x}_t \in \mathbb{R}^F$ and previous hidden state $\mathbf{h}_{t-1} \in \mathbb{R}^H$ to produce new hidden state $\mathbf{h}_t$ and cell state $\mathbf{c}_t$.

**Forget gate** (determines what to remove from cell state):
$$\mathbf{f}_t = \sigma\!\left(\mathbf{W}_f [\mathbf{h}_{t-1},\ \mathbf{x}_t] + \mathbf{b}_f\right) \tag{1}$$

**Input gate** (determines what new information to store):
$$\mathbf{i}_t = \sigma\!\left(\mathbf{W}_i [\mathbf{h}_{t-1},\ \mathbf{x}_t] + \mathbf{b}_i\right) \tag{2}$$

**Candidate cell state** (new information to potentially add):
$$\tilde{\mathbf{c}}_t = \tanh\!\left(\mathbf{W}_c [\mathbf{h}_{t-1},\ \mathbf{x}_t] + \mathbf{b}_c\right) \tag{3}$$

**Cell state update**:
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \tag{4}$$

**Output gate** (determines what to expose from cell state):
$$\mathbf{o}_t = \sigma\!\left(\mathbf{W}_o [\mathbf{h}_{t-1},\ \mathbf{x}_t] + \mathbf{b}_o\right) \tag{5}$$

**Hidden state output**:
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t) \tag{6}$$

where $\sigma(\cdot)$ is the sigmoid function, $\tanh(\cdot)$ is the hyperbolic tangent, and $\odot$ is the Hadamard (element-wise) product.

### 2.2 Parameter Count

For a single LSTM layer with input size $F$ and hidden size $H$:

$$\text{params} = 4 \times (H \times (F + H) + H) = 4H(F + H + 1) \tag{7}$$

For a stacked LSTM with $L$ layers:
- Layer 1: $4H(F + H + 1)$
- Layers 2 to $L$: $4H(H + H + 1) = 4H(2H + 1)$ each
- Linear output head: $H + 1$

**Example:** $L=2$, $H=128$, $F=75$:
$$\text{total} = 4 \times 128 \times (75 + 128 + 1) + 4 \times 128 \times (128 + 128 + 1) + 128 + 1 \approx 266K \text{ params}$$

---

## 3. Network Architecture

### 3.1 Full Architecture Diagram

```
Input:  [batch_size × T × F]
          │
          │  F = selected features (~75)
          │  T = lookback window (10/30/60/120 mins)
          ▼
┌────────────────────────────────────┐
│  LSTM Layer 1                      │
│  - Input size:  F                  │
│  - Hidden size: H                  │
│  - Dropout:     p (after output)   │
│  - Output:      [B × T × H]        │
└─────────────────┬──────────────────┘
                  │
                  ▼ (if num_layers ≥ 2)
┌────────────────────────────────────┐
│  LSTM Layer 2                      │
│  - Input size:  H                  │
│  - Hidden size: H                  │
│  - Dropout:     p (after output)   │
│  - Output:      [B × T × H]        │
└─────────────────┬──────────────────┘
                  │
                  ▼ (if num_layers ≥ 3)
              [LSTM Layer 3, 4 ...]
                  │
                  ▼
          Select last timestep:
          output[:, -1, :]  →  [B × H]
                  │
                  ▼
┌────────────────────────────────────┐
│  Dropout Layer (p)                 │
│  Output: [B × H]                  │
└─────────────────┬──────────────────┘
                  │
                  ▼
┌────────────────────────────────────┐
│  Linear (Fully Connected) Layer    │
│  - Input:  H                       │
│  - Output: 1                       │
│  - No activation (raw regression)  │
└─────────────────┬──────────────────┘
                  │
                  ▼
Output: [batch_size × 1]
        (predicted log return at t+1)
```

### 3.2 Layer Configuration by Particle

The PSO particle directly controls the following architecture variables:

| Variable | Symbol | Source |
|---|---|---|
| Number of LSTM layers | $L$ | `particle[0]` decoded |
| Hidden units per layer | $H$ | `particle[1]` decoded |
| Dropout rate | $p$ | `particle[2]` decoded |
| Learning rate | $\eta$ | `particle[3]` decoded |
| Lookback window | $T$ | `particle[4]` decoded |

All layers share the same hidden size $H$ and dropout rate $p$ for simplicity and to reduce the search space.

---

## 4. Input/Output Shapes

### 4.1 Feature Tensor

```
X: np.ndarray of shape [N_samples, T_lookback, F_features]

N_samples = total number of 1-minute bars in the set - T_lookback
T_lookback ∈ {10, 30, 60, 120}   (minutes)
F_features ≈ 75                   (after XGBoost selection)
```

**Sliding window construction:**

For target timestep $n$ (predicting $r_{n+1}$), the input window is:
$$\mathbf{X}_n = \left[\mathbf{x}_{n-T+1},\ \mathbf{x}_{n-T+2},\ \ldots,\ \mathbf{x}_n\right] \in \mathbb{R}^{T \times F}$$

### 4.2 Target Vector

```
y: np.ndarray of shape [N_samples, 1]

y_n = log(close_{n+1} / close_n)   (next-period log return)
```

### 4.3 Batch Tensor (During Training)

PyTorch DataLoader produces batches of shape:
```
x_batch: Tensor of shape [B, T, F]   (B = batch_size = 256)
y_batch: Tensor of shape [B, 1]
```

---

## 5. Many-to-One Design

### 5.1 Rationale

In a **Many-to-One** configuration, the LSTM processes all $T$ timesteps but only the **last hidden state** $\mathbf{h}_T$ is passed to the output layer. This design:

1. Leverages the full historical context via the LSTM's internal memory
2. Focuses prediction energy on a single target (next-bar return)
3. Avoids auxiliary losses and multi-step forecast uncertainty
4. Is computationally simpler than sequence-to-sequence models

**Alternative considered:** Many-to-Many (predict all future bars in a window). Rejected due to compounding errors and difficulty integrating into the PSO fitness function (which single target metric to use?).

### 5.2 Sequence Alignment

```
Input window:
 t-T+1  t-T+2  ...  t-1    t
   │      │           │     │
 LSTM₁→LSTM₂→...→LSTMₜ₋₁→LSTMₜ
                              │
                           h_T (last hidden state)
                              │
                           Linear
                              │
                           ŷ_{t+1}  ←  predicts log return at t+1
```

### 5.3 Batched Many-to-One in PyTorch

```python
class LSTMModel(nn.Module):
    def forward(self, x):  # x: [B, T, F]
        # lstm_out: [B, T, H], (h_n, c_n): each [L, B, H]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Many-to-One: take only the last timestep output
        last_hidden = lstm_out[:, -1, :]  # [B, H]
        
        out = self.dropout(last_hidden)
        out = self.fc(out)               # [B, 1]
        return out
```

---

## 6. Training Loop Design

### 6.1 Overview

The training loop is a standard PyTorch minibatch SGD loop with Adam optimizer, MSE loss, and early stopping.

```python
def fit(self, X_train, y_train, X_val, y_val):
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
    
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(self.max_epochs):
        # --- Training phase ---
        self.model.train()
        for x_batch, y_batch in train_dl:
            optimizer.zero_grad()
            y_pred = self.model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # --- Validation phase ---
        self.model.eval()
        with torch.no_grad():
            val_pred = self.model(torch.FloatTensor(X_val))
            val_loss = criterion(val_pred, torch.FloatTensor(y_val)).item()
        
        # --- Early stopping ---
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            self._save_best_weights()
        else:
            patience_counter += 1
            if patience_counter >= self.patience:
                break
    
    self._restore_best_weights()
    return {'best_val_loss': best_val_loss, 'epochs_trained': epoch+1}
```

### 6.2 Training Parameters

| Parameter | Value | Notes |
|---|---|---|
| Optimizer | Adam | Adaptive learning rate |
| Loss function | MSE | Squared error on log returns |
| Batch size | 256 | Fixed across all PSO evaluations |
| Max epochs | 100 | Hard cap |
| Early stopping patience | 10 | Halt after 10 non-improving epochs |
| Gradient clipping | 1.0 (L2 norm) | Prevents exploding gradients |
| Shuffle train data | True | Reduces sequential correlation bias |
| Weight initialization | PyTorch default (orthogonal for LSTM) | — |

### 6.3 Gradient Clipping

$$\text{if } \|\nabla \theta\|_2 > \text{clip\_val}:\ \nabla \theta \leftarrow \frac{\text{clip\_val}}{\|\nabla \theta\|_2} \nabla \theta$$

Max norm = 1.0. Critical for LSTM training on financial return series which can exhibit sudden volatility spikes.

### 6.4 Adam Optimizer

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t)$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

Default: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. Learning rate $\eta$ is determined by the PSO particle.

---

## 7. Regularization Strategy

### 7.1 Dropout

Applied **between LSTM layers** and **before the output linear layer**. The dropout rate $p$ is determined by the PSO particle.

**PyTorch LSTM dropout parameter:** Applies dropout on the outputs of each LSTM layer except the last (consistent with `nn.LSTM(dropout=p)` behavior for multi-layer LSTMs).

### 7.2 L2 Weight Decay (Optional)

A small L2 penalty on model parameters can be added through Adam's `weight_decay` parameter:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
```

This is fixed at $10^{-5}$ and not included in the PSO search space to avoid increasing dimensionality.

### 7.3 Early Stopping (Implicit Regularization)

By halting training when validation loss stops improving, early stopping prevents overfitting to the training set. With patience=10, the model is allowed up to 10 epochs of non-improvement before halting. Best weights from the optimal epoch are restored.

### 7.4 Batch Normalization (Optional)

If very deep LSTMs (4 layers) exhibit training instability, Layer Normalization (not Batch Normalization, which is problematic for sequence models) can be applied between LSTM layers.

---

## 8. Baseline Models

All baselines use the **same data pipeline** (features, normalization, splits) as the LSTM.

### 8.1 Persistence Model (Naive Forecast)

$$\hat{y}_{t+1} = y_t$$

The simplest possible baseline: the next-period return equals the current return. Represents the "random walk" hypothesis.

### 8.2 Vanilla LSTM (Manual Hyperparameters)

Same architecture as above but with fixed hyperparameters chosen by manual inspection:

| Parameter | Value |
|---|---|
| num_layers | 2 |
| hidden_units | 128 |
| dropout | 0.2 |
| learning_rate | 0.001 |
| lookback | 30 |

This establishes whether IPSO adds value over a reasonable manual configuration.

### 8.3 XGBoost Regressor

A gradient boosting model operating on **flattened** features (no temporal structure):

```
Input: X_flat = [N_samples × (T × F)]   (flattened sequence)
Output: ŷ_{t+1}
```

```python
import xgboost as xgb
model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    random_state=42
)
```

XGBoost provides a strong non-deep-learning baseline that captures non-linear feature interactions without temporal structure.

---

## 9. PyTorch Implementation Reference

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional
import numpy as np

class LSTMModel(nn.Module):
    """
    Configurable stacked LSTM for many-to-one time series regression.
    All architecture parameters set by the PSO optimizer.
    """
    def __init__(self,
                 input_size: int,
                 num_layers: int,
                 hidden_units: int,
                 dropout: float,
                 output_size: int = 1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        
        # Stacked LSTM
        # Note: dropout in nn.LSTM applies between layers (not after last layer)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_units,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True  # Input: [batch, seq, features]
        )
        
        # Post-LSTM dropout
        self.dropout = nn.Dropout(p=dropout)
        
        # Output projection (many-to-one)
        self.fc = nn.Linear(hidden_units, output_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal init for LSTM weights, zero init for biases."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 (Jozefowicz et al., 2015)
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_size]
        Returns:
            out: [batch_size, 1]
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # lstm_out: [B, T, H]
        
        # Many-to-One: use only the last timestep's output
        last_out = lstm_out[:, -1, :]  # [B, H]
        
        # Dropout + linear projection
        out = self.dropout(last_out)
        out = self.fc(out)  # [B, 1]
        return out
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Inference wrapper (returns numpy array)."""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X)
            preds = self.forward(x_tensor)
        return preds.numpy().flatten()
```