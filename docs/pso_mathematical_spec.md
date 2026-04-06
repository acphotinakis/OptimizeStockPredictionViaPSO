# PSO Mathematical Specification
## Improved Particle Swarm Optimization for LSTM Hyperparameter Tuning

**Document Version:** 1.0 | March 2026

---

## Table of Contents

1. [Standard PSO Formulation](#1-standard-pso-formulation)
2. [Improved PSO (IPSO) — Modifications](#2-improved-pso-ipso--modifications)
3. [Particle Encoding](#3-particle-encoding)
4. [Search Space Definition](#4-search-space-definition)
5. [Fitness Function](#5-fitness-function)
6. [Full Algorithm Pseudocode](#6-full-algorithm-pseudocode)
7. [Convergence Analysis](#7-convergence-analysis)
8. [Hyperparameter Settings](#8-hyperparameter-settings)

---

## 1. Standard PSO Formulation

Particle Swarm Optimization (PSO), introduced by Kennedy and Eberhart (1995), is a population-based metaheuristic that models cooperative search behavior inspired by bird flocking and fish schooling.

### 1.1 Problem Setup

Given a fitness function $f: \mathbb{R}^D \to \mathbb{R}$ to minimize, a swarm of $M$ particles search a $D$-dimensional continuous space.

**Swarm state at iteration $t$:**

- $\mathbf{x}_i^t \in \mathbb{R}^D$ — position of particle $i$
- $\mathbf{v}_i^t \in \mathbb{R}^D$ — velocity of particle $i$
- $\mathbf{p}_i \in \mathbb{R}^D$ — personal best position (pbest) of particle $i$
- $\mathbf{g} \in \mathbb{R}^D$ — global best position (gbest) across all particles

### 1.2 Velocity Update

$$\mathbf{v}_i^{t+1} = \omega^t \mathbf{v}_i^t + c_1 r_1 (\mathbf{p}_i - \mathbf{x}_i^t) + c_2 r_2 (\mathbf{g} - \mathbf{x}_i^t) \tag{1}$$

where:
- $\omega^t$ — inertia weight (controls momentum)
- $c_1$ — cognitive coefficient (attraction to personal best)
- $c_2$ — social coefficient (attraction to global best)
- $r_1, r_2 \sim \mathcal{U}(0, 1)$ — independent uniform random numbers, resampled each iteration

### 1.3 Position Update

$$\mathbf{x}_i^{t+1} = \mathbf{x}_i^t + \mathbf{v}_i^{t+1} \tag{2}$$

### 1.4 Best Position Updates

**Personal best:**
$$\mathbf{p}_i \leftarrow \begin{cases} \mathbf{x}_i^{t+1} & \text{if } f(\mathbf{x}_i^{t+1}) < f(\mathbf{p}_i) \\ \mathbf{p}_i & \text{otherwise} \end{cases} \tag{3}$$

**Global best:**
$$\mathbf{g} \leftarrow \arg\min_{i \in \{1,\ldots,M\}} f(\mathbf{p}_i) \tag{4}$$

### 1.5 Standard PSO Limitations

1. **Premature convergence**: The swarm collapses too quickly around a suboptimal gbest.
2. **Static inertia**: Linear decreasing $\omega$ does not adapt to the current diversity of the swarm.
3. **No escape mechanism**: Once trapped in a local optimum, particles cannot escape without mutation.

---

## 2. Improved PSO (IPSO) — Modifications

The IPSO algorithm addresses all three limitations through two key innovations, following Ji et al. (2021): a **non-linear inertia weight** and an **adaptive mutation factor**.

### 2.1 Non-Linear Inertia Weight (Tanh Decay)

**Standard linear decay:**
$$\omega_{\text{linear}}^t = \omega_{\max} - \frac{(\omega_{\max} - \omega_{\min}) \cdot t}{T_{\max}}$$

**Proposed non-linear (tanh) decay:**
$$\boxed{\omega_i^t = \omega_{\max} - (\omega_{\max} - \omega_{\min}) \cdot \tanh\!\left(\frac{4t}{T_{\max}}\right)} \tag{5}$$

where:
- $\omega_{\min} = 0.4$ — minimum inertia weight
- $\omega_{\max} = 0.9$ — maximum inertia weight
- $t$ — current iteration number (1-indexed)
- $T_{\max}$ — maximum number of iterations

**Key properties of tanh inertia:**

| Phase | $t/T_{\max}$ | $\omega$ behavior | Effect |
|---|---|---|---|
| Early | 0.0 – 0.3 | High, slow decrease | Global exploration |
| Mid | 0.3 – 0.6 | Rapid nonlinear decrease | Balanced search |
| Late | 0.6 – 1.0 | Low, near $\omega_{\min}$ | Local exploitation |

**Comparison with linear decay:**

```
ω
0.9 |●────                     (tanh: slow initial drop)
    |     ╲───────
    |          ╲───            (linear: constant rate)
    |               ╲
0.4 |────────────────●
    0              T_max        t
```

The tanh schedule preserves large $\omega$ values (exploration) for longer before rapidly switching to small values (exploitation), which is better suited for high-dimensional, noisy fitness landscapes.

**Note:** Each particle $i$ has its own inertia weight $\omega_i^t$ updated independently, creating **diversity in search behavior** across the swarm.

### 2.2 Adaptive Mutation Factor

The adaptive mutation factor $\mu_{\text{mf}}$ provides a stochastic escape mechanism from local optima.

$$\boxed{\mu_{\text{mf}}^t = 0.7 + 0.3 \cdot \frac{t}{T_{\max}}} \tag{6}$$

Note that $\mu_{\text{mf}}^t \in (0.7, 1.0]$, monotonically **increasing** with iterations.

**Mutation decision rule:**

Let $\xi \sim \mathcal{U}(0, 1)$ be a uniform random variable drawn independently for each particle at each iteration.

$$\text{Action} = \begin{cases}
\text{MUTATE} & \text{if } \xi > \mu_{\text{mf}}^t \\
\text{STANDARD UPDATE} & \text{if } \xi \leq \mu_{\text{mf}}^t
\end{cases} \tag{7}$$

**Mutation probability:**
$$P(\text{mutate at iteration } t) = 1 - \mu_{\text{mf}}^t = 0.3 \cdot \left(1 - \frac{t}{T_{\max}}\right)$$

This decays from 30% at $t=0$ to 0% at $t=T_{\max}$, which is the **reverse** of the inertia schedule: high mutation early (when global search is needed) decreasing to zero late (when exploitation dominates).

**Mutation operation (when triggered):**

The particle's position is **randomly reinitialized** within the search space bounds:

$$\mathbf{x}_i^{t+1} \sim \mathcal{U}(\mathbf{lb}, \mathbf{ub}) \tag{8}$$

where $\mathbf{lb}$ and $\mathbf{ub}$ are the lower and upper bounds of each dimension.

### 2.3 Velocity Clamping

To prevent particles from flying out of the search space, velocity is clamped:

$$v_{i,d}^{t+1} \leftarrow \text{clip}(v_{i,d}^{t+1},\ -v_{\max,d},\ v_{\max,d})$$

where $v_{\max,d} = 0.2 \cdot (ub_d - lb_d)$.

### 2.4 Position Clamping

After the position update:

$$x_{i,d}^{t+1} \leftarrow \text{clip}(x_{i,d}^{t+1},\ lb_d,\ ub_d)$$

---

## 3. Particle Encoding

### 3.1 Particle Representation

Each particle encodes a 5-dimensional vector of LSTM hyperparameters in a **continuous space**, which is decoded to discrete values at each fitness evaluation.

$$\mathbf{x}_i = [x_1,\ x_2,\ x_3,\ x_4,\ x_5] \in \mathbb{R}^5 \tag{9}$$

### 3.2 Encoding and Decoding

| Dim | Variable | Continuous Range | Decoding |
|---|---|---|---|
| $x_1$ | `num_layers` | $[1.0, 4.99]$ | $\lfloor x_1 \rfloor \in \{1,2,3,4\}$ |
| $x_2$ | `hidden_units` | $[32, 512]$ | Round to nearest multiple of 32 |
| $x_3$ | `dropout` | $[0.0, 0.5]$ | Direct (float) |
| $x_4$ | `learning_rate` | $[\log(10^{-5}), \log(10^{-1})]$ | $\exp(x_4)$ |
| $x_5$ | `lookback` | $[0, 3.99]$ | $\{0,1,2,3\} \to \{10,30,60,120\}$ |

**Decoding function:**
```python
def decode(x: np.ndarray) -> dict:
    return {
        'num_layers':   int(np.clip(x[0], 1, 4)),
        'hidden_units': int(round(np.clip(x[1], 32, 512) / 32) * 32),
        'dropout':      float(np.clip(x[2], 0.0, 0.5)),
        'learning_rate': float(np.exp(np.clip(x[3], np.log(1e-5), np.log(1e-1)))),
        'lookback':     [10, 30, 60, 120][int(np.clip(x[4], 0, 3.99))]
    }
```

### 3.3 Initialization

Particle positions are initialized uniformly within the search space bounds:

$$x_{i,d}^0 \sim \mathcal{U}(lb_d, ub_d), \quad d = 1,\ldots,5 \tag{10}$$

Particle velocities are initialized to a fraction of the search range:

$$v_{i,d}^0 \sim \mathcal{U}\!\left(-\frac{ub_d - lb_d}{4},\ \frac{ub_d - lb_d}{4}\right) \tag{11}$$

---

## 4. Search Space Definition

| Dimension | Variable | Lower Bound ($lb_d$) | Upper Bound ($ub_d$) | Scale |
|---|---|---|---|---|
| 1 | `num_layers` | 1.0 | 4.99 | Linear |
| 2 | `hidden_units` | 32 | 512 | Linear |
| 3 | `dropout` | 0.0 | 0.5 | Linear |
| 4 | `learning_rate` (log) | $\ln(10^{-5}) = -11.51$ | $\ln(10^{-1}) = -2.30$ | Log |
| 5 | `lookback_idx` | 0.0 | 3.99 | Categorical index |

**Effective hyperparameter ranges (decoded):**

| Hyperparameter | Values / Range |
|---|---|
| Number of LSTM layers | {1, 2, 3, 4} |
| Hidden units per layer | {32, 64, 96, …, 512} (16 values) |
| Dropout rate | [0.0, 0.5] (continuous) |
| Learning rate | [$10^{-5}$, $10^{-1}$] (log scale) |
| Lookback window | {10, 30, 60, 120} minutes |

**Search space volume:**

$$|\mathcal{X}| = 4 \times 16 \times 0.5 \times 4 \text{ orders of magnitude} \times 4 \approx 2,048 \text{ (discrete combinations)}$$

The actual continuous search volume is $\approx 4.99 \times 480 \times 0.5 \times 9.21 \times 3.99 \approx 46,580$ (product of ranges), which is modest but non-convex due to the LSTM training in the fitness function.

---

## 5. Fitness Function

### 5.1 Composite Fitness

The fitness function combines three metrics to balance prediction accuracy and trading quality:

$$\boxed{F(\mathbf{x}) = \alpha_1 \cdot \hat{\text{RMSE}}(\mathbf{x}) + \alpha_2 \cdot [1 - \hat{\text{SR}}(\mathbf{x})] + \alpha_3 \cdot \hat{\text{MDD}}(\mathbf{x})} \tag{12}$$

where $\hat{\cdot}$ denotes min-max normalization to [0,1] using running bounds across all particle evaluations, and:

- $\alpha_1 = 0.4$ — weight for prediction error
- $\alpha_2 = 0.4$ — weight for Sharpe ratio (inverted, so lower is better)
- $\alpha_3 = 0.2$ — weight for maximum drawdown

**Lower $F(\mathbf{x})$ is better.** The optimizer minimizes $F$.

### 5.2 Component Definitions

**RMSE (Root Mean Squared Error):**

$$\text{RMSE}(\mathbf{x}) = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(\hat{y}_i - y_i\right)^2} \tag{13}$$

where $\hat{y}_i$ are LSTM predictions on the validation set and $y_i$ are true log returns.

**Annualized Sharpe Ratio:**

$$\text{SR}(\mathbf{x}) = \frac{\bar{r}_p - r_f}{\sigma_p} \cdot \sqrt{252 \times 390} \tag{14}$$

where $\bar{r}_p$ is the mean per-bar portfolio return from a simple threshold-based signal:

$$\text{signal}_t = \begin{cases} +1 & \hat{y}_t > \theta \\ -1 & \hat{y}_t < -\theta \\ 0 & \text{otherwise} \end{cases}, \quad \theta = 10^{-4}$$

$r_f = 0$ (zero risk-free rate at bar-level), $\sigma_p$ is the standard deviation of bar-level returns. The factor $\sqrt{252 \times 390}$ annualizes from 1-minute to yearly frequency.

**Maximum Drawdown:**

$$\text{MDD}(\mathbf{x}) = \max_{s \leq t}\left(\frac{V_s - V_t}{V_s}\right) \tag{15}$$

where $V_t$ is the running portfolio value from applying the signal sequence.

### 5.3 Online Normalization

To ensure each component contributes equally regardless of scale:

$$\hat{m}(\mathbf{x}) = \frac{m(\mathbf{x}) - m_{\min}}{m_{\max} - m_{\min} + \epsilon}, \quad \epsilon = 10^{-8} \tag{16}$$

where $m_{\min}$ and $m_{\max}$ are the running minimum and maximum of metric $m$ observed across all particle evaluations so far.

---

## 6. Full Algorithm Pseudocode

```
ALGORITHM: IPSO-LSTM Hyperparameter Optimization

INPUT:
  M = 30          (number of particles)
  T = 50          (maximum iterations)
  ω_min = 0.4, ω_max = 0.9
  c1 = 1.5, c2 = 1.5
  lb = [1.0, 32, 0.0, ln(1e-5), 0.0]
  ub = [4.99, 512, 0.5, ln(1e-1), 3.99]
  X_train, y_train, X_val, y_val

OUTPUT: gbest_params, gbest_fitness

INITIALIZATION:
  For i = 1 to M:
    x_i ~ Uniform(lb, ub)                   # Random position
    v_i ~ Uniform(-(ub-lb)/4, (ub-lb)/4)   # Random velocity
    p_i ← x_i                               # Personal best = initial position
    f_i ← EVALUATE(x_i, X_train, y_train, X_val, y_val)
  g ← argmin_{i} f(p_i)                    # Global best

MAIN LOOP:
  For t = 1 to T:
    For i = 1 to M:
      # --- Mutation check ---
      ξ ~ Uniform(0, 1)
      μ_mf ← 0.7 + 0.3 × (t / T)
      
      IF ξ > μ_mf:                          # MUTATION triggered
        x_i ← Uniform(lb, ub)              # Reinitialize position randomly
        v_i ← Uniform(-(ub-lb)/4, (ub-lb)/4)
      ELSE:                                 # STANDARD update
        # Inertia weight (per particle)
        ω_i ← ω_max - (ω_max - ω_min) × tanh(4t / T)
        
        # Velocity update
        r1, r2 ~ Uniform(0, 1)
        v_i ← ω_i × v_i
              + c1 × r1 × (p_i - x_i)
              + c2 × r2 × (g - x_i)
        
        # Velocity clamping
        v_i ← clip(v_i, -0.2(ub-lb), 0.2(ub-lb))
        
        # Position update
        x_i ← x_i + v_i
        x_i ← clip(x_i, lb, ub)
      END IF
      
      # Evaluate fitness
      f_i ← EVALUATE(x_i, X_train, y_train, X_val, y_val)
      
      # Update personal best
      IF f_i < f(p_i):
        p_i ← x_i
      
    # Update global best
    g ← argmin_{i} f(p_i)
    
    # Log convergence
    Log(t, f(g), decode(g))
    
    # Checkpoint every 10 iterations
    IF t mod 10 == 0:
      SAVE_CHECKPOINT(t, swarm, g, f(g))
  
  RETURN decode(g), f(g)

SUBROUTINE: EVALUATE(x, X_train, y_train, X_val, y_val)
  params ← decode(x)
  model ← LSTMModel(params)
  trainer ← LSTMTrainer(model, lr=params['learning_rate'])
  trainer.fit(X_train[:, :params['lookback'], :], y_train,
              X_val[:, :params['lookback'], :], y_val)
  y_pred ← trainer.predict(X_val)
  rmse ← RMSE(y_val, y_pred)
  sharpe ← SHARPE(y_val, y_pred)
  mdd ← MAX_DRAWDOWN(y_val, y_pred)
  F ← COMPOSITE_FITNESS(rmse, sharpe, mdd)  # Online normalized
  RETURN F
```

---

## 7. Convergence Analysis

### 7.1 Theoretical Convergence Conditions

For standard PSO with constant $\omega$, convergence to a fixed point is guaranteed when (Clerc & Kennedy, 2002):

$$\omega < 1, \quad c_1 + c_2 < 4, \quad \omega > \frac{c_1 + c_2}{2} - 1$$

For our settings: $\omega \in [0.4, 0.9]$, $c_1 = c_2 = 1.5$, $c_1 + c_2 = 3 < 4$. ✓

The tanh schedule ensures $\omega^t$ satisfies these conditions throughout optimization.

### 7.2 Diversity Metric

We track swarm diversity to detect premature convergence:

$$D_t = \frac{1}{M}\sum_{i=1}^{M} \|\mathbf{x}_i^t - \bar{\mathbf{x}}^t\|_2$$

where $\bar{\mathbf{x}}^t = \frac{1}{M}\sum_i \mathbf{x}_i^t$.

Premature convergence is flagged when $D_t < \epsilon_D$ for 5 consecutive iterations, triggering a random reinitializion of the worst 20% of particles.

### 7.3 Expected Convergence Behavior

Based on results from Ji et al. (2021) on the ASX200 dataset:

| Method | Iterations to 80% of gbest | Final RMSE |
|---|---|---|
| Standard PSO | ~15 | Higher |
| IPSO (tanh + mutation) | ~25 (slower but better) | Lower |

The slower initial convergence of IPSO is by design — it maintains exploration longer, enabling escape from early local optima.

### 7.4 Computational Complexity

Per PSO iteration:
- $M$ LSTM training runs (dominant cost)
- Each LSTM training: $O(\text{epochs} \times N \times H^2)$ where $N$ = training samples, $H$ = hidden units

**Total evaluation budget:**
$$\text{Budget} = M \times T = 30 \times 50 = 1{,}500 \text{ LSTM training runs}$$

With early stopping (patience=10) reducing actual epochs to ~20–40 on average, each run takes approximately 30–120 seconds on a GPU, giving a total runtime of 12–50 hours per ticker.

**Parallelization speedup:** 8× with particle-level parallelism → 1.5–6 hours per ticker.

---

## 8. Hyperparameter Settings

### 8.1 PSO Configuration

| Parameter | Symbol | Value | Rationale |
|---|---|---|---|
| Swarm size | $M$ | 30 | Balance between exploration and budget |
| Max iterations | $T$ | 50 | 1,500 total evaluations |
| Min inertia | $\omega_{\min}$ | 0.4 | Standard recommendation (Shi & Eberhart, 1998) |
| Max inertia | $\omega_{\max}$ | 0.9 | Standard recommendation |
| Cognitive coefficient | $c_1$ | 1.5 | Matches Ji et al. (2021) |
| Social coefficient | $c_2$ | 1.5 | Matches Ji et al. (2021) |
| Mutation base | — | 0.7 | From Ji et al. (2021) eq. 14 |
| Velocity clamp fraction | — | 0.2 | 20% of search range per dim |
| Random seed | — | 42 | For reproducibility |

### 8.2 Sensitivity of Key Parameters

**Inertia weight range:** The most sensitive parameters are $\omega_{\min}$ and $\omega_{\max}$. Values outside [0.4, 0.9] can cause either non-convergence ($\omega$ too high) or premature collapse ($\omega$ too low).

**Particle count $M$:** Empirically, $M \geq 20$ provides sufficient diversity for a 5-dimensional space. $M = 30$ offers a comfortable margin.

**c1 vs. c2 ratio:** Setting $c_1 = c_2$ creates symmetric cognitive/social balance. A higher $c_2$ (e.g., 2.0) accelerates convergence but risks premature settling; a higher $c_1$ promotes individual exploration.