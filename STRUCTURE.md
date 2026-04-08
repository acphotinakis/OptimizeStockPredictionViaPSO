# Project Structure

This document describes the reorganized codebase structure for better modularity and maintainability.

## Directory Structure

```
ClaudePaper/
├── src/                    # Source code - Model definitions and utilities
│   ├── models/            # Model classes only (no training logic)
│   │   ├── lstm/          # LSTM model definitions
│   │   ├── xgboost/       # XGBoost model definitions
│   │   ├── rl/            # Reinforcement learning models
│   │   └── baselines.py   # Baseline model classes
│   ├── data/              # Data processing modules
│   ├── features/          # Feature engineering
│   ├── evaluation/        # Evaluation metrics and backtesting
│   ├── optimizer/         # PSO and other optimizers
│   └── utils/             # Utility functions
│
├── pipelines/             # Pipeline execution scripts (.py files)
│   ├── ingest_data.py     # Data ingestion pipeline
│   ├── build_features.py  # Feature engineering pipeline
│   ├── run_lstm_baseline.py   # LSTM training/val/test pipeline
│   ├── run_xgboost.py     # XGBoost main pipeline runner
│   ├── train_xgboost.py   # XGBoost training logic
│   ├── validate_xgboost.py # XGBoost validation logic
│   ├── test_xgboost.py    # XGBoost testing logic
│   ├── run_pso.py         # PSO optimization pipeline
│   ├── train_rl_agent.py  # RL training pipeline
│   ├── evaluate_rl_agent.py # RL evaluation pipeline
│   ├── backtest.py        # Backtesting pipeline
│   ├── evaluate.py        # Model evaluation pipeline
│   └── plots_*.py         # Plotting utilities
│
├── scripts/               # Shell scripts only (.sh files)
│   ├── run_data_ingestion.sh      # Run data ingestion
│   ├── run_feature_engineering.sh # Run feature engineering
│   ├── run_xgboost_pipeline.sh    # Run XGBoost pipeline
│   ├── run_lstm_pipeline.sh       # Run LSTM pipeline
│   ├── run_rl_pipeline.sh         # Run RL pipeline
│   └── run_spy_analysis.sh        # Run full analysis
│
├── config/                # Configuration files
├── data/                  # Data directories
│   ├── raw/              # Raw downloaded data
│   ├── cleaned/          # Cleaned data
│   ├── processed/        # Aligned data
│   └── features/         # Feature matrices
│
├── results/              # Model outputs and results
└── tests/                # Unit tests
```

## Design Principles

### 1. Separation of Concerns

- **src/models/**: Contains only model class definitions
  - Model architecture
  - Forward pass logic
  - Prediction methods
  - Save/load functionality
  - NO training loops, validation logic, or testing code

- **pipelines/**: Contains execution logic
  - Training pipelines
  - Validation pipelines
  - Testing pipelines
  - Data processing pipelines
  - All .py files that orchestrate the workflow

- **scripts/**: Contains shell scripts only
  - Wrapper scripts to run pipelines
  - Multi-step workflows
  - All .sh files

### 2. Model Organization

Each model type follows this structure:

```
src/models/<model_type>/
├── <model_name>_model.py   # Model class definition
└── __init__.py             # Exports
```

Training, validation, and testing logic lives in `pipelines/`:

```
pipelines/
├── train_<model_type>.py    # Training logic
├── validate_<model_type>.py # Validation logic
├── test_<model_type>.py     # Testing logic
└── run_<model_type>.py      # Main runner that dispatches to above
```

### 3. Pipeline Execution Flow

1. **Data Ingestion**: `scripts/run_data_ingestion.sh`
   - Downloads raw data
   - Cleans and validates
   - Aligns to benchmark (SPY)

2. **Feature Engineering**: `scripts/run_feature_engineering.sh`
   - Builds technical indicators
   - Creates cross-ticker features
   - Applies feature selection

3. **Model Training**: `scripts/run_<model>_pipeline.sh`
   - Trains model
   - Validates performance
   - Tests on held-out set

## Usage Examples

### Complete Workflow

```bash
# 1. Ingest data
bash scripts/run_data_ingestion.sh config/default_config.yaml config/tickers.txt

# 2. Build features
bash scripts/run_feature_engineering.sh config/default_config.yaml config/tickers.txt 4

# 3. Train XGBoost
bash scripts/run_xgboost_pipeline.sh SPY 42 default

# 4. Train LSTM
bash scripts/run_lstm_pipeline.sh SPY 42
```

### Individual Pipeline Steps

```bash
# Run only XGBoost training
python pipelines/run_xgboost.py --ticker SPY --mode train --seed 42

# Run only XGBoost validation
python pipelines/run_xgboost.py --ticker SPY --mode val --seed 42

# Run only XGBoost testing
python pipelines/run_xgboost.py --ticker SPY --mode test --seed 42
```

## Benefits of This Structure

1. **Modularity**: Models are independent of training logic
2. **Reusability**: Model classes can be imported and used anywhere
3. **Testability**: Easy to unit test models separately from pipelines
4. **Clarity**: Clear separation between "what" (models) and "how" (pipelines)
5. **Maintainability**: Changes to training logic don't affect model definitions
6. **Consistency**: All shell scripts in one place, all Python pipelines in another

## Migration Notes

### Old Structure → New Structure

- `src/models/xgboost/xgboost_train.py` → `pipelines/train_xgboost.py`
- `src/models/xgboost/xgboost_val.py` → `pipelines/validate_xgboost.py`
- `src/models/xgboost/xgboost_test.py` → `pipelines/test_xgboost.py`
- `scripts/backtest.py` → `pipelines/backtest.py`
- `scripts/evaluate.py` → `pipelines/evaluate.py`
- `scripts/run_pso.py` → `pipelines/run_pso.py`

### Removed Files

- `src/models/xgboost/helpers.py` - Functions moved to pipelines
- `src/models/xgboost/consts.py` - Constants moved to pipeline files
- `src/models/lstm/lstm_baseline/` - Duplicate code removed

## Testing

Run tests to ensure the reorganization works:

```bash
# Test model imports
python -c "from src.models.xgboost.xgboost_model import XGBoostModel; print('✓ XGBoost model imports')"
python -c "from src.models.lstm.lstm_model import LSTMModel; print('✓ LSTM model imports')"
python -c "from src.models.baselines import VanillaLSTM; print('✓ Baselines import')"

# Test pipeline imports
python -c "from pipelines.train_xgboost import run_train; print('✓ XGBoost training pipeline imports')"
python -c "from pipelines.run_lstm_baseline import main; print('✓ LSTM pipeline imports')"
```
