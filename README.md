pso_lstm_stock_tuner/
├── configs/                # Hydra/YAML configuration files
│   ├── data/               # Tickers, Alpaca settings, cleaning rules
│   ├── features/           # Indicator params, wavelet denoising settings
│   ├── model/              # LSTM architecture defaults
│   ├── optimization/       # PSO search space & swarm parameters
│   └── config.yaml         # Main entry point for configs
├── data/                   # Data versioned and isolated
│   ├── raw/                # Unmodified Alpaca OHLCV (Parquet)
│   ├── interim/            # Cleaned, joined, but unscaled data
│   ├── processed/          # Final features + scaled tensors
│   └── results/            # Model predictions & metric CSVs
├── logs/                   # Timestamped execution logs
├── models/                 # Versioned artifacts
│   ├── checkpoints/        # PyTorch .pt weights
│   ├── scalers/            # Saved MinMax/Standard scalers
│   └── pso_best/           # JSON files of "Global Best" configs
├── notebooks/              # EDA and result visualization
├── reports/                # Generated figures (loss curves, Sharpe plots)
├── scripts/                # Shell utilities and automation
├── src/                    # Primary source code
│   ├── data/               # Ingestion (Alpaca) & Integrity checks
│   ├── features/           # Feature engineering & Denoising (Wavelets)
│   ├── models/             # PyTorch LSTM definitions
│   ├── optimization/       # Improved PSO logic (Mutation/Inertia)
│   ├── training/           # Training loops & Walk-forward validation
│   ├── evaluation/         # Performance metrics (Sharpe, MDD, RMSE)
│   └── utils/              # Logging setup, seed control, pathing
├── tests/                  # Unit tests (Pytest)
├── .env                    # API Keys (gitignored)
├── .env.example            # Template for team members
├── .gitignore
├── Makefile                # Workflow automation
├── pyproject.toml          # Dependency management (Poetry)
├── README.md               # Setup & execution guide
└── main.py                 # CLI Entry point