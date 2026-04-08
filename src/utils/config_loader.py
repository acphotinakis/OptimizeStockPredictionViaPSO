# src/utils/config_loader.py

from pathlib import Path
import yaml

from src.utils.config_schema import *


def load_config(path: str | Path) -> Config:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return Config(
        pso=PSOConfig(**raw["pso"]),
        lstm=LSTMConfig(
            num_layers=RangeInt(**raw["lstm"]["num_layers"]),
            hidden_units=RangeInt(**raw["lstm"]["hidden_units"]),
            dropout=RangeFloat(**raw["lstm"]["dropout"]),
            learning_rate=RangeFloat(**raw["lstm"]["learning_rate"]),
            lookback=LookbackConfig(**raw["lstm"]["lookback"]),
            max_epochs=raw["lstm"]["max_epochs"],
            batch_size=raw["lstm"]["batch_size"],
            early_stopping_patience=raw["lstm"]["early_stopping_patience"],
            grad_clip=raw["lstm"]["grad_clip"],
        ),
        lstm_baseline=LSTMBaselineConfig(**raw["lstm_baseline"]),
        xgboost=XGBoostConfig(**raw["xgboost"]),
        fitness=FitnessConfig(**raw["fitness"]),
        data=DataConfig(**raw["data"]),
        features=FeaturesConfig(
            selector=SelectorConfig(
                method=raw["features"]["selector"]["method"],
                importance_threshold=raw["features"]["selector"][
                    "importance_threshold"
                ],
                xgb_params=XGBParams(**raw["features"]["selector"]["xgb_params"]),
            )
        ),
        backtesting=BacktestingConfig(**raw["backtesting"]),
        logging=LoggingConfig(**raw["logging"]),
    )
