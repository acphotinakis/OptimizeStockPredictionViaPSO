from enum import Enum
from pathlib import Path
from datetime import datetime

# =========================================================
# ENUMS
# =========================================================


class ModelType(str, Enum):
    LSTM = "lstm"
    XGBOOST = "xgboost"


class Phase(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class RunMode(str, Enum):
    CREATE = "create"
    ATTACH = "attach"


class ArtifactType(Enum):
    MODEL = "model"
    METRICS = "metrics"
    HISTORY = "history"
    PREDICTIONS = "predictions"
    TRUTH = "truth"
    IMPORTANCES = "importances"
    PLOT = "plot"
    CONFIG = "config"
