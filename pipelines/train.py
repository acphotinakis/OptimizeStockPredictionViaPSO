#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plots.plots_lstm import plot_lstm_pnl
from src.evaluation.metrics import all_statistical_metrics
from src.models.baselines import VanillaLSTM
from src.data.splitter import build_windows
from src.evaluation import all_statistical_metrics
from src.utils import set_all_seeds, setup_logger
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)
