"""Tests for safe data storage operations and JSON-based feature metadata loading."""

import json
from pathlib import Path
import pytest

from src.utils.data_storage import load_feature_names


def test_load_feature_names_from_json(tmp_path: Path):
    """Verify load_feature_names safely parses JSON metadata without pickle deserialization."""
    ticker = "AAPL"
    ticker_dir = tmp_path / ticker
    ticker_dir.mkdir(parents=True)

    expected_features = ["open", "high", "low", "close", "ema12", "rsi"]

    # 1. Direct feature_names.json
    fn_path = ticker_dir / "feature_names.json"
    with open(fn_path, "w", encoding="utf-8") as f:
        json.dump(expected_features, f)

    loaded = load_feature_names(tmp_path, ticker)
    assert loaded == expected_features

    # 2. If feature_names.json removed, fallback to selected_features.json
    fn_path.unlink()
    sf_path = ticker_dir / "selected_features.json"
    with open(sf_path, "w", encoding="utf-8") as f:
        json.dump(expected_features[:4], f)

    loaded_sf = load_feature_names(tmp_path, ticker)
    assert loaded_sf == expected_features[:4]

    # 3. If selected_features.json removed, fallback to frozen state JSON
    sf_path.unlink()
    frozen_path = ticker_dir / f"{ticker}_frozen_state.json"
    with open(frozen_path, "w", encoding="utf-8") as f:
        json.dump({"feature_names_selected": ["ema12", "rsi"]}, f)

    loaded_frozen = load_feature_names(tmp_path, ticker)
    assert loaded_frozen == ["ema12", "rsi"]

    # 4. If all JSON files missing, raise FileNotFoundError
    frozen_path.unlink()
    with pytest.raises(FileNotFoundError, match="No valid feature names JSON file found"):
        load_feature_names(tmp_path, ticker)
