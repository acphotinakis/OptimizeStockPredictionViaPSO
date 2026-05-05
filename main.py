from __future__ import annotations

import argparse
from typing import Optional, Sequence


def _extract_provider(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--provider", choices=["yfinance", "alpaca"], default="yfinance")
    args, remaining = parser.parse_known_args(argv)
    return args.provider, remaining


def main(argv: Optional[Sequence[str]] = None) -> None:
    provider, remaining = _extract_provider(argv)
    if provider == "alpaca":
        from src.utils.alpaca_cli import main as provider_main
    else:
        from src.utils.yfinance_cli import main as provider_main
    provider_main(remaining)


if __name__ == "__main__":
    main()
