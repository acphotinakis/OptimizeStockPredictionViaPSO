#!/usr/bin/env python3
"""
scripts/plot_xgboost_results.py

Convenience script to generate XGBoost result plots.
Wrapper around plots.xgboost_plots module.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plots.xgboost_plots import plot_all_results, create_summary_dashboard


def main():
    parser = argparse.ArgumentParser(
        description="Generate XGBoost result plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--mode", default="train", choices=["train", "val", "test"],
                       help="Training mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    parser.add_argument("--dashboard", action="store_true", 
                       help="Create single dashboard instead of individual plots")
    
    args = parser.parse_args()
    
    print(f"Generating plots for {args.ticker}...")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        if args.dashboard:
            # Create single dashboard
            print("\nCreating summary dashboard...")
            create_summary_dashboard(
                args.results_dir,
                args.ticker,
                args.mode,
                args.seed,
                save_path=Path(args.output_dir) / f"xgb_dashboard_{args.ticker}_{args.mode}_seed{args.seed}.png",
                show=args.show,
            )
            print("✓ Dashboard created successfully")
        else:
            # Create all individual plots
            print("\nGenerating individual plots...")
            figures = plot_all_results(
                args.results_dir,
                args.ticker,
                args.mode,
                args.seed,
                args.output_dir,
                args.show,
            )
            print(f"✓ Generated {len(figures)} plots successfully")
    
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print(f"\nMake sure you have run training for {args.ticker} first:")
        print(f"  python scripts/xgboost/run_xgboost.py --ticker {args.ticker} --mode train")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
