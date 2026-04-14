def _plot_equity(
    result,
    ticker: str,
    model: str,
    closes: np.ndarray,
    initial_capital: float,
    save_path: Path,
) -> None:
    """3-panel summary plot."""
    equity = result.equity_curve
    bar_ret = result.bar_returns
    bah_curve = initial_capital * np.exp(
        np.cumsum(np.log(closes[1:] / closes[:-1] + 1e-10))
    )
    bah_curve = np.concatenate([[initial_capital], bah_curve])[: len(equity)]

    fig, axes = plt.subplots(
        3, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [3, 1.5, 1.5]}
    )
    fig.suptitle(
        f"{ticker} — {model.upper()} Test Evaluation\n"
        f"Sharpe={result.sharpe:.3f}  MDD={result.mdd:.2%}  "
        f"CAGR={result.cagr_:.2%}  Trades={result.n_trades}",
        fontsize=13,
        fontweight="bold",
    )

    # Panel 1: Equity
    ax = axes[0]
    x = np.arange(len(equity))
    ax.plot(x, equity, color="royalblue", linewidth=1.5, label="Strategy")
    ax.plot(
        x,
        bah_curve,
        color="grey",
        linewidth=1.0,
        linestyle="--",
        alpha=0.7,
        label="Buy-and-hold",
    )
    ax.axhline(initial_capital, color="black", linewidth=0.8, linestyle=":")
    ax.fill_between(
        x,
        initial_capital,
        equity,
        where=(equity >= initial_capital),
        color="green",
        alpha=0.15,
    )
    ax.fill_between(
        x,
        initial_capital,
        equity,
        where=(equity < initial_capital),
        color="red",
        alpha=0.15,
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_ylabel("Equity ($)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # Panel 2: Drawdown
    ax = axes[1]
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-10) * 100
    ax.fill_between(x, 0, dd, color="red", alpha=0.5)
    ax.plot(x, dd, color="darkred", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Drawdown (%)")
    ax.grid(alpha=0.3)

    # Panel 3: Return distribution
    ax = axes[2]
    nonzero = bar_ret[bar_ret != 0]
    ax.hist(nonzero * 100, bins=80, color="steelblue", alpha=0.7, edgecolor="none")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(
        nonzero.mean() * 100,
        color="green",
        linewidth=1.5,
        linestyle="--",
        label=f"Mean {nonzero.mean()*100:.3f}%",
    )
    ax.set_xlabel("Bar Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Equity plot saved --> %s", save_path)
