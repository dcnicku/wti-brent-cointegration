"""
The three charts I put in the write-up.

  1. The spread with its entry bands, shaded where a position was open
  2. Equity curves at each cost assumption  <- the one that matters
  3. The hedge ratio estimated on a rolling window, to check it is stable
"""

import matplotlib
matplotlib.use("Agg")          # save to file, no display needed
import matplotlib.pyplot as plt
import pandas as pd

NAVY = "#1f3a5f"
RED = "#c0392b"
GREY = "#888888"


def plot_spread(df, entry=2.0, stop=3.5, path="results/fig1_spread.png"):
    """Z-scored spread with the entry and stop bands."""
    fig, ax = plt.subplots(figsize=(11, 5))
    z = df["z"]

    ax.plot(z.index, z.values, lw=0.8, color=NAVY, label="z-score")
    ax.axhline(entry, ls="--", lw=0.9, color=RED, label=f"entry +/-{entry}")
    ax.axhline(-entry, ls="--", lw=0.9, color=RED)
    ax.axhline(stop, ls=":", lw=0.9, color=RED, label=f"stop +/-{stop}")
    ax.axhline(-stop, ls=":", lw=0.9, color=RED)
    ax.axhline(0, lw=0.6, color=GREY)

    ax.fill_between(z.index, -5, 5, where=(df["position"] != 0),
                    alpha=0.10, color="#2e7d32", label="in position")

    ax.set_ylim(-5, 5)
    ax.set_title("WTI-Brent spread (rolling z-score), out of sample")
    ax.set_ylabel("standard deviations")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_equity(sweep_curves, path="results/fig2_equity.png"):
    """
    One equity curve per cost level.

    sweep_curves: dict of {cost_bps: (equity Series, net Sharpe)}
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#1b5e20", "#2e7d32", "#f9a825", RED]

    for (c, (equity, sr)), col in zip(sorted(sweep_curves.items()), colors):
        ax.plot(equity.index, equity.values, lw=1.3, color=col,
                label=f"{c}bps  (Sharpe {sr:.2f})")

    ax.axhline(1.0, lw=0.7, color=GREY, ls="--")
    ax.set_title("Cumulative return by transaction cost assumption")
    ax.set_ylabel("growth of 1")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_rolling_beta(log_x, log_y, window=250, path="results/fig3_beta.png"):
    """
    Hedge ratio re-estimated on a rolling window.

    A flat line supports holding beta fixed, as I do. Visible drift is evidence
    against it, and would be the argument for letting beta move over time.

    The OLS slope of x on y is just Cov(x, y) / Var(y), so this needs no
    regression loop.
    """
    idx = log_x.index.intersection(log_y.index)
    x, y = log_x.loc[idx], log_y.loc[idx]

    betas = x.rolling(window).cov(y) / y.rolling(window).var()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(betas.index, betas.values, lw=1.2, color=NAVY)
    ax.axhline(1.0, ls="--", lw=0.8, color=GREY, label="beta = 1 (parity)")
    ax.set_title(f"Rolling hedge ratio ({window}-day window)")
    ax.set_ylabel("beta")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
