"""
Checking whether the backtest result is real.

The first version of this project produced a net Sharpe of 0.92 out of sample,
which is a suspiciously good number for a rule this simple on a market this
liquid. Rather than report it, I decomposed where the P&L actually came from.

It turns out almost all of it lands on the first trading day of each month,
which is when the front-month futures contracts roll. That is not a tradeable
move, it is an artefact of how the continuous price series is stitched together
(see METHODOLOGY.md section 8). This module is the test that shows it.
"""

import numpy as np
import pandas as pd

from backtest import performance


def first_trading_day_mask(index):
    """
    True on the first trading day of each calendar month.

    Front-month crude futures roll monthly, so this is where the continuous
    series jumps from the expiring contract to the next one.
    """
    idx = pd.DatetimeIndex(index)
    positions = pd.Series(range(len(idx)), index=idx).groupby(
        [idx.year, idx.month]).head(1)

    mask = np.zeros(len(idx), dtype=bool)
    mask[positions.values] = True
    return pd.Series(mask, index=idx)


def roll_attribution(df):
    """
    Compare month-start days against every other day, on two measures:
    how much the spread moves, and how much money the strategy makes.

    If the strategy had a real edge, P&L would be spread across the sample
    roughly in proportion to the number of days. If it is harvesting roll
    jumps, it will be concentrated on month-start days.
    """
    is_first = first_trading_day_mask(df.index)
    moves = df["spread"].diff().abs()
    gross = df["gross"]

    total = gross.sum()
    biggest = moves.nlargest(20)

    return {
        "n_month_start_days": int(is_first.sum()),
        "share_of_days": float(is_first.mean()),
        "mean_abs_move_month_start": float(moves[is_first].mean()),
        "mean_abs_move_other_days": float(moves[~is_first].mean()),
        "move_ratio": float(moves[is_first].mean() / moves[~is_first].mean()),
        "top20_moves_on_month_start": int(biggest.index.isin(
            df.index[is_first.values]).sum()),
        "pnl_month_start": float(gross[is_first].sum()),
        "pnl_other_days": float(gross[~is_first].sum()),
        "share_of_pnl_month_start": float(gross[is_first].sum() / total),
    }


def performance_excluding_rolls(df):
    """
    Re-score the strategy with month-start returns set to zero.

    This is the pessimistic reading: assume none of the roll-day move was
    capturable, and see what edge is left on the days you could actually trade.
    """
    mask = first_trading_day_mask(df.index)

    clean = df.copy()
    clean.loc[mask.values, "gross"] = 0.0
    clean.loc[mask.values, "net"] = 0.0
    clean["equity"] = (1 + clean["net"]).cumprod()

    return performance(clean)


def concentration(df, n=5):
    """What share of gross P&L came from the n single best days?"""
    gross = df["gross"]
    total = gross.sum()
    return {
        "best_day": float(gross.max()),
        "share_best_day": float(gross.max() / total),
        f"share_best_{n}_days": float(gross.nlargest(n).sum() / total),
        "best_day_date": gross.idxmax(),
    }
