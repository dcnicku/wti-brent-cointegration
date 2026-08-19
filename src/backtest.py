"""
Turning the spread into positions, and scoring the result.

Covers METHODOLOGY.md sections 5-6.

Two places where it would be easy to cheat, and what I do about each:
  1. The z-score uses a trailing window only, never the full-sample mean.
  2. A signal from the close of day t is traded at the close of day t+1.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------

def rolling_zscore(spread, window):
    """
    Standardise the spread against its own recent history.

    pandas .rolling() only ever looks backwards: the window at t covers
    [t-window+1, t]. Including today is fine, since today's close is known at
    today's close. What would NOT be fine is using spread.mean() over the whole
    sample, which puts information from 2024 into a signal dated 2022.
    """
    s = pd.Series(spread)
    mu = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std(ddof=1)
    return (s - mu) / sd


def choose_window(half_life, lo=30, hi=120, mult=5):
    """Window of 5 half-lives, capped to a sensible range."""
    if not np.isfinite(half_life) or half_life <= 0:
        return 60
    return int(np.clip(round(mult * half_life), lo, hi))


def generate_positions(z, entry=2.0, exit_=0.5, stop=3.5, max_hold=None):
    """
    Turn z-scores into positions in {-1, 0, +1}.

        +1 = long the spread  (it is unusually low, I expect it to rise)
        -1 = short the spread (it is unusually high, I expect it to fall)

    Written as a loop rather than vectorised, because what I do today depends
    on whether I am already in a position.

    Note the "blocked" flag. Without it the stop-loss does nothing: if I am
    stopped out at z = 3.6 then z is still above the entry threshold of 2.0 the
    next day, so I would re-enter the same losing trade immediately. After a
    stop I stay flat until the spread comes back inside the entry band.
    """
    z = pd.Series(z)
    pos = np.zeros(len(z))
    state, held, blocked = 0, 0, False

    for i, zi in enumerate(z.values):
        if np.isnan(zi):
            pos[i] = 0
            state, held, blocked = 0, 0, False
            continue

        if state == 0:
            if blocked:
                if abs(zi) < entry:
                    blocked = False          # spread has calmed down
            elif zi > entry:
                state, held = -1, 0          # too high, short it
            elif zi < -entry:
                state, held = +1, 0          # too low, buy it
        else:
            held += 1
            if abs(zi) > stop:
                state, held, blocked = 0, 0, True
            elif abs(zi) < exit_ or (max_hold is not None and held >= max_hold):
                state, held = 0, 0

        pos[i] = state

    return pd.Series(pos, index=z.index, name="position")


# ---------------------------------------------------------------------------
# P&L
# ---------------------------------------------------------------------------

def run_backtest(log_x, log_y, beta, half_life, cost_bps=10.0,
                 entry=2.0, exit_=0.5, stop=3.5, window=None):
    """
    Run the strategy over one period.

    beta is the hedge ratio estimated on the training window and then held
    fixed. cost_bps is the one-way cost per leg.

    Returns (daily DataFrame, metrics dict).
    """
    idx = log_x.index.intersection(log_y.index)
    log_x, log_y = log_x.loc[idx], log_y.loc[idx]

    spread = log_x - beta * log_y

    if window is None:
        window = choose_window(half_life)

    z = rolling_zscore(spread, window)
    max_hold = int(round(3 * half_life)) if np.isfinite(half_life) else None
    pos = generate_positions(z, entry, exit_, stop, max_hold)

    # Execution lag: act on yesterday's signal, at today's close.
    pos_traded = pos.shift(1).fillna(0)

    # Return on being long 1 unit of x and short beta units of y.
    spread_return = log_x.diff() - beta * log_y.diff()
    gross = (pos_traded * spread_return).fillna(0)

    # Costs are charged when the position ACTUALLY changes, which is the traded
    # series, not the signal series. Using pos.diff() here would bill the trade
    # a day before it happens.
    turnover = pos_traded.diff().abs().fillna(pos_traded.abs())
    cost = turnover * (1 + abs(beta)) * (cost_bps / 10_000.0)

    net = gross - cost

    out = pd.DataFrame({
        "spread": spread,
        "z": z,
        "position": pos,
        "position_traded": pos_traded,
        "gross": gross,
        "turnover": turnover,
        "cost": cost,
        "net": net,
        "equity": (1 + net).cumprod(),
    })
    return out, performance(out)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def max_drawdown(equity):
    """Worst peak-to-trough fall in the equity curve."""
    eq = pd.Series(equity).dropna()
    if eq.empty:
        return 0.0
    return float((eq / eq.cummax() - 1).min())


def sharpe(returns, periods=TRADING_DAYS):
    """
    Annualised Sharpe. No risk-free rate subtracted: the spread is funded by
    its own short leg, so the return is already close to an excess return.
    """
    r = pd.Series(returns).dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(np.sqrt(periods) * r.mean() / r.std(ddof=1))


def trade_details(df):
    """
    Split the run into individual trades, for the win rate and holding period.

    A trade opens on the day the signal turns non-zero (index "start") and the
    money starts moving the day after, because of the execution lag.
    """
    p = df["position"].fillna(0)
    trades, start = [], None

    for i in range(len(p)):
        if p.iloc[i] != 0 and (i == 0 or p.iloc[i - 1] == 0):
            start = i
        elif p.iloc[i] == 0 and start is not None:
            trades.append({"days": i - start,
                           "pnl": float(df["net"].iloc[start + 1:i + 1].sum())})
            start = None

    if start is not None:                      # still open when the data ends
        trades.append({"days": len(p) - start,
                       "pnl": float(df["net"].iloc[start + 1:].sum())})

    return pd.DataFrame(trades)


def performance(df):
    """Headline numbers for one run."""
    net = df["net"].dropna()
    pos = df["position"].fillna(0)
    n_years = len(net) / TRADING_DAYS if len(net) else 0
    total = float(df["equity"].iloc[-1] - 1) if len(df) else 0.0
    td = trade_details(df)

    # Guard against a total loss, where the compounding formula is undefined.
    if n_years > 0 and total > -1:
        ann_return = float((1 + total) ** (1 / n_years) - 1)
    else:
        ann_return = float("nan")

    return {
        "sharpe_gross": sharpe(df["gross"]),
        "sharpe_net": sharpe(df["net"]),
        "total_return": total,
        "ann_return": ann_return,
        "ann_vol": float(net.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(net) > 1 else 0.0,
        "max_drawdown": max_drawdown(df["equity"]),
        "n_trades": int(((pos != 0) & (pos.shift(1).fillna(0) == 0)).sum()),
        "win_rate": float((td["pnl"] > 0).mean()) if len(td) else 0.0,
        "avg_hold_days": float(td["days"].mean()) if len(td) else 0.0,
        "total_costs": float(df["cost"].sum()),
        "time_in_market": float((pos != 0).mean()),
    }


def cost_scenarios(log_x, log_y, beta, half_life, cost_grid=(0, 5, 10, 25), **kw):
    """
    Run the strategy under each cost assumption.

    Costs do not change which trades I take, only what they earn, so I run the
    backtest once at zero cost and then subtract each cost level from it.

    Returns {cost_bps: (daily DataFrame, metrics dict)}.
    """
    base, _ = run_backtest(log_x, log_y, beta, half_life, cost_bps=0.0, **kw)

    scenarios = {}
    for c in cost_grid:
        df = base.copy()
        df["cost"] = base["turnover"] * (1 + abs(beta)) * (c / 10_000.0)
        df["net"] = df["gross"] - df["cost"]
        df["equity"] = (1 + df["net"]).cumprod()
        scenarios[c] = (df, performance(df))

    return scenarios


def cost_sweep(log_x, log_y, beta, half_life, cost_grid=(0, 5, 10, 25), **kw):
    """
    How quickly does the edge disappear as costs rise? This is the main result.
    """
    scenarios = cost_scenarios(log_x, log_y, beta, half_life, cost_grid, **kw)
    rows = [{"cost_bps": c, **m} for c, (_, m) in scenarios.items()]
    return pd.DataFrame(rows).set_index("cost_bps")


def buy_and_hold(log_price):
    """Benchmark: just hold WTI over the same window."""
    simple = np.exp(log_price.diff().fillna(0)) - 1
    equity = (1 + simple).cumprod()
    return {
        "sharpe": sharpe(simple),
        "total_return": float(equity.iloc[-1] - 1),
        "max_drawdown": max_drawdown(equity),
    }
