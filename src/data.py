"""
Downloading and preparing the price data.

I use front-month continuous futures from Yahoo Finance:
    CL=F  WTI crude
    BZ=F  Brent crude

Caveat on these series: they are "continuous front-month", meaning that when a
contract expires the series jumps to the next one. That jump is a bookkeeping
step, not a real price move. I have not corrected for it (see METHODOLOGY.md,
section 7) so some of the spread volatility I measure is roll, not economics.
"""

import numpy as np
import pandas as pd

WTI = "CL=F"
BRENT = "BZ=F"


def download_prices(start="2016-01-01", end=None):
    """
    Download daily closes for both contracts, aligned, with no gaps.

    Returns a DataFrame with columns ["WTI", "BRENT"] indexed by date.
    """
    import yfinance as yf

    raw = yf.download(
        [WTI, BRENT],
        start=start,
        end=end,
        # Futures have no splits or dividends so this flag changes nothing here.
        # I set it explicitly so the result doesn't shift if yfinance changes
        # its default.
        auto_adjust=True,
        progress=False,
    )

    prices = raw["Close"].rename(columns={WTI: "WTI", BRENT: "BRENT"})

    # Keep only days when BOTH contracts traded. US and UK holidays differ, and
    # forward-filling one leg through a holiday would invent a spread move that
    # never happened.
    prices = prices[["WTI", "BRENT"]].dropna()

    prices.index = pd.to_datetime(prices.index)
    return prices.sort_index()


def log_prices(prices):
    """Work in logs: differences are percentage returns and beta is a ratio."""
    return np.log(prices)


def split_train_test(df, split_date="2022-01-01"):
    """
    Split into a training period (where I estimate everything) and a test
    period (where I only trade). Nothing from the test period is allowed to
    influence a parameter choice.
    """
    split = pd.Timestamp(split_date)
    return df.loc[df.index < split], df.loc[df.index >= split]


def summarise(prices):
    """Sanity check: date range, number of days, price range, missing values."""
    rows = []
    for col in prices.columns:
        s = prices[col]
        rows.append({
            "series": col,
            "start": s.index.min().date(),
            "end": s.index.max().date(),
            "n_obs": len(s),
            "min": round(float(s.min()), 2),
            "max": round(float(s.max()), 2),
            "n_missing": int(s.isna().sum()),
        })
    return pd.DataFrame(rows).set_index("series")
