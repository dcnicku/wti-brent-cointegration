"""
Cointegration testing and the half-life estimate.

Covers METHODOLOGY.md sections 2-4.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


# ---------------------------------------------------------------------------
# Are the individual price series I(1)?
# ---------------------------------------------------------------------------

def test_unit_root(series, name=""):
    """
    Augmented Dickey-Fuller test, run on the levels and on the differences.

    I want to FAIL to reject in levels (there is a unit root) and to reject in
    differences (the differences are stationary). That pair of results is what
    "I(1)" means, and I(1) is the precondition for cointegration to be the
    right tool.
    """
    s = pd.Series(series).dropna()

    p_levels = adfuller(s, autolag="AIC")[1]
    p_diffs = adfuller(s.diff().dropna(), autolag="AIC")[1]

    return {
        "name": name,
        "p_levels": p_levels,
        "p_diffs": p_diffs,
        "is_I1": (p_levels > 0.05) and (p_diffs < 0.05),
    }


# ---------------------------------------------------------------------------
# Engle-Granger cointegration
# ---------------------------------------------------------------------------

@dataclass
class CointResult:
    dependent: str
    independent: str
    alpha: float
    beta: float
    stat: float
    pvalue: float
    crit_5pct: float
    spread: pd.Series
    cointegrated: bool


def engle_granger(x, y, x_name="x", y_name="y"):
    """
    Engle-Granger, regressing x on y.

    Step 1:  x_t = alpha + beta*y_t + s_t     (OLS)
    Step 2:  test whether the residual s_t is stationary

    The step-2 test needs Engle-Granger critical values, not ordinary ADF ones.
    The reason: s_t is not observed data, it is a residual whose variance the
    step-1 regression has already minimised by choosing beta. That makes it look
    more stationary than it is, so the correct critical values are stricter.
    statsmodels' coint() applies them; running adfuller() on the residual myself
    would overstate the significance.
    """
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()
    idx = x.index.intersection(y.index)
    x, y = x.loc[idx], y.loc[idx]

    model = sm.OLS(x.values, sm.add_constant(y.values)).fit()
    alpha, beta = float(model.params[0]), float(model.params[1])

    spread = pd.Series(x.values - alpha - beta * y.values, index=idx, name="spread")

    stat, pvalue, crit = coint(x.values, y.values)

    return CointResult(
        dependent=x_name,
        independent=y_name,
        alpha=alpha,
        beta=beta,
        stat=float(stat),
        pvalue=float(pvalue),
        crit_5pct=float(crit[1]),
        spread=spread,
        cointegrated=bool(pvalue < 0.05),
    )


# ---------------------------------------------------------------------------
# How fast does the spread revert?
# ---------------------------------------------------------------------------

@dataclass
class OUResult:
    phi: float          # AR(1) coefficient
    mu: float           # long-run level of the spread
    half_life: float    # days to close half the gap
    stationary: bool


def fit_ou(spread):
    """
    Fit an AR(1) to the spread and read off the half-life.

        ds_t = a + b * s_{t-1} + u_t
        phi  = 1 + b
        half_life = ln(0.5) / ln(phi)

    Intuition: phi is how much of today's deviation survives to tomorrow, so
    the half-life is how many days until half of it is gone. It only makes
    sense if the spread is stationary, which needs -2 < b < 0.
    """
    s = pd.Series(spread).dropna()
    ds = s.diff().dropna()
    lag = s.shift(1).loc[ds.index]

    fit = sm.OLS(ds.values, sm.add_constant(lag.values)).fit()
    a, b = float(fit.params[0]), float(fit.params[1])

    phi = 1.0 + b
    stationary = (b < 0) and (abs(phi) < 1)

    if stationary:
        half_life = float(np.log(0.5) / np.log(phi))
        mu = -a / b            # the level where the drift is zero
    else:
        half_life = float("inf")
        mu = float(s.mean())

    return OUResult(phi=phi, mu=mu, half_life=half_life, stationary=stationary)
