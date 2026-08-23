# Methodology

The maths behind the WTI-Brent cointegration model.

---

## 1. Notation

`P^W_t` and `P^B_t` are daily closing prices of front month WTI and Brent futures.

    x_t = ln(P^W_t),    y_t = ln(P^B_t)

Log are used as they show differences are percentage returns and `β` comes out as a
proportional relationship so it tells how much Brent to hold against a unit of WTI.

---

## 2. Unit root

Commodity prices are usually I(1) (have a unit root):

    x_t = x_{t-1} + e_t

A random walk wanders, so there is no mean reversion trade
Augmented Dickey–Fuller test is used on both series to test for unit root. 
That pair of results need to be I(1); no rejection in log prices but rejection in log returns
Unit root has to be confirmed before cointegration.

---

## 3. Engle-Granger cointegration

Two I(1) series are cointegrated if some linear combination of them is stationary.
They share a common trend and the gap between them is temporary.

### Step 1: the cointegrating regression

    x_t = α + β·y_t + s_t                                            (1)

by OLS. The fitted residual

    ŝ_t = x_t - α̂ - β̂·y_t                                           (2)

is the spread. `β̂` is the hedge ratio: 1 unit of notional in WTI against `β̂` units
of Brent, opposite directions.

The test isn't symmetric in finite samples, so regressing WTI on Brent gives a
different answer from the reverse. I fixed it as WTI on Brent since Brent is the
global benchmark and WTI the regional grade. The reverse regression is reported as a
robustness check, not as a second candidate.

### Step 2: is the residual stationary?

ADF on the spread:

    Δŝ_t = γ·ŝ_{t-1} + Σ δ_i·Δŝ_{t-i} + u_t                          (3)

    H₀: γ = 0   unit root, NOT cointegrated
    H₁: γ < 0   stationary, cointegrated

The critical values are not the ordinary ADF ones. `ŝ_t` isn't observed data, it's a
residual, and step 1 has already picked `β̂` to minimise its variance. That tilts the
test toward stationarity whether or not it's there, so the Engle-Granger critical
values are stricter than the standard ADF table. `statsmodels.tsa.stattools.coint`
uses the right ones; `adfuller` on the residual would overstate significance.

---

## 4. Speed of reversion

Model the spread as an Ornstein-Uhlenbeck process:

    ds_t = θ(μ - s_t)dt + σ·dW_t                                     (4)

Drift is negative above `μ` and positive below, pulling back at a rate proportional
to the distance. Expectations:

    E[s_t | s_0] = μ + (s_0 - μ)·e^(-θt)                             (5)

Deviations decay exponentially, so the half-life is

    τ = ln(2) / θ                                                    (6)

### Estimating it

Discretise (4) daily and it's an AR(1), so OLS:

    Δs_t = a + b·s_{t-1} + u_t                                       (7)
    φ = 1 + b                                                        (8)

`φ` is the fraction of today's deviation still there tomorrow. Stationarity needs
`-2 < b < 0`. Then

    τ = ln(0.5) / ln(φ)                                              (9)

I use this rather than `τ = -ln(2)/b`, which is the small-`b` approximation of the
same thing. Long-run level is `μ = -a/b`, where the drift in (7) is zero.

Same parameter the ADF test looks at: `γ = 0` in (3) is `θ = 0` in (4), which sends
`τ` to infinity. The test asks whether reversion speed differs from zero. The
half-life asks by how much.

---

## 5. Trading rules

### Standardising the spread

    z_t = (s_t - μ̂_t) / σ̂_t                                          (10)

`μ̂_t` and `σ̂_t` come from a trailing window of length `L` ending at `t`, not the full
sample. Full-sample moments put future information into every past signal.

`L = clip(5τ, 30, 120)` days. Long enough for a stable mean, short enough to follow
drift in the equilibrium level.

### Rules

| Event | Condition | Action |
|---|---|---|
| Entry, short spread | `z_t > +2.0` | short WTI, long β·Brent |
| Entry, long spread | `z_t < -2.0` | long WTI, short β·Brent |
| Exit | `\|z_t\| < 0.5` | flat |
| Stop loss | `\|z_t\| > 3.5` | flat, no re-entry until `\|z_t\| < 2.0` |
| Time stop | held longer than `3τ` | flat |

Exit at 0.5σ rather than 0 because the spread crosses exactly zero rarely and
briefly. Waiting for it gives long holding periods and exits that sometimes never
fire. 0.5σ picks up most of the convergence.

The cooldown on the stop is necessary. Going flat at `z = 3.6` with nothing else
changed leaves `z` above the 2.0 entry threshold the next day, so a naive
implementation re-enters the same losing trade and the stop does nothing. The spread
has to come back inside the entry band first.

Time stop follows from (5): a 2σ deviation should be around 0.25σ after three
half-lives. Still open past `3τ` and the reversion thesis has failed on its own
terms, most likely because the relationship has shifted, so I close it wherever `z`
is.

Execution lag: a signal from the close of day `t` is traded at the close of day
`t+1`.

---

## 6. P&L and costs

From (2), the daily change in the spread is

    Δs_t = Δx_t - β̂·Δy_t                                            (11)

which in log terms is the return on long 1 unit of WTI, short `β̂` units of Brent.
With position `p_t ∈ {-1, 0, +1}`:

    r_t = p_{t-1} · (Δx_t - β̂·Δy_t)                                 (12)

`p_{t-1}` rather than `p_t`, per the execution lag.

### Costs

Any change in position trades both legs. One-way cost `c` (0.0010 for 10bps) on each
leg's notional:

    cost_t = |p_{t-1} - p_{t-2}| · (1 + β̂) · c                      (13)

Lagged positions again, since the cost lands when the trade happens rather than when
the signal fires.

A full round trip is `2·(1 + β̂)·c`. At `β̂ ≈ 1` and 10bps that's about 40bps per
completed trade, which the convergence has to clear.

    r^net_t = r_t - cost_t                                          (14)

---

## 7. Evaluation

| Period | Purpose |
|---|---|
| 2016-01-01 to 2021-12-31 | Train. Estimate `β̂`, run the cointegration test, estimate `τ`. |
| 2022-01-01 to present | Test. Trade on the training `β̂` only. |

The rolling z-score in the test period uses trailing test-period data only. The only
thing carried across is the fixed `β̂`.

Metrics: annualised Sharpe (`√252 · mean(r) / std(r)`, no risk-free rate subtracted
since the spread is largely self-funding), total and annualised return, max drawdown,
number of trades, win rate, mean holding period.

Cost sensitivity at `c ∈ {0, 5, 10, 25}` bps one-way. How fast the result decays from
gross to net matters as much as the gross number.

Benchmark is buy-and-hold WTI over the same window, to check whether the
market-neutral version adds anything over directional exposure.
