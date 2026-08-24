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

Two I(1) series are cointegrated if a weighted difference between the two is stationary.
They share a common trend once that is gone the rest mean reverts.

### Step 1: cointegrating regression

 Regress log WTI on log Brent:

    x_t = α + β·y_t + s_t                                            (1)

The gap between the actual price and the fitted one is the spread.

    ŝ_t = x_t - α̂ - β̂·y_t                                           (2)


The test isn't symmetric in finite samples, so regressing WTI on Brent and regressing Brent
on WTI do not give the same result. I chose WTI on Brent.
Brent is the global benchmark and WTI the regional grader. 

### Step 2: is the residual stationary?

Augmented Dickey–Fuller test on the spread:

    Δŝ_t = γ·ŝ_{t-1} + Σ δ_i·Δŝ_{t-i} + u_t                          (3)

    H₀: γ = 0   unit root, not cointegrated
    H₁: γ < 0   stationary, cointegrated

Normal ADF critical values do not apply as `ŝ_t` is not real data, it is a
residual and step 1 chose `β̂` to make it as small as possible (minimise variance). That
makes it look stationary even when it is not, so Engle-Granger uses stricter critical values.
`statsmodels.tsa.stattools.coint` applies them. Running `adfuller` on the residual
instead would make the result look more significant than it is(false relationship).

---

## 4. Timing of the reversion

The spread is modelled as an Ornstein-Uhlenbeck process:

    ds_t = θ(μ - s_t)dt + σ·dW_t                                     (4)

It is a random walk that drags towards `μ`. `μ` is the resting level, `θ` is the
strength of the spring, `σ` is the noise. Expected:

    E[s_t | s_0] = μ + (s_0 - μ)·e^(-θt)                             (5)

Deviations decay exponentially. The half-life is;

    τ = ln(2) / θ                                                    (6)

### Estimating Half life τ

Sampled daily (4) becomes a AR(1) which can be fitted with OLS:

    Δs_t = a + b·s_{t-1} + u_t                                       (7)
    φ = 1 + b                                                        (8) 

- `Δs_t` — Δ spread today
- `s_{t-1}` — where the spread was t-1
- `b` —  slope. How today's move depends on t-1
- `a` — intercept
- `u_t` — noise
- `φ` — amount of t-1 deviation is present in t
    
For the spread to revert `-2 < b < 0` same as `-1 < φ < 1`.

    φ^τ = 0.5
    τ·ln(φ) = ln(0.5)
    τ = ln(0.5) / ln(φ)                                              (9)

I use `τ = ln(0.5) / ln(φ)` instead of `τ = ln(0.5)/b` as it is used when `b` is small.

`γ` in (3) and `b` in (7) are the same coefficient. ADF test proves if the
spread reverts at all; the half-life asks how quickly.

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
