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

- `Δs_t` -  Δ spread today
- `s_{t-1}` -  where the spread was t-1
- `b` -   slope. How today's move depends on t-1
- `a` - intercept
- `u_t` - noise
- `φ` -  amount of t-1 deviation is present in t
    
For the spread to revert `-2 < b < 0` same as `-1 < φ < 1`.

    φ^τ = 0.5
    τ·ln(φ) = ln(0.5)
    τ = ln(0.5) / ln(φ)                                              (9)

I use `τ = ln(0.5) / ln(φ)` instead of `τ = ln(0.5)/b` as it is used when `b` is small.

`γ` in (3) and `b` in (7) are the same coefficient. ADF test proves if the
spread reverts at all; the half-life asks how quickly.

---

## 5. Strategy

### z-score

    z_t = (s_t - μ̂_t) / σ̂_t                                          (10)

- `z_t` -  how many standard deviations the spread sits from the mean
- `s_t` -  t spread
- `μ̂_t` -  average spread over the last `L` days
- `σ̂_t` -  standard deviation of the spread over `L` days
-  `L`  -  amount of trailing days. 5τ, bounded to [30, 120] days

### Rules

| Event | Condition | Action |
|---|---|---|
| Enter short | `z_t > +2.0` | short WTI long `β` Brent |
| Enter long | `z_t < -2.0` | long WTI short `β` Brent |
| Exit | `\|z_t\| < 0.5` |
| Stop loss | `\|z_t\| > 3.5` | exit, do not enter again until `\|z_t\| < 2.0` |
| Time stop | held past `3τ` | exit, reversion failed within expected period |

Exit at 0.5σ rather than 0 because the spread almost never sits exactly at zero.
Waiting causes long holds and exits that may never close.

The cooldown on the stop is necessary. Exiting at eg. `z = 3.7`
leaves `z` above 2.0 entry level the next day. `z` has to go under 2.0 before re-entry.
So the same losing trade is not re-entered.

Time stop: after three half-lives a 2σ gap should be around 0.25σ. Still in the
trade after`3τ` means the reversion is not happening therefore it is closed.

Execution lag: signal from close of day `t` is traded at close of day `t+1`.

---

## 6. Returns after costs

The daily Δ spread:

    Δs_t = Δx_t - β̂·Δy_t                                            (11)

Also the daily return on long 1 unit of WTI against `β̂` units of Brent.
The position `p_t` is +1 (long), -1 (short ) or 0 (not in a trade):

    r_t = p_{t-1} · (Δx_t - β̂·Δy_t)                                 (12)


- `r_t` - today's return
- `p_{t-1}` -  yesterday's position
- `(Δx_t - β̂·Δy_t)` - Δ  spread today

### Costs

A position change trades both WTI and Brent. One-way cost `c` (10bps)

    cost_t = |p_{t-1} - p_{t-2}| · (1 + β̂) · c                      (13)

The executed positions are lagged so cost occur when
the trade happens not when the signal appears.

A full round trip is `2·(1 + β̂)·c`. `β̂ ≈ 1` `c = 10bps`.
Approximately 40bps cost when any trade is opened and closed.
Therefore convergence has to beat 40bps before profit.

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
