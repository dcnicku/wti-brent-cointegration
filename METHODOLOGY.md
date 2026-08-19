# Methodology

The maths behind the WTI–Brent cointegration study, in the order the code runs.

---

## 1. Notation and why logs

Let `P^W_t` and `P^B_t` be the daily closing prices of front-month WTI and Brent
futures. I work with log prices:

    x_t = ln(P^W_t),    y_t = ln(P^B_t)

Three reasons for logs:

1. Log differences are approximately percentage returns, and they add up over
   time — the sum of daily log returns is the total log return.
2. The regression coefficient `β` becomes a proportional relationship rather
   than a dollar-for-dollar one. That matters here because crude traded between
   roughly $20 and $120 over the sample, so a fixed dollar hedge would mean
   something very different at each end.
3. Position sizing follows directly: a one-unit move in log price is a 1% move
   in price, so `β` is the relative size of the two legs.

---

## 2. Order of integration

Commodity prices are usually I(1) — they contain a unit root:

    x_t = x_{t-1} + e_t

A random walk has no fixed level to return to, so it cannot be traded on a
mean-reversion argument. I check this with an Augmented Dickey–Fuller test and
expect two things: it should **fail to reject** a unit root in levels, and
should **reject** it in first differences. That combination is what I(1) means,
and I(1) is the precondition for cointegration to be the right tool.

---

## 3. Engle–Granger cointegration

Two I(1) series are **cointegrated** if some linear combination of them is
stationary. In words: they share a common trend, and the gap between them is
temporary.

### Step 1 — the cointegrating regression

    x_t = α + β·y_t + s_t                                            (1)

by OLS. The fitted residual

    ŝ_t = x_t − α̂ − β̂·y_t                                           (2)

is the **spread**, and `β̂` is the **hedge ratio**: hold 1 unit of notional in
WTI against `β̂` units of Brent, in opposite directions.

**Direction.** The test is not symmetric in finite samples — regressing WTI on
Brent is not the same as regressing Brent on WTI. I fix the direction as WTI on
Brent, because Brent is the global benchmark grade and WTI is the regional one,
which makes Brent the more natural explanatory variable. The reverse regression
is reported alongside as a robustness check rather than used to pick a winner.

### Step 2 — is the residual stationary?

An ADF regression on the spread:

    Δŝ_t = γ·ŝ_{t-1} + Σ δ_i·Δŝ_{t-i} + u_t                          (3)

    H₀: γ = 0   unit root — NOT cointegrated
    H₁: γ < 0   stationary — cointegrated

**The critical values are not the ordinary ADF ones.** `ŝ_t` is not observed
data, it is a residual, and the step-1 regression has already chosen `β̂` to make
its variance as small as possible. That biases the test toward finding
stationarity, so the correct Engle–Granger critical values are stricter than
standard ADF values. `statsmodels.tsa.stattools.coint` applies them; running
`adfuller` on the residual directly would overstate the significance.

---

## 4. How fast does the spread close?

Model the spread as a mean-reverting process:

    ds_t = θ(μ − s_t)dt + σ·dW_t                                     (4)

The drift term is negative when `s_t` is above `μ` and positive when it is
below, so it pulls the spread back at a speed proportional to how far away it
is. Taking expectations gives

    E[s_t | s_0] = μ + (s_0 − μ)·e^(−θt)                             (5)

Deviations decay exponentially. Setting the remaining deviation to half its
starting size gives the **half-life**:

    τ = ln(2) / θ                                                    (6)

### Estimating it

Discretising (4) at one day gives an AR(1), which is just OLS:

    Δs_t = a + b·s_{t-1} + u_t                                       (7)
    φ = 1 + b                                                        (8)

`φ` is how much of today's deviation is still there tomorrow. Stationarity needs
`−2 < b < 0`. The half-life follows exactly:

    τ = ln(0.5) / ln(φ)                                              (9)

(The commonly quoted `τ = −ln(2)/b` is the small-`b` approximation of this; I
use the exact form.) The long-run level is `μ = −a/b`, the point where the drift
in (7) is zero.

**This is the same parameter the ADF test looks at.** The null `γ = 0` in (3) is
the statement `θ = 0` in (4), which sends `τ → ∞`. The cointegration test asks
whether the reversion speed differs from zero; the half-life asks by how much.

---

## 5. Trading rules

### Standardising the spread

    z_t = (s_t − μ̂_t) / σ̂_t                                          (10)

where `μ̂_t` and `σ̂_t` come from a **trailing** window of length `L` ending at
`t`. Using the full-sample mean and standard deviation would embed future
information in every past signal, which is the single most common way a backtest
like this produces a result that cannot be repeated.

I set `L = clip(5τ, 30, 120)` days — long enough to estimate the mean stably,
short enough to track slow drift in the equilibrium level.

### Rules

| Event | Condition | Action |
|---|---|---|
| Entry, short spread | `z_t > +2.0` | short WTI, long β·Brent |
| Entry, long spread | `z_t < −2.0` | long WTI, short β·Brent |
| Exit | `\|z_t\| < 0.5` | flat |
| Stop loss | `\|z_t\| > 3.5` | flat, and do not re-enter until `\|z_t\| < 2.0` |
| Time stop | held longer than `3τ` | flat |

**Why exit at 0.5σ rather than 0.** The spread crosses exactly zero rarely and
briefly. Waiting for it means long holding periods and exits that never fire.
0.5σ captures most of the convergence.

**Why the stop needs a cooldown.** If I go flat at `z = 3.6` and nothing else
changes, then `z` is still above the entry threshold of 2.0 the next day, so a
naive implementation re-enters the same losing trade immediately and the stop
does nothing at all. Requiring the spread to come back inside the entry band
first is what makes the stop meaningful.

**Why a time stop.** From (5), a 2σ deviation should decay to about 0.25σ after
three half-lives. If the position is still open past `3τ`, the reversion thesis
has failed by its own logic — most likely the relationship has changed — and I
close regardless of where `z` is.

**Execution lag.** A signal from the close of day `t` is traded at the close of
day `t+1`. Trading on the same close that generated the signal is a small
assumption with a large effect on results.

---

## 6. P&L and costs

From (2), the daily change in the spread is

    Δs_t = Δx_t − β̂·Δy_t                                            (11)

which in log prices is the return on being long 1 unit of WTI and short `β̂`
units of Brent. With position `p_t ∈ {−1, 0, +1}`:

    r_t = p_{t-1} · (Δx_t − β̂·Δy_t)                                 (12)

Note `p_{t-1}`, not `p_t` — that is the execution lag.

### Costs

Every change in position trades both legs. With one-way cost `c` (0.0010 for
10bps) on the notional of each leg:

    cost_t = |p_{t-1} − p_{t-2}| · (1 + β̂) · c                      (13)

The lagged positions appear here, not the raw signal, because the cost is
incurred when the trade actually happens.

A full round trip (flat → in → flat) therefore costs `2·(1 + β̂)·c`. With
`β̂ ≈ 1` and `c = 10bps` that is roughly 40bps per completed trade, which the
convergence has to cover before anything is left over.

    r^net_t = r_t − cost_t                                          (14)

---

## 7. Evaluation

| Period | Purpose |
|---|---|
| 2016-01-01 → 2021-12-31 | **Train.** Estimate `β̂`, run the cointegration test, estimate `τ`. |
| 2022-01-01 → present | **Test.** Trade using the training `β̂` only. |

The rolling z-score in the test period uses only trailing test-period data, so
nothing from the training period leaks in beyond the fixed `β̂`.

**Metrics.** Annualised Sharpe (`√252 · mean(r) / std(r)`, no risk-free rate
subtracted, since the spread is largely self-funding), total and annualised
return, maximum drawdown, trade count, win rate, and mean holding period.

**Cost sensitivity.** Results are reported at `c ∈ {0, 5, 10, 25}` bps one-way.
How fast the result decays from gross to net is the actual finding: a strategy
with a positive gross Sharpe that turns negative at 10bps has no economic
content.

**Benchmark.** Buy and hold WTI over the same window, to show whether the
market-neutral spread adds anything over simple directional exposure.

---

## 8. Limitations

1. **Roll artefacts.** Front-month continuous series contain jumps at contract
   expiry that are accounting artefacts, not price moves. Not corrected here, so
   some of the measured spread volatility is roll. Back-adjusted continuous
   contracts would fix this.
2. **No financing costs.** Borrow and margin on the short leg are not modelled.
3. **Close-to-close execution.** Assumes fills at the closing price, with no
   slippage or market impact.
4. **A single pair.** No diversification; everything depends on one
   relationship continuing to hold.
5. **A fixed hedge ratio.** `β̂` is estimated once on the training window. The
   rolling-β chart shows how much it actually moves, and letting it vary over
   time is the obvious extension.
