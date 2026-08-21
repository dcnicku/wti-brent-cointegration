# Does the WTI–Brent spread mean revert enough to trade?

An econometrics project on 2 crude oil benchmarks. I test whether the
gap between WTI and Brent is cointegrated, estimate how quickly it closes, and
then check whether a mean reversion rule built on the estimate is profitable
after transaction costs.

**Headline result: the strategy appears to work, and doesn't.** The backtest
returns a net Sharpe of 0.92 out of sample after 10bps costs, against 0.26 for
buying and holding WTI. That number is not real. Decomposing the P&L shows that
**94% of it is earned on the 4.8% of days when the futures contracts roll** —
days on which the price series jumps for accounting reasons and no trader could
have captured the move. Strip those days out and the Sharpe is **−0.21**.

The finding of this project is the artefact, not the Sharpe.

---

## The question

WTI and Brent are two grades of crude oil. They differ in sulphur content,
density and where they are delivered, but they are close substitutes, so their
prices are driven by the same global supply and demand. That suggests the gap
between them should be temporary rather than permanent: when it widens without
anything real changing, it should close again.

The trade that follows is market-neutral. Buy the cheap grade, sell the
expensive one, in the proportion given by the hedge ratio. If crude rallies,
both legs move together and largely cancel — the position is a bet on the gap,
not on the oil price.

**Why cointegration and not correlation.** Correlation is measured on returns
and says nothing about whether the *level* gap closes. Two series can have
almost perfectly correlated returns and still drift apart forever if their
trends differ. Cointegration is a statement about levels sharing a common trend,
and that is the property the trade actually depends on.

---

## Method

| Step | What I do |
|---|---|
| Transform | Work in log prices, so differences are percentage returns |
| Cointegration | Engle–Granger two-step, WTI regressed on Brent |
| Hedge ratio | The OLS slope β, estimated on the training window only |
| Reversion speed | AR(1) fit on the spread, giving a half-life τ |
| Signal | Rolling z-score of the spread, window = 5τ capped to [30, 120] days |
| Rules | Enter at \|z\|>2, exit at \|z\|<0.5, stop at \|z\|>3.5, time stop at 3τ |
| Costs | 0 / 5 / 10 / 25 bps one-way, charged on both legs every time I trade |
| Split | Estimate on 2016–2021, trade 2022 onwards, one-day execution lag |

The maths is written out in [METHODOLOGY.md](METHODOLOGY.md).

---

## Results

Estimated on 2016–2021, traded on 2022-01-03 to 2026-08-18 (1,162 days).

- Hedge ratio β = **1.020** (train window, held fixed)
- Half-life = **13.5 trading days** → z-score window 67 days, time stop 40 days

| One-way cost | Gross Sharpe | Net Sharpe | Total return | Max drawdown | Trades |
|---|---|---|---|---|---|
| 0bps  | 1.13 | 1.13 | 67.1% | −7.8% | 26 |
| 5bps  | 1.13 | 1.03 | 58.6% | −7.9% | 26 |
| 10bps | 1.13 | 0.92 | 50.5% | −8.0% | 26 |
| 25bps | 1.13 | 0.59 | 28.6% | −8.3% | 26 |

Buy and hold WTI over the identical window: Sharpe 0.26, total return 11.0%,
max drawdown −55.3%.

Taken at face value this looks good: the edge decays roughly linearly in cost
and is still positive at 25bps, the strategy is in the market only 29% of the
time, and its worst drawdown is 8% against 55% for holding the outright. The win
rate is 92% on 26 trades.

That win rate is what made me check further.

![equity curves](results/fig2_equity.png)

Notice the step in early 2026. One day carries a large share of the whole
result, and the curve is nearly flat around it.

---

## Why the result is not real

**Where the money comes from.** The single best day, 2026-04-01, is **27.5%** of
all gross P&L. The best five days are **64.1%**. A strategy with a genuine edge
spreads its returns across the sample; one that harvests a recurring artefact
concentrates them.

**Those days are contract roll dates.** These are front-month continuous
futures, so when a contract expires the series jumps to the next one. That jump
is bookkeeping, not a price move. Splitting the sample on whether a day is the
first trading day of a month:

| | Month-start days | All other days |
|---|---|---|
| Share of sample | 4.8% (56 days) | 95.2% (1,105 days) |
| **Share of gross P&L** | **94.0%** | **6.0%** |
| Mean daily \|spread move\| | 0.0181 | 0.0038 |

The spread moves **4.8× more** on roll days, and **8 of the 20 largest spread
moves** in the sample fall on them, where chance would give about 1.

**What is left without them.** Setting every month-start return to zero — the
pessimistic reading, that none of the jump was capturable:

| | With roll days | Roll days zeroed |
|---|---|---|
| Net Sharpe (10bps) | 0.92 | **−0.21** |
| Total return | 50.5% | **−5.8%** |
| Annualised return | 9.3% | **−1.3%** |

So the entire result is the artefact. On the days you could actually trade, the
strategy loses money slowly.

This is what the mechanism looks like in the data — the entry on 2026-03-31 and
the "profit" the next day:

| Date | spread | z | position | gross return |
|---|---|---|---|---|
| 2026-03-31 | −0.2489 | −3.36 | enters long | — |
| 2026-04-01 | −0.1014 | **+2.25** | long | **+14.8%** |

The spread moved 15% and the z-score travelled from −3.4 to +2.3 in a single
session. Nothing in the oil market did that. The contract rolled.

`src/diagnostics.py` runs this decomposition; it is step 8 of the pipeline.

### Two further caveats

**WTI does not cleanly pass the I(1) precondition.** On the training window the
ADF test rejects a unit root in WTI levels at 5% (p = 0.026) while Brent does
not (p = 0.067). Cointegration analysis assumes both series are I(1), so the
setup is not cleanly satisfied to begin with.

**The cointegration evidence straddles the 5% line.** WTI on Brent gives
p = 0.023; the reverse regression gives p = 0.052. The conclusion depends on
which variable sits on the left-hand side.

### What would settle it

Rerun on **back-adjusted continuous contracts**, where the roll is spliced out
rather than left in as a jump. My prediction is that the remaining edge is
somewhere near the −0.21 figure above, i.e. nothing. The cheap alternative is to
simply refuse to trade on roll dates, which the diagnostic above already
simulates.

---

## Two ways this could have gone wrong

Both of these make a backtest look good for reasons that have nothing to do with
the strategy, so I handled them explicitly.

**Using information I would not have had at the time.** The z-score is computed
on a strictly trailing window. Standardising against the full-sample mean and
standard deviation would put 2024 information into a signal dated 2022, which is
the most common way this kind of backtest produces a fake result. Check 5 in the
test file verifies this: the z-score at day 800 is the same whether it is
computed on 800 days of data or on all 1,500.

**Trading at a price I could not have got.** A signal from the close of day *t*
is traded at the close of day *t+1*. Positions are lagged one day before being
applied to returns, and costs are charged on the day the position actually
changes rather than the day the signal fires.

The hedge ratio β is estimated on the training window and then held fixed. The
test period never influences any parameter.

---

## Checking the code is right

The problem with a project like this is that on real data I have no idea what
the true hedge ratio or half-life is, so a broken estimator would look exactly
like a real result. `tests/test_synthetic.py` builds two series that are
cointegrated *by construction*, with a β and a half-life I picked, and checks
the code recovers them. It also runs a negative control on two unrelated random
walks, which must fail the cointegration test, and a check that the stop-loss
does not immediately re-enter the trade it just exited.

```bash
python tests/test_synthetic.py
```

---

## Running it

```bash
pip install -r requirements.txt
```

```bash
python tests/test_synthetic.py
```

```bash
python run_analysis.py
```

Charts, the daily backtest, the cost sweep and a summary table are written to
`results/`.

---

## Layout

```
├── README.md
├── METHODOLOGY.md          the maths written out
├── run_analysis.py         runs everything end to end
├── src/
│   ├── data.py             download, align, split
│   ├── stats.py            unit root tests, cointegration, half-life
│   ├── backtest.py         signal, P&L, costs, metrics
│   ├── diagnostics.py      roll attribution -- the part that kills the result
│   └── plots.py            the three charts
├── tests/
│   └── test_synthetic.py   checks the estimators on data with a known answer
└── results/                output (created on first run)
```

---

## What this does not do

1. **Roll artefacts.** Not corrected, and as shown above they account for
   essentially the whole result. Back-adjusted continuous contracts would fix
   this and are the single most important next step.
2. **No financing costs.** Borrow and margin on the short leg are not modelled.
3. **Close-to-close fills.** No slippage, no market impact.
4. **One pair.** Everything rests on a single relationship holding up.
5. **A fixed hedge ratio.** β is estimated once. `results/fig3_beta.png` shows
   how much it actually moves over the sample, which is the evidence for or
   against that assumption.

## Where I would take it next

- Rerun on back-adjusted continuous contracts, to see whether anything survives
  once the roll jumps are spliced out
- Let β drift over time instead of fixing it
- Allow different reversion speeds in calm and stressed periods
- Extend to other energy spreads (heating oil, crack spreads)

---

*Written in Python with AI assistance on the implementation. The question,
the method, the validation design and the interpretation are mine.*
