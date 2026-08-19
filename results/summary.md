# Results

- Hedge ratio (train): **1.020**
- Half-life: **13.5 trading days**
- Z-score window: **67 days**

## Out of sample, 2022 onwards

| One-way cost | Gross Sharpe | Net Sharpe | Total return | Max drawdown | Trades |
|---|---|---|---|---|---|
| 0bps | 1.13 | 1.13 | 67.1% | -7.8% | 26 |
| 5bps | 1.13 | 1.03 | 58.6% | -7.9% | 26 |
| 10bps | 1.13 | 0.92 | 50.5% | -8.0% | 26 |
| 25bps | 1.13 | 0.59 | 28.6% | -8.3% | 26 |

Buy and hold WTI over the same window: Sharpe 0.26, total return 11.0%, max drawdown -55.3%.

## But the P&L is a roll artefact

- Month-start days (contract roll) are **4.8%** of the sample but earn **94.0%** of gross P&L
- The spread moves **4.8x** more on those days
- **8 of the 20** biggest spread moves are month-start (chance would give about 1)
- The single best day (2026-04-01) is **27.5%** of all gross P&L; the best 5 days are **64.1%**

With month-start returns zeroed out, net Sharpe falls from **0.92** to **-0.21** and total return from **50.5%** to **-5.8%**.
