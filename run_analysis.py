"""
WTI-Brent cointegration study: the whole pipeline in one run.

    python run_analysis.py

Steps:
  1. Download and check the data
  2. Confirm both price series are I(1)
  3. Test for cointegration on the TRAINING window, and take the hedge ratio
  4. Estimate the half-life of the spread
  5. Trade the TEST window using the training hedge ratio
  6. Repeat under different transaction cost assumptions
  7. Compare against buying and holding WTI
  8. Decompose the P&L to check the result is not a contract-roll artefact

Everything it prints is also written to results/.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import data as D
import stats as S
import backtest as B
import diagnostics as G


pd.set_option("display.width", 100)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

START = "2016-01-01"
SPLIT = "2022-01-01"
COST_GRID = (0, 5, 10, 25)
RESULTS = "results"


def heading(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    os.makedirs(RESULTS, exist_ok=True)

    # ---- 1. data --------------------------------------------------------
    heading("1. DATA")
    prices = D.download_prices(start=START)
    print(D.summarise(prices))

    logp = D.log_prices(prices)
    train, test = D.split_train_test(logp, SPLIT)
    print(f"\nTrain: {train.index.min().date()} to {train.index.max().date()}  ({len(train)} days)")
    print(f"Test:  {test.index.min().date()} to {test.index.max().date()}  ({len(test)} days)")

    # ---- 2. order of integration ----------------------------------------
    heading("2. UNIT ROOT TESTS (training window)")
    print("Expect: levels p > 0.05 (has a unit root), differences p < 0.05 => I(1)\n")
    for col in ["WTI", "BRENT"]:
        r = S.test_unit_root(train[col], col)
        print(f"  {col:6s} levels p={r['p_levels']:.4f}   diffs p={r['p_diffs']:.4f}   I(1)={r['is_I1']}")

    # ---- 3. cointegration -----------------------------------------------
    heading("3. ENGLE-GRANGER COINTEGRATION (training window only)")
    # I regress WTI on Brent rather than the other way round because Brent is
    # the global benchmark grade and WTI is the regional one, so Brent is the
    # more natural explanatory variable. The test is not symmetric in finite
    # samples, so I report the reverse regression underneath as a check.
    fit = S.engle_granger(train["WTI"], train["BRENT"], "WTI", "BRENT")
    reverse = S.engle_granger(train["BRENT"], train["WTI"], "BRENT", "WTI")

    print(f"  WTI on BRENT:  beta={fit.beta:.4f}  stat={fit.stat:.3f}  p={fit.pvalue:.4f}")
    print(f"  BRENT on WTI:  beta={reverse.beta:.4f}  stat={reverse.stat:.3f}  p={reverse.pvalue:.4f}   (check)")
    print(f"\n  5% critical value: {fit.crit_5pct:.3f}")
    print(f"  Cointegrated at 5%: {fit.cointegrated}")
    print(f"  Hedge ratio beta = {fit.beta:.4f}")

    beta = fit.beta

    # ---- 4. half-life ----------------------------------------------------
    heading("4. HALF-LIFE OF THE SPREAD (training window)")
    ou = S.fit_ou(fit.spread)
    print(f"  phi = {ou.phi:.5f}   long-run level mu = {ou.mu:.5f}   stationary = {ou.stationary}")
    print(f"  Half-life = {ou.half_life:.1f} trading days")

    window = B.choose_window(ou.half_life)
    print(f"  Rolling z-score window = {window} days")
    print(f"  Time stop = {int(round(3 * ou.half_life))} days")

    if not (5 <= ou.half_life <= 60):
        print("\n  NOTE: half-life is outside the 5-60 day range this setup is built for.")

    # ---- 5. out-of-sample backtest ---------------------------------------
    heading("5. OUT-OF-SAMPLE BACKTEST (test window, training beta)")
    lx, ly = test["WTI"], test["BRENT"]

    df, metrics = B.run_backtest(lx, ly, beta, ou.half_life, cost_bps=10.0, window=window)
    print("  At 10bps one-way per leg:\n")
    for k, v in metrics.items():
        print(f"    {k:16s} {v:>12,.4f}")

    # ---- 6. cost sensitivity ---------------------------------------------
    heading("6. COST SENSITIVITY (one-way bps per leg)")
    scenarios = B.cost_scenarios(lx, ly, beta, ou.half_life,
                                 cost_grid=COST_GRID, window=window)
    sweep = pd.DataFrame([{"cost_bps": c, **m} for c, (_, m) in scenarios.items()]
                         ).set_index("cost_bps")
    print(sweep[["sharpe_gross", "sharpe_net", "total_return", "max_drawdown", "n_trades"]])

    # ---- 7. benchmark ----------------------------------------------------
    heading("7. BENCHMARK: BUY AND HOLD WTI (same window)")
    bh = B.buy_and_hold(test["WTI"])
    for k, v in bh.items():
        print(f"    {k:16s} {v:>12,.4f}")

    # ---- 8. is the edge real? ---------------------------------------------
    heading("8. WHERE DOES THE P&L ACTUALLY COME FROM?")
    roll = G.roll_attribution(df)
    conc = G.concentration(df, n=5)

    print(f"  Best single day was {conc['best_day_date'].date()}, worth "
          f"{conc['share_best_day']:.1%} of all gross P&L")
    print(f"  The best 5 days account for {conc['share_best_5_days']:.1%} of it\n")

    print("  Splitting by whether the day is the first trading day of a month,")
    print("  which is when the front-month futures contracts roll:\n")
    print(f"    month-start days are          {roll['share_of_days']:.1%} of the sample "
          f"({roll['n_month_start_days']} days)")
    print(f"    but earn                      {roll['share_of_pnl_month_start']:.1%} of gross P&L")
    print(f"    spread moves on those days    {roll['move_ratio']:.1f}x more than on other days")
    print(f"    of the 20 biggest spread moves, {roll['top20_moves_on_month_start']} are month-start "
          f"(chance would give about 1)")

    print("\n  Re-scored with every month-start return set to zero:\n")
    clean = G.performance_excluding_rolls(df)
    for k in ["sharpe_net", "total_return", "ann_return", "max_drawdown"]:
        print(f"    {k:16s} {clean[k]:>12,.4f}")
    print(f"\n  (for comparison, with roll days included: sharpe_net "
          f"{metrics['sharpe_net']:.4f}, total_return {metrics['total_return']:.4f})")

    # ---- charts and files -------------------------------------------------
    heading("SAVING RESULTS")
    if "--no-plots" in sys.argv:
        print("  (skipping charts: --no-plots)")
    else:
        import plots as P
        P.plot_spread(df, path=f"{RESULTS}/fig1_spread.png")
        P.plot_equity({c: (d["equity"], m["sharpe_net"]) for c, (d, m) in scenarios.items()},
                      path=f"{RESULTS}/fig2_equity.png")
        P.plot_rolling_beta(logp["WTI"], logp["BRENT"], path=f"{RESULTS}/fig3_beta.png")

    df.to_csv(f"{RESULTS}/backtest_daily.csv")
    sweep.to_csv(f"{RESULTS}/cost_sweep.csv")
    write_summary(sweep, bh, beta, ou, window, roll, conc, clean)

    for f in sorted(os.listdir(RESULTS)):
        print(f"  {RESULTS}/{f}")
    print("\nPaste the table in results/summary.md into the README.")


def write_summary(sweep, bh, beta, ou, window, roll, conc, clean):
    """Write the results table as markdown, ready to paste into the README."""
    lines = [
        "# Results",
        "",
        f"- Hedge ratio (train): **{beta:.3f}**",
        f"- Half-life: **{ou.half_life:.1f} trading days**",
        f"- Z-score window: **{window} days**",
        "",
        "## Out of sample, 2022 onwards",
        "",
        "| One-way cost | Gross Sharpe | Net Sharpe | Total return | Max drawdown | Trades |",
        "|---|---|---|---|---|---|",
    ]
    for c, row in sweep.iterrows():
        lines.append(
            f"| {c}bps | {row['sharpe_gross']:.2f} | {row['sharpe_net']:.2f} | "
            f"{row['total_return']:.1%} | {row['max_drawdown']:.1%} | {int(row['n_trades'])} |"
        )
    lines += [
        "",
        f"Buy and hold WTI over the same window: Sharpe {bh['sharpe']:.2f}, "
        f"total return {bh['total_return']:.1%}, max drawdown {bh['max_drawdown']:.1%}.",
        "",
        "## But the P&L is a roll artefact",
        "",
        f"- Month-start days (contract roll) are **{roll['share_of_days']:.1%}** of the sample "
        f"but earn **{roll['share_of_pnl_month_start']:.1%}** of gross P&L",
        f"- The spread moves **{roll['move_ratio']:.1f}x** more on those days",
        f"- **{roll['top20_moves_on_month_start']} of the 20** biggest spread moves are month-start "
        "(chance would give about 1)",
        f"- The single best day ({conc['best_day_date'].date()}) is "
        f"**{conc['share_best_day']:.1%}** of all gross P&L; the best 5 days are "
        f"**{conc['share_best_5_days']:.1%}**",
        "",
        f"With month-start returns zeroed out, net Sharpe falls from "
        f"**{sweep.loc[10, 'sharpe_net']:.2f}** to **{clean['sharpe_net']:.2f}** and total return "
        f"from **{sweep.loc[10, 'total_return']:.1%}** to **{clean['total_return']:.1%}**.",
        "",
    ]

    with open(f"{RESULTS}/summary.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
