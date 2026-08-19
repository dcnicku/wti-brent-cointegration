"""
Checking the estimators against data where I know the right answer.

On real data I cannot tell whether the code is correct, because the true hedge
ratio and half-life are unobservable. So I build two series that are
cointegrated by construction, with a beta and a half-life I choose myself, and
check that the code recovers them. I also run a negative control: two unrelated
random walks, which must NOT come back as cointegrated.

    python tests/test_synthetic.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import stats as S
import backtest as B

RNG = np.random.default_rng(42)

TRUE_BETA = 0.98
TRUE_HALF_LIFE = 12.0


def make_cointegrated(n=1500, beta=TRUE_BETA, half_life=TRUE_HALF_LIFE,
                      sigma_s=0.02, sigma_y=0.015):
    """
    Build y as a random walk and x = alpha + beta*y + s, where s is a
    mean-reverting series with the half-life I asked for.

        phi = 0.5 ** (1 / half_life)      inverts half_life = ln(0.5)/ln(phi)
    """
    phi = 0.5 ** (1.0 / half_life)

    y = np.cumsum(RNG.normal(0, sigma_y, n)) + np.log(60.0)

    s = np.zeros(n)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + RNG.normal(0, sigma_s * np.sqrt(1 - phi ** 2))

    x = 0.05 + beta * y + s
    idx = pd.bdate_range("2016-01-01", periods=n)
    return pd.Series(x, idx, name="x"), pd.Series(y, idx, name="y"), phi


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{'   ' + detail if detail else ''}")
    return bool(ok)


def close(label, got, want, tol):
    return check(label, abs(got - want) < tol, f"got={got:.4f} want={want:.4f}")


def main():
    print("=" * 72)
    print("SYNTHETIC DATA CHECKS")
    print("=" * 72)

    x, y, phi = make_cointegrated()
    print(f"Built with beta={TRUE_BETA}, half-life={TRUE_HALF_LIFE}d, "
          f"phi={phi:.5f}, n={len(x)}")

    results = []

    # --- 1. both series should look I(1) ---
    print("\n[1] Order of integration")
    for name, ser in [("x", x), ("y", y)]:
        r = S.test_unit_root(ser, name)
        results.append(check(f"{name} is I(1)", r["is_I1"],
                             f"levels p={r['p_levels']:.3f} diffs p={r['p_diffs']:.3f}"))

    # --- 2. the test should find cointegration and recover beta ---
    print("\n[2] Engle-Granger")
    fit = S.engle_granger(x, y, "x", "y")
    results.append(check("finds cointegration", fit.pvalue < 0.05,
                         f"p={fit.pvalue:.4f}"))
    results.append(close("recovers beta", fit.beta, TRUE_BETA, 0.05))

    # --- 3. the half-life should come back close to what I put in ---
    print("\n[3] Half-life")
    ou = S.fit_ou(fit.spread)
    results.append(close("recovers half-life", ou.half_life, TRUE_HALF_LIFE, 3.0))
    results.append(close("recovers phi", ou.phi, phi, 0.03))

    # --- 4. negative control: unrelated random walks must fail the test ---
    print("\n[4] Negative control (two unrelated random walks)")
    a = pd.Series(np.cumsum(RNG.normal(0, 0.015, 1500)), x.index)
    b = pd.Series(np.cumsum(RNG.normal(0, 0.015, 1500)), x.index)
    nc = S.engle_granger(a, b, "a", "b")
    results.append(check("correctly finds NO cointegration", nc.pvalue > 0.05,
                         f"p={nc.pvalue:.4f}"))

    # --- 5. the z-score must not peek at future data ---
    print("\n[5] No lookahead in the z-score")
    # If the z-score only uses trailing data, then computing it on the first 800
    # observations must give the same answer at day 800 as computing it on all
    # 1500. If it used the full-sample mean, the two would differ.
    cut = 800
    z_full = B.rolling_zscore(fit.spread, 60)
    z_trunc = B.rolling_zscore(fit.spread.iloc[:cut], 60)
    same = np.isclose(z_full.iloc[cut - 1], z_trunc.iloc[-1], equal_nan=True)
    results.append(check("z-score uses only trailing data", same,
                         f"full={z_full.iloc[cut-1]:.6f} truncated={z_trunc.iloc[-1]:.6f}"))

    # --- 6. the stop-loss must actually stop ---
    print("\n[6] Stop-loss does not immediately re-enter")
    # z climbs past the stop and then sits between the entry and stop levels.
    # A naive implementation goes flat for one day and then re-enters, because
    # z is still above the entry threshold. It should stay flat instead.
    z_stop = pd.Series([0.0, 2.5, 3.0, 4.0] + [3.0] * 10)
    pos = B.generate_positions(z_stop, entry=2.0, exit_=0.5, stop=3.5)
    results.append(check("stays flat after being stopped out",
                         (pos.iloc[4:] == 0).all(),
                         f"positions after stop: {list(pos.iloc[4:8].astype(int))}"))

    # --- 7. the backtest should run and make money on genuinely reverting data ---
    print("\n[7] Backtest on the synthetic spread")
    df, m = B.run_backtest(x, y, fit.beta, ou.half_life, cost_bps=10.0)
    print(f"       trades={m['n_trades']}  gross Sharpe={m['sharpe_gross']:.2f}  "
          f"net Sharpe={m['sharpe_net']:.2f}  win rate={m['win_rate']:.0%}")
    results.append(check("produces a sensible number of trades", m["n_trades"] > 5))
    results.append(check("positive gross edge on data that really does revert",
                         m["sharpe_gross"] > 0))

    # --- 8. more cost must mean less profit ---
    print("\n[8] Cost sweep")
    sweep = B.cost_sweep(x, y, fit.beta, ou.half_life)
    print(sweep[["sharpe_gross", "sharpe_net", "total_return", "n_trades"]].to_string())
    results.append(check("net Sharpe falls as costs rise",
                         sweep["sharpe_net"].is_monotonic_decreasing))

    print("\n" + "=" * 72)
    print(f"RESULT: {sum(results)}/{len(results)} checks passed")
    print("=" * 72)
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
