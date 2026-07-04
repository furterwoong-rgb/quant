#!/usr/bin/env python
"""
reversal_signal.py — Stage 3 (signal construction) for Candidate-1: vol-conditioned
short-horizon reversal on QQQ.

Builds the causal features on 1-min bars, fits the OU mean-reversion to set the horizon,
and runs the in-sample (January) predictive regression that characterizes the signal.
This is SIGNAL CONSTRUCTION + IN-SAMPLE CHARACTERIZATION. Frozen out-of-sample validation
on May (DSR / PBO / Jensen-alpha vs B&H) is Stage 4 and lives in validation/.

Gamma is intentionally OFF here (historical OI unavailable — see probe). The gamma gate
is a separable forward-test overlay. What we validate historically is the z×RV core.

Features (all causal, reset per session — flat overnight):
    p_t   = ln(VWAP_t)                                   # VWAP avoids bid-ask bounce
    mu_t  = EWMA(p, halflife=h_mu)                        # adaptive reference, seeded at the open
    d_t   = p_t - mu_t                                    # deviation (the OU variable)
    sig_t = sqrt(EWMA(d^2, halflife=h_sigma))             # deviation scale
    z_t   = d_t / sig_t                                   # standardized stretch (reversal: theta<0)
    rv_t  = sqrt(EWMA(r1^2, halflife=rv_halflife))        # local realized vol (Nagel conditioner)

Model:  r_{t->t+h} = a + theta0*z + theta1*(z * rv_z) + e     (reversal => theta0 < 0,
        Nagel => theta1 < 0 : reversal stronger in higher vol)

Run
---
    PY=/opt/miniconda3/envs/quant-env/bin/python
    $PY analysis/reversal_signal.py                              # January in-sample fit
    $PY analysis/reversal_signal.py --dates 2026-05-01:2026-05-29  # peek (do NOT tune on this)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

CACHE_DIR_DEFAULT = "/Users/woongyeol/quant_trading/logs/tick_cache"
OUT_DIR_DEFAULT = "/Users/woongyeol/quant_trading/analysis/output"


@dataclass
class Params:
    h_mu: float = 20.0          # reference EWMA half-life (minutes)
    h_sigma: float = 60.0       # deviation-scale EWMA half-life (minutes)
    rv_halflife: float = 30.0   # realized-vol EWMA half-life (minutes)
    burn_in: int = 20           # drop first N bars/session (EWMA warm-up)
    price_col: str = "vwap"
    horizon: int | None = None  # forward horizon (min); None => set from OU half-life


# ------------------------------------------------------------------------ estimators
def ols_hac(y, X, L):
    """OLS with Newey-West (Bartlett) HAC covariance. Returns (beta, tstat, se)."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    Xe = X * (y - X @ beta)[:, None]
    S = Xe.T @ Xe
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        G = Xe[l:].T @ Xe[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return beta, beta / se, se


# --------------------------------------------------------------------------- features
def available_dates(cache_dir, symbol="QQQ"):
    dates = []
    for p in glob.glob(os.path.join(cache_dir, f"{symbol}_*_bars.pkl")):
        m = re.search(rf"{symbol}_(\d{{4}}-\d{{2}}-\d{{2}})_bars\.pkl", os.path.basename(p))
        if m:
            dates.append(m.group(1))
    return sorted(dates)


def session_features(bars: pd.DataFrame, p: Params) -> pd.DataFrame:
    px = bars[p.price_col].where(bars[p.price_col] > 0).fillna(bars["close"])
    lp = np.log(px)
    mu = lp.ewm(halflife=p.h_mu, adjust=False).mean()              # seeded at open bar
    d = lp - mu
    sig = np.sqrt((d ** 2).ewm(halflife=p.h_sigma, adjust=False).mean())
    z = d / sig.replace(0, np.nan)
    r1 = lp.diff()
    rv = np.sqrt((r1 ** 2).ewm(halflife=p.rv_halflife, adjust=False).mean())
    out = pd.DataFrame({"lp": lp, "d": d, "z": z, "rv": rv, "r1": r1}, index=bars.index)
    return out.iloc[p.burn_in:]


def build_features(dates, cache_dir, p: Params, symbol="QQQ") -> pd.DataFrame:
    frames = []
    for dt in dates:
        f = os.path.join(cache_dir, f"{symbol}_{dt}_bars.pkl")
        if not os.path.exists(f):
            continue
        b = pd.read_pickle(f).between_time("09:30", "16:00")
        if b.empty:
            continue
        fe = session_features(b, p)
        fe["date"] = dt
        frames.append(fe)
    if not frames:
        raise SystemExit("No bar data for requested dates.")
    return pd.concat(frames)


def add_forward(feat: pd.DataFrame, h: int) -> pd.DataFrame:
    feat = feat.copy()
    # within-session forward return p_{t+h} - p_t (no cross-day leakage)
    feat["r_fwd"] = feat.groupby("date")["lp"].transform(lambda s: s.shift(-h) - s)
    return feat


# ------------------------------------------------------------------------------- fits
def estimate_ou(feat: pd.DataFrame) -> dict:
    """Pooled within-day AR(1) on d_t: d_{t+1} = c + phi*d_t + e. half-life sets horizon."""
    d0, d1 = [], []
    for _, g in feat.groupby("date"):
        dd = g["d"].to_numpy()
        d0.append(dd[:-1]); d1.append(dd[1:])
    d0 = np.concatenate(d0); d1 = np.concatenate(d1)
    X = np.column_stack([np.ones_like(d0), d0])
    beta, t, _ = ols_hac(d1, X, L=5)
    phi = float(beta[1])
    kappa = -np.log(phi) if 0 < phi < 1 else np.nan
    hl = float(np.log(2) / kappa) if kappa == kappa else float("inf")
    return dict(phi=phi, half_life_min=hl, t_phi=float(t[1]), n_pairs=int(len(d0)))


def predictive(feat: pd.DataFrame, h: int) -> dict:
    d = feat.dropna(subset=["z", "r_fwd", "rv"]).copy()
    z = d["z"].to_numpy()
    r = d["r_fwd"].to_numpy()
    rv_z = ((d["rv"] - d["rv"].mean()) / d["rv"].std()).to_numpy()
    X = np.column_stack([np.ones_like(z), z, z * rv_z])
    L = int(1.5 * h) + 1
    beta, t, se = ols_hac(r, X, L)
    ic = float(np.corrcoef(z, r)[0, 1])
    ic_s = float(pd.Series(z).corr(pd.Series(r), method="spearman"))
    n_eff = len(z) / h
    # quantile monotonicity (reversal => mean r_fwd decreasing in z)
    qb = pd.qcut(d["z"], 10, labels=False, duplicates="drop")
    decile = d.groupby(qb)["r_fwd"].mean().mul(1e4).round(3).to_dict()  # bps
    return dict(
        n=int(len(z)), n_eff=float(n_eff), horizon=h,
        theta0=float(beta[1]), t_theta0=float(t[1]),
        theta1_rv=float(beta[2]), t_theta1=float(t[2]),
        ic_pearson=ic, ic_spearman=ic_s, ic_tstat=float(ic * np.sqrt(n_eff)),
        decile_rfwd_bps=decile,
    )


def gross_sharpe_nonoverlap(feat: pd.DataFrame, h: int, cost_bps: float) -> dict:
    """IS-only diagnostic: non-overlapping h-spaced bets, position = -sign(z)."""
    rets = []
    for _, g in feat.dropna(subset=["z", "r_fwd"]).groupby("date"):
        gg = g.iloc[::h]
        rets.append(-np.sign(gg["z"].to_numpy()) * gg["r_fwd"].to_numpy())
    pnl = np.concatenate(rets)
    net = pnl - cost_bps * 1e-4
    bets_per_day = np.mean([len(r) for r in rets])
    ann = np.sqrt(252 * max(bets_per_day, 1))
    g_sharpe = float(pnl.mean() / pnl.std() * ann) if pnl.std() > 0 else 0.0
    n_sharpe = float(net.mean() / net.std() * ann) if net.std() > 0 else 0.0
    return dict(gross_sharpe=g_sharpe, net_sharpe=n_sharpe,
                mean_gross_bps=float(pnl.mean() * 1e4), n_bets=int(len(pnl)),
                bets_per_day=float(bets_per_day))


# -------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", default="2026-01-01:2026-01-31", help="'YYYY-MM-DD:YYYY-MM-DD' or 'all'")
    ap.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--cost-bps", type=float, default=0.16, help="round-trip cost for net diagnostic")
    ap.add_argument("--h-mu", type=float, default=20.0)
    ap.add_argument("--h-sigma", type=float, default=60.0)
    ap.add_argument("--horizon", type=int, default=0, help="0 => from OU half-life")
    args = ap.parse_args()

    p = Params(h_mu=args.h_mu, h_sigma=args.h_sigma)
    alld = available_dates(args.cache_dir)
    if args.dates == "all":
        dates = alld
    else:
        lo, hi = args.dates.split(":")
        dates = [d for d in alld if lo <= d <= hi]
    if not dates:
        raise SystemExit(f"No dates match {args.dates}. Available: {alld[:2]}..{alld[-2:]}")

    feat = build_features(dates, args.cache_dir, p)
    ou = estimate_ou(feat)
    h = args.horizon or max(2, min(30, round(ou["half_life_min"])))
    feat = add_forward(feat, h)

    grid = {}
    for hh in sorted({max(2, h - 3), max(2, h - 1), h, h + 2, h + 5}):
        grid[hh] = round(predictive(add_forward(feat, hh), hh)["ic_pearson"], 5)

    reg = predictive(feat, h)
    diag = gross_sharpe_nonoverlap(feat, h, args.cost_bps)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{dates[0]}_{dates[-1]}"
    feat.to_pickle(os.path.join(args.out_dir, f"features_{tag}.pkl"))
    report = dict(params=asdict(p), dates=[dates[0], dates[-1], len(dates)],
                  horizon=h, ou=ou, regression=reg, ic_grid=grid, diagnostic=diag)
    with open(os.path.join(args.out_dir, f"stage3_report_{tag}.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n=== Stage 3 — Candidate-1 reversal  ({len(dates)} days {dates[0]}..{dates[-1]}) ===")
    print(f"obs={reg['n']:,}  N_eff≈{reg['n_eff']:,.0f}")
    print(f"\nOU:  phi={ou['phi']:.4f}  half-life={ou['half_life_min']:.1f} min  "
          f"(t_phi={ou['t_phi']:.1f})  ->  horizon h={h} min")
    print(f"\nPredictive  r_fwd = a + theta0*z + theta1*(z*rv_z):")
    print(f"  theta0 (reversal)     = {reg['theta0']*1e4:+.3f} bps/z   HAC t = {reg['t_theta0']:+.2f}")
    print(f"  theta1 (vol interact) = {reg['theta1_rv']*1e4:+.3f} bps/z   HAC t = {reg['t_theta1']:+.2f}")
    print(f"  IC(pearson)={reg['ic_pearson']:+.4f}  IC(spearman)={reg['ic_spearman']:+.4f}  "
          f"IC·√N_eff={reg['ic_tstat']:+.2f}")
    print(f"\nIC vs horizon (min): " + "  ".join(f"{k}:{v:+.4f}" for k, v in grid.items()))
    print(f"\nMean r_fwd by z-decile (bps, reversal => decreasing):")
    print("  " + "  ".join(f"{k}:{v:+.2f}" for k, v in reg["decile_rfwd_bps"].items()))
    print(f"\nIS diagnostic (non-overlapping, position=-sign(z)):")
    print(f"  mean gross={diag['mean_gross_bps']:+.3f} bps/bet  gross Sharpe={diag['gross_sharpe']:.2f}  "
          f"net@{args.cost_bps}bps Sharpe={diag['net_sharpe']:.2f}  (bets/day≈{diag['bets_per_day']:.0f})")
    print(f"\n[!] In-sample characterization only. Stage-4 freezes these params and tests May OOS.")
    print(f"saved: {args.out_dir}/features_{tag}.pkl  +  stage3_report_{tag}.json")


if __name__ == "__main__":
    main()
