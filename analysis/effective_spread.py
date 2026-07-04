#!/usr/bin/env python
"""
effective_spread.py — Liquidity / transaction-cost measurement (the "cost wall").

Measures, per time-of-day bucket, the cost a SMALL liquidity-demanding order pays:

    quoted spread          = ask - bid                       (posted full spread)
    quoted half-spread     = (ask - bid) / 2                 (taker cost to cross, one way)
    effective half-spread  = |trade_price - mid|             (realized cost vs mid)
    round-trip (taker)     ~ quoted spread                   (cross half each way)

Why this exists
---------------
Candidate-1 (vol-conditioned reversal) is a thin-edge strategy. Its gross per-trade
edge must clear the ROUND-TRIP cost of the ETF it trades on. This produces that number
from your own data so signal thresholds are set on measured cost, not on guesses. The
output feeds the position-entry thresholds in the Candidate-1 spec (Part A6).

Data-quality note (important)
-----------------------------
Time & sales contains block trades and late/derivatively-priced prints stamped at a
time when the prevailing NBBO does not reflect their true execution. Those inflate a
naive size-weighted effective spread far above what a small order actually pays. We
therefore (a) measure effective spread only on trades AT/INSIDE the contemporaneous
NBBO, (b) report a size-weighted MEDIAN (robust to the remaining tail), and (c) anchor
the headline cost on the QUOTED spread, which bounds a small marketable order's cost.
The share of out-of-NBBO prints is reported as a diagnostic.

Offline batch (pandas). NOT part of the live asyncio trader — async constraint applies
to the execution path, not to research tooling.

Usage
-----
    PY=/opt/miniconda3/envs/quant-env/bin/python
    $PY analysis/effective_spread.py                                  # all available QQQ dates
    $PY analysis/effective_spread.py --dates 2026-01-02:2026-01-30    # in-sample (January)
    $PY analysis/effective_spread.py --dates 2026-05-01:2026-05-29    # out-of-sample (May)
    $PY analysis/effective_spread.py --symbol TQQQ --dates 2026-05-01:2026-05-29
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from datetime import time

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- config
CACHE_DIR_DEFAULT = "/Users/woongyeol/quant_trading/logs/tick_cache"
OUT_DIR_DEFAULT = "/Users/woongyeol/quant_trading/analysis/output"
RTH_START, RTH_END = time(9, 30), time(16, 0)
MERGE_TOLERANCE = pd.Timedelta("2s")    # max staleness allowed for the prevailing quote
NBBO_TOL = 0.01                         # price tolerance ($) for "inside NBBO" (sub-penny/rounding)
MAX_SANE_HALF_BPS = 50.0               # final guard against broken prints
SIZE_FLOOR = 1.0                        # min trade size (set 100 to drop odd lots)
BUCKET_MINUTES = 30


# --------------------------------------------------------------------------- helpers
def _wavg(x, w) -> float:
    x = np.asarray(x, float); w = np.asarray(w, float)
    s = w.sum()
    return float(np.dot(x, w) / s) if s > 0 else np.nan


def _wmedian(x, w) -> float:
    x = np.asarray(x, float); w = np.asarray(w, float)
    if x.size == 0 or w.sum() == 0:
        return np.nan
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w)
    return float(x[np.searchsorted(c, 0.5 * c[-1])])


# --------------------------------------------------------------------------- discovery
def available_dates(cache_dir: str, symbol: str) -> list[str]:
    pat = os.path.join(cache_dir, f"{symbol}_*_quotes.pkl")
    dates = []
    for p in glob.glob(pat):
        m = re.search(rf"{re.escape(symbol)}_(\d{{4}}-\d{{2}}-\d{{2}})_quotes\.pkl", os.path.basename(p))
        if m:
            dates.append(m.group(1))
    return sorted(dates)


def _load(cache_dir: str, symbol: str, date: str, kind: str) -> pd.DataFrame | None:
    f = os.path.join(cache_dir, f"{symbol}_{date}_{kind}.pkl")
    if not os.path.exists(f):
        return None
    df = pd.read_pickle(f)
    if not isinstance(df.index, pd.DatetimeIndex):
        ts = next((c for c in ("timestamp", "t", "time") if c in df.columns), None)
        if ts is None:
            raise ValueError(f"{f}: no DatetimeIndex and no timestamp column")
        df = df.set_index(pd.DatetimeIndex(df[ts]))
    if df.index.tz is None:                      # data should already be ET; defensive
        df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
    return df.sort_index()


# --------------------------------------------------------------------------- core calc
def measure_day(cache_dir: str, symbol: str, date: str) -> pd.DataFrame | None:
    """Per-trade frame with effective spread for one day, or None if data missing."""
    q = _load(cache_dir, symbol, date, "quotes")
    t = _load(cache_dir, symbol, date, "trades")
    if q is None or t is None:
        return None

    q = q.between_time(RTH_START, RTH_END)
    t = t.between_time(RTH_START, RTH_END)

    q = q[(q["bid_price"] > 0) & (q["ask_price"] > 0) & (q["ask_price"] >= q["bid_price"])]
    q = q[["bid_price", "ask_price"]].copy()
    q["mid"] = (q["bid_price"] + q["ask_price"]) / 2.0

    t = t[t["size"] >= SIZE_FLOOR][["price", "size"]].copy()
    if t.empty or q.empty:
        return None

    m = pd.merge_asof(
        t, q, left_index=True, right_index=True,
        direction="backward", tolerance=MERGE_TOLERANCE,
    ).dropna(subset=["mid"])
    if m.empty:
        return None

    dev = m["price"] - m["mid"]
    half_spread = (m["ask_price"] - m["bid_price"]) / 2.0
    m["quoted_bps"] = (m["ask_price"] - m["bid_price"]) / m["mid"] * 1e4
    m["eff_half_bps"] = dev.abs() / m["mid"] * 1e4
    m["signed_bps"] = dev / m["mid"] * 1e4                       # +ve => above mid (buyer-initiated)
    # a small order cannot print materially outside the prevailing NBBO; flag the rest
    m["inside_nbbo"] = (m["price"] >= m["bid_price"] - NBBO_TOL) & (m["price"] <= m["ask_price"] + NBBO_TOL)
    m = m[m["eff_half_bps"] <= MAX_SANE_HALF_BPS]

    mins = m.index.hour * 60 + m.index.minute
    bstart = (mins // BUCKET_MINUTES) * BUCKET_MINUTES
    m["bucket"] = [f"{x // 60:02d}:{x % 60:02d}" for x in bstart]
    m["date"] = date
    return m


def aggregate_day(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bucket, d in m.groupby("bucket"):
        ins = d[d["inside_nbbo"]]
        rows.append(dict(
            date=d["date"].iloc[0],
            bucket=bucket,
            n_trades=int(len(d)),
            dollar_vol=float((d["price"] * d["size"]).sum()),
            outside_nbbo_share=float(1.0 - d["inside_nbbo"].mean()),
            quoted_bps=float(d["quoted_bps"].median()),                       # full posted spread
            eff_half_bps=_wmedian(ins["eff_half_bps"], ins["size"]),          # robust, inside-NBBO
            eff_half_bps_wmean=_wavg(ins["eff_half_bps"], ins["size"]),       # diagnostic
            signed_bps=_wavg(ins["signed_bps"], ins["size"]),                 # adverse-selection tilt
        ))
    return pd.DataFrame(rows)


def grand_summary(daily: pd.DataFrame) -> pd.DataFrame:
    out = []
    for bucket, d in daily.groupby("bucket"):
        w = d["dollar_vol"]
        quoted = float(d["quoted_bps"].median())
        eff_half = _wavg(d["eff_half_bps"], w)
        out.append(dict(
            bucket=bucket,
            n_days=int(d["date"].nunique()),
            avg_trades_per_day=float(d["n_trades"].mean()),
            quoted_bps=quoted,
            quoted_half_bps=quoted / 2.0,
            eff_half_bps=eff_half,
            roundtrip_taker_bps=quoted,             # cross half-spread each way ~ full spread
            roundtrip_eff_bps=2.0 * eff_half,       # realized vs mid, both legs
            outside_nbbo_share=_wavg(d["outside_nbbo_share"], d["n_trades"]),
            signed_bps=_wavg(d["signed_bps"], w),
        ))
    return pd.DataFrame(out).sort_values("bucket").reset_index(drop=True)


# --------------------------------------------------------------------------------- cli
def parse_dates(spec: str, all_dates: list[str]) -> list[str]:
    if not spec or spec == "all":
        return all_dates
    if ":" in spec:
        lo, hi = spec.split(":", 1)
        return [d for d in all_dates if lo <= d <= hi]
    wanted = {s.strip() for s in spec.split(",")}
    return [d for d in all_dates if d in wanted]


def main() -> None:
    ap = argparse.ArgumentParser(description="Effective-spread / cost-wall measurement.")
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    ap.add_argument("--dates", default="all", help="'all' | 'YYYY-MM-DD:YYYY-MM-DD' | comma list")
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    args = ap.parse_args()

    all_dates = available_dates(args.cache_dir, args.symbol)
    dates = parse_dates(args.dates, all_dates)
    if not dates:
        raise SystemExit(f"No cached dates for {args.symbol} matching '{args.dates}'.")

    os.makedirs(args.out_dir, exist_ok=True)
    daily_rows = []
    for date in dates:
        m = measure_day(args.cache_dir, args.symbol, date)
        if m is None:
            print(f"  {date}: no usable data, skipped")
            continue
        daily_rows.append(aggregate_day(m))
        ins = m[m["inside_nbbo"]]
        print(f"  {date}: {len(m):>9,} trades  quoted={m['quoted_bps'].median():.3f}bps  "
              f"eff_half(wmed)={_wmedian(ins['eff_half_bps'], ins['size']):.3f}bps  "
              f"outside_nbbo={1 - m['inside_nbbo'].mean():.1%}")
        del m

    if not daily_rows:
        raise SystemExit("No days produced results.")

    daily = pd.concat(daily_rows, ignore_index=True)
    summary = grand_summary(daily)

    daily_path = os.path.join(args.out_dir, f"effective_spread_{args.symbol}_daily.csv")
    summ_path = os.path.join(args.out_dir, f"effective_spread_{args.symbol}_summary.csv")
    daily.to_csv(daily_path, index=False)
    summary.to_csv(summ_path, index=False)

    print(f"\n=== {args.symbol}  cost by time-of-day  ({len(dates)} days: {dates[0]}..{dates[-1]}) ===")
    show = summary[["bucket", "avg_trades_per_day", "quoted_bps", "eff_half_bps",
                    "roundtrip_taker_bps", "outside_nbbo_share", "signed_bps"]]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    rt = _wavg(summary["roundtrip_taker_bps"], summary["avg_trades_per_day"])
    print(f"\nSession round-trip taker cost (trade-weighted): ~{rt:.2f} bps")
    print(f"  -> Candidate-1 entries on {args.symbol} must clear ~{rt:.2f} bps gross edge per round trip.")
    print(f"saved: {summ_path}\n       {daily_path}")


if __name__ == "__main__":
    main()
