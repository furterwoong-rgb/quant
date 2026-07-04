#!/usr/bin/env python
"""
snapshot_option_state.py — Daily QQQ option-state capture (greeks + IV + open interest).

WHY: Alpaca serves option greeks/OI as "latest" only, so Jan/May 2026 gamma is not
back-testable (see probe_options_history.py). The fix is to start capturing the chain
DAILY now, building our own option-state history → the gamma gate becomes back-testable
from today forward. Run it today to seed the series; the scheduler runs it every weekday
near the US close.

WHAT IT STORES (per run):
  logs/option_cache/QQQ_{YYYY-MM-DD}_optionstate.pkl   full per-contract frame:
      symbol, type, strike, expiry, dte, gamma, delta, vega, theta, iv, open_interest,
      oi_asof, dollar_gamma, signed_dollar_gamma   (+ spot, asof_ts as attrs/columns)
  logs/option_cache/gex_daily.csv                      one summary row per day:
      date, asof_ts, spot, net_gex, regime, call_dollar_gamma, put_dollar_gamma,
      atm_iv, n_contracts, oi_asof

Raw per-contract components are stored so GEX can be recomputed under any dealer-sign
convention later. Default convention (index-ETF standard): dealers long calls / short
puts → net_gex>0 = long-gamma (mean-reverting) regime.

Run
---
    PY=/opt/miniconda3/envs/quant-env/bin/python
    $PY analysis/snapshot_option_state.py            # capture now
    $PY analysis/snapshot_option_state.py --force    # overwrite today's file

Read-only against Alpaca (chain + contracts + latest trade). No orders.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = "/Users/woongyeol/quant_trading"
OUT_DIR = os.path.join(PROJECT_ROOT, "logs", "option_cache")
UNDERLYING = "QQQ"
STRIKE_BAND = 0.15     # capture strikes within ±15% of spot (covers the bulk of dealer gamma)
MAX_DTE = 60           # capture expirations within 60 days (0-60 DTE dominates gamma)
ET = "America/New_York"

API_KEY_ID = ""        # optional; else env / config.settings
API_SECRET = ""


def load_credentials():
    key = API_KEY_ID or os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_KEY_ID")
    sec = API_SECRET or os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
    if not (key and sec):
        try:
            sys.path.insert(0, PROJECT_ROOT)
            from config import settings as st  # type: ignore
            for k in ("ALPACA_API_KEY", "API_KEY", "ALPACA_KEY_ID", "APCA_API_KEY_ID", "KEY_ID"):
                key = key or getattr(st, k, None)
            for s in ("ALPACA_SECRET_KEY", "API_SECRET", "ALPACA_SECRET", "APCA_API_SECRET_KEY", "SECRET_KEY"):
                sec = sec or getattr(st, s, None)
        except Exception:
            pass
    if not (key and sec):
        raise SystemExit("No Alpaca credentials (env APCA_API_KEY_ID/APCA_API_SECRET_KEY or config.settings).")
    return key, sec


def build(cls, **kw):
    fields = getattr(cls, "model_fields", None)
    if fields is not None:
        kw = {k: v for k, v in kw.items() if k in fields}
    return cls(**kw)


def fetch_spot(stk, StockLatestTradeRequest) -> float:
    lt = stk.get_stock_latest_trade(build(StockLatestTradeRequest, symbol_or_symbols=UNDERLYING))
    return float(lt[UNDERLYING].price)


def fetch_chain(opt, OptionChainRequest, FEEDS, spot):
    lo, hi = round(spot * (1 - STRIKE_BAND)), round(spot * (1 + STRIKE_BAND))
    exp_lte = date.today() + timedelta(days=MAX_DTE)
    for feed in FEEDS:
        try:
            req = build(OptionChainRequest, underlying_symbol=UNDERLYING, feed=feed,
                        strike_price_gte=lo, strike_price_lte=hi, expiration_date_lte=exp_lte)
            chain = opt.get_option_chain(req)
            if chain:
                return chain
        except Exception as e:
            print(f"  chain feed={feed}: {type(e).__name__}: {e}")
    return {}


def fetch_oi(trd, GetOptionContractsRequest, AssetStatus, spot) -> dict:
    """Paginate all near-dated QQQ contracts; return {symbol: (oi, oi_asof)}."""
    lo, hi = round(spot * (1 - STRIKE_BAND)), round(spot * (1 + STRIKE_BAND))
    out, token = {}, None
    while True:
        kw = dict(underlying_symbols=[UNDERLYING], limit=10000, page_token=token,
                  strike_price_gte=str(lo), strike_price_lte=str(hi),
                  expiration_date_gte=date.today(),
                  expiration_date_lte=date.today() + timedelta(days=MAX_DTE))
        if AssetStatus is not None:
            kw["status"] = AssetStatus.ACTIVE
        resp = trd.get_option_contracts(build(GetOptionContractsRequest, **kw))
        for c in getattr(resp, "option_contracts", []) or []:
            oi = getattr(c, "open_interest", None)
            if oi not in (None, ""):
                out[c.symbol] = (float(oi), getattr(c, "open_interest_date", None))
        token = getattr(resp, "next_page_token", None)
        if not token:
            break
    return out


def parse_occ(symbol: str):
    """OCC: <root><YYMMDD><C/P><strike*1000 (8)>. Returns (type, expiry_date, strike)."""
    strike = int(symbol[-8:]) / 1000.0
    cp = symbol[-9]
    ymd = symbol[-15:-9]
    expiry = datetime.strptime(ymd, "%y%m%d").date()
    return ("C" if cp == "C" else "P"), expiry, strike


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite today's snapshot")
    args = ap.parse_args()

    key, sec = load_credentials()
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import OptionChainRequest, StockLatestTradeRequest
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOptionContractsRequest
    try:
        from alpaca.data.enums import OptionsFeed
        FEEDS = [OptionsFeed.OPRA, OptionsFeed.INDICATIVE]
    except Exception:
        FEEDS = [None]
    try:
        from alpaca.trading.enums import AssetStatus
    except Exception:
        AssetStatus = None

    opt = OptionHistoricalDataClient(key, sec)
    stk = StockHistoricalDataClient(key, sec)
    trd = TradingClient(key, sec, paper=True)

    asof = pd.Timestamp.now(tz="UTC")
    today_et = asof.tz_convert(ET).date().isoformat()
    os.makedirs(OUT_DIR, exist_ok=True)
    pkl_path = os.path.join(OUT_DIR, f"{UNDERLYING}_{today_et}_optionstate.pkl")
    csv_path = os.path.join(OUT_DIR, "gex_daily.csv")
    if os.path.exists(pkl_path) and not args.force:
        raise SystemExit(f"{pkl_path} exists (use --force to overwrite).")

    spot = fetch_spot(stk, StockLatestTradeRequest)
    chain = fetch_chain(opt, OptionChainRequest, FEEDS, spot)
    oi_map = fetch_oi(trd, GetOptionContractsRequest, AssetStatus, spot)
    print(f"spot={spot:.2f}  chain={len(chain)}  contracts_with_oi={len(oi_map)}")

    rows = []
    for sym, snap in chain.items():
        g = getattr(snap, "greeks", None)
        if not g or getattr(g, "gamma", None) is None:
            continue
        oi, oi_asof = oi_map.get(sym, (np.nan, None))
        typ, expiry, strike = parse_occ(sym)
        gamma = float(g.gamma)
        dollar_gamma = gamma * (oi if oi == oi else 0.0) * 100 * spot * spot * 0.01
        rows.append(dict(
            symbol=sym, type=typ, strike=strike, expiry=expiry,
            dte=(expiry - date.today()).days,
            gamma=gamma, delta=getattr(g, "delta", np.nan), vega=getattr(g, "vega", np.nan),
            theta=getattr(g, "theta", np.nan), iv=getattr(snap, "implied_volatility", np.nan),
            open_interest=oi, oi_asof=oi_asof,
            dollar_gamma=dollar_gamma,
            signed_dollar_gamma=dollar_gamma if typ == "C" else -dollar_gamma,  # dealers long C / short P
        ))

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No contracts with greeks captured — check entitlement/feed.")
    df.attrs["spot"] = spot
    df.attrs["asof"] = asof.isoformat()
    df.to_pickle(pkl_path)

    matched = df["open_interest"].notna()
    net_gex = float(df.loc[matched, "signed_dollar_gamma"].sum())
    call_dg = float(df.loc[matched & (df.type == "C"), "dollar_gamma"].sum())
    put_dg = float(df.loc[matched & (df.type == "P"), "dollar_gamma"].sum())
    atm = df.iloc[(df["strike"] - spot).abs().argsort()[:8]]
    atm_iv = float(atm["iv"].mean())
    oi_asof = next((str(v) for v in df["oi_asof"] if v is not None), "")
    regime = "long_gamma" if net_gex > 0 else "short_gamma"

    summary = dict(
        date=today_et, asof_ts=asof.isoformat(), spot=round(spot, 4),
        net_gex=round(net_gex, 2), regime=regime,
        call_dollar_gamma=round(call_dg, 2), put_dollar_gamma=round(put_dg, 2),
        atm_iv=round(atm_iv, 5), n_contracts=int(matched.sum()), oi_asof=oi_asof,
    )
    hdr = not os.path.exists(csv_path)
    prior = pd.read_csv(csv_path) if not hdr else None
    if prior is not None and (prior["date"] == today_et).any():
        prior = prior[prior["date"] != today_et]                      # idempotent on re-run
        pd.concat([prior, pd.DataFrame([summary])], ignore_index=True).to_csv(csv_path, index=False)
    else:
        pd.DataFrame([summary]).to_csv(csv_path, mode="a", header=hdr, index=False)

    print(f"net_GEX={net_gex:,.0f} -> {regime}   ATM_IV={atm_iv:.3f}   matched={int(matched.sum())}")
    print(f"saved: {pkl_path}\n       {csv_path}")


if __name__ == "__main__":
    main()
