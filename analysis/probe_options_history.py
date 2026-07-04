#!/usr/bin/env python
"""
probe_options_history.py — Can we reconstruct QQQ dealer-gamma (GEX) for Jan & May 2026
from Alpaca, or is gamma a forward-test-only overlay?

The Candidate-1 reversal signal conditions on a gamma regime (GEX sign). To put that in
the *backtest* (Jan in-sample / May out-of-sample) we need historical per-strike
GREEKS and OPEN INTEREST. Alpaca's chain/snapshot endpoints are "latest" only, so this
script empirically discovers, for THIS account + alpaca-py version, exactly what is and
isn't available, and prints a verdict:

    gamma backtestable on Jan/May  =  YES  /  PARTIAL  /  NO   (with reasons)

Tests
-----
  0. SDK version + available option methods (introspection)
  1. LIVE chain snapshot: greeks + IV present?           -> forward/paper-trading gamma
  2. Open-interest source (trading OptionContracts):     -> OI present? how fresh?
  3. Historical option BARS for expired Jan & May 2026 contracts: retrievable? carry OI?
  4. Any AS-OF-DATE historical chain/greeks?             -> (expected: no)
  5. BONUS: end-to-end LIVE GEX from chain greeks x contract OI (proves the forward path)

Run
---
    PY=/opt/miniconda3/envs/quant-env/bin/python
    APCA_API_KEY_ID=... APCA_API_SECRET_KEY=... $PY analysis/probe_options_history.py
  (or export the keys in your shell / config and just run it)

Read-only. No orders are placed.
"""
from __future__ import annotations

import os
import sys
import warnings
from datetime import date, datetime, timedelta

warnings.filterwarnings("ignore")

PROJECT_ROOT = "/Users/woongyeol/quant_trading"
UNDERLYING = "QQQ"
# Monthly (3rd-Friday) expiries; pick a few strikes around each month's ATM for the bars test.
JAN = dict(yymmdd="260116", strikes=[615, 620, 625, 630, 635],
           start=datetime(2026, 1, 2), end=datetime(2026, 2, 1))
MAY = dict(yymmdd="260515", strikes=[665, 670, 675, 680, 685],
           start=datetime(2026, 5, 1), end=datetime(2026, 5, 30))

# Optionally hard-code keys here (leave blank to use env / config.settings instead).
API_KEY_ID = ""
API_SECRET = ""


# --------------------------------------------------------------------------- utilities
def sep(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def tag(ok) -> str:
    return {True: "PASS", False: "FAIL", None: "UNKNOWN"}[ok]


def mask(s: str) -> str:
    return f"{s[:4]}…{s[-2:]}" if s and len(s) > 6 else "(set)"


def occ(root: str, yymmdd: str, cp: str, strike: float) -> str:
    return f"{root}{yymmdd}{cp}{int(round(strike * 1000)):08d}"


def build(cls, **kw):
    """Instantiate a pydantic request, silently dropping kwargs the version doesn't support."""
    fields = getattr(cls, "model_fields", None)
    if fields is not None:
        kw = {k: v for k, v in kw.items() if k in fields}
    return cls(**kw)


def load_credentials():
    key = API_KEY_ID or os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_KEY_ID")
    sec = API_SECRET or os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
    src = "env / constant"
    if not (key and sec):
        try:
            sys.path.insert(0, PROJECT_ROOT)
            from config import settings as st  # type: ignore
            for k in ("ALPACA_API_KEY", "API_KEY", "ALPACA_KEY_ID", "APCA_API_KEY_ID", "KEY_ID"):
                key = key or getattr(st, k, None)
            for s in ("ALPACA_SECRET_KEY", "API_SECRET", "ALPACA_SECRET", "APCA_API_SECRET_KEY", "SECRET_KEY"):
                sec = sec or getattr(st, s, None)
            src = "config.settings"
        except Exception:
            pass
    return key, sec, src


# ------------------------------------------------------------------------------- probe
def main() -> None:
    key, sec, src = load_credentials()
    if not (key and sec):
        raise SystemExit(
            "No Alpaca credentials found. Set env vars and re-run, e.g.:\n"
            "  APCA_API_KEY_ID=... APCA_API_SECRET_KEY=... "
            "/opt/miniconda3/envs/quant-env/bin/python analysis/probe_options_history.py\n"
            "or fill API_KEY_ID/API_SECRET at the top of this file, or expose them in config/settings.py."
        )
    print(f"credentials: {mask(key)} (source: {src})")

    # imports (report precisely what the installed SDK exposes)
    import alpaca
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import (OptionChainRequest, OptionBarsRequest,
                                      OptionSnapshotRequest, StockLatestTradeRequest)
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOptionContractsRequest
    try:
        from alpaca.data.enums import OptionsFeed
        FEEDS = [OptionsFeed.OPRA, OptionsFeed.INDICATIVE]
    except Exception:
        FEEDS = [None]
    try:
        from alpaca.trading.enums import AssetStatus, ContractType
    except Exception:
        AssetStatus = ContractType = None

    opt = OptionHistoricalDataClient(key, sec)
    stk = StockHistoricalDataClient(key, sec)
    trd = TradingClient(key, sec, paper=True)

    results = dict(live_greeks=None, oi=None, hist_bars_jan=None, hist_bars_may=None,
                   hist_oi=None, asof_chain=False, live_gex=None)

    # --- Test 0: SDK surface -------------------------------------------------------
    sep("TEST 0  SDK surface")
    print("alpaca-py version:", getattr(alpaca, "__version__", "unknown"))
    print("OptionHistoricalDataClient methods:",
          [m for m in dir(opt) if m.startswith("get_")])
    chain_fields = list(getattr(OptionChainRequest, "model_fields", {}).keys())
    print("OptionChainRequest fields:", chain_fields)
    results["asof_chain"] = any(f in chain_fields for f in ("as_of", "date", "start", "asof"))
    print(f"  -> historical as-of-date chain param present? {results['asof_chain']}")

    # --- live underlying spot (for strike bounds + GEX) ----------------------------
    spot = None
    try:
        lt = stk.get_stock_latest_trade(build(StockLatestTradeRequest, symbol_or_symbols=UNDERLYING))
        spot = float(lt[UNDERLYING].price)
        print(f"live {UNDERLYING} spot: {spot:.2f}")
    except Exception as e:
        spot = 680.0
        print(f"live spot fetch failed ({type(e).__name__}); assuming {spot}")

    # --- Test 1: live chain snapshot (greeks + IV) --------------------------------
    sep("TEST 1  Live chain snapshot — greeks + implied vol")
    chain = None
    for feed in FEEDS:
        try:
            req = build(OptionChainRequest, underlying_symbol=UNDERLYING, feed=feed,
                        strike_price_gte=round(spot * 0.95), strike_price_lte=round(spot * 1.05),
                        expiration_date_lte=date.today() + timedelta(days=45))
            chain = opt.get_option_chain(req)
            if chain:
                print(f"feed={feed}: {len(chain)} contracts returned")
                break
        except Exception as e:
            print(f"feed={feed}: {type(e).__name__}: {e}")
    if chain:
        n_g = sum(1 for s in chain.values() if getattr(s, "greeks", None))
        n_iv = sum(1 for s in chain.values() if getattr(s, "implied_volatility", None))
        results["live_greeks"] = n_g > 0
        print(f"  with greeks: {n_g}/{len(chain)}   with IV: {n_iv}/{len(chain)}   [{tag(results['live_greeks'])}]")
        for sym, s in list(chain.items())[:1]:
            g = getattr(s, "greeks", None)
            if g:
                print(f"  sample {sym}: gamma={getattr(g,'gamma',None)} delta={getattr(g,'delta',None)} "
                      f"iv={getattr(s,'implied_volatility',None)}")
    else:
        print("  no chain returned  [FAIL]  (check options entitlement / feed / plan)")

    # --- Test 2: open interest via trading OptionContracts ------------------------
    sep("TEST 2  Open interest (trading OptionContracts)")
    contracts = []
    try:
        kw = dict(underlying_symbols=[UNDERLYING], limit=200,
                  expiration_date_gte=date.today(), expiration_date_lte=date.today() + timedelta(days=45))
        if AssetStatus is not None:
            kw["status"] = AssetStatus.ACTIVE
        resp = trd.get_option_contracts(build(GetOptionContractsRequest, **kw))
        contracts = getattr(resp, "option_contracts", []) or []
        with_oi = [c for c in contracts if getattr(c, "open_interest", None) not in (None, "")]
        results["oi"] = len(with_oi) > 0
        print(f"contracts: {len(contracts)}   with open_interest: {len(with_oi)}   [{tag(results['oi'])}]")
        for c in with_oi[:3]:
            print(f"  {c.symbol}: OI={c.open_interest}  as_of={getattr(c,'open_interest_date',None)}  "
                  f"strike={c.strike_price}")
        if with_oi:
            oid = getattr(with_oi[0], "open_interest_date", None)
            if isinstance(oid, str):
                try: oid = datetime.fromisoformat(oid).date()
                except Exception: oid = None
            if isinstance(oid, date):
                print(f"  OI freshness: {(date.today() - oid).days} days old "
                      f"(latest-only; NOT an as-of-Jan/May value)")
    except Exception as e:
        results["oi"] = False
        print(f"  {type(e).__name__}: {e}  [FAIL]")

    # --- Test 3: historical option BARS for expired Jan & May contracts -----------
    sep("TEST 3  Historical option bars (expired Jan & May 2026 contracts)")

    def try_bars(month, label):
        for k in month["strikes"]:
            sym = occ(UNDERLYING, month["yymmdd"], "C", k)
            try:
                req = build(OptionBarsRequest, symbol_or_symbols=sym, timeframe=TimeFrame.Day,
                            start=month["start"], end=month["end"])
                bs = opt.get_option_bars(req)
                df = bs.df
                if df is not None and len(df) > 0:
                    has_oi = "open_interest" in [c.lower() for c in df.columns]
                    print(f"  {label} {sym}: {len(df)} daily bars  cols={list(df.columns)}  OI_in_bars={has_oi}")
                    return True, has_oi
            except Exception as e:
                print(f"  {label} {sym}: {type(e).__name__}: {e}")
        print(f"  {label}: no bars returned for any tried strike")
        return False, False

    ok_jan, oi_jan = try_bars(JAN, "JAN")
    ok_may, oi_may = try_bars(MAY, "MAY")
    results["hist_bars_jan"] = ok_jan
    results["hist_bars_may"] = ok_may
    results["hist_oi"] = bool(oi_jan or oi_may)
    print(f"  historical bars: Jan={tag(ok_jan)} May={tag(ok_may)}   OI in historical bars: {tag(results['hist_oi'])}")

    # --- Test 4: as-of-date greeks on an expired symbol (snapshot is latest-only) --
    sep("TEST 4  As-of-date greeks on expired contract (expected: none)")
    try:
        sym = occ(UNDERLYING, JAN["yymmdd"], "C", JAN["strikes"][2])
        snap = opt.get_option_snapshot(build(OptionSnapshotRequest, symbol_or_symbols=sym))
        s = snap.get(sym) if isinstance(snap, dict) else None
        has = bool(getattr(s, "greeks", None)) if s else False
        print(f"  snapshot for expired {sym}: greeks present? {has} "
              f"(if present they are stale/now, not as-of January)")
    except Exception as e:
        print(f"  {type(e).__name__}: {e}  (expired-contract snapshot unavailable — expected)")

    # --- Test 5 (bonus): end-to-end LIVE GEX (forward-path proof) -----------------
    sep("TEST 5  BONUS — live GEX from chain greeks x contract OI")
    try:
        if chain and contracts and spot:
            oi_by_sym = {c.symbol: float(c.open_interest) for c in contracts
                         if getattr(c, "open_interest", None) not in (None, "")}
            gex = 0.0
            n = 0
            for sym, s in chain.items():
                g = getattr(s, "greeks", None)
                oi = oi_by_sym.get(sym)
                if g and getattr(g, "gamma", None) is not None and oi:
                    dollar = g.gamma * oi * 100 * spot * spot * 0.01
                    gex += dollar if sym[-9] == "C" else -dollar   # OCC: char before 8-digit strike is C/P
                    n += 1
            if n:
                results["live_gex"] = gex
                regime = "LONG-gamma (mean-revert → Candidate-1 ON)" if gex > 0 else "SHORT-gamma (trend → Candidate-1 OFF)"
                print(f"  matched {n} strikes;  live net GEX = {gex:,.0f}  ->  {regime}")
            else:
                print("  could not match greeks to OI (bounds mismatch) — fix strike bounds and retry")
        else:
            print("  skipped (need both chain greeks and contract OI from tests 1 & 2)")
    except Exception as e:
        print(f"  {type(e).__name__}: {e}")

    # --- Verdict -------------------------------------------------------------------
    sep("VERDICT  — gamma in the Jan/May backtest")
    hist_bars = bool(results["hist_bars_jan"] and results["hist_bars_may"])
    if results["asof_chain"] or (hist_bars and results["hist_oi"]):
        print("YES — historical greeks/OI are retrievable; GEX can go directly into the Jan/May backtest.")
    elif results["live_greeks"] and results["oi"] and hist_bars:
        print("PARTIAL (most likely outcome):")
        print("  • Forward/paper gamma: FULLY supported — live chain greeks + contract OI work")
        print(f"    (live GEX computed end-to-end: {results['live_gex'] is not None}).")
        print("  • Historical per-contract GAMMA: reconstructable from historical option BARS")
        print("    (Black-Scholes from option price + underlying + IV).")
        print("  • Historical OPEN INTEREST: NOT available as-of Jan/May (snapshot/contracts are latest-only)")
        print("    → historical GEX *weights* are missing, so GEX is NOT cleanly backtestable on Jan/May.")
        print("\n  RECOMMENDATION:")
        print("  1. Validate the z×RV reversal CORE on Jan/May historically (gamma layer OFF).")
        print("  2. Run the gamma gate as a FORWARD-TEST overlay during Stage-7 paper trading.")
        print("  3. Start a daily QQQ OI+greeks snapshot NOW (cron via your scheduler) to build")
        print("     your own option-state history → gamma becomes backtestable from today forward.")
    elif results["live_greeks"]:
        print("FORWARD-ONLY — live greeks work but OI/contracts unclear; verify plan entitlement.")
    else:
        print("NO — options data not returning. Check options entitlement, feed (OPRA), and keys.")
    print("\nPaste this entire output back so we can lock the gamma plan.")


if __name__ == "__main__":
    main()
