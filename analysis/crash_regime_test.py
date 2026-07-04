#!/usr/bin/env python3
"""
2025-04-03/04 타리프 쇼크 크래시 레짐 테스트
────────────────────────────────────────────────
확정 전략 (VWAP-only, Long/Cash, 0.20~0.30) 이
하락 레짐에서 어떻게 동작하는지 검증

2025-04-03: SPY -4.9% (Liberation Day 관세 충격)
2025-04-04: SPY -6.0% (추가 급락)
"""

import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytz, datetime as dt
from pathlib import Path

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests   import (StockTradesRequest, StockQuotesRequest,
                                     StockBarsRequest, TimeFrame)
from config.settings import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY,
    STOP_LOSS_PCT, PARTIAL_PCT, TRAIL_PULLBACK,
    CONVICTION_LONG_EXIT, ENTRY_MIN_CONVICTION,
    EOD_TIGHT_STOP, EOD_TIGHT_TRAIL, EOD_TIGHT_PARTIAL,
    SWITCH_COOLDOWN_MIN, CONVICTION_COOLDOWN,
)

ET        = pytz.timezone("America/New_York")
CACHE_DIR = Path("logs/tick_cache")

CRASH_DAYS = ["2025-04-03", "2025-04-04"]

PORTFOLIO  = 100_000.0
KELLY      = 0.10
SPREAD     = 0.00015
OBI_ALPHA  = 0.08
TFI_ALPHA  = 0.15
MIN_CONV   = 0.20
MAX_CONV   = 0.30

SESSION_OPEN  = dt.time(9, 45)
SESSION_CLOSE = dt.time(15, 55)
EOD_TIME      = dt.time(14, 55)
LONG_SYMS     = ["SSO", "QLD"]


# ── 데이터 로더 ───────────────────────────────────────────────────────────────

def load_tick(sym: str, date_str: str, client) -> tuple:
    cache_t = CACHE_DIR / f"{sym}_{date_str}_trades.pkl"
    cache_q = CACHE_DIR / f"{sym}_{date_str}_quotes.pkl"
    d = pd.Timestamp(date_str)
    s_et  = ET.localize(dt.datetime(d.year, d.month, d.day, 9, 30))
    e_et  = ET.localize(dt.datetime(d.year, d.month, d.day, 16, 0))
    s_utc = s_et.astimezone(dt.timezone.utc)
    e_utc = e_et.astimezone(dt.timezone.utc)

    def _load(cache, req_fn):
        if cache.exists():
            with open(cache, "rb") as f: return pickle.load(f)
        print(f"    다운로드: {sym} {date_str} {cache.stem.split('_')[-1]}...", end=" ", flush=True)
        df = req_fn()
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(sym, level="symbol")
        df.index = pd.DatetimeIndex(df.index).tz_convert(ET)
        with open(cache, "wb") as f: pickle.dump(df, f)
        print(f"✓ ({len(df):,}행)")
        return df

    trades = _load(cache_t, lambda: client.get_stock_trades(
        StockTradesRequest(symbol_or_symbols=sym, start=s_utc, end=e_utc, feed="sip")).df)
    quotes = _load(cache_q, lambda: client.get_stock_quotes(
        StockQuotesRequest(symbol_or_symbols=sym, start=s_utc, end=e_utc, feed="sip")).df)
    return trades, quotes


def load_bars(sym: str, date_str: str, client) -> pd.DataFrame:
    cache = CACHE_DIR / f"{sym}_{date_str}_bars.pkl"
    d = pd.Timestamp(date_str)
    s_et  = ET.localize(dt.datetime(d.year, d.month, d.day, 9, 30))
    e_et  = ET.localize(dt.datetime(d.year, d.month, d.day, 16, 0))
    s_utc = s_et.astimezone(dt.timezone.utc)
    e_utc = e_et.astimezone(dt.timezone.utc)

    if not cache.exists():
        print(f"    다운로드: {sym} {date_str} bars...", end=" ", flush=True)
        bars = client.get_stock_bars(
            StockBarsRequest(symbol_or_symbols=sym, start=s_utc, end=e_utc,
                             timeframe=TimeFrame.Minute, feed="sip")).df
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(sym, level="symbol")
        with open(cache, "wb") as f: pickle.dump(bars, f)
        print(f"✓ ({len(bars)}행)")
    else:
        with open(cache, "rb") as f: bars = pickle.load(f)

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(sym, level="symbol")
    bars.index = pd.DatetimeIndex(bars.index)
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize(ET)
    else:
        bars.index = bars.index.tz_convert(ET)
    day = pd.Timestamp(date_str).date()
    return bars[bars.index.date == day].between_time("09:30", "16:00")


# ── 신호 계산 ─────────────────────────────────────────────────────────────────

def compute_obi(quotes: pd.DataFrame) -> pd.Series:
    q = quotes[["bid_size", "ask_size"]].copy()
    q = q[(q["bid_size"] > 0) & (q["ask_size"] > 0)]
    if q.empty: return pd.Series(dtype=float)
    raw = (q["bid_size"] - q["ask_size"]) / (q["bid_size"] + q["ask_size"])
    return raw.ewm(alpha=OBI_ALPHA, adjust=False).mean().resample("1min").last().ffill()


def compute_tfi(quotes: pd.DataFrame, trades: pd.DataFrame) -> pd.Series:
    q_slim = quotes[["bid_price", "ask_price"]].rename(
        columns={"bid_price": "bid", "ask_price": "ask"})
    t_slim = trades[["price", "size"]].copy()
    if t_slim.empty or q_slim.empty: return pd.Series(dtype=float)
    combined = pd.concat([q_slim, t_slim]).sort_index()
    combined["bid"] = combined["bid"].ffill()
    combined["ask"] = combined["ask"].ffill()
    trd = combined[combined["price"].notna()].copy()
    trd["direction"] = np.where(trd["price"] >= trd["ask"],  1,
                       np.where(trd["price"] <= trd["bid"], -1, 0))
    trd["buy_vol"]  = np.where(trd["direction"] ==  1, trd["size"], 0.0)
    trd["sell_vol"] = np.where(trd["direction"] == -1, trd["size"], 0.0)
    buy_r  = trd["buy_vol"].rolling("5min").sum()
    sell_r = trd["sell_vol"].rolling("5min").sum()
    total  = buy_r + sell_r
    raw    = ((buy_r - sell_r) / total.replace(0, np.nan)).fillna(0.0)
    return raw.ewm(alpha=TFI_ALPHA, adjust=False).mean().resample("1min").last().ffill()


def compute_signals(date_str: str, spy_tr, spy_qt, qqq_tr, qqq_qt) -> pd.Series:
    d = pd.Timestamp(date_str)
    idx = pd.date_range(
        ET.localize(dt.datetime(d.year, d.month, d.day, 9, 30)),
        ET.localize(dt.datetime(d.year, d.month, d.day, 16, 0)),
        freq="1min")
    def align(s):
        return s.reindex(idx, method="ffill").fillna(0) if not s.empty else pd.Series(0.0, index=idx)
    spy_obi = align(compute_obi(spy_qt))
    qqq_obi = align(compute_obi(qqq_qt))
    spy_tfi = align(compute_tfi(spy_qt, spy_tr))
    qqq_tfi = align(compute_tfi(qqq_qt, qqq_tr))
    return (((spy_obi * 0.35 + spy_tfi * 0.65)
            + (qqq_obi * 0.35 + qqq_tfi * 0.65)) / 2).clip(-1, 1)


# ── 가격 조회 ─────────────────────────────────────────────────────────────────

def get_px(sym: str, ts, col: str, bars_dict: dict) -> float:
    df = bars_dict.get(sym)
    if df is None or df.empty: return np.nan
    before = df.index[df.index <= ts]
    if before.empty: return np.nan
    v = df.at[before[-1], col]
    return float(v) if pd.notna(v) else np.nan


def make_vwap_filter(sso_bars: pd.DataFrame):
    def ok(ts):
        before = sso_bars.index[sso_bars.index <= ts]
        if before.empty: return True
        row = sso_bars.loc[before[-1]]
        close = float(row.get("close", np.nan))
        vwap  = float(row.get("vwap",  np.nan))
        if not np.isfinite(close) or not np.isfinite(vwap): return True
        return close >= vwap
    return ok


# ── 시뮬레이션 ────────────────────────────────────────────────────────────────

def run_sim(date_str: str, sig_series: pd.Series, bars_dict: dict,
            vwap_filter_fn=None) -> tuple:
    """
    Look-ahead bias 수정: 신호 t → t+1 open 진입 (pending 패턴)
    """
    all_ts      = sig_series.index
    portfolio   = PORTFOLIO
    st = {"direction": "flat", "entry_px": {}, "qty": {},
          "pnl_peak": 0.0, "partial_done": False,
          "last_switch": None, "last_conv_exit": None,
          "entry_comp": 0.0}
    trades_log  = []
    daily_pnls  = []
    pending_sig = None

    def switch_ok(ts):
        if st["last_switch"] is None: return True
        return (ts - st["last_switch"]).total_seconds() >= SWITCH_COOLDOWN_MIN * 60

    def conv_ok(ts):
        if st["last_conv_exit"] is None: return True
        return (ts - st["last_conv_exit"]).total_seconds() >= CONVICTION_COOLDOWN * 60

    def get_open(s, ts):  return get_px(s, ts, "open",  bars_dict)
    def get_close(s, ts): return get_px(s, ts, "close", bars_dict)

    def open_pos(ts, sig):
        st["direction"] = "long"; st["entry_px"] = {}; st["qty"] = {}
        st["pnl_peak"]  = 0.0;   st["partial_done"] = False
        st["last_switch"] = ts;  st["entry_comp"] = sig
        for s in LONG_SYMS:
            px = get_open(s, ts)
            if not np.isfinite(px) or px <= 0: continue
            q = max(1, int(portfolio * KELLY / px))
            st["entry_px"][s] = px; st["qty"][s] = q
        if not st["qty"]: st["direction"] = "flat"; return
        vwap_ok = vwap_filter_fn(ts) if vwap_filter_fn else True
        trades_log.append({"time": ts.strftime("%H:%M"), "action": "ENTER",
                            "sig": round(sig, 3), "pnl": 0.0, "vwap_ok": vwap_ok})

    def close_pos(reason: str, ts, sig):
        nonlocal portfolio
        c_pnl = c_cost = 0.0
        for s in st["qty"]:
            px = get_close(s, ts)
            if not np.isfinite(px) or px <= 0: px = st["entry_px"][s]
            c_pnl  += (px - st["entry_px"][s]) * st["qty"][s]
            c_cost += st["qty"][s] * st["entry_px"][s] * SPREAD * 2
        net = c_pnl - c_cost
        portfolio += net
        daily_pnls.append(net)
        vwap_ok = vwap_filter_fn(ts) if vwap_filter_fn else True
        trades_log.append({"time": ts.strftime("%H:%M"), "action": reason,
                            "sig": round(sig, 3), "pnl": round(net, 2), "vwap_ok": vwap_ok})
        st["direction"] = "flat"; st["entry_px"] = {}; st["qty"] = {}
        st["pnl_peak"] = 0.0; st["partial_done"] = False

    for ts in all_ts:
        if not (SESSION_OPEN <= ts.time() < SESSION_CLOSE):
            pending_sig = None; continue
        sig = float(sig_series.get(ts, 0.0))
        eod = ts.time() >= EOD_TIME

        # Step 1: 이전 봉 pending → t+1 open 진입
        if pending_sig is not None and st["direction"] == "flat" and not eod:
            open_pos(ts, pending_sig)
        pending_sig = None

        # Step 2: 기존 포지션 평가 (ts close)
        if st["direction"] == "long" and st["qty"]:
            pnl   = sum((get_close(s, ts) - st["entry_px"][s]) * st["qty"][s]
                        for s in st["qty"] if np.isfinite(get_close(s, ts)))
            denom = sum(st["entry_px"][s] * st["qty"][s] for s in st["qty"])
            pct   = pnl / denom if denom else 0.0
            st["pnl_peak"] = max(st["pnl_peak"], pnl)

            stop = EOD_TIGHT_STOP    if eod else STOP_LOSS_PCT
            ptgt = EOD_TIGHT_PARTIAL if eod else PARTIAL_PCT
            tpct = EOD_TIGHT_TRAIL   if eod else TRAIL_PULLBACK

            if pct <= -stop:
                close_pos("손절", ts, sig); continue
            if sig < CONVICTION_LONG_EXIT:
                close_pos("신호소멸", ts, sig)
                st["last_conv_exit"] = ts; continue
            if not st["partial_done"] and pct >= ptgt:
                for s in st["qty"]: st["qty"][s] = max(1, st["qty"][s] // 2)
                st["partial_done"] = True; continue
            if st["partial_done"] and st["pnl_peak"] > 0:
                if (st["pnl_peak"] - pnl) / st["pnl_peak"] >= tpct:
                    close_pos("트레일링", ts, sig); continue

        # Step 3: ts 신호로 t+1 진입 예약
        if st["direction"] == "flat" and not eod:
            if MIN_CONV <= sig < MAX_CONV and switch_ok(ts) and conv_ok(ts):
                if vwap_filter_fn and not vwap_filter_fn(ts):
                    continue
                pending_sig = sig

    if st["direction"] == "long" and st["qty"]:
        close_pos("세션종료", all_ts[-1], 0.0)

    n_wins = sum(1 for p in daily_pnls if p > 0)
    return portfolio - PORTFOLIO, trades_log, n_wins, len(daily_pnls) - n_wins


# ── 결과 출력 ─────────────────────────────────────────────────────────────────

def print_day_detail(date_str: str, pnl: float, trades_log: list,
                     sig_series: pd.Series, bars_dict: dict):
    sso = bars_dict.get("SSO")
    if sso is not None and not sso.empty:
        day_open  = float(sso["open"].iloc[0])
        day_close = float(sso["close"].iloc[-1])
        sso_ret   = (day_close - day_open) / day_open * 100
    else:
        sso_ret = float("nan")

    # 신호 분포
    session = sig_series.between_time("09:45", "15:55")
    n_entry_zone = ((session >= MIN_CONV) & (session < MAX_CONV)).sum()
    sig_max = session.max(); sig_min = session.min()

    print(f"\n  {'─'*70}")
    print(f"  📅 {date_str}")
    print(f"     SSO 당일 수익률: {sso_ret:+.2f}%")
    print(f"     신호 범위: {sig_min:.3f} ~ {sig_max:.3f} | 진입구간(0.20~0.30) 분봉: {n_entry_zone}개")
    print(f"     전략 PnL: ${pnl:+.2f}")
    print(f"     거래 내역:")
    for t in trades_log:
        vwap_str = "✓VWAP" if t.get("vwap_ok", True) else "✗VWAP"
        pnl_str  = f"  {t['pnl']:+.2f}" if t["action"] != "ENTER" else ""
        print(f"       {t['time']}  {t['action']:<10}  sig={t['sig']:+.3f}  {vwap_str}{pnl_str}")

    if not trades_log:
        print(f"       (진입 없음 — VWAP 필터가 모두 차단하거나 신호 부재)")


def print_signal_profile(date_str: str, sig_series: pd.Series):
    """신호가 어떻게 분포했는지 5분 단위로 출력"""
    session = sig_series.between_time("09:45", "15:55")
    print(f"\n     신호 프로파일 (5분 샘플):")
    for ts in session.index[::5]:
        v = float(session[ts])
        bar = "█" * int(abs(v) * 20)
        flag = " ← 진입구간" if MIN_CONV <= v < MAX_CONV else ""
        print(f"       {ts.strftime('%H:%M')}  {v:+.3f}  {bar}{flag}")


def main():
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"  크래시 레짐 테스트 — 2025년 4월 타리프 쇼크")
    print(f"  전략: Long/Cash | 진입: 0.20 ≤ sig < 0.30 | VWAP 필터")
    print(f"  기준선: 5월 11일 +$175.49")
    print(f"{'='*72}")

    print(f"\n  [데이터 다운로드]")
    day_data = {}
    for date_str in CRASH_DAYS:
        print(f"\n  {date_str}:")
        try:
            spy_tr, spy_qt = load_tick("SPY", date_str, client)
            qqq_tr, qqq_qt = load_tick("QQQ", date_str, client)
            bars = {}
            for sym in LONG_SYMS:
                bars[sym] = load_bars(sym, date_str, client)
            sig = compute_signals(date_str, spy_tr, spy_qt, qqq_tr, qqq_qt)
            day_data[date_str] = {"bars": bars, "sig": sig}
            print(f"    신호 계산 완료 ({len(sig)}분봉)")
        except Exception as e:
            print(f"    ❌ {e}")

    print(f"\n\n  [전략 시뮬레이션]")
    total_pnl = 0.0

    for date_str in CRASH_DAYS:
        if date_str not in day_data:
            print(f"  {date_str}: 데이터 없음"); continue
        bars    = day_data[date_str]["bars"]
        sig     = day_data[date_str]["sig"]
        vwap_fn = make_vwap_filter(bars["SSO"]) if "SSO" in bars else None

        pnl, trades_log, n_win, n_loss = run_sim(date_str, sig, bars, vwap_fn)
        total_pnl += pnl
        print_day_detail(date_str, pnl, trades_log, sig, bars)
        if len(trades_log) == 0:
            # 신호가 아예 없으면 프로파일 출력
            print_signal_profile(date_str, sig)

    print(f"\n{'='*72}")
    print(f"  크래시 2일 합산 PnL : ${total_pnl:+.2f}")
    print(f"  5월 11일 합산 PnL   : +$175.49")
    print(f"  {'─'*50}")

    if total_pnl > -50:
        verdict = "✅ 크래시 레짐에서도 손실 제어 성공 (VWAP 필터 효과)"
    elif total_pnl > -200:
        verdict = "⚠️  소폭 손실 — 크래시에서 일부 진입이 발생하나 제한적"
    else:
        verdict = "🔴 크래시 레짐에서 큰 손실 — 추가 하락 방어 장치 필요"

    print(f"\n  판정: {verdict}")
    print(f"\n  💡 해석 포인트:")
    print(f"     - VWAP 필터: SSO < VWAP 구간(=하락장) 진입 차단 효과 확인")
    print(f"     - 크래시 시 OBI/TFI 신호가 0.20~0.30 구간에 진입했는지 여부가 핵심")
    print(f"     - 신호가 아예 발생 안 했으면: 하락장은 신호 자체가 없는 구조 → 자연적 보호")
    print(f"     - 신호가 발생 후 VWAP 차단: 필터 방어력 입증")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
