#!/usr/bin/env python3
"""
5/07 하락장 분석 + VWAP Regime 필터 전략 테스트
────────────────────────────────────────────────
1. 2026-05-07 분봉 시각화: 신호 / SSO 가격 / VWAP / 진입 포인트
2. VWAP 필터 효과: "SSO 가격 < VWAP → 롱 진입 거부"
3. VWAP 필터 ON/OFF 5월 전체 비교
"""

import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pytz, datetime as dt
from pathlib import Path

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests   import StockTradesRequest, StockQuotesRequest
from config.settings import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY,
    STOP_LOSS_PCT, PARTIAL_PCT, TRAIL_PULLBACK,
    CONVICTION_LONG_EXIT, ENTRY_MIN_CONVICTION,
    EOD_TIGHT_STOP, EOD_TIGHT_TRAIL, EOD_TIGHT_PARTIAL,
    SWITCH_COOLDOWN_MIN, CONVICTION_COOLDOWN,
)

ET        = pytz.timezone("America/New_York")
CACHE_DIR = Path("logs/tick_cache")
LOG_DIR   = Path("logs")

MAY_TRADING_DAYS = [
    "2026-05-01", "2026-05-04", "2026-05-05", "2026-05-06",
    "2026-05-07", "2026-05-08", "2026-05-11", "2026-05-12",
    "2026-05-13", "2026-05-14", "2026-05-15",
]

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


# ── 공통 로더 ────────────────────────────────────────────────────────────────

def load_tick(sym, date_str, client):
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
        df = req_fn()
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(sym, level="symbol")
        df.index = pd.DatetimeIndex(df.index).tz_convert(ET)
        with open(cache, "wb") as f: pickle.dump(df, f)
        return df

    trades = _load(cache_t, lambda: client.get_stock_trades(
        StockTradesRequest(symbol_or_symbols=sym, start=s_utc, end=e_utc, feed="sip")).df)
    quotes = _load(cache_q, lambda: client.get_stock_quotes(
        StockQuotesRequest(symbol_or_symbols=sym, start=s_utc, end=e_utc, feed="sip")).df)
    return trades, quotes


def load_bars(sym, date_str):
    with open(CACHE_DIR / f"{sym}_{date_str}_bars.pkl", "rb") as f:
        bars = pickle.load(f)
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(sym, level="symbol")
    bars.index = pd.DatetimeIndex(bars.index)
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize(ET)
    else:
        bars.index = bars.index.tz_convert(ET)
    d = pd.Timestamp(date_str).date()
    return bars[bars.index.date == d].between_time("09:30", "16:00")


# ── 신호 계산 ────────────────────────────────────────────────────────────────

def compute_obi(quotes):
    q = quotes[["bid_size", "ask_size"]].copy()
    q = q[(q["bid_size"] > 0) & (q["ask_size"] > 0)]
    if q.empty: return pd.Series(dtype=float)
    raw = (q["bid_size"] - q["ask_size"]) / (q["bid_size"] + q["ask_size"])
    return raw.ewm(alpha=OBI_ALPHA, adjust=False).mean().resample("1min").last().ffill()


def compute_tfi(quotes, trades):
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


def compute_signals(date_str, spy_tr, spy_qt, qqq_tr, qqq_qt):
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


# ── 가격 조회 ────────────────────────────────────────────────────────────────

def get_px(sym, ts, col, bars_dict):
    df = bars_dict.get(sym)
    if df is None or df.empty: return np.nan
    before = df.index[df.index <= ts]
    if before.empty: return np.nan
    v = df.at[before[-1], col]
    return float(v) if pd.notna(v) else np.nan


# ── VWAP 레짐 필터 ────────────────────────────────────────────────────────────

def make_vwap_filter(sso_bars: pd.DataFrame):
    """ts → bool: SSO 종가 >= VWAP이면 True (롱 진입 허용)"""
    def ok(ts):
        before = sso_bars.index[sso_bars.index <= ts]
        if before.empty: return True
        row = sso_bars.loc[before[-1]]
        close = float(row.get("close", np.nan))
        vwap  = float(row.get("vwap",  np.nan))
        if not np.isfinite(close) or not np.isfinite(vwap): return True
        return close >= vwap
    return ok


# ── 시뮬레이션 ───────────────────────────────────────────────────────────────

def run_sim(date_str, sig_series, bars_dict,
            vwap_filter_fn=None,
            use_dcomp: bool = False,
            persist_bars: int = 1) -> tuple[float, list, int, int]:
    """
    Look-ahead bias 수정 적용:
      신호 t → t+1 open 진입 (pending_entry 패턴)
      평가/청산은 현재 봉 close 사용 (진입은 이전 봉에서 이미 완료)

    vwap_filter_fn : ts → bool (None이면 미사용)
    use_dcomp      : True → d_comp > 0 필터
    persist_bars   : N → 직전 N분간 신호 유지 시만 진입 (기본 1 = 비활성)
    """
    all_ts        = sig_series.index
    d_comp_series = sig_series.diff().fillna(0)
    portfolio     = PORTFOLIO
    st = {"direction": "flat", "entry_px": {}, "qty": {},
          "pnl_peak": 0.0, "partial_done": False,
          "last_switch": None, "last_conv_exit": None,
          "entry_comp": 0.0}
    trades_log    = []
    daily_pnls    = []
    pending_sig   = None   # t 봉에서 조건 충족 → t+1 open 진입 대기

    def switch_ok(ts):
        if st["last_switch"] is None: return True
        return (ts - st["last_switch"]).total_seconds() >= SWITCH_COOLDOWN_MIN * 60

    def conv_ok(ts):
        if st["last_conv_exit"] is None: return True
        return (ts - st["last_conv_exit"]).total_seconds() >= CONVICTION_COOLDOWN * 60

    def get_open(s, ts):  return get_px(s, ts, "open",  bars_dict)
    def get_close(s, ts): return get_px(s, ts, "close", bars_dict)

    def open_pos(ts, sig):
        """ts+1 봉의 open으로 진입 (pending에서 호출)"""
        st["direction"] = "long"
        st["entry_px"]  = {}; st["qty"] = {}
        st["pnl_peak"]  = 0.0; st["partial_done"] = False
        st["last_switch"] = ts; st["entry_comp"] = sig
        for s in LONG_SYMS:
            px = get_open(s, ts)
            if not np.isfinite(px) or px <= 0: continue
            q = max(1, int(portfolio * KELLY / px))
            st["entry_px"][s] = px; st["qty"][s] = q
        if not st["qty"]: st["direction"] = "flat"; return
        trades_log.append({"time": ts.strftime("%H:%M"), "action": "ENTER",
                            "sig": round(sig, 3), "pnl": 0.0,
                            "vwap_ok": vwap_filter_fn(ts) if vwap_filter_fn else True})

    def close_pos(reason, ts, sig):
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
        trades_log.append({"time": ts.strftime("%H:%M"), "action": reason,
                            "sig": round(sig, 3), "pnl": round(net, 2),
                            "vwap_ok": vwap_filter_fn(ts) if vwap_filter_fn else True})
        st["direction"] = "flat"; st["entry_px"] = {}; st["qty"] = {}
        st["pnl_peak"] = 0.0; st["partial_done"] = False

    for ts in all_ts:
        if not (SESSION_OPEN <= ts.time() < SESSION_CLOSE):
            pending_sig = None   # 세션 외 구간에서 pending 파기
            continue
        sig = float(sig_series.get(ts, 0.0))
        eod = ts.time() >= EOD_TIME

        # ── Step 1: 이전 봉 신호로 발생한 pending 진입 실행 (t+1 open) ──
        if pending_sig is not None and st["direction"] == "flat" and not eod:
            open_pos(ts, pending_sig)
        pending_sig = None

        # ── Step 2: 기존 포지션 평가 (ts close — 이전 봉 open 진입 후 완전한 봉) ──
        if st["direction"] == "long" and st["qty"]:
            pnl   = sum((get_close(s, ts) - st["entry_px"][s]) * st["qty"][s]
                        for s in st["qty"]
                        if np.isfinite(get_close(s, ts)))
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

        # ── Step 3: ts 신호로 진입 조건 검사 → 다음 봉(t+1) 진입 예약 ──
        if st["direction"] == "flat" and not eod:
            if MIN_CONV <= sig < MAX_CONV and switch_ok(ts) and conv_ok(ts):
                if vwap_filter_fn and not vwap_filter_fn(ts):
                    continue
                if use_dcomp and d_comp_series.get(ts, 0) <= 0:
                    continue
                if persist_bars > 1:
                    ts_pos = all_ts.get_loc(ts)
                    if ts_pos < persist_bars - 1:
                        continue
                    recent = [float(sig_series.iloc[ts_pos - k])
                              for k in range(persist_bars)]
                    if not all(MIN_CONV <= v < MAX_CONV for v in recent):
                        continue
                pending_sig = sig   # t+1 봉 open에서 실행

    if st["direction"] == "long" and st["qty"]:
        close_pos("세션종료", all_ts[-1], 0.0)

    n_wins = sum(1 for p in daily_pnls if p > 0)
    return portfolio - PORTFOLIO, trades_log, n_wins, len(daily_pnls) - n_wins


# ── 5/07 상세 시각화 ─────────────────────────────────────────────────────────

def plot_0507(sig_series, sso_bars, trades_no, trades_vwap):
    session = sig_series.between_time("09:45", "15:55")
    sso_close = sso_bars["close"].reindex(session.index, method="ffill")
    sso_vwap  = sso_bars["vwap"].reindex(session.index, method="ffill")

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("2026-05-07 분석: 하락장에서 VWAP 필터 효과", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 1, hspace=0.35)

    # ① SSO 가격 + VWAP
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(sso_close.index, sso_close.values, color="#2196F3", lw=1.5, label="SSO close")
    ax1.plot(sso_vwap.index,  sso_vwap.values,  color="#FF9800", lw=1.2, ls="--", label="SSO VWAP")
    ax1.fill_between(sso_close.index,
                     sso_close.values, sso_vwap.values,
                     where=(sso_close.values < sso_vwap.values),
                     alpha=0.12, color="red", label="Price < VWAP (bearish)")

    # 진입 마커 (필터 없음)
    for t in trades_no:
        if t["action"] == "ENTER":
            ts = pd.Timestamp(f"2026-05-07 {t['time']}").tz_localize(ET)
            px = sso_close.asof(ts)
            ax1.axvline(ts, color="red", alpha=0.3, lw=1)
            ax1.scatter(ts, px, color="red", s=60, zorder=5, marker="^")
    # VWAP 필터 진입 마커
    for t in trades_vwap:
        if t["action"] == "ENTER":
            ts = pd.Timestamp(f"2026-05-07 {t['time']}").tz_localize(ET)
            px = sso_close.asof(ts)
            ax1.scatter(ts, px, color="green", s=80, zorder=6, marker="^",
                        edgecolors="darkgreen", label="_")
    ax1.set_ylabel("SSO ($)", fontsize=10)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("SSO 가격 vs VWAP  |  빨간 삼각=기존 진입  초록 삼각=VWAP 허용 진입", fontsize=9)

    # ② 합성 신호
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(session.index, session.values, color="#9C27B0", lw=1.2)
    ax2.axhline(MIN_CONV, color="green",  ls="--", lw=0.8, alpha=0.7, label=f"MIN {MIN_CONV}")
    ax2.axhline(MAX_CONV, color="red",    ls="--", lw=0.8, alpha=0.7, label=f"MAX {MAX_CONV}")
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.fill_between(session.index, session.values, MIN_CONV,
                     where=(session.values >= MIN_CONV) & (session.values < MAX_CONV),
                     alpha=0.2, color="green", label="진입 가능 구간")
    ax2.set_ylabel("Composite Signal", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("합성 신호 (0.20~0.30 진입 구간)", fontsize=9)

    # ③ VWAP 대비 이격 (%)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    spread_pct = (sso_close - sso_vwap) / sso_vwap * 100
    ax3.bar(spread_pct.index, spread_pct.values,
            color=np.where(spread_pct.values >= 0, "#4CAF50", "#F44336"),
            width=pd.Timedelta("50s"), alpha=0.7)
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_ylabel("(Close - VWAP) / VWAP (%)", fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_title("SSO 가격의 VWAP 이격도  |  음수 구간 = 하락 레짐", fontsize=9)

    plt.tight_layout()
    path = LOG_DIR / "regime_0507_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  ✅ 차트 저장: {path}")
    plt.close()


# ── 전체 5월 필터 A/B/C 비교 ─────────────────────────────────────────────────

FILTER_CONFIGS = [
    # (label,              vwap,  dcomp, persist)
    ("A. VWAP only",       True,  False, 1),   # 현재 기준선
    ("B. VWAP + d_comp>0", True,  True,  1),   # 신호 가속 필터
    ("C. VWAP + 2-bar",    True,  False, 2),   # 2분 지속 필터
]


def run_full_may(client):
    print(f"\n{'='*80}")
    print(f"  5월 전체 필터 비교 — Long-only 0.20~0.30")
    print(f"  A: VWAP only (기준선)  B: +d_comp>0  C: +2-bar persistence")
    print(f"  ※ 자유파라미터 없음 — 과적합 방지")
    print(f"{'='*80}")

    # 날짜별 신호/바 사전 계산 (1회)
    day_cache = {}
    for date_str in MAY_TRADING_DAYS:
        try:
            spy_tr, spy_qt = load_tick("SPY", date_str, client)
            qqq_tr, qqq_qt = load_tick("QQQ", date_str, client)
            bars = {s: load_bars(s, date_str) for s in LONG_SYMS}
            sig  = compute_signals(date_str, spy_tr, spy_qt, qqq_tr, qqq_qt)
            day_cache[date_str] = {"bars": bars, "sig": sig}
        except Exception as e:
            print(f"  {date_str} 로드 실패: {e}")

    # 설정별 실행
    results = {lbl: [] for lbl, *_ in FILTER_CONFIGS}

    for date_str in MAY_TRADING_DAYS:
        if date_str not in day_cache: continue
        bars = day_cache[date_str]["bars"]
        sig  = day_cache[date_str]["sig"]
        vwap_fn = make_vwap_filter(bars["SSO"])

        for lbl, use_vwap, use_dc, persist in FILTER_CONFIGS:
            fn = vwap_fn if use_vwap else None
            pnl, _, w, l = run_sim(date_str, sig, bars,
                                   vwap_filter_fn=fn,
                                   use_dcomp=use_dc,
                                   persist_bars=persist)
            results[lbl].append({"date": date_str, "pnl": pnl, "w": w, "l": l})

    # 출력
    labels = [lbl for lbl, *_ in FILTER_CONFIGS]
    hdr = f"  {'날짜':<12}"
    for lbl in labels:
        hdr += f"  {lbl[:16]:>18}"
    print(hdr)
    print(f"  {'─'*76}")

    totals = {lbl: 0.0 for lbl in labels}
    tw = {lbl: 0 for lbl in labels}
    tl = {lbl: 0 for lbl in labels}

    for i, date_str in enumerate([d for d in MAY_TRADING_DAYS if d in day_cache]):
        row = f"  {date_str:<12}"
        for lbl in labels:
            r = results[lbl][i]
            row += f"  ${r['pnl']:>+8.2f}({r['w']}승/{r['l']}패)"
            totals[lbl] += r["pnl"]
            tw[lbl]     += r["w"]
            tl[lbl]     += r["l"]
        print(row)

    print(f"  {'─'*76}")

    # 요약
    print(f"\n  {'설정':<22}  {'총 PnL':>10}  {'수익률':>9}  {'승률':>8}  "
          f"{'거래':>5}  {'흑자일':>7}  {'기준선 대비':>10}")
    print(f"  {'─'*76}")
    base_pnl = totals[labels[0]]
    for lbl in labels:
        nw = tw[lbl]; nl = tl[lbl]
        wr = nw/(nw+nl)*100 if (nw+nl) else 0
        n_green = sum(1 for r in results[lbl] if r["pnl"] > 0)
        vs_base = totals[lbl] - base_pnl
        flag = " ★" if totals[lbl] == max(totals.values()) else ""
        vs_str = f"{vs_base:>+.2f}" if lbl != labels[0] else "—"
        print(f"  {lbl:<22}  ${totals[lbl]:>+9.2f}  "
              f"{totals[lbl]/PORTFOLIO*100:>+8.4f}%  {wr:>7.1f}%  "
              f"{nw+nl:>5}  {n_green}/11일  ${vs_str:>9}{flag}")

    # 과적합 경고
    print(f"\n  ⚠️  과적합 주의사항:")
    print(f"     - 11거래일은 통계적으로 부족 (표준오차 큼)")
    print(f"     - 필터 추가 시 거래 수 감소 → 분산 증가")
    print(f"     - 다음 검증: 하락장(2025-04-03/04) 등 다른 레짐 테스트 필요")
    print(f"{'='*80}\n")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    # ── 5/07 상세 분석 ────────────────────────────────────────────────────────
    print("\n  [1] 2026-05-07 상세 분석")
    DATE = "2026-05-07"
    spy_tr, spy_qt = load_tick("SPY", DATE, client)
    qqq_tr, qqq_qt = load_tick("QQQ", DATE, client)
    bars_07 = {s: load_bars(s, DATE) for s in LONG_SYMS}
    sig_07  = compute_signals(DATE, spy_tr, spy_qt, qqq_tr, qqq_qt)

    pnl_no,  t_no,  w_no,  l_no  = run_sim(DATE, sig_07, bars_07, None)
    vwap_fn = make_vwap_filter(bars_07["SSO"])
    pnl_vw,  t_vw,  w_vw,  l_vw  = run_sim(DATE, sig_07, bars_07, vwap_fn)

    print(f"\n  ── 2026-05-07 거래 로그 (VWAP 필터 OFF) ──")
    print(f"  {'시간':<7}  {'행동':<10}  {'신호':>8}  {'PnL':>9}")
    print(f"  {'─'*40}")
    for t in t_no:
        print(f"  {t['time']:<7}  {t['action']:<10}  {t['sig']:>+8.3f}  "
              f"${t['pnl']:>+8.2f}" if t["pnl"] != 0 else
              f"  {t['time']:<7}  {t['action']:<10}  {t['sig']:>+8.3f}  {'—':>9}")
    print(f"  {'─'*40}")
    print(f"  VWAP 필터 OFF:  PnL ${pnl_no:>+8.2f}  ({w_no}승/{l_no}패)")

    print(f"\n  ── 2026-05-07 거래 로그 (VWAP 필터 ON) ──")
    print(f"  {'시간':<7}  {'행동':<10}  {'신호':>8}  {'PnL':>9}")
    print(f"  {'─'*40}")
    for t in t_vw:
        print(f"  {t['time']:<7}  {t['action']:<10}  {t['sig']:>+8.3f}  "
              f"${t['pnl']:>+8.2f}" if t["pnl"] != 0 else
              f"  {t['time']:<7}  {t['action']:<10}  {t['sig']:>+8.3f}  {'—':>9}")
    print(f"  {'─'*40}")
    print(f"  VWAP 필터 ON:   PnL ${pnl_vw:>+8.2f}  ({w_vw}승/{l_vw}패)")
    print(f"  개선폭: ${pnl_vw - pnl_no:>+.2f}")

    # 차트
    plot_0507(sig_07, bars_07["SSO"], t_no, t_vw)

    # ── 5월 전체 A/B/C 필터 비교 ─────────────────────────────────────────────
    print("\n  [2] 5월 전체 필터 A/B/C 비교")
    run_full_may(client)


if __name__ == "__main__":
    main()
