#!/usr/bin/env python3
"""
Forward Return Analysis — 신호 방향성 검증
────────────────────────────────────────────
"OBI+TFI 합성신호가 실제로 미래 가격 방향을 예측하는가?"
"강한 신호(0.30+)가 추세추종인가, exhaustion(반전) 신호인가?"

5월 전체 11거래일 캐시 데이터 사용
신호 발생 시점 기준 +1분 / +3분 / +5분 / +10분 수익률 집계
"""

import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytz, datetime as dt
from pathlib import Path

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests  import StockTradesRequest, StockQuotesRequest
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY

ET        = pytz.timezone("America/New_York")
CACHE_DIR = Path("logs/tick_cache")

MAY_TRADING_DAYS = [
    "2026-05-01", "2026-05-04", "2026-05-05", "2026-05-06",
    "2026-05-07", "2026-05-08", "2026-05-11", "2026-05-12",
    "2026-05-13", "2026-05-14", "2026-05-15",
]

OBI_ALPHA = 0.08
TFI_ALPHA = 0.15

# 신호 구간 정의 (왼쪽 포함, 오른쪽 미포함)
BINS   = [-1.0, -0.30, -0.20, -0.10, 0.10, 0.20, 0.30, 1.0]
LABELS = ["≤-0.30", "-0.30~-0.20", "-0.20~-0.10",
          "FLAT(-0.10~0.10)",
          "0.10~0.20", "0.20~0.30", "≥0.30"]

HORIZONS = [1, 3, 5, 10]   # 분 단위 forward window


# ── 데이터 로드 ─────────────────────────────────────────────────────────────

def load_tick(sym: str, date_str: str, client) -> tuple:
    cache_t = CACHE_DIR / f"{sym}_{date_str}_trades.pkl"
    cache_q = CACHE_DIR / f"{sym}_{date_str}_quotes.pkl"
    d = pd.Timestamp(date_str)
    s_et = ET.localize(dt.datetime(d.year, d.month, d.day, 9, 30))
    e_et = ET.localize(dt.datetime(d.year, d.month, d.day, 16, 0))
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


def load_bars(sym: str, date_str: str) -> pd.DataFrame:
    cache = CACHE_DIR / f"{sym}_{date_str}_bars.pkl"
    with open(cache, "rb") as f:
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


def compute_signals(date_str, spy_tr, spy_qt, qqq_tr, qqq_qt) -> pd.Series:
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
    composite = (((spy_obi * 0.35 + spy_tfi * 0.65)
                + (qqq_obi * 0.35 + qqq_tfi * 0.65)) / 2).clip(-1, 1)
    return composite


# ── Forward Return 계산 ───────────────────────────────────────────────────────

def calc_forward_returns(sig: pd.Series, sso_bars: pd.DataFrame) -> pd.DataFrame:
    """
    sig      : 분봉 composite 신호 (index = ET datetime)
    sso_bars : SSO 1분봉 (open/close)
    반환: 각 분봉별 {signal, bin, fwd_1m, fwd_3m, fwd_5m, fwd_10m, direction_correct_Xm}
    """
    rows = []
    sso_close = sso_bars["close"]

    for ts, s_val in sig.items():
        if not (dt.time(9, 45) <= ts.time() <= dt.time(15, 45)):
            continue   # 세션 범위 제한 (forward window 확보)
        if abs(s_val) < 0.01:
            continue   # 완전 flat 제외

        entry_idx = sso_close.index.searchsorted(ts, side="right") - 1
        if entry_idx < 0: continue
        entry_px = float(sso_close.iloc[entry_idx])
        if not np.isfinite(entry_px) or entry_px <= 0: continue

        row = {"ts": ts, "signal": round(float(s_val), 4),
               "bin": pd.cut([s_val], bins=BINS, labels=LABELS)[0]}

        for h in HORIZONS:
            fwd_ts = ts + pd.Timedelta(minutes=h)
            fwd_idx = sso_close.index.searchsorted(fwd_ts, side="right") - 1
            if fwd_idx < 0 or fwd_idx >= len(sso_close):
                row[f"fwd_{h}m_ret"] = np.nan
                row[f"correct_{h}m"]  = np.nan
            else:
                fwd_px = float(sso_close.iloc[fwd_idx])
                if not np.isfinite(fwd_px) or fwd_px <= 0:
                    row[f"fwd_{h}m_ret"] = np.nan
                    row[f"correct_{h}m"]  = np.nan
                else:
                    ret = (fwd_px - entry_px) / entry_px * 10_000   # bps
                    # 신호 방향 보정: 양수 신호 → SSO 상승이 "correct"
                    # 음수 신호 → SSO 하락(SDS 상승)이 "correct" → ret 부호 반전
                    signed_ret = ret * np.sign(s_val)
                    row[f"fwd_{h}m_ret"]  = round(signed_ret, 2)
                    row[f"correct_{h}m"]  = 1 if signed_ret > 0 else 0
        rows.append(row)

    return pd.DataFrame(rows)


# ── 집계 & 출력 ──────────────────────────────────────────────────────────────

def summarize(df: pd.DataFrame):
    print(f"\n{'='*80}")
    print(f"  Forward Return Analysis — 5월 전체 ({len(MAY_TRADING_DAYS)}거래일)")
    print(f"  지표: 신호 발생 후 N분 뒤 SSO 방향 정확도 + 평균 수익(bps)")
    print(f"  ※ 부호 보정 적용: 양수신호→SSO 상승, 음수신호→SSO 하락이 '정답'")
    print(f"{'='*80}")

    # 전체 분포
    print(f"\n  신호 구간별 발생 빈도:")
    bin_counts = df["bin"].value_counts().reindex(LABELS, fill_value=0)
    total = len(df)
    for b, cnt in bin_counts.items():
        bar = "█" * int(cnt / total * 40)
        print(f"  {str(b):<22}  {cnt:>5}건  {cnt/total*100:>5.1f}%  {bar}")

    # 구간별 forward return 집계
    print(f"\n  {'신호 구간':<22}  {'N':>5}  ", end="")
    for h in HORIZONS:
        print(f"  +{h}분 정확도   +{h}분 평균(bps)", end="")
    print()
    print(f"  {'─'*100}")

    g = df.groupby("bin", observed=True)

    for label in LABELS:
        if label not in g.groups: continue
        grp = g.get_group(label)
        n = len(grp)
        row = f"  {label:<22}  {n:>5}"
        for h in HORIZONS:
            acc_col = f"correct_{h}m"
            ret_col = f"fwd_{h}m_ret"
            acc = grp[acc_col].mean() * 100 if acc_col in grp else np.nan
            avg = grp[ret_col].mean()       if ret_col in grp else np.nan
            acc_str = f"{acc:>6.1f}%" if np.isfinite(acc) else "   N/A "
            avg_str = f"{avg:>+8.1f}" if np.isfinite(avg) else "     N/A"
            row += f"  {acc_str}  {avg_str}    "
        print(row)

    print(f"  {'─'*100}")

    # 양수 신호 vs 음수 신호 요약
    print(f"\n  📌 롱 신호(양수) vs 숏 신호(음수) 전체 요약:")
    for direction, mask in [("롱 신호 (sig > 0)", df["signal"] > 0),
                            ("숏 신호 (sig < 0)", df["signal"] < 0)]:
        sub = df[mask]
        print(f"\n    [{direction}]  N={len(sub)}")
        for h in HORIZONS:
            acc = sub[f"correct_{h}m"].mean() * 100
            avg = sub[f"fwd_{h}m_ret"].mean()
            flag = "✅" if acc > 52 else ("🔴" if acc < 48 else "⚠️ ")
            print(f"      +{h:>2}분:  정확도 {acc:>5.1f}%  평균 {avg:>+6.1f}bps  {flag}")

    # 핵심 진단
    print(f"\n  🔍 핵심 진단:")
    for h in HORIZONS:
        ret_col = f"fwd_{h}m_ret"
        strong_long  = df[df["signal"] >= 0.30][ret_col].mean()
        mod_long     = df[(df["signal"] >= 0.20) & (df["signal"] < 0.30)][ret_col].mean()
        weak_long    = df[(df["signal"] >= 0.10) & (df["signal"] < 0.20)][ret_col].mean()
        strong_short = df[df["signal"] <= -0.30][ret_col].mean()
        mod_short    = df[(df["signal"] <= -0.20) & (df["signal"] > -0.30)][ret_col].mean()

        if np.isfinite(strong_long) and strong_long < 0:
            verdict = f"⚠️  +{h}분: 강한 롱신호(≥0.30) 평균 {strong_long:+.1f}bps → 반전 가능성"
        elif np.isfinite(strong_long) and strong_long > 0:
            verdict = f"✅  +{h}분: 강한 롱신호(≥0.30) 평균 {strong_long:+.1f}bps → 추세 추종"
        else:
            verdict = f"❓  +{h}분: 강한 롱신호 데이터 부족"
        print(f"      {verdict}")

    print(f"\n{'='*80}\n")


def main():
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    all_frames = []

    print(f"\n  Forward Return Analysis 시작 — 5월 11거래일")
    print(f"  {'─'*50}")

    for date_str in MAY_TRADING_DAYS:
        print(f"  {date_str} 계산 중...", end=" ", flush=True)
        try:
            spy_tr, spy_qt = load_tick("SPY", date_str, client)
            qqq_tr, qqq_qt = load_tick("QQQ", date_str, client)
            sso_bars = load_bars("SSO", date_str)

            sig = compute_signals(date_str, spy_tr, spy_qt, qqq_tr, qqq_qt)
            df_day = calc_forward_returns(sig, sso_bars)
            df_day["date"] = date_str
            all_frames.append(df_day)
            print(f"✅  {len(df_day)}행")
        except Exception as e:
            print(f"❌  {e}")

    if not all_frames:
        print("  데이터 없음"); return

    full_df = pd.concat(all_frames, ignore_index=True)
    full_df.to_csv("logs/forward_return_may2026.csv", index=False)
    print(f"\n  전체 {len(full_df)}행 → logs/forward_return_may2026.csv 저장")

    summarize(full_df)


if __name__ == "__main__":
    main()
