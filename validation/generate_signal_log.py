#!/usr/bin/env python3
"""
generate_signal_log.py — tick cache → signal_log.csv 생성 (Tier 3용)
───────────────────────────────────────────────────────────────────────────────
backtest_spy_qqq.py 수정 없이, 기존 compute_obi/compute_tfi 함수를 임포트해
실제 tick 기반 OBI+TFI 신호를 추출한다.

실행:
  cd /Users/woongyeol/quant_trading

  # 기본 (1분봉 forward return)
  python validation/generate_signal_log.py --sym QQQ --out ./data/signal_log.csv

  # tick 모드 (초 단위 forward return — edge decay 검증용)
  python validation/generate_signal_log.py --sym QQQ --mode tick \
    --out ./data/signal_log_tick.csv

출력 signal_log.csv 컬럼:
  bars 모드: signal_time, symbol, signal_score, obi_raw, tfi_raw, is_signal,
             ret_1m, ret_3m, ret_5m, ret_10m
  tick 모드: signal_time, symbol, signal_score, obi_raw, tfi_raw, is_signal,
             ret_1s, ret_5s, ret_10s, ret_30s, ret_60s
"""

import argparse
import sys
import warnings
from datetime import datetime, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings("ignore")

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent                          # quant_trading/
BACKTEST  = ROOT / "backtesting" / "engine"
CACHE_DIR = ROOT / "logs" / "tick_cache"

sys.path.insert(0, str(BACKTEST))
sys.path.insert(0, str(ROOT))

# backtest_spy_qqq.py 의 CACHE_DIR는 __file__ 기준 상대경로이므로
# 실제 캐시 위치(quant_trading/logs/tick_cache)로 패치 후 임포트
import backtest_spy_qqq as _bt_module
_bt_module.CACHE_DIR = CACHE_DIR

from backtest_spy_qqq import (
    load_day,
    compute_obi,
    compute_tfi,
    OBI_ALPHA,
    TFI_ALPHA,
    SESSION_OPEN,
    SESSION_CLOSE,
)

ET = pytz.timezone("America/New_York")

# bars 모드 (분 단위)
FWD_HORIZONS_MIN  = [1, 3, 5, 10]
# tick 모드 (초 단위 — edge decay 검증)
FWD_HORIZONS_SEC  = [1, 5, 10, 30, 60]

SIGNAL_THRESHOLD  = 0.25
COOLDOWN_BARS     = 5


# ══════════════════════════════════════════════════════════════════════════════
# tick cache에서 사용 가능한 날짜 목록
# ══════════════════════════════════════════════════════════════════════════════

def get_available_dates(sym: str) -> list[str]:
    """tick_cache에서 sym의 완성된 날짜(quotes+trades+bars 모두 있는) 목록."""
    dates = set()
    for f in CACHE_DIR.glob(f"{sym}_*_quotes.pkl"):
        date_str = f.stem.replace(f"{sym}_", "").replace("_quotes", "")
        bars_ok   = (CACHE_DIR / f"{sym}_{date_str}_bars.pkl").exists()
        trades_ok = (CACHE_DIR / f"{sym}_{date_str}_trades.pkl").exists()
        if bars_ok and trades_ok:
            dates.add(date_str)
    return sorted(dates)


# ══════════════════════════════════════════════════════════════════════════════
# 하루치 신호 + forward return 추출
# ══════════════════════════════════════════════════════════════════════════════

def extract_day_signals(sym: str, date_str: str) -> list[dict]:
    """
    하루치 tick 데이터에서 1분봉 OBI/TFI 신호와 forward return을 추출.
    매매 룰 없음 — 순수 신호만 기록.
    """
    try:
        quotes, trades, bars = load_day(sym, date_str)
    except FileNotFoundError:
        return []

    # 실제 tick 기반 OBI/TFI 계산
    obi_1m = compute_obi(quotes)
    tfi_1m = compute_tfi(quotes, trades)

    # bars: ET 인덱스 (load_day에서 이미 변환됨)
    session_bars = bars[
        (bars.index.time >= SESSION_OPEN) &
        (bars.index.time <= SESSION_CLOSE)
    ].copy()

    if len(session_bars) < 15:
        return []

    # 신호를 bars 인덱스에 정렬
    obi_aligned = obi_1m.reindex(session_bars.index, method="ffill").fillna(0.0)
    tfi_aligned = tfi_1m.reindex(session_bars.index, method="ffill").fillna(0.0)
    sig_aligned = np.clip(obi_aligned * 0.35 + tfi_aligned * 0.65, -1.0, 1.0)

    closes    = session_bars["close"].values
    bar_times = session_bars.index
    n_bars    = len(session_bars)

    rows: list[dict] = []
    last_signal_bar  = -COOLDOWN_BARS

    for i in range(n_bars):
        sig     = float(sig_aligned.iloc[i])
        obi_raw = float(obi_aligned.iloc[i])
        tfi_raw = float(tfi_aligned.iloc[i])

        is_sig = sig >= SIGNAL_THRESHOLD

        # cooldown: is_signal=True 신호만 적용
        if is_sig:
            if i - last_signal_bar < COOLDOWN_BARS:
                continue
            last_signal_bar = i

        # forward return 계산 (bar close 기준, 분 단위)
        entry_close = closes[i]
        fwd = {}
        for h in FWD_HORIZONS_MIN:
            fwd_idx = i + h
            if fwd_idx < n_bars and entry_close > 0:
                fwd[f"ret_{h}m"] = round(
                    float((closes[fwd_idx] - entry_close) / entry_close), 8)
            else:
                fwd[f"ret_{h}m"] = np.nan

        rows.append({
            "signal_time":  bar_times[i],
            "symbol":       sym,
            "signal_score": round(sig,     6),
            "obi_raw":      round(obi_raw, 6),
            "tfi_raw":      round(tfi_raw, 6),
            "is_signal":    is_sig,
            **fwd,
        })

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# tick 모드: 초 단위 forward return (edge decay 검증)
# ══════════════════════════════════════════════════════════════════════════════

def extract_day_signals_tick(sym: str, date_str: str) -> list[dict]:
    """
    tick 모드: 1분봉 신호 기준, trade 체결가로 초 단위 forward return 계산.

    신호 발생 시점 = 해당 bar의 마지막 trade 체결가 (bar close 근사)
    Forward return:
      +1s:  [ts, ts+1s] 마지막 체결가
      +5s:  [ts, ts+5s] 마지막 체결가
      +10s: [ts, ts+10s] 마지막 체결가
      +30s: [ts, ts+30s] 마지막 체결가
      +60s: [ts, ts+60s] 마지막 체결가

    lookup: searchsorted O(log n) — 하루 100만 건 이상에서도 고속
    """
    try:
        quotes, trades, bars = load_day(sym, date_str)
    except FileNotFoundError:
        return []

    # 실제 tick 기반 OBI/TFI
    obi_1m = compute_obi(quotes)
    tfi_1m = compute_tfi(quotes, trades)

    session_bars = bars[
        (bars.index.time >= SESSION_OPEN) &
        (bars.index.time <= SESSION_CLOSE)
    ].copy()
    if len(session_bars) < 15:
        return []

    obi_aligned = obi_1m.reindex(session_bars.index, method="ffill").fillna(0.0)
    tfi_aligned = tfi_1m.reindex(session_bars.index, method="ffill").fillna(0.0)
    sig_aligned = np.clip(obi_aligned * 0.35 + tfi_aligned * 0.65, -1.0, 1.0)

    # trades: ET 변환 후 numpy 배열로 (searchsorted용)
    trades_et   = trades.copy()
    trades_et.index = trades_et.index.tz_convert(ET)
    trade_ts_ns = trades_et.index.view("int64")   # nanoseconds
    trade_px    = trades_et["price"].values

    def last_price_at(ts_ns: int, delta_ns: int) -> float:
        """[ts_ns, ts_ns+delta_ns] 구간의 마지막 trade 체결가."""
        lo = int(np.searchsorted(trade_ts_ns, ts_ns,            side="left"))
        hi = int(np.searchsorted(trade_ts_ns, ts_ns + delta_ns, side="right"))
        if hi <= lo:
            return np.nan
        return float(trade_px[hi - 1])

    NS = 1_000_000_000  # 1초 = 1e9 나노초

    rows: list[dict] = []
    last_signal_bar  = -COOLDOWN_BARS

    for i in range(len(session_bars)):
        sig     = float(sig_aligned.iloc[i])
        obi_raw = float(obi_aligned.iloc[i])
        tfi_raw = float(tfi_aligned.iloc[i])
        is_sig  = sig >= SIGNAL_THRESHOLD

        if is_sig:
            if i - last_signal_bar < COOLDOWN_BARS:
                continue
            last_signal_bar = i

        bar_ts    = session_bars.index[i]
        bar_ts_ns = int(bar_ts.value)

        # 진입가: bar timestamp 직전까지의 마지막 trade
        entry_px = last_price_at(bar_ts_ns - NS, NS)
        if np.isnan(entry_px) or entry_px <= 0:
            continue

        fwd = {}
        for h_sec in FWD_HORIZONS_SEC:
            fwd_px = last_price_at(bar_ts_ns, h_sec * NS)
            if not np.isnan(fwd_px) and fwd_px > 0:
                fwd[f"ret_{h_sec}s"] = round((fwd_px - entry_px) / entry_px, 8)
            else:
                fwd[f"ret_{h_sec}s"] = np.nan

        rows.append({
            "signal_time":  bar_ts,
            "symbol":       sym,
            "signal_score": round(sig,     6),
            "obi_raw":      round(obi_raw, 6),
            "tfi_raw":      round(tfi_raw, 6),
            "is_signal":    is_sig,
            **fwd,
        })

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

def generate(sym: str, out_path: str, start: str = None, end: str = None,
             mode: str = "bars"):
    dates = get_available_dates(sym)
    if not dates:
        print(f"오류: {CACHE_DIR} 에서 {sym} tick cache를 찾을 수 없습니다.")
        sys.exit(1)

    if start:
        dates = [d for d in dates if d >= start]
    if end:
        dates = [d for d in dates if d <= end]

    if not dates:
        print(f"오류: {start} ~ {end} 범위에 해당하는 날짜 없음")
        sys.exit(1)

    mode_label = "tick (초 단위 forward return)" if mode == "tick" else "bars (분 단위 forward return)"
    print(f"  {sym} 신호 추출: {len(dates)}일 ({dates[0]} ~ {dates[-1]})")
    print(f"  모드: {mode_label}")
    print(f"  tick cache 경로: {CACHE_DIR}")
    print(f"  출력: {out_path}")
    print()

    extractor = extract_day_signals_tick if mode == "tick" else extract_day_signals

    all_rows: list[dict] = []
    for i, date_str in enumerate(dates):
        rows = extractor(sym, date_str)
        all_rows.extend(rows)
        n_sig = sum(1 for r in rows if r["is_signal"])
        print(f"  [{i+1:3d}/{len(dates)}] {date_str}  "
              f"rows={len(rows):>4,}  signal={n_sig:>3,}", end="\r")

    print()  # 줄 정리

    if not all_rows:
        print("오류: 추출된 신호 없음")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    n_sig  = int(df["is_signal"].sum())
    n_base = int((~df["is_signal"]).sum())

    print(f"\n  완료: {out_path}")
    print(f"  total={len(df):,}  is_signal={n_sig:,}  non-signal={n_base:,}")
    print(f"  기간: {df['signal_time'].min()} ~ {df['signal_time'].max()}")
    print(f"  score 범위: {df['signal_score'].min():.4f} ~ "
          f"{df['signal_score'].max():.4f}")
    return df


def _build_parser():
    p = argparse.ArgumentParser(description="tick cache → signal_log.csv")
    p.add_argument("--sym",   default="QQQ",  help="종목 (기본: QQQ)")
    p.add_argument("--mode",  default="bars", choices=["bars", "tick"],
                   help="bars: 분 단위 forward return / tick: 초 단위 (edge decay 검증)")
    p.add_argument("--start", default=None,   help="시작일 YYYY-MM-DD")
    p.add_argument("--end",   default=None,   help="종료일 YYYY-MM-DD")
    p.add_argument("--out",   default=str(ROOT / "data" / "signal_log.csv"))
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    generate(args.sym, args.out, args.start, args.end, mode=args.mode)
