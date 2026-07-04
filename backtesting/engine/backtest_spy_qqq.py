#!/usr/bin/env python3
"""
SPY Live Trader 백테스터 — 틱 데이터 완전 재현
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
spy_qqq_live_trader.py 의 신호·실행 로직을 1:1 재현.

데이터 (로컬 캐시):
  logs/tick_cache/SPY_YYYY-MM-DD_{quotes|trades|bars}.pkl

신호 계산 (벡터화):
  TFI : trades + Lee-Ready + 5분 롤링 → ewm(α=0.15)
  Signal = clip(TFI, -1, 1) → 1분봉 마지막 값  [OBI 제거됨 — Tier3 FAIL]

실행 가정:
  진입   : 신호 예약(t) → 다음 bar(t+1).open 가격
  손절   : bar.low 가 stop 가격 이하 → stop 가격 체결
  로컬저점: bar.low 가 local_min_stop 이하 → local_min_stop 가격 체결
  부분익절: bar.high 가 partial 가격 이상 → partial 가격 체결
  신호소멸: bar.close 기준
  트레일링: bar.close 기준 (peak → drawdown)
  EOD 15:45: bar.open 에서 50% 분할청산
  EOD 15:55: bar.close 에서 잔여 전량 청산
  슬리피지: SPREAD_BPS × 2 (왕복) 추가 비용

ATR 스케일:
  yfinance 일봉 → ATR14 / 30일 평균 → clip(0.6, 2.0)
"""

import os, sys, pickle, warnings
from datetime import datetime, time as dtime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pytz
import yfinance as yf

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

ET        = pytz.timezone("America/New_York")
CACHE_DIR = Path(__file__).parent.parent.parent / "logs" / "tick_cache"
OUT_DIR   = Path(__file__).parent.parent.parent / "logs"

# ── 전략 파라미터 (spy_qqq_live_trader.py 와 완전 동일) ────────────────────────
INITIAL_CAPITAL      = 100_000.0
KELLY                = 0.025
STOP_LOSS_PCT        = 0.010
PARTIAL_PCT          = 0.008
TRAIL_PULLBACK       = 0.30
CONVICTION_LONG_EXIT    = -0.10
MIN_CONV                = 0.20
MIN_LOCAL_STOP_DIST     = 0.003   # 로컬저점 최소 거리: 진입가 대비 0.3% 이상이어야 활성화
SWITCH_COOLDOWN_MIN  = 15
CONVICTION_COOLDOWN  = 5
SPREAD_BPS           = 0.5
DAILY_TURNOVER_LIMIT = 0.50
OBI_ALPHA            = 0.08
TFI_ALPHA            = 0.15
TFI_WINDOW           = "5min"

SESSION_OPEN    = dtime(9, 45)
SESSION_CLOSE   = dtime(15, 55)
EOD_TIGHT_TIME  = dtime(14, 55)
EOD_REDUCE_TIME = dtime(15, 45)

ATR_PERIOD   = 14
ATR_LOOKBACK = 30
ATR_MIN      = 0.6
ATR_MAX      = 2.0


# ══════════════════════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ══════════════════════════════════════════════════════════════════════════════

def load_day(sym: str, date_str: str):
    """(quotes_df, trades_df, bars_df) 반환. bars는 UTC→ET 변환."""
    def _load(tag):
        p = CACHE_DIR / f"{sym}_{date_str}_{tag}.pkl"
        if not p.exists():
            raise FileNotFoundError(f"캐시 없음: {p}")
        with open(p, "rb") as f:
            return pickle.load(f)

    quotes = _load("quotes")
    trades = _load("trades")
    bars   = _load("bars")

    # bars 인덱스: UTC → ET
    bars.index = bars.index.tz_convert(ET)
    return quotes, trades, bars


# ══════════════════════════════════════════════════════════════════════════════
# 2. OBI / TFI 벡터 계산 (live trader 와 동일 로직)
# ══════════════════════════════════════════════════════════════════════════════

def compute_obi(quotes: pd.DataFrame) -> pd.Series:
    q = quotes[["bid_size", "ask_size"]].copy()
    q = q[(q["bid_size"] > 0) & (q["ask_size"] > 0)]
    raw   = (q["bid_size"] - q["ask_size"]) / (q["bid_size"] + q["ask_size"])
    ticks = raw.ewm(alpha=OBI_ALPHA, adjust=False).mean()
    return ticks.resample("1min").last().ffill()


def compute_tfi(quotes: pd.DataFrame, trades: pd.DataFrame) -> pd.Series:
    q_slim = quotes[["bid_price", "ask_price"]].rename(
        columns={"bid_price": "bid", "ask_price": "ask"})
    t_slim = trades[["price", "size"]].copy()

    combined = pd.concat([q_slim, t_slim]).sort_index()
    combined["bid"] = combined["bid"].ffill()
    combined["ask"] = combined["ask"].ffill()

    trd = combined[combined["price"].notna()].copy()
    trd["direction"] = np.where(
        trd["price"] >= trd["ask"],  1,
        np.where(trd["price"] <= trd["bid"], -1, 0))
    trd["buy_vol"]  = np.where(trd["direction"] ==  1, trd["size"], 0.0)
    trd["sell_vol"] = np.where(trd["direction"] == -1, trd["size"], 0.0)

    buy_r  = trd["buy_vol"].rolling(TFI_WINDOW).sum()
    sell_r = trd["sell_vol"].rolling(TFI_WINDOW).sum()
    total  = (buy_r + sell_r).replace(0, np.nan)
    raw    = ((buy_r - sell_r) / total).fillna(0.0)
    ticks  = raw.ewm(alpha=TFI_ALPHA, adjust=False).mean()
    return ticks.resample("1min").last().ffill()


def compute_signals(sym: str, date_str: str) -> pd.Series:
    """1분봉 신호 Series (index = ET timezone) 반환.
    TFI 단독 사용 — OBI는 alpha=0.08 per-tick EWM이 QQQ 실제 틱레이트(~264/sec)에서
    half-life ~31ms로 사실상 노이즈가 되어 제외 (Tier3 검증 FAIL, 2026-06-30 재확인)."""
    quotes, trades, _ = load_day(sym, date_str)

    tfi = compute_tfi(quotes, trades)

    d     = pd.Timestamp(date_str)
    idx   = pd.date_range(
        ET.localize(datetime(d.year, d.month, d.day, 9, 30)),
        ET.localize(datetime(d.year, d.month, d.day, 16, 0)),
        freq="1min",
    )
    tfi = tfi.reindex(idx, method="ffill").fillna(0.0)
    sig = np.clip(tfi, -1.0, 1.0)
    return sig


# ══════════════════════════════════════════════════════════════════════════════
# 3. ATR 스케일 (yfinance 일봉)
# ══════════════════════════════════════════════════════════════════════════════

_atr_cache: dict[str, pd.DataFrame] = {}

def _get_daily_bars(sym: str) -> pd.DataFrame:
    if sym not in _atr_cache:
        df = yf.download(sym, period="120d", interval="1d",
                         auto_adjust=True, progress=False)
        _atr_cache[sym] = df
    return _atr_cache[sym]


def get_atr_scale(sym: str, date_str: str) -> float:
    df  = _get_daily_bars(sym).copy()
    cut = pd.Timestamp(date_str).tz_localize(None)
    df  = df[df.index < cut]          # 해당일 이전 데이터만 사용
    if len(df) < ATR_PERIOD + 5:
        return 1.0

    high, low, close = df["High"], df["Low"], df["Close"]
    prev_c = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_c).abs(),
        (low  - prev_c).abs(),
    ], axis=1).max(axis=1)
    atr_series  = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    current_atr = float(atr_series.iloc[-1])
    avg_atr     = float(atr_series.iloc[-ATR_LOOKBACK:].mean())
    if avg_atr <= 0:
        return 1.0
    return float(np.clip(current_atr / avg_atr, ATR_MIN, ATR_MAX))


# ══════════════════════════════════════════════════════════════════════════════
# 4. 로컬 저점 계산 (live trader latest_local_min 동일)
# ══════════════════════════════════════════════════════════════════════════════

def latest_local_min(bar_lows: list[float]) -> float:
    lows = [l for l in bar_lows if np.isfinite(l) and l > 0]
    if len(lows) < 3:
        return 0.0
    for i in range(len(lows) - 2, 0, -1):
        if lows[i] <= lows[i - 1] and lows[i] <= lows[i + 1]:
            return lows[i]
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 5. 포지션 상태
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    direction:      str                    = "flat"
    entry_px:       float                  = 0.0
    qty:            int                    = 0
    pnl_peak:       float                  = 0.0
    partial_done:   bool                   = False
    eod_reduced:    bool                   = False
    local_min_stop: float                  = 0.0
    pending_sig:    Optional[float]        = None
    last_switch:    Optional[pd.Timestamp] = None
    last_conv_exit: Optional[pd.Timestamp] = None
    entry_bar_idx:  int                    = -1   # 시간 기반 exit용

    def reset(self):
        self.direction = "flat"; self.entry_px = 0.0; self.qty = 0
        self.pnl_peak  = 0.0;  self.partial_done = False
        self.local_min_stop = 0.0; self.pending_sig = None
        self.last_switch = None; self.entry_bar_idx = -1

    def switch_ok(self, ts: pd.Timestamp) -> bool:
        if self.last_switch is None: return True
        return (ts - self.last_switch).total_seconds() >= SWITCH_COOLDOWN_MIN * 60

    def conv_ok(self, ts: pd.Timestamp) -> bool:
        if self.last_conv_exit is None: return True
        return (ts - self.last_conv_exit).total_seconds() >= CONVICTION_COOLDOWN * 60


# ══════════════════════════════════════════════════════════════════════════════
# 6. 1일 시뮬레이션
# ══════════════════════════════════════════════════════════════════════════════

def simulate_day(
    sym:              str,
    date_str:         str,
    bars:             pd.DataFrame,
    signals:          pd.Series,
    atr_scale:        float,
    pos:              Position,
    equity:           float,
    daily_notional:   float,
    bar_low_history:  list[float],
) -> tuple[list[dict], float, float]:
    """
    하루치 1분봉을 순차 시뮬레이션.
    반환: (trades_list, updated_equity, updated_daily_notional)
    """
    trades = []

    bars_et = bars[
        (bars.index.time >= dtime(9, 30)) &
        (bars.index.time <= dtime(16, 0))
    ].copy()

    # 신호 인덱스를 bars 인덱스에 맞춤
    sig_aligned = signals.reindex(bars_et.index, method="ffill").fillna(0.0)
    bar_list    = list(bars_et.iterrows())

    for i, (ts, bar) in enumerate(bar_list):
        t_et = ts.time()

        # ── bar OHLC ──────────────────────────────────────────────────────
        bar_open  = float(bar["open"])
        bar_high  = float(bar["high"])
        bar_low   = float(bar["low"])
        bar_close = float(bar["close"])
        sig       = float(sig_aligned.iloc[i])

        # ── OHLC 이력 갱신 (로컬저점 계산용) ─────────────────────────────
        bar_low_history.append(bar_low)
        if len(bar_low_history) > 30:
            bar_low_history.pop(0)

        # ── ATR 동적 파라미터 ─────────────────────────────────────────────
        eod      = t_et >= EOD_TIGHT_TIME
        stop_pct = STOP_LOSS_PCT * atr_scale / (2 if eod else 1)
        part_pct = PARTIAL_PCT   * atr_scale / (2 if eod else 1)
        eod_tag  = "[EOD⚡]" if eod else ""

        # ── 15:45 선제청산 (1회) ──────────────────────────────────────────
        if t_et >= EOD_REDUCE_TIME and not pos.eod_reduced:
            pos.eod_reduced = True
            pos.pending_sig = None
            if pos.direction == "long" and pos.qty > 0:
                close_qty = max(1, pos.qty // 2)
                px        = bar_open
                slip      = px * (SPREAD_BPS / 10_000)
                gross     = (px - pos.entry_px) * close_qty
                net       = gross - slip * close_qty
                equity   += net
                pos.qty  -= close_qty
                if pos.qty == 0:
                    pos.direction = "flat"
                    pos.reset()
                pos.partial_done = True
                trades.append(_trade(ts, sym, "EOD_REDUCE_50%",
                                     px, close_qty, net, sig, equity))

        # ── 15:55 강제청산 ─────────────────────────────────────────────────
        if t_et >= SESSION_CLOSE:
            if pos.direction == "long" and pos.qty > 0:
                px     = bar_close
                slip   = px * (SPREAD_BPS / 10_000)
                gross  = (px - pos.entry_px) * pos.qty
                net    = gross - slip * pos.qty
                equity += net
                trades.append(_trade(ts, sym, "EOD_FORCED",
                                     px, pos.qty, net, sig, equity))
                pos.reset()
            continue

        # ── 세션 외 ────────────────────────────────────────────────────────
        if t_et < SESSION_OPEN:
            continue

        # ── Step 1: 예약 진입 (t+1 bar open) ─────────────────────────────
        if pos.pending_sig is not None and pos.direction == "flat":
            notional = equity * KELLY
            qty_try  = int(notional / bar_open)
            if qty_try >= 1:
                cost_total = qty_try * bar_open * (1 + SPREAD_BPS / 10_000)
                limit = equity * DAILY_TURNOVER_LIMIT
                if daily_notional + cost_total <= limit:
                    slip              = bar_open * (SPREAD_BPS / 10_000)
                    pos.direction     = "long"
                    pos.entry_px      = bar_open + slip   # 슬리피지 포함
                    pos.qty           = qty_try
                    pos.pnl_peak      = 0.0
                    pos.partial_done  = False
                    pos.last_switch   = ts
                    raw_lmin = latest_local_min(bar_low_history[:-1])
                    # 진입가 대비 0.3% 이상 아래인 경우만 로컬저점 스톱 활성화
                    if raw_lmin > 0 and (bar_open - raw_lmin) / bar_open >= MIN_LOCAL_STOP_DIST:
                        pos.local_min_stop = raw_lmin
                    else:
                        pos.local_min_stop = 0.0
                    daily_notional   += cost_total
                    trades.append(_trade(ts, sym, "ENTER",
                                         pos.entry_px, qty_try, 0.0,
                                         pos.pending_sig, equity,
                                         extra=f"atr={atr_scale:.3f} "
                                               f"stop={stop_pct:.3%} "
                                               f"lmin={pos.local_min_stop:.2f}"))
        pos.pending_sig = None

        # ── Step 2: 포지션 평가 ───────────────────────────────────────────
        if pos.direction == "long" and pos.qty > 0:
            pnl_abs   = (bar_close - pos.entry_px) * pos.qty
            pnl_pct   = (bar_close - pos.entry_px) / pos.entry_px
            pos.pnl_peak = max(pos.pnl_peak, pnl_abs)

            stop_px   = pos.entry_px * (1 - stop_pct)
            part_px   = pos.entry_px * (1 + part_pct)

            # ① 고정 손절 (bar.low 기준)
            if bar_low <= stop_px:
                px     = stop_px
                slip   = px * (SPREAD_BPS / 10_000)
                gross  = (px - pos.entry_px) * pos.qty
                net    = gross - slip * pos.qty
                equity += net
                trades.append(_trade(ts, sym, f"EXIT_손절{eod_tag}",
                                     px, pos.qty, net, sig, equity))
                pos.reset()
                continue

            # ② 로컬저점 손절 (bar.low 기준)
            if pos.local_min_stop > 0 and bar_low <= pos.local_min_stop:
                px     = pos.local_min_stop
                slip   = px * (SPREAD_BPS / 10_000)
                gross  = (px - pos.entry_px) * pos.qty
                net    = gross - slip * pos.qty
                equity += net
                trades.append(_trade(ts, sym, "EXIT_로컬저점",
                                     px, pos.qty, net, sig, equity))
                pos.last_conv_exit = ts
                pos.reset()
                continue

            # ③ 신호소멸 청산 (bar.close 기준)
            #    부분익절 완료 후에는 트레일링에 위임 → 신호소멸 skip
            if sig < CONVICTION_LONG_EXIT:
                px     = bar_close
                slip   = px * (SPREAD_BPS / 10_000)
                gross  = (px - pos.entry_px) * pos.qty
                net    = gross - slip * pos.qty
                equity += net
                trades.append(_trade(ts, sym, "EXIT_신호소멸",
                                     px, pos.qty, net, sig, equity))
                pos.last_conv_exit = ts
                pos.reset()
                continue

            # ④ 부분익절 (bar.high 기준)
            if not pos.partial_done and bar_high >= part_px:
                px        = part_px
                slip      = px * (SPREAD_BPS / 10_000)
                close_qty = max(1, pos.qty // 2)
                gross     = (px - pos.entry_px) * close_qty
                net       = gross - slip * close_qty
                equity   += net
                pos.qty  -= close_qty
                pos.partial_done = True
                trades.append(_trade(ts, sym, "PARTIAL_EXIT",
                                     px, close_qty, net, sig, equity))
                if pos.qty == 0:
                    pos.reset()
                continue

            # ⑤ 트레일링 스탑 (bar.close 기준)
            if pos.partial_done and pos.pnl_peak > 0:
                pnl_now  = (bar_close - pos.entry_px) * pos.qty
                drawdown = (pos.pnl_peak - pnl_now) / pos.pnl_peak
                if drawdown >= TRAIL_PULLBACK:
                    px     = bar_close
                    slip   = px * (SPREAD_BPS / 10_000)
                    gross  = (px - pos.entry_px) * pos.qty
                    net    = gross - slip * pos.qty
                    equity += net
                    trades.append(_trade(ts, sym, "EXIT_트레일링",
                                         px, pos.qty, net, sig, equity))
                    pos.reset()
                    continue

        # ── Step 3: 신규 진입 예약 (15:45 이전만) ─────────────────────────
        if (pos.direction == "flat"
                and t_et < EOD_REDUCE_TIME
                and sig >= MIN_CONV
                and pos.switch_ok(ts)
                and pos.conv_ok(ts)):
            pos.pending_sig = sig

    return trades, equity, daily_notional


# ══════════════════════════════════════════════════════════════════════════════
# 6-B. 시간 기반 Exit 시뮬레이션 (stripped-down: 하드스탑 + N분 홀딩만)
# ══════════════════════════════════════════════════════════════════════════════

def simulate_day_timed(
    sym:              str,
    date_str:         str,
    bars:             pd.DataFrame,
    signals:          pd.Series,
    atr_scale:        float,
    pos:              Position,
    equity:           float,
    daily_notional:   float,
    bar_low_history:  list[float],
    n_hold_bars:      int = 5,
) -> tuple[list[dict], float, float]:
    """
    극도로 단순화된 시뮬레이션: TFI 진입 → N분 뒤 bar.close 청산.
    유일한 조기 청산: 하드 스탑(ATR 기반).
    부분익절·트레일링·신호소멸·EOD_REDUCE 모두 제거.

    목적: "TFI signal edge가 transaction cost 이후에도 N분 홀딩에서 살아남는가?"
          Tier 3 결과(5m/10m 유의)와 1:1 대응.
    """
    trades = []

    bars_et = bars[
        (bars.index.time >= dtime(9, 30)) &
        (bars.index.time <= dtime(16, 0))
    ].copy()
    sig_aligned = signals.reindex(bars_et.index, method="ffill").fillna(0.0)
    bar_list    = list(bars_et.iterrows())

    for i, (ts, bar) in enumerate(bar_list):
        t_et      = ts.time()
        bar_open  = float(bar["open"])
        bar_low   = float(bar["low"])
        bar_close = float(bar["close"])
        sig       = float(sig_aligned.iloc[i])

        bar_low_history.append(bar_low)
        if len(bar_low_history) > 30:
            bar_low_history.pop(0)

        stop_pct = STOP_LOSS_PCT * atr_scale

        # 15:55 강제 청산
        if t_et >= SESSION_CLOSE:
            if pos.direction == "long" and pos.qty > 0:
                px     = bar_close
                slip   = px * (SPREAD_BPS / 10_000)
                net    = (px - pos.entry_px) * pos.qty - slip * pos.qty
                equity += net
                trades.append(_trade(ts, sym, "EOD_FORCED",
                                     px, pos.qty, net, sig, equity))
                pos.reset()
            continue

        if t_et < SESSION_OPEN:
            continue

        # ── Step 1: 예약 진입 (t+1 bar open) ─────────────────────────────
        if pos.pending_sig is not None and pos.direction == "flat":
            notional   = equity * KELLY
            qty_try    = int(notional / bar_open)
            if qty_try >= 1:
                cost_total = qty_try * bar_open * (1 + SPREAD_BPS / 10_000)
                if daily_notional + cost_total <= equity * DAILY_TURNOVER_LIMIT:
                    slip             = bar_open * (SPREAD_BPS / 10_000)
                    pos.direction    = "long"
                    pos.entry_px     = bar_open + slip
                    pos.qty          = qty_try
                    pos.entry_bar_idx = i
                    pos.last_switch  = ts
                    daily_notional  += cost_total
                    trades.append(_trade(ts, sym, "ENTER",
                                         pos.entry_px, qty_try, 0.0,
                                         pos.pending_sig, equity,
                                         extra=f"hold={n_hold_bars}m "
                                               f"atr={atr_scale:.3f}"))
        pos.pending_sig = None

        # ── Step 2: 포지션 관리 (하드스탑 → 시간 청산) ────────────────────
        if pos.direction == "long" and pos.qty > 0:
            stop_px = pos.entry_px * (1 - stop_pct)

            # ① 하드 스탑 (bar.low 기준)
            if bar_low <= stop_px:
                px     = stop_px
                slip   = px * (SPREAD_BPS / 10_000)
                net    = (px - pos.entry_px) * pos.qty - slip * pos.qty
                equity += net
                trades.append(_trade(ts, sym, "EXIT_손절",
                                     px, pos.qty, net, sig, equity))
                pos.reset()
                continue

            # ② N분 홀딩 만료 → bar.close 청산
            if (i - pos.entry_bar_idx) >= n_hold_bars:
                px     = bar_close
                slip   = px * (SPREAD_BPS / 10_000)
                net    = (px - pos.entry_px) * pos.qty - slip * pos.qty
                equity += net
                trades.append(_trade(ts, sym, f"EXIT_시간{n_hold_bars}m",
                                     px, pos.qty, net, sig, equity))
                pos.reset()
                continue

        # ── Step 3: 신규 진입 예약 ────────────────────────────────────────
        if (pos.direction == "flat"
                and t_et < EOD_REDUCE_TIME
                and sig >= MIN_CONV
                and pos.switch_ok(ts)):
            pos.pending_sig = sig

    return trades, equity, daily_notional


def _trade(ts, sym, action, px, qty, net, sig, equity, extra=""):
    return {
        "timestamp": ts,
        "symbol":    sym,
        "action":    action,
        "price":     round(px, 4),
        "qty":       qty,
        "net_pnl":   round(net, 2),
        "signal":    round(sig, 4),
        "equity":    round(equity, 2),
        "extra":     extra,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. 멀티-데이 백테스트 실행
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(sym: str = "SPY",
                 start: str = "2026-05-01",
                 end:   str = "2026-05-15",
                 n_hold_bars: int = None) -> pd.DataFrame:
    """n_hold_bars=None → 기존 복합 exit / int → 시간 기반 stripped exit."""

    # 사용 가능한 날짜 필터
    avail = sorted(
        p.name.split(f"{sym}_")[1].split("_quotes")[0]
        for p in CACHE_DIR.glob(f"{sym}_*_quotes.pkl")
    )
    days = [d for d in avail if start <= d <= end]
    if not days:
        raise RuntimeError(f"[{sym}] {start}~{end} 구간 tick 데이터 없음")

    print(f"\n{'═'*60}")
    print(f"  SPY/QQQ Backtest  |  {sym}  |  {days[0]} ~ {days[-1]}")
    print(f"  거래일 {len(days)}일  |  초기자본 ${INITIAL_CAPITAL:,.0f}")
    print(f"{'═'*60}")

    equity          = INITIAL_CAPITAL
    pos             = Position()
    bar_low_history: list[float] = []
    all_trades:      list[dict]  = []
    equity_curve:    list[dict]  = []

    for date_str in days:
        print(f"\n  ── {date_str} ──────────────────────────────")

        # ATR 스케일
        atr_scale = get_atr_scale(sym, date_str)
        print(f"  ATR scale={atr_scale:.3f}  "
              f"stop={STOP_LOSS_PCT*atr_scale*100:.3f}%  "
              f"partial={PARTIAL_PCT*atr_scale*100:.3f}%")

        # 신호 계산
        try:
            signals = compute_signals(sym, date_str)
        except Exception as e:
            print(f"  [SKIP] 신호 계산 실패: {e}")
            continue

        # bars 로드
        try:
            _, _, bars = load_day(sym, date_str)
        except Exception as e:
            print(f"  [SKIP] bars 로드 실패: {e}")
            continue

        day_start_equity = equity
        daily_notional   = 0.0
        pos.eod_reduced  = False

        sim_fn = simulate_day_timed if n_hold_bars is not None else simulate_day
        sim_kwargs = {"n_hold_bars": n_hold_bars} if n_hold_bars is not None else {}
        day_trades, equity, daily_notional = sim_fn(
            sym, date_str, bars, signals,
            atr_scale, pos, equity, daily_notional, bar_low_history,
            **sim_kwargs,
        )

        day_pnl = equity - day_start_equity
        print(f"  당일 거래 {len(day_trades)}건  "
              f"PnL=${day_pnl:+,.2f}  "
              f"equity=${equity:,.2f}")
        for t in day_trades:
            tag = "✅" if t["net_pnl"] >= 0 else "🔴"
            print(f"    {tag} {t['timestamp'].strftime('%H:%M')}  "
                  f"{t['action']:<22}  "
                  f"${t['price']:.2f} × {t['qty']}주  "
                  f"PnL=${t['net_pnl']:+,.2f}  "
                  f"sig={t['signal']:+.3f}")

        all_trades.extend(day_trades)
        equity_curve.append({
            "date":   date_str,
            "equity": equity,
            "pnl":    day_pnl,
        })

    df_trades = pd.DataFrame(all_trades)
    df_curve  = pd.DataFrame(equity_curve)
    _print_summary(df_trades, df_curve)
    _plot(df_trades, df_curve, sym)
    return df_trades


# ══════════════════════════════════════════════════════════════════════════════
# 8. 통계 출력
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(df: pd.DataFrame, curve: pd.DataFrame):
    if df.empty:
        print("\n  [결과 없음]"); return

    exits = df[df["action"].str.startswith("EXIT") |
               df["action"].str.startswith("EOD") |
               df["action"].str.startswith("PARTIAL")]

    # 완전 청산(ENTER와 짝지어진 거래)만 추출
    closed = df[df["action"].str.startswith("EXIT") |
                df["action"].str.startswith("EOD")]

    total_pnl   = closed["net_pnl"].sum()
    total_ret   = total_pnl / INITIAL_CAPITAL * 100
    n_trades    = len(closed)
    wins        = closed[closed["net_pnl"] > 0]
    losses      = closed[closed["net_pnl"] <= 0]
    win_rate    = len(wins) / n_trades * 100 if n_trades else 0
    avg_win     = wins["net_pnl"].mean()   if len(wins)   else 0
    avg_loss    = losses["net_pnl"].mean() if len(losses) else 0
    pf          = wins["net_pnl"].sum() / abs(losses["net_pnl"].sum()) \
                  if losses["net_pnl"].sum() != 0 else float("inf")

    # 부분익절 포함 전체 PnL
    total_pnl_all = df[~df["action"].str.startswith("ENTER")]["net_pnl"].sum()

    # 최대 낙폭
    eq_vals = [INITIAL_CAPITAL] + list(curve["equity"])
    peak = eq_vals[0]; mdd = 0.0
    for v in eq_vals:
        peak = max(peak, v)
        mdd  = max(mdd, (peak - v) / peak)

    print(f"\n{'═'*60}")
    print(f"  📊 백테스트 결과 요약  ({len(curve)}거래일)")
    print(f"{'─'*60}")
    print(f"  총 수익        : ${total_pnl_all:>+10,.2f}  ({total_pnl_all/INITIAL_CAPITAL*100:+.3f}%)")
    print(f"  총 거래 수     : {n_trades}건  (부분익절 {len(df[df['action']=='PARTIAL_EXIT'])}건 별도)")
    print(f"  승률           : {win_rate:.1f}%  ({len(wins)}승 {len(losses)}패)")
    print(f"  평균 수익      : ${avg_win:>+8,.2f}")
    print(f"  평균 손실      : ${avg_loss:>+8,.2f}")
    print(f"  손익비 (PF)    : {pf:.2f}")
    print(f"  최대 낙폭(MDD) : {mdd*100:.2f}%")
    print(f"{'─'*60}")
    print(f"  청산 유형 분포:")
    for action, cnt in df["action"].value_counts().items():
        if not action.startswith("ENTER"):
            pnl_sum = df[df["action"] == action]["net_pnl"].sum()
            print(f"    {action:<25} {cnt:>3}건  합계 ${pnl_sum:>+9,.2f}")
    print(f"{'═'*60}")

    # 일별 P&L
    print(f"\n  📅 일별 손익:")
    for _, row in curve.iterrows():
        tag = "✅" if row["pnl"] >= 0 else "🔴"
        bar = "█" * int(abs(row["pnl"]) / 50)
        print(f"  {tag} {row['date']}  ${row['pnl']:>+8,.2f}  {bar}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. 차트 생성
# ══════════════════════════════════════════════════════════════════════════════

def _plot(df: pd.DataFrame, curve: pd.DataFrame, sym: str):
    if curve.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                              facecolor="#0d1117", gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle(f"{sym} TFI-Only Strategy Backtest  |  ATR Dynamic Scaling + EOD Staggered Exit",
                 color="#e6edf3", fontsize=12, fontweight="bold")

    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        ax.spines[:].set_color("#30363d")

    # ── 상단: 에쿼티 곡선 ──────────────────────────────────────────────────
    ax1 = axes[0]
    dates  = [pd.Timestamp(d) for d in curve["date"]]
    equities = [INITIAL_CAPITAL] + list(curve["equity"])
    date_axis = [dates[0] - pd.Timedelta(days=1)] + dates

    ax1.plot(date_axis, equities, color="#58a6ff", lw=2.0, label="Equity")
    ax1.axhline(INITIAL_CAPITAL, color="#484f58", lw=0.8, ls="--", label="Initial")
    ax1.fill_between(date_axis, INITIAL_CAPITAL, equities,
                     where=[e >= INITIAL_CAPITAL for e in equities],
                     alpha=0.15, color="#2ea043")
    ax1.fill_between(date_axis, INITIAL_CAPITAL, equities,
                     where=[e < INITIAL_CAPITAL for e in equities],
                     alpha=0.15, color="#f85149")
    ax1.set_ylabel("Equity ($)", color="#8b949e")
    ax1.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3",
               fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # ── 하단: 일별 PnL 막대 ────────────────────────────────────────────────
    ax2 = axes[1]
    colors = ["#2ea043" if p >= 0 else "#f85149" for p in curve["pnl"]]
    ax2.bar(dates, curve["pnl"], color=colors, width=0.6, alpha=0.85)
    ax2.axhline(0, color="#484f58", lw=0.8)
    ax2.set_ylabel("Daily PnL ($)", color="#8b949e")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:>+,.0f}"))

    total_pnl = curve["pnl"].sum()
    ret_pct   = total_pnl / INITIAL_CAPITAL * 100
    wins      = (curve["pnl"] > 0).sum()
    fig.text(0.99, 0.02,
             f"Total: ${total_pnl:+,.2f} ({ret_pct:+.3f}%)  |  "
             f"Win days: {wins}/{len(curve)}",
             ha="right", va="bottom", color="#8b949e", fontsize=9)

    plt.tight_layout()
    out = OUT_DIR / "backtest_spy_qqq.png"
    plt.savefig(out, dpi=130, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close()
    print(f"\n  차트 저장: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. 시간 기반 Exit 스윕 (Tier 3 검증용)
# ══════════════════════════════════════════════════════════════════════════════

def run_sweep(
    sym:   str       = "QQQ",
    start: str       = "2026-01-02",
    end:   str       = "2026-05-19",
    hold_bars: list  = [3, 5, 7, 10, 15],
) -> pd.DataFrame:
    """
    n_hold_bars 를 여러 값으로 스윕하여 비교표 출력.
    Tier 3 결과(5m/10m 유의)와 직접 대응.
    """
    print(f"\n{'═'*60}")
    print(f"  TFI-Only Time-Exit Sweep  |  {sym}  |  {start} ~ {end}")
    print(f"  테스트 홀딩: {hold_bars}분")
    print(f"{'═'*60}")

    summary_rows = []
    for n in hold_bars:
        df = run_backtest(sym=sym, start=start, end=end, n_hold_bars=n)

        if df.empty:
            summary_rows.append({"hold_min": n})
            continue

        exits = df[~df["action"].str.startswith("ENTER")]
        closed = exits[~exits["action"].isin(["EOD_REDUCE_50%", "PARTIAL_EXIT"])]

        wins   = closed[closed["net_pnl"] > 0]
        losses = closed[closed["net_pnl"] <= 0]
        total_pnl = closed["net_pnl"].sum()
        n_trades  = len(closed)
        win_rate  = len(wins) / n_trades * 100 if n_trades else 0
        pf = (wins["net_pnl"].sum() / abs(losses["net_pnl"].sum())
              if losses["net_pnl"].sum() != 0 else float("inf"))
        avg_pnl = total_pnl / n_trades if n_trades else 0

        summary_rows.append({
            "hold_min":   n,
            "n_trades":   n_trades,
            "total_pnl":  round(total_pnl, 2),
            "ret_pct":    round(total_pnl / INITIAL_CAPITAL * 100, 4),
            "win_rate":   round(win_rate, 1),
            "PF":         round(pf, 3) if pf != float("inf") else None,
            "avg_pnl":    round(avg_pnl, 3),
        })

    print(f"\n{'─'*60}")
    print(f"  {'Hold':>5}  {'Trades':>7}  {'TotalPnL':>10}  "
          f"{'Ret%':>7}  {'WinRate':>8}  {'PF':>6}  {'AvgPnL':>8}")
    print(f"  {'─'*5}  {'─'*7}  {'─'*10}  {'─'*7}  {'─'*8}  {'─'*6}  {'─'*8}")
    for r in summary_rows:
        if "n_trades" not in r:
            print(f"  {r['hold_min']:>4}m  (데이터 없음)")
            continue
        pf_str = f"{r['PF']:.3f}" if r["PF"] is not None else "  inf"
        tier1  = "✅" if (r["PF"] or 0) >= 1.25 else "❌"
        print(f"  {r['hold_min']:>4}m  {r['n_trades']:>7,}  "
              f"${r['total_pnl']:>+9,.2f}  "
              f"{r['ret_pct']:>+6.3f}%  "
              f"{r['win_rate']:>7.1f}%  "
              f"{pf_str:>6}  "
              f"${r['avg_pnl']:>+7.3f}  {tier1}")
    print(f"{'─'*60}")

    return pd.DataFrame(summary_rows)


# ══════════════════════════════════════════════════════════════════════════════
# 11. 진입점
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_sweep(sym="QQQ", start="2026-01-02", end="2026-05-19",
              hold_bars=[3, 5, 7, 10, 15])
