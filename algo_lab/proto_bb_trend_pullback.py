"""
proto_bb_trend_pullback.py — 볼린저밴드 + 이동평균 정배열 눌림목 전략 프로토타입
─────────────────────────────────────────────────────────────────────────────
전제조건 (일봉 기준, 전일 종가까지 데이터만 사용 — lookahead 방지):
  - MA5 > MA20 > MA60 > MA120 정배열, 또는 최소 MA120이 4개 중 가장 아래
  - MA120이 4개 중 가장 위면(역배열) 매매 안 함

진입 (3분봉 기준, 눌림목 분할매수):
  1차: 종가 <= 볼린저밴드 하단 AND 종가가 일봉 MA5 근접(±0.5%)
  2차: 1차 보유 중 종가가 일봉 MA20까지 근접(±0.5%) 추가 하락

손절: 종가 < 직전 "거래량 실린 장대양봉" 시가, 또는 종가 < 일봉 MA120 (먼저 닿는 것)
익절: 고가 >= 진입 직전 50봉 스윙하이(전고점)

데이터: Alpaca 일봉(MA 계산) + 3분봉(진입/청산) — 로컬 pkl 캐시
실행:
    python algo_lab/proto_bb_trend_pullback.py QQQ 2026-02-01 2026-05-19
"""
import sys
import pickle
import warnings
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY

ET = pytz.timezone("America/New_York")
CACHE_DIR = ROOT / "logs" / "bar_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = ROOT / "logs"

# ── 전략 파라미터 ───────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 100_000.0
KELLY_LEG       = 0.025          # 1차/2차 각각 2.5% (최대 5%)

MA_PERIODS      = [5, 20, 60, 120]   # 일봉 기준 정배열 판단용
BB_PERIOD       = 20              # 볼린저밴드 기간 (3분봉)
BB_STD          = 2.0
PROXIMITY_PCT   = 0.005           # MA5/MA20 "근접" 판정 ±0.5%

VOL_SURGE_MULT  = 1.5             # 장대양봉 거래량 기준: 20봉 평균 대비 배수
BODY_SURGE_MULT = 1.5             # 장대양봉 몸통 기준: 20봉 평균 대비 배수
CANDLE_LOOKBACK = 20

SWING_LOOKBACK  = 50               # 스윙하이(전고점) 탐색 구간 (3분봉 개수)

SESSION_OPEN  = dtime(9, 45)
SESSION_CLOSE = dtime(15, 55)

SPREAD_BPS = 0.5


# ══════════════════════════════════════════════════════════════════════════════
# 1. 데이터 로드 (일봉 + 3분봉, 로컬 캐시)
# ══════════════════════════════════════════════════════════════════════════════

_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


def _cache_path(sym: str, tag: str, start: str, end: str) -> Path:
    return CACHE_DIR / f"{sym}_{tag}_{start}_{end}.pkl"


def load_daily_bars(sym: str, start: str, end: str) -> pd.DataFrame:
    """MA120 계산을 위해 start 이전 200일을 더 끌어와 로드."""
    cache = _cache_path(sym, "daily", start, end)
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    fetch_start = (pd.Timestamp(start) - pd.Timedelta(days=300)).strftime("%Y-%m-%d")
    req = StockBarsRequest(
        symbol_or_symbols=sym, timeframe=TimeFrame.Day,
        start=fetch_start, end=end, feed="sip",
    )
    df = _client.get_stock_bars(req).df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(sym, level="symbol")
    df.index = pd.DatetimeIndex(df.index).tz_convert(ET).normalize()
    with open(cache, "wb") as f:
        pickle.dump(df, f)
    return df


TICK_CACHE_DIR = ROOT / "logs" / "tick_cache"


def load_3min_from_tick_cache(sym: str, start: str, end: str) -> pd.DataFrame:
    """기존 tick_cache의 1분봉(bars.pkl)을 모아 3분봉으로 리샘플링.
    Alpaca API 재호출 없이, OBI+TFI 백테스트와 동일한 원천 데이터를 사용."""
    days = sorted(
        p.name.split(f"{sym}_")[1].split("_bars")[0]
        for p in TICK_CACHE_DIR.glob(f"{sym}_*_bars.pkl")
    )
    days = [d for d in days if start <= d <= end]
    if not days:
        raise RuntimeError(f"[{sym}] {start}~{end} 구간 tick_cache bars 없음")

    frames = []
    for d in days:
        with open(TICK_CACHE_DIR / f"{sym}_{d}_bars.pkl", "rb") as f:
            df = pickle.load(f)
        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True)
        frames.append(df)
    bars1 = pd.concat(frames).sort_index()
    bars1.index = bars1.index.tz_convert(ET)

    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    bars3 = (bars1.resample("3min", label="left", closed="left")
                  .agg(agg).dropna(subset=["open"]))
    return bars3


def load_3min_bars(sym: str, start: str, end: str) -> pd.DataFrame:
    cache = _cache_path(sym, "3min", start, end)
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    req = StockBarsRequest(
        symbol_or_symbols=sym, timeframe=TimeFrame(3, TimeFrameUnit.Minute),
        start=start, end=end, feed="sip",
    )
    df = _client.get_stock_bars(req).df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(sym, level="symbol")
    df.index = pd.DatetimeIndex(df.index).tz_convert(ET)
    with open(cache, "wb") as f:
        pickle.dump(df, f)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. 일봉 MA 정배열 필터
# ══════════════════════════════════════════════════════════════════════════════

def compute_daily_trend(daily: pd.DataFrame) -> pd.DataFrame:
    """
    일봉 MA5/20/60/120 계산 + 매일의 트렌드 상태 플래그.
    반환 인덱스 = 일봉 날짜(00:00 ET 정규화). 당일 컬럼은 '당일 종가까지' 반영된 값이므로
    백테스트에서 사용할 땐 반드시 D-1(전일) 값을 참조해야 lookahead가 안 생김.
    """
    df = daily.copy()
    for p in MA_PERIODS:
        df[f"ma{p}"] = df["close"].rolling(p).mean()

    ma_cols = [f"ma{p}" for p in MA_PERIODS]

    def _status(row):
        vals = row[ma_cols]
        if vals.isna().any():
            return "insufficient"
        if vals["ma120"] == vals.max():
            return "blocked"          # 120일선이 가장 위 → 역배열, 매매 금지
        aligned = vals["ma5"] > vals["ma20"] > vals["ma60"] > vals["ma120"]
        if aligned or vals["ma120"] == vals.min():
            return "ok"
        return "neutral"              # 정배열도 역배열도 아닌 애매한 구간 → 매매 안 함

    df["trend_status"] = df.apply(_status, axis=1)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. 3분봉 지표 (볼린저밴드, 장대양봉, 스윙하이)
# ══════════════════════════════════════════════════════════════════════════════

def compute_3min_indicators(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.copy()
    df["bb_mid"] = df["close"].rolling(BB_PERIOD).mean()
    bb_std = df["close"].rolling(BB_PERIOD).std()
    df["bb_lower"] = df["bb_mid"] - BB_STD * bb_std
    df["bb_upper"] = df["bb_mid"] + BB_STD * bb_std

    df["body"] = (df["close"] - df["open"]).abs()
    avg_vol  = df["volume"].rolling(CANDLE_LOOKBACK).mean()
    avg_body = df["body"].rolling(CANDLE_LOOKBACK).mean()
    df["is_surge_candle"] = (
        (df["close"] > df["open"]) &                      # 양봉
        (df["volume"] >= VOL_SURGE_MULT * avg_vol) &
        (df["body"]   >= BODY_SURGE_MULT * avg_body)
    )
    return df


def latest_swing_high(highs: list[float]) -> float:
    """완성된 bar 기준 가장 최근 스윙하이. lows 버전(latest_local_min)과 대칭 로직."""
    hs = [h for h in highs if np.isfinite(h) and h > 0]
    if len(hs) < 3:
        return 0.0
    for i in range(len(hs) - 2, 0, -1):
        if hs[i] >= hs[i - 1] and hs[i] >= hs[i + 1]:
            return hs[i]
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 4. 포지션 상태 + 시뮬레이션
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    direction:      str   = "flat"     # flat / leg1 / leg2
    entry_px1:      float = 0.0
    qty1:           int   = 0
    entry_px2:      float = 0.0
    qty2:           int   = 0
    stop_px:        float = 0.0        # 직전 장대양봉 시가 기준
    target_px:      float = 0.0        # 진입 직전 스윙하이

    @property
    def total_qty(self) -> int:
        return self.qty1 + self.qty2

    @property
    def avg_entry(self) -> float:
        if self.total_qty == 0:
            return 0.0
        return (self.entry_px1 * self.qty1 + self.entry_px2 * self.qty2) / self.total_qty

    def reset(self):
        self.__init__()


def _trade(ts, sym, action, px, qty, net, equity):
    return {"timestamp": ts, "symbol": sym, "action": action,
            "price": round(px, 4), "qty": qty,
            "net_pnl": round(net, 2), "equity": round(equity, 2)}


def simulate(sym: str, bars3: pd.DataFrame, daily_trend: pd.DataFrame,
            equity: float) -> tuple[list[dict], float]:
    trades: list[dict] = []
    pos = Position()
    surge_history: list[tuple] = []   # (ts, open) — 최근 SWING_LOOKBACK봉 내 장대양봉
    high_history: list[float] = []

    daily_idx = daily_trend.index   # normalized dates, ascending

    for i in range(len(bars3)):
        row   = bars3.iloc[i]
        ts    = bars3.index[i]
        t_et  = ts.time()
        if t_et < SESSION_OPEN or t_et >= SESSION_CLOSE:
            continue

        today = ts.normalize()
        prior_days = daily_idx[daily_idx < today]
        if len(prior_days) == 0:
            continue
        d_row = daily_trend.loc[prior_days[-1]]   # 전일까지의 MA — lookahead 방지

        high_history.append(float(row["high"]))
        if len(high_history) > SWING_LOOKBACK:
            high_history.pop(0)

        if bool(row.get("is_surge_candle", False)):
            surge_history.append((ts, float(row["open"])))
        surge_history = [(t, o) for t, o in surge_history
                          if t >= ts - pd.Timedelta(minutes=3 * SWING_LOOKBACK)]

        close = float(row["close"])
        high  = float(row["high"])
        low   = float(row["low"])

        # ── 포지션 보유 중: 손절/익절 판단 ──────────────────────────────────
        if pos.direction != "flat":
            ma120 = d_row["ma120"]
            # 두 손절 기준 중 "먼저 닿는" 쪽(=더 높은 가격)을 실제 체결가로 사용.
            # 둘 다 안 닿았으면 stop_hit=False — min()으로 안 닿은 기준을 체결가로
            # 잘못 쓰지 않도록 트리거된 기준만 후보에 넣는다.
            candidates = []
            if pos.stop_px > 0 and low <= pos.stop_px:
                candidates.append(pos.stop_px)
            if np.isfinite(ma120) and low <= ma120:
                candidates.append(ma120)
            if candidates:
                px = max(candidates)   # 더 높게 걸린(먼저 닿은) 손절선이 실제 체결가
                slip = px * (SPREAD_BPS / 10_000)
                net  = (px - pos.avg_entry) * pos.total_qty - slip * pos.total_qty
                equity += net
                trades.append(_trade(ts, sym, "EXIT_손절", px, pos.total_qty, net, equity))
                pos.reset()
                continue

            if pos.target_px > 0 and high >= pos.target_px:
                px   = pos.target_px
                slip = px * (SPREAD_BPS / 10_000)
                net  = (px - pos.avg_entry) * pos.total_qty - slip * pos.total_qty
                equity += net
                trades.append(_trade(ts, sym, "EXIT_익절", px, pos.total_qty, net, equity))
                pos.reset()
                continue

        # ── 트렌드 필터 ──────────────────────────────────────────────────────
        if d_row["trend_status"] != "ok":
            continue

        ma5, ma20 = d_row["ma5"], d_row["ma20"]
        bb_lower  = row.get("bb_lower", np.nan)

        # ── 1차 매수 ──────────────────────────────────────────────────────────
        if pos.direction == "flat" and np.isfinite(bb_lower):
            near_ma5 = abs(close - ma5) / ma5 <= PROXIMITY_PCT if np.isfinite(ma5) else False
            if close <= bb_lower and near_ma5:
                notional = equity * KELLY_LEG
                qty = int(notional / close)
                if qty >= 1:
                    pos.direction = "leg1"
                    pos.entry_px1 = close
                    pos.qty1      = qty
                    # 손절 기준: 최근 SWING_LOOKBACK봉 내 장대양봉 중 "진입가보다 아래"인
                    # 가장 최근 것만 사용 — 진입 직전 캔들처럼 진입가에 바짝 붙은 경우
                    # (시가>=진입가) 손절선으로 의미가 없으므로 제외.
                    below = [o for t, o in surge_history if o < close]
                    pos.stop_px   = below[-1] if below else 0.0
                    pos.target_px = latest_swing_high(high_history[:-1])
                    trades.append(_trade(ts, sym, "ENTER_1차", close, qty, 0.0, equity))
            continue

        # ── 2차 매수 (1차 보유 중, MA20까지 추가 하락) ─────────────────────────
        if pos.direction == "leg1" and np.isfinite(ma20):
            near_ma20 = abs(close - ma20) / ma20 <= PROXIMITY_PCT
            if near_ma20 and close < pos.entry_px1:
                notional = equity * KELLY_LEG
                qty = int(notional / close)
                if qty >= 1:
                    pos.direction = "leg2"
                    pos.entry_px2 = close
                    pos.qty2      = qty
                    trades.append(_trade(ts, sym, "ENTER_2차", close, qty, 0.0, equity))

    return trades, equity


# ══════════════════════════════════════════════════════════════════════════════
# 5. 실행 + 요약
# ══════════════════════════════════════════════════════════════════════════════

def run(sym: str = "QQQ", start: str = "2026-02-01", end: str = "2026-05-19",
       use_tick_cache: bool = False):
    print(f"\n{'═'*60}\n  BB+추세정배열 눌림목 전략 백테스트  |  {sym}  |  {start}~{end}"
          f"{'  [tick_cache 기반]' if use_tick_cache else ''}\n{'═'*60}")

    daily = load_daily_bars(sym, start, end)
    daily_trend = compute_daily_trend(daily)
    n_ok = (daily_trend.loc[start:end, "trend_status"] == "ok").sum()
    n_tot = len(daily_trend.loc[start:end])
    print(f"  트렌드 필터 통과일: {n_ok}/{n_tot}일")

    if use_tick_cache:
        bars3 = load_3min_from_tick_cache(sym, start, end)
    else:
        bars3 = load_3min_bars(sym, start, end)
    bars3 = compute_3min_indicators(bars3)
    print(f"  3분봉 {len(bars3)}개 로드")

    trades, final_equity = simulate(sym, bars3, daily_trend, INITIAL_CAPITAL)

    df = pd.DataFrame(trades)
    if df.empty:
        print("\n  [거래 없음 — 조건이 너무 엄격할 수 있음]")
        return df

    closed = df[df["action"].str.startswith("EXIT")]
    wins   = closed[closed["net_pnl"] > 0]
    losses = closed[closed["net_pnl"] <= 0]
    total_pnl = closed["net_pnl"].sum()
    pf = wins["net_pnl"].sum() / abs(losses["net_pnl"].sum()) if len(losses) and losses["net_pnl"].sum() != 0 else float("inf")

    print(f"\n{'─'*60}")
    print(f"  총 수익      : ${total_pnl:+,.2f}  ({total_pnl/INITIAL_CAPITAL*100:+.3f}%)")
    print(f"  진입(1차) 수 : {(df['action']=='ENTER_1차').sum()}건  "
          f"(2차 추가 {(df['action']=='ENTER_2차').sum()}건)")
    print(f"  청산 수      : {len(closed)}건")
    print(f"  승률         : {len(wins)/len(closed)*100:.1f}%  ({len(wins)}승 {len(losses)}패)" if len(closed) else "  승률: N/A")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  청산 유형    : {dict(closed['action'].value_counts())}")
    print(f"{'─'*60}")

    return df


if __name__ == "__main__":
    sym   = sys.argv[1] if len(sys.argv) > 1 else "QQQ"
    start = sys.argv[2] if len(sys.argv) > 2 else "2026-02-01"
    end   = sys.argv[3] if len(sys.argv) > 3 else "2026-05-19"
    run(sym, start, end)
