#!/usr/bin/env python3
"""
BB Trend Pullback Live Trader — WebSocket 기반 (spy_qqq_live_trader.py 구조 이식)
──────────────────────────────────────────────────────────────────────────────
spy_qqq_live_trader.py와 다른 점은 매매 알고리즘(신호·진입·청산)뿐 —
WebSocket 연결/재연결, 상태 영속, 주문 실행, 일일 한도, 로깅 구조는 동일.

BTC/SPY·QQQ 트레이더와 달라진 점:
  - SignalEngine(OBI+TFI tick) → Bar3Engine(1분봉 3개 합성 → 3분봉 + BB/장대양봉/스윙하이)
  - subscribe_quotes+subscribe_trades → subscribe_bars (Alpaca가 만든 1분봉 직접 구독)
  - 신호 기반 청산 → 가격 레벨(BB하단/일봉MA/스윙하이/장대양봉) 기반 진입·청산
  - ⚠️ 멀티데이 보유 설계 — EOD 강제청산 없음 (스윙 전략 특성상 의도된 차이,
    algo_lab/proto_bb_trend_pullback.py 백테스트와 동일 가정)

전략 (algo_lab/proto_bb_trend_pullback.py와 동일 로직):
  전제조건(일봉, 전일 종가까지): MA5>MA20>MA60>MA120 정배열 또는 MA120이 4개 중 최저
  1차 진입: 3분봉 종가 <= BB하단(20,2) AND 일봉MA5 근접(±0.5%)
  2차 진입: 1차 보유 중 종가가 일봉MA20 근접(±0.5%)까지 추가 하락
  손절: 직전 장대양봉(거래량·몸통 20봉평균×1.5) 시가, 또는 일봉MA120 — 먼저 닿는 쪽
  익절: 진입 직전 50봉 스윙하이(전고점) 터치
"""

import asyncio
import csv
import json
import logging
import os
import signal
import sys
from collections import deque
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(__file__))

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY

# ── 로거 ──────────────────────────────────────────────────────────────────────
ET = pytz.timezone("America/New_York")

LOG_DIR = Path(__file__).parent / "logs" / "bb_trend" / datetime.now(ET).strftime("%Y/%m/%d")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "bb_trend_trader.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("bb_trend_trader")

# ── 종목 설정 (런타임에 결정) ─────────────────────────────────────────────────
INSTRUMENTS: list[str] = []

# ── 세션 (ET 기준) ─────────────────────────────────────────────────────────────
SESSION_OPEN  = dtime(9, 45)
SESSION_CLOSE = dtime(15, 55)

# ── 전략 파라미터 (algo_lab/proto_bb_trend_pullback.py와 동일) ────────────────
KELLY_LEG       = 0.025
MA_PERIODS      = [5, 20, 60, 120]
BB_PERIOD       = 20
BB_STD          = 2.0
PROXIMITY_PCT   = 0.005
VOL_SURGE_MULT  = 1.5
BODY_SURGE_MULT = 1.5
CANDLE_LOOKBACK = 20
SWING_LOOKBACK  = 50
SPREAD_BPS      = 0.5
BAR_MIN         = 3              # 3분봉 (1분봉 3개 합성)

# ── 일일 매매 한도 (spy_qqq_live_trader.py와 동일 안전장치) ──────────────────
DAILY_TURNOVER_LIMIT = 0.50

STATE_FILE = LOG_DIR / "bb_trend_state.json"
TRADE_CSV  = LOG_DIR / "bb_trend_trades.csv"


# ── 매매 CSV 로거 ─────────────────────────────────────────────────────────────

def log_trade_csv(sym: str, action: str, price: float, qty: int,
                  pnl: float = 0.0, slippage_bps: float = 0.0, portfolio: float = 0.0):
    write_header = not TRADE_CSV.exists()
    with open(TRADE_CSV, mode="a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp_et", "symbol", "action", "price", "qty",
                        "pnl_usd", "slippage_bps", "portfolio"])
        w.writerow([
            datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S"),
            sym, action, round(price, 4), qty,
            round(pnl, 2), round(slippage_bps, 2), round(portfolio, 2),
        ])


# ── 3분봉 합성 엔진 (종목별 독립, SignalEngine 대응) ──────────────────────────

class Bar3Engine:
    """
    Alpaca가 만든 1분봉 3개를 모아 3분봉으로 합성.
    완성된 3분봉마다 볼린저밴드 하단/장대양봉 여부를 계산해 이력에 저장.
    """

    def __init__(self):
        self._bucket_start: Optional[int] = None     # epoch seconds, 3분 정렬
        self._open  = np.nan
        self._high  = -np.inf
        self._low   = +np.inf
        self._close = np.nan
        self._vol   = 0.0
        self.bar3_history: deque = deque(maxlen=max(SWING_LOOKBACK, CANDLE_LOOKBACK) + 10)

    @staticmethod
    def _bucket_for(ts_epoch: float) -> int:
        return int(ts_epoch) - (int(ts_epoch) % (BAR_MIN * 60))

    def on_minute_bar(self, o: float, h: float, l: float, c: float,
                      v: float, ts_epoch: float) -> Optional[dict]:
        """1분봉 1개 수신. 3분 버킷이 바뀌면 직전 3분봉을 확정해 반환."""
        bucket = self._bucket_for(ts_epoch)
        completed = None

        if self._bucket_start is not None and bucket != self._bucket_start:
            completed = self._finalize(self._bucket_start)
            self._reset()

        if self._bucket_start is None or bucket != self._bucket_start:
            self._bucket_start = bucket

        self._open  = o if np.isnan(self._open) else self._open
        self._high  = max(self._high, h)
        self._low   = min(self._low, l)
        self._close = c
        self._vol  += v

        return completed

    def _reset(self):
        self._open, self._high, self._low, self._close, self._vol = np.nan, -np.inf, np.inf, np.nan, 0.0

    def _finalize(self, bucket_start: int) -> dict:
        bar = {
            "ts":    datetime.fromtimestamp(bucket_start, tz=timezone.utc).astimezone(ET),
            "open":  self._open, "high": self._high, "low": self._low,
            "close": self._close, "volume": self._vol,
        }
        self._compute_indicators(bar)
        self.bar3_history.append(bar)
        return bar

    def _compute_indicators(self, bar: dict):
        closes = [b["close"] for b in self.bar3_history] + [bar["close"]]
        closes = closes[-BB_PERIOD:]
        if len(closes) >= BB_PERIOD:
            mid = float(np.mean(closes))
            std = float(np.std(closes, ddof=1))
            bar["bb_lower"] = mid - BB_STD * std
        else:
            bar["bb_lower"] = np.nan

        vols  = [b["volume"] for b in self.bar3_history][-CANDLE_LOOKBACK:]
        bodies = [abs(b["close"] - b["open"]) for b in self.bar3_history][-CANDLE_LOOKBACK:]
        body = abs(bar["close"] - bar["open"])
        if len(vols) >= CANDLE_LOOKBACK:
            avg_vol  = float(np.mean(vols))
            avg_body = float(np.mean(bodies)) if bodies else 0.0
            bar["is_surge_candle"] = (
                bar["close"] > bar["open"]
                and bar["volume"] >= VOL_SURGE_MULT * avg_vol
                and body >= BODY_SURGE_MULT * avg_body
            )
        else:
            bar["is_surge_candle"] = False

    # ── 조회 헬퍼 ──────────────────────────────────────────────────────────────

    def swing_high(self, exclude_last: bool = True) -> float:
        bars = list(self.bar3_history)[-SWING_LOOKBACK:]
        if exclude_last and bars:
            bars = bars[:-1]
        hs = [b["high"] for b in bars if np.isfinite(b["high"]) and b["high"] > 0]
        if len(hs) < 3:
            return 0.0
        for i in range(len(hs) - 2, 0, -1):
            if hs[i] >= hs[i - 1] and hs[i] >= hs[i + 1]:
                return hs[i]
        return 0.0

    def surge_opens_below(self, px: float) -> list[float]:
        bars = list(self.bar3_history)[-SWING_LOOKBACK:]
        return [b["open"] for b in bars if b.get("is_surge_candle") and b["open"] < px]

    @property
    def latest_bar(self) -> Optional[dict]:
        return self.bar3_history[-1] if self.bar3_history else None


# ── 포지션 상태 (종목별 독립, JSON 영속) ─────────────────────────────────────

class PositionState:
    def __init__(self, sym: str):
        self.sym       = sym
        self.direction = "flat"     # flat / leg1 / leg2
        self.qty1      = 0
        self.entry_px1 = 0.0
        self.qty2      = 0
        self.entry_px2 = 0.0
        self.stop_px   = 0.0
        self.target_px = 0.0

    @property
    def total_qty(self) -> int:
        return self.qty1 + self.qty2

    @property
    def avg_entry(self) -> float:
        if self.total_qty == 0:
            return 0.0
        return (self.entry_px1 * self.qty1 + self.entry_px2 * self.qty2) / self.total_qty

    def to_dict(self) -> dict:
        return {
            "direction": self.direction, "qty1": self.qty1, "entry_px1": self.entry_px1,
            "qty2": self.qty2, "entry_px2": self.entry_px2,
            "stop_px": self.stop_px, "target_px": self.target_px,
        }

    def from_dict(self, data: dict):
        self.direction = data.get("direction", "flat")
        self.qty1      = data.get("qty1", 0)
        self.entry_px1 = data.get("entry_px1", 0.0)
        self.qty2      = data.get("qty2", 0)
        self.entry_px2 = data.get("entry_px2", 0.0)
        self.stop_px   = data.get("stop_px", 0.0)
        self.target_px = data.get("target_px", 0.0)

    def reset(self):
        self.direction = "flat"
        self.qty1 = self.qty2 = 0
        self.entry_px1 = self.entry_px2 = 0.0
        self.stop_px = self.target_px = 0.0


# ── 일봉 트렌드 필터 (VolatilityScaler 대응 — 하루 1회 갱신) ─────────────────

class DailyTrendFilter:
    """일봉 MA5/20/60/120 기반 정배열 트렌드 상태. 하루 1회 refresh()."""

    def __init__(self):
        self._client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        self._state: dict[str, dict] = {}

    def refresh(self, sym: str) -> None:
        try:
            end   = datetime.now(ET).date()
            start = end - timedelta(days=300)
            req = StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Day,
                                   start=start.isoformat(), end=end.isoformat(), feed="sip")
            df = self._client.get_stock_bars(req).df
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(sym, level="symbol")
            df.index = pd.DatetimeIndex(df.index).tz_convert(ET).normalize()

            today = pd.Timestamp(datetime.now(ET).date(), tz=ET)
            df = df[df.index < today]   # 전일까지만 — lookahead 방지

            for p in MA_PERIODS:
                df[f"ma{p}"] = df["close"].rolling(p).mean()
            last = df.iloc[-1]
            vals = {f"ma{p}": float(last[f"ma{p}"]) for p in MA_PERIODS}

            if any(np.isnan(v) for v in vals.values()):
                status = "insufficient"
            elif vals["ma120"] == max(vals.values()):
                status = "blocked"
            elif (vals["ma5"] > vals["ma20"] > vals["ma60"] > vals["ma120"]) or \
                 vals["ma120"] == min(vals.values()):
                status = "ok"
            else:
                status = "neutral"

            vals["status"] = status
            vals["asof"]   = str(df.index[-1].date())
            self._state[sym] = vals
            log.info(f"[{sym}] 일봉 트렌드 갱신: status={status}  asof={vals['asof']}  "
                     f"MA5={vals['ma5']:.2f} MA20={vals['ma20']:.2f} "
                     f"MA60={vals['ma60']:.2f} MA120={vals['ma120']:.2f}")
        except Exception as e:
            log.warning(f"[{sym}] 일봉 트렌드 갱신 실패: {e} → 직전 상태 유지")

    def status(self, sym: str) -> str:
        return self._state.get(sym, {}).get("status", "insufficient")

    def ma(self, sym: str, period: int) -> float:
        return self._state.get(sym, {}).get(f"ma{period}", np.nan)


# ── 주문 실행 (spy_qqq_live_trader.py OrderExecutor와 동일) ─────────────────

class OrderExecutor:
    def __init__(self, trading_client: TradingClient):
        self.client = trading_client

    def buy(self, sym: str, qty: int) -> bool:
        try:
            req = MarketOrderRequest(symbol=sym, qty=qty, side=OrderSide.BUY,
                                     time_in_force=TimeInForce.DAY)
            order = self.client.submit_order(req)
            log.info(f"BUY 주문 제출: {sym} {qty}주  order_id={order.id}")
            return True
        except Exception as e:
            log.error(f"BUY 주문 실패 [{sym}]: {e}")
            return False

    def close_position(self, sym: str) -> bool:
        try:
            self.client.close_position(sym)
            log.info(f"포지션 청산 완료: {sym}")
            return True
        except Exception as e:
            log.error(f"포지션 청산 실패 [{sym}]: {e}")
            return False

    def cancel_open_orders(self, sym: str) -> int:
        try:
            orders = self.client.get_orders(GetOrdersRequest(
                status=QueryOrderStatus.OPEN, symbols=[sym]))
            cancelled = 0
            for o in orders:
                try:
                    self.client.cancel_order_by_id(o.id)
                    cancelled += 1
                except Exception:
                    pass
            if cancelled:
                log.info(f"[{sym}] 미체결 주문 {cancelled}건 취소")
            return cancelled
        except Exception as e:
            log.warning(f"[{sym}] 미체결 주문 조회/취소 실패: {e}")
            return 0

    def get_last_fill_info(self, sym: str) -> tuple[float, int]:
        try:
            today_start = ET.localize(
                datetime.combine(datetime.now(ET).date(), dtime(0, 0, 0))
            ).astimezone(timezone.utc)
            orders = self.client.get_orders(GetOrdersRequest(
                status=QueryOrderStatus.CLOSED, symbols=[sym], limit=10, after=today_start))
            for o in orders:
                if o.filled_avg_price is not None and o.filled_qty is not None:
                    return float(o.filled_avg_price), int(float(o.filled_qty))
            return 0.0, 0
        except Exception as e:
            log.warning(f"[{sym}] 체결 정보 조회 실패: {e}")
            return 0.0, 0

    def get_position_snapshot(self, sym: str) -> tuple[int, float]:
        try:
            pos = self.client.get_open_position(sym)
            return int(float(pos.qty)), float(pos.avg_entry_price)
        except Exception:
            return 0, 0.0

    def get_account_snapshot(self) -> tuple[float, float]:
        try:
            acct = self.client.get_account()
            return float(acct.equity), float(acct.cash)
        except Exception as e:
            log.warning(f"계좌 정보 조회 실패: {e}")
            return 0.0, 0.0


# ── 메인 트레이더 ─────────────────────────────────────────────────────────────

class BBTrendLiveTrader:
    def __init__(self):
        self.engines  = {sym: Bar3Engine()      for sym in INSTRUMENTS}
        self.states   = {sym: PositionState(sym) for sym in INSTRUMENTS}
        self.trend    = DailyTrendFilter()
        self.executor = OrderExecutor(
            TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True))
        self.stream   = None
        self._last_processed_bar_ts: dict[str, Optional[datetime]] = {
            sym: None for sym in INSTRUMENTS}
        self._trade_locks = {sym: asyncio.Lock() for sym in INSTRUMENTS}
        self._running            = True
        self._current_date:      Optional[date] = None
        self._daily_buy_notional = 0.0

        self._load_state()
        self._sync_with_broker()

        log.info("일봉 트렌드 초기 로드 중...")
        for sym in INSTRUMENTS:
            self.trend.refresh(sym)

        log.info("3분봉 워밍업 중 (최근 거래 이력으로 스윙하이/장대양봉 이력 확보)...")
        for sym in INSTRUMENTS:
            self._warmup(sym)

    # ── 상태 영속 ──────────────────────────────────────────────────────────────

    def _save_state(self):
        data = {
            "date":               self._current_date.isoformat() if self._current_date else None,
            "daily_buy_notional": self._daily_buy_notional,
        }
        for sym in INSTRUMENTS:
            data[sym] = self.states[sym].to_dict()
        STATE_FILE.write_text(json.dumps(data, indent=2))

    def _load_state(self):
        if not STATE_FILE.exists():
            return
        try:
            data  = json.loads(STATE_FILE.read_text())
            today = datetime.now(ET).date()
            saved = data.get("date")
            if saved and date.fromisoformat(saved) == today:
                self._daily_buy_notional = data.get("daily_buy_notional", 0.0)
                self._current_date = today
            for sym in INSTRUMENTS:
                if sym in data:
                    self.states[sym].from_dict(data[sym])
            log.info(f"상태 복구: 일일누적=${self._daily_buy_notional:,.0f}")
            for sym in INSTRUMENTS:
                st = self.states[sym]
                log.info(f"  [{sym}] direction={st.direction}  qty={st.total_qty}주")
        except Exception as e:
            log.warning(f"상태 파일 로드 실패 ({e}) — 초기화로 시작")

    def _sync_with_broker(self):
        equity, cash = self.executor.get_account_snapshot()
        log.info(f"계좌 현황: equity=${equity:,.2f}  cash=${cash:,.2f}")
        for sym in INSTRUMENTS:
            actual_qty, _ = self.executor.get_position_snapshot(sym)
            st = self.states[sym]
            if actual_qty > 0 and st.direction == "flat":
                log.warning(f"[{sym}] 실제 포지션 {actual_qty}주 있음 — 상태 flat 불일치. 수동 확인 필요.")
            elif actual_qty == 0 and st.direction != "flat":
                log.warning(f"[{sym}] 상태는 {st.direction}이나 실제 포지션 없음 — 상태 초기화.")
                st.reset()
            elif actual_qty > 0 and actual_qty != st.total_qty:
                log.info(f"[{sym}] 포지션 수량 보정: {st.total_qty} → {actual_qty}주")
                if st.direction == "leg1":
                    st.qty1 = actual_qty
                else:
                    st.qty2 = actual_qty - st.qty1
        self._save_state()

    def _warmup(self, sym: str):
        """최근 거래일 1분봉으로 Bar3Engine 워밍업 (스윙하이/장대양봉 이력 확보)."""
        try:
            client = self.trend._client
            end   = datetime.now(ET)
            start = end - timedelta(days=7)
            req = StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Minute,
                                   start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                                   feed="sip")
            df = client.get_stock_bars(req).df
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(sym, level="symbol")
            df.index = pd.DatetimeIndex(df.index).tz_convert(ET)
            df = df[(df.index.time >= SESSION_OPEN) & (df.index.time < SESSION_CLOSE)]

            engine = self.engines[sym]
            for ts, row in df.iterrows():
                engine.on_minute_bar(float(row["open"]), float(row["high"]),
                                     float(row["low"]), float(row["close"]),
                                     float(row["volume"]), ts.timestamp())
            log.info(f"[{sym}] 3분봉 워밍업 완료: {len(engine.bar3_history)}개")
        except Exception as e:
            log.warning(f"[{sym}] 워밍업 실패: {e}")

    # ── 세션 헬퍼 ─────────────────────────────────────────────────────────────

    def _in_session(self, now: datetime) -> bool:
        now_et = now.astimezone(ET).time()
        return SESSION_OPEN <= now_et < SESSION_CLOSE

    async def _maybe_reset_day(self, now: datetime):
        today = now.astimezone(ET).date()
        if self._current_date == today:
            return
        self._current_date       = today
        self._daily_buy_notional = 0.0
        log.info(f"=== 새 거래일: {today} | 일일 한도 초기화 ===")
        for sym in INSTRUMENTS:
            await asyncio.to_thread(self.trend.refresh, sym)

    def _can_enter(self, notional: float, cash: float) -> tuple[bool, str]:
        spread_cost = notional * (SPREAD_BPS / 10_000)
        total = notional + spread_cost
        limit = cash * DAILY_TURNOVER_LIMIT
        if self._daily_buy_notional + total > limit:
            return False, (f"일일 한도 초과: ${self._daily_buy_notional + total:,.0f} > "
                           f"cash×{DAILY_TURNOVER_LIMIT:.0%}=${limit:,.0f}")
        return True, ""

    # ── WebSocket 핸들러 ──────────────────────────────────────────────────────

    async def _on_bar(self, bar):
        """Alpaca subscribe_bars로 들어오는 1분봉. 3개 모이면 3분봉 처리."""
        sym = getattr(bar, "symbol", None)
        if sym not in self.engines:
            return
        try:
            completed = self.engines[sym].on_minute_bar(
                float(bar.open), float(bar.high), float(bar.low),
                float(bar.close), float(bar.volume or 0), bar.timestamp.timestamp())
            if completed is not None:
                await self._maybe_process_bar(sym)
        except Exception as e:
            log.warning(f"[{sym}] bar 처리 오류: {e}")

    # ── bar 처리 ───────────────────────────────────────────────────────────────

    async def _maybe_process_bar(self, sym: str):
        engine = self.engines[sym]
        bar = engine.latest_bar
        if bar is None or bar["ts"] == self._last_processed_bar_ts[sym]:
            return
        self._last_processed_bar_ts[sym] = bar["ts"]

        now = datetime.now(tz=timezone.utc)
        await self._maybe_reset_day(now)

        if not self._in_session(now):
            return

        log.info(f"[{sym}] 3분봉 {bar['ts'].strftime('%H:%M ET')}  "
                 f"O={bar['open']:.2f} H={bar['high']:.2f} L={bar['low']:.2f} C={bar['close']:.2f}  "
                 f"position={self.states[sym].direction}")

        await self._step(sym, bar, now)
        self._save_state()

    # ── 전략 로직 (종목별 독립) ───────────────────────────────────────────────

    async def _step(self, sym: str, bar: dict, now: datetime):
        st = self.states[sym]
        close, high, low = bar["close"], bar["high"], bar["low"]

        # Step 1: 포지션 보유 중 — 손절/익절 판단
        if st.direction != "flat":
            ma120 = self.trend.ma(sym, 120)
            candidates = []
            if st.stop_px > 0 and low <= st.stop_px:
                candidates.append(st.stop_px)
            if np.isfinite(ma120) and low <= ma120:
                candidates.append(ma120)
            if candidates:
                await self._exit(sym, "손절", now)
                return
            if st.target_px > 0 and high >= st.target_px:
                await self._exit(sym, "익절", now)
                return
            log.info(f"  [{sym}] 보유중 {st.direction} qty={st.total_qty}  "
                     f"stop=${st.stop_px:.2f}  target=${st.target_px:.2f}")
            return

        # Step 2: 트렌드 필터
        if self.trend.status(sym) != "ok":
            return

        ma5, ma20 = self.trend.ma(sym, 5), self.trend.ma(sym, 20)
        bb_lower  = bar.get("bb_lower", np.nan)
        engine    = self.engines[sym]

        # Step 3: 1차 매수
        if np.isfinite(bb_lower) and np.isfinite(ma5):
            near_ma5 = abs(close - ma5) / ma5 <= PROXIMITY_PCT
            if close <= bb_lower and near_ma5:
                await self._enter(sym, "leg1", close, now)
            return

    async def _enter(self, sym: str, leg: str, ref_px: float, now: datetime):
        async with self._trade_locks[sym]:
            st = self.states[sym]
            if leg == "leg1" and st.direction != "flat":
                return
            if leg == "leg2" and st.direction != "leg1":
                return

            equity, cash = await asyncio.to_thread(self.executor.get_account_snapshot)
            if equity <= 0:
                log.warning(f"[{sym}] 진입 실패: 계좌 조회 실패")
                return

            notional = equity * KELLY_LEG
            qty = int(notional / ref_px)
            if qty < 1:
                log.warning(f"[{sym}] 진입 실패: qty={qty}주 (최소 1주 미만)")
                return

            can, reason = self._can_enter(qty * ref_px, cash)
            if not can:
                log.info(f"  [{sym}] 진입 차단 — {reason}")
                return

            await asyncio.to_thread(self.executor.cancel_open_orders, sym)
            ok = await asyncio.to_thread(self.executor.buy, sym, qty)
            if not ok:
                return

            await asyncio.sleep(2.0)
            actual_qty, actual_px = await asyncio.to_thread(
                self.executor.get_position_snapshot, sym)
            fill_px = actual_px if actual_px > 0 else ref_px

            self._daily_buy_notional += qty * fill_px * (1 + SPREAD_BPS / 10_000)

            engine = self.engines[sym]
            if leg == "leg1":
                st.direction = "leg1"
                st.qty1 = qty
                st.entry_px1 = fill_px
                below = engine.surge_opens_below(fill_px)
                st.stop_px   = below[-1] if below else 0.0
                st.target_px = engine.swing_high(exclude_last=True)
                log.info(f"  [{sym}] ✅ 1차 진입: {qty}주 @ ${fill_px:.2f}  "
                         f"stop=${st.stop_px:.2f}  target=${st.target_px:.2f}")
                log_trade_csv(sym, "ENTER_1차", fill_px, qty, 0.0, 0.0, equity)
            else:
                st.direction = "leg2"
                st.qty2 = qty
                st.entry_px2 = fill_px
                log.info(f"  [{sym}] ✅ 2차 진입: {qty}주 @ ${fill_px:.2f}")
                log_trade_csv(sym, "ENTER_2차", fill_px, qty, 0.0, 0.0, equity)

    async def _exit(self, sym: str, reason: str, now: datetime):
        async with self._trade_locks[sym]:
            st = self.states[sym]
            if st.total_qty <= 0:
                return

            ok = await asyncio.to_thread(self.executor.close_position, sym)
            if not ok:
                await asyncio.sleep(1.0)
                await asyncio.to_thread(self.executor.cancel_open_orders, sym)
                ok = await asyncio.to_thread(self.executor.close_position, sym)
                if not ok:
                    log.warning(f"  [{sym}] 청산 재시도도 실패 — 다음 bar에서 재평가")
                    return

            await asyncio.sleep(1.0)
            fill_px, fill_qty = await asyncio.to_thread(self.executor.get_last_fill_info, sym)
            qty = fill_qty if fill_qty > 0 else st.total_qty
            px  = fill_px if fill_px > 0 else st.avg_entry
            gross  = (px - st.avg_entry) * qty
            equity, _ = await asyncio.to_thread(self.executor.get_account_snapshot)
            sign = "✅" if gross >= 0 else "🔴"
            log.info(f"  [{sym}] {sign} {reason}: {qty}주 @ ${px:.2f}  PnL=${gross:+.2f}  "
                     f"portfolio=${equity:,.2f}")
            await asyncio.to_thread(
                log_trade_csv, sym, f"EXIT_{reason}", px, qty, gross, 0.0, equity)
            st.reset()

    # ── 실행 진입점 ────────────────────────────────────────────────────────────

    async def _heartbeat(self):
        while self._running:
            await asyncio.sleep(60)
            equity, cash = await asyncio.to_thread(self.executor.get_account_snapshot)
            now_et = datetime.now(ET).strftime("%H:%M ET")
            positions = " | ".join(
                f"{sym}: {self.states[sym].direction}({self.states[sym].total_qty}주)"
                for sym in INSTRUMENTS)
            log.info(f"[heartbeat] {now_et}  {positions}  일일누적=${self._daily_buy_notional:,.0f}")

    def _on_shutdown(self, *_):
        log.info("종료 신호 수신 — 안전 종료 중...")
        self._running = False
        self._save_state()
        log.info("상태 저장 완료 (포지션은 유지 — 멀티데이 보유 설계, 재시작 시 복구)")
        sys.exit(0)

    def run(self):
        equity, cash = self.executor.get_account_snapshot()
        now_et = datetime.now(ET)

        log.info("=" * 65)
        log.info("BB Trend Pullback Live Trader 시작 (Paper Trading)")
        log.info(f"종목     : {', '.join(INSTRUMENTS)}")
        log.info(f"Kelly    : {KELLY_LEG*100:.1f}%/leg  (1차+2차 최대 {KELLY_LEG*2*100:.0f}%)")
        log.info(f"세션     : {SESSION_OPEN.strftime('%H:%M')}~{SESSION_CLOSE.strftime('%H:%M')} ET")
        log.info(f"일일 한도: cash × {DAILY_TURNOVER_LIMIT*100:.0f}% = ${cash * DAILY_TURNOVER_LIMIT:,.0f}")
        log.info(f"계좌     : equity=${equity:,.2f}  cash=${cash:,.2f}")
        log.info(f"현재시각 : {now_et.strftime('%Y-%m-%d %H:%M ET')}")
        log.info("⚠️  멀티데이 보유 설계 — EOD 강제청산 없음")
        log.info("=" * 65)

        signal.signal(signal.SIGINT,  self._on_shutdown)
        signal.signal(signal.SIGTERM, self._on_shutdown)

        log.info(f"스트림 구독 시작 (1분봉): {', '.join(INSTRUMENTS)}")

        async def _run():
            hb_task = asyncio.create_task(self._heartbeat())
            reconnect_delay = 5
            while self._running:
                try:
                    self.stream = StockDataStream(
                        ALPACA_API_KEY, ALPACA_SECRET_KEY, feed=DataFeed.SIP)
                    for sym in INSTRUMENTS:
                        self.stream.subscribe_bars(self._on_bar, sym)
                    reconnect_delay = 5
                    await self.stream._run_forever()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    if not self._running:
                        break
                    log.warning(f"WebSocket 연결 끊김: {e}")
                    log.info(f"{reconnect_delay}초 후 재연결 시도...")
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 60)
            hb_task.cancel()

        try:
            asyncio.run(_run())
        except KeyboardInterrupt:
            self._on_shutdown()


# ── 진입점 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ans = sys.argv[1].strip().upper()
    INSTRUMENTS.extend(["SPY", "QQQ"] if ans == "BOTH" else [ans])
    print(f"  선택된 종목: {', '.join(INSTRUMENTS)}  (Paper Trading)\n")

    trader = BBTrendLiveTrader()
    trader.run()
