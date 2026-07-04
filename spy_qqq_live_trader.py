#!/usr/bin/env python3
"""
SPY/QQQ Live Trader — OBI+TFI Signal Architecture (BTC 구조 이식)
──────────────────────────────────────────────────────────────────
BTC Live Trader와 동일한 신호 아키텍처를 미국 주식(SPY/QQQ)에 적용.

BTC와 달라진 점:
  - CryptoDataStream → StockDataStream (SIP feed)
  - 24h 운영 → 미국 정규장만 (09:45~15:55 ET)
  - qty: float(BTC 소수점) → int(주식 정수 주수)
  - close_all() → 종목별 close_position(sym)
  - 종목 2개 독립 운영: SignalEngine × 2, PositionState × 2
  - 일일 매수 금액 누적 추적 → cash × 50% 초과 시 진입 차단

파라미터 (백테스터 수치 × 2배, Kelly 2.5%):
  - STOP_LOSS:  0.5% → 1.0%
  - PARTIAL:    0.4% → 0.8%
  - KELLY:      2.5% / 진입 (최대 20회 × 2.5% = 50%)
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

# 시스템 로컬 타임존(KST 등)이 아닌 ET 거래일 기준으로 로그 폴더 결정.
# 로컬 날짜로 잡으면 자정 부근 재시작 시 _current_date(ET 기준)와 어긋날 수 있음.
LOG_DIR = Path(__file__).parent / "logs" / datetime.now(ET).strftime("%Y/%m/%d")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "spy_qqq_trader.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("spy_qqq_trader")

# ── 종목 설정 (런타임에 결정) ─────────────────────────────────────────────────
INSTRUMENTS: list[str] = []   # main()에서 채워짐

# ── 세션 (ET 기준) ─────────────────────────────────────────────────────────────
SESSION_OPEN    = dtime(9, 45)   # 오프닝 레인지 15분 건너뜀
SESSION_CLOSE   = dtime(15, 55)  # EOD 잔여 포지션 강제 청산
EOD_TIGHT_TIME  = dtime(14, 55)  # 이후 손절/익절 기준 절반으로 강화
EOD_REDUCE_TIME = dtime(15, 45)  # 신규 진입 차단 + 보유 포지션 50% 선제청산

# ── 전략 파라미터 (백테스터 수치 × 2배) ──────────────────────────────────────
KELLY                = 0.025      # 진입당 2.5% 배분
STOP_LOSS_PCT        = 0.010      # -1.0%   (백테스터 0.5% × 2)
PARTIAL_PCT          = 0.008      # +0.8%   (백테스터 0.4% × 2)
TRAIL_PULLBACK       = 0.30       # 고점 대비 30% 되돌림
CONVICTION_LONG_EXIT = -0.10      # 신호소멸 청산 기준
MIN_CONV             = 0.20       # 진입 최소 신호 (이상이면 진입 예약)
MIN_LOCAL_STOP_DIST  = 0.003      # 로컬저점 최소 거리: 진입가 대비 0.3% 이상이어야 활성화
SWITCH_COOLDOWN_MIN  = 15         # 진입 후 재진입 최소 간격 (분)
CONVICTION_COOLDOWN  = 5          # 신호소멸 청산 후 재진입 대기 (분)

OBI_ALPHA       = 0.08
TFI_ALPHA       = 0.15
TFI_WINDOW_SEC  = 300             # TFI rolling 5분 window

SPREAD_BPS      = 0.5             # SPY/QQQ 편도 스프레드 추정 (~0.5bp)

# ── 일일 매매 한도 ─────────────────────────────────────────────────────────────
DAILY_TURNOVER_LIMIT = 0.50       # 하루 누적 매수 금액 ≤ 전체 cash × 50%

STATE_FILE = LOG_DIR / "spy_qqq_state.json"
TRADE_CSV  = LOG_DIR / "spy_qqq_trades.csv"


# ── 매매 CSV 로거 ─────────────────────────────────────────────────────────────

def log_trade_csv(sym: str, action: str, price: float, qty: int,
                  pnl: float = 0.0, slippage_bps: float = 0.0,
                  sig: float = 0.0, portfolio: float = 0.0):
    """매매 이벤트를 CSV에 한 줄 추가."""
    write_header = not TRADE_CSV.exists()
    with open(TRADE_CSV, mode="a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp_et", "symbol", "action", "price", "qty",
                        "pnl_usd", "slippage_bps", "signal", "portfolio"])
        w.writerow([
            datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S"),
            sym, action,
            round(price, 4), qty,
            round(pnl, 2),
            round(slippage_bps, 2),
            round(sig, 4),
            round(portfolio, 2),
        ])


# ── 신호 엔진 (종목별 독립) ───────────────────────────────────────────────────

class SignalEngine:
    """
    실시간 tick 데이터로 OBI+TFI 신호를 1분 bar 단위로 계산.
    BTC Live Trader와 동일한 구조.

    OBI: 호가 잔량 불균형의 EWM
    TFI: Lee-Ready 분류 기반 체결 방향 비율의 EWM
    """

    def __init__(self):
        self._obi_ewm    = 0.0
        self._tfi_ewm    = 0.0
        self._last_bid   = np.nan
        self._last_ask   = np.nan
        self._trade_buf: deque = deque()   # (ts_sec, buy_vol, sell_vol)
        self._rolling_buy_vol  = 0.0       # O(1) 롤링 합산
        self._rolling_sell_vol = 0.0
        self._bar_start_sec: int = 0
        self.bar_signals: deque = deque(maxlen=60)
        # ── bar OHLC 추적 (로컬저점 손절용) ──────────────────────────────
        self._bar_open:  float = np.nan
        self._bar_high:  float = -np.inf
        self._bar_low:   float = +np.inf
        self._bar_close: float = np.nan
        self.bar_prices: deque = deque(maxlen=30)  # 최근 30 bar OHLC

    def on_quote(self, bid_px: float, ask_px: float,
                 bid_sz: float, ask_sz: float, ts_sec: float):
        if bid_sz <= 0 or ask_sz <= 0 or bid_px <= 0 or ask_px <= 0:
            return
        self._last_bid = bid_px
        self._last_ask = ask_px
        raw = (bid_sz - ask_sz) / (bid_sz + ask_sz)
        self._obi_ewm = OBI_ALPHA * raw + (1 - OBI_ALPHA) * self._obi_ewm
        self._check_bar(ts_sec)

    def on_trade(self, price: float, size: float, ts_sec: float):
        if not (np.isfinite(self._last_bid) and np.isfinite(self._last_ask)):
            return
        if price >= self._last_ask:
            buy_vol, sell_vol = size, 0.0
        elif price <= self._last_bid:
            buy_vol, sell_vol = 0.0, size
        else:
            buy_vol, sell_vol = 0.0, 0.0
        self._trade_buf.append((ts_sec, buy_vol, sell_vol))
        self._rolling_buy_vol  += buy_vol
        self._rolling_sell_vol += sell_vol
        self._purge_old_trades(ts_sec)
        self._update_tfi_ewm()
        self._check_bar(ts_sec)       # ← 먼저: 이전 bar 확정 & OHLC 저장
        # ── 현재 bar OHLC 업데이트 (bar 경계 이후 실행) ─────────────────
        if np.isnan(self._bar_open):
            self._bar_open = price
        self._bar_high  = max(self._bar_high, price)
        self._bar_low   = min(self._bar_low,  price)
        self._bar_close = price

    def _purge_old_trades(self, now_sec: float):
        cutoff = now_sec - TFI_WINDOW_SEC
        while self._trade_buf and self._trade_buf[0][0] < cutoff:
            _, b_vol, s_vol = self._trade_buf.popleft()
            self._rolling_buy_vol  -= b_vol
            self._rolling_sell_vol -= s_vol

    def _update_tfi_ewm(self):
        total = self._rolling_buy_vol + self._rolling_sell_vol
        if total <= 0:
            return
        raw = (self._rolling_buy_vol - self._rolling_sell_vol) / total
        self._tfi_ewm = TFI_ALPHA * raw + (1 - TFI_ALPHA) * self._tfi_ewm

    def _check_bar(self, ts_sec: float):
        """1분 경계 감지 → 이전 bar 신호 & OHLC 확정."""
        bar_sec = int(ts_sec // 60) * 60
        if bar_sec != self._bar_start_sec:
            if self._bar_start_sec > 0:
                sig = float(np.clip(
                    self._obi_ewm * 0.35 + self._tfi_ewm * 0.65, -1, 1))
                bar_ts = datetime.fromtimestamp(self._bar_start_sec, tz=timezone.utc)
                self.bar_signals.append({"ts": bar_ts, "signal": sig})
                # bar OHLC 저장 (trade 데이터가 있는 bar만)
                if np.isfinite(self._bar_low) and self._bar_low < np.inf:
                    self.bar_prices.append({
                        "ts":    bar_ts,
                        "open":  self._bar_open  if np.isfinite(self._bar_open)  else self._bar_low,
                        "high":  self._bar_high  if np.isfinite(self._bar_high)  else self._bar_low,
                        "low":   self._bar_low,
                        "close": self._bar_close if np.isfinite(self._bar_close) else self._bar_low,
                    })
            self._bar_start_sec = bar_sec
            # 새 bar OHLC 초기화
            self._bar_open  = np.nan
            self._bar_high  = -np.inf
            self._bar_low   = +np.inf
            self._bar_close = np.nan

    @property
    def latest_signal(self) -> float:
        if not self.bar_signals:
            return 0.0
        return self.bar_signals[-1]["signal"]

    @property
    def latest_bar_ts(self) -> Optional[datetime]:
        if not self.bar_signals:
            return None
        return self.bar_signals[-1]["ts"]

    @property
    def latest_local_min(self) -> float:
        """
        완성된 bar 기준 가장 최근의 로컬 저점(low).

        정의: lows[i] <= lows[i-1] AND lows[i] <= lows[i+1] 을 만족하는
              가장 오른쪽(최근) bar 의 low 값.
        최소 3 bar 필요. 없으면 0.0 반환.
        """
        lows = [b["low"] for b in self.bar_prices
                if np.isfinite(b["low"]) and b["low"] > 0]
        if len(lows) < 3:
            return 0.0
        for i in range(len(lows) - 2, 0, -1):  # 최근 → 과거 순 탐색
            if lows[i] <= lows[i - 1] and lows[i] <= lows[i + 1]:
                return lows[i]
        return 0.0  # 로컬 저점 없음


# ── 포지션 상태 (종목별 독립, JSON 영속) ─────────────────────────────────────

class PositionState:
    """포지션 상태. JSON으로 영속 (재시작 시 복구)."""

    def __init__(self, sym: str):
        self.sym             = sym
        self.direction       = "flat"
        self.entry_px        = 0.0
        self.qty             = 0          # 정수 주수 (BTC와 달리 소수점 없음)
        self.pnl_peak        = 0.0
        self.partial_done    = False
        self.last_switch:    Optional[datetime] = None
        self.last_conv_exit: Optional[datetime] = None
        self.pending_sig:    Optional[float] = None
        self.local_min_stop: float = 0.0  # 진입 전 가장 최근 로컬저점 (고정 손절선)
        self.eod_reduced:    bool  = False # 당일 15:45 50% 선제청산 완료 여부

    def to_dict(self) -> dict:
        return {
            "direction":      self.direction,
            "entry_px":       self.entry_px,
            "qty":            self.qty,
            "pnl_peak":       self.pnl_peak,
            "partial_done":   self.partial_done,
            "pending_sig":    self.pending_sig,
            "local_min_stop": self.local_min_stop,
            "eod_reduced":    self.eod_reduced,
            "last_switch":    self.last_switch.isoformat() if self.last_switch else None,
            "last_conv_exit": self.last_conv_exit.isoformat() if self.last_conv_exit else None,
        }

    def from_dict(self, data: dict):
        self.direction      = data.get("direction", "flat")
        self.entry_px       = data.get("entry_px", 0.0)
        self.qty            = int(data.get("qty", 0))
        self.pnl_peak       = data.get("pnl_peak", 0.0)
        self.partial_done   = data.get("partial_done", False)
        self.pending_sig    = data.get("pending_sig")
        self.local_min_stop = data.get("local_min_stop", 0.0)
        self.eod_reduced    = data.get("eod_reduced", False)
        ls  = data.get("last_switch")
        lce = data.get("last_conv_exit")
        self.last_switch    = datetime.fromisoformat(ls)  if ls  else None
        self.last_conv_exit = datetime.fromisoformat(lce) if lce else None

    def reset(self):
        self.direction      = "flat"
        self.entry_px       = 0.0
        self.qty            = 0
        self.pnl_peak       = 0.0
        self.partial_done   = False
        self.last_switch    = None
        self.pending_sig    = None
        self.local_min_stop = 0.0
        self.eod_reduced    = False

    def switch_ok(self, now: datetime) -> bool:
        if self.last_switch is None:
            return True
        return (now - self.last_switch).total_seconds() >= SWITCH_COOLDOWN_MIN * 60

    def conv_ok(self, now: datetime) -> bool:
        if self.last_conv_exit is None:
            return True
        return (now - self.last_conv_exit).total_seconds() >= CONVICTION_COOLDOWN * 60


# ── 변동성 스케일러 (ATR14 기반 동적 파라미터 조정) ─────────────────────────

class VolatilityScaler:
    """
    일봉 ATR14 기반으로 손절/익절 폭을 동적 조정.

    current_atr / avg_atr(30일) 비율을 [0.6, 2.0] 범위로 클램핑.
    - 저변동성(ratio < 1): 손절·익절 폭 축소 → 타이트한 리스크 관리
    - 고변동성(ratio > 1): 손절·익절 폭 확대 → 노이즈 회피
    _maybe_reset_day() 호출 시 하루 1회 갱신.
    """

    _ATR_PERIOD = 14
    _LOOKBACK   = 30
    _SCALE_MIN  = 0.6
    _SCALE_MAX  = 2.0

    def __init__(self):
        self._ratios: dict[str, float] = {}
        self._client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    def refresh(self, sym: str) -> None:
        """새 거래일 시작 시 ATR 비율 갱신. 실패하면 1.0(무스케일) 유지."""
        bars = self._fetch_daily_bars(sym)
        if bars is None or len(bars) < self._ATR_PERIOD + 5:
            log.warning(f"[{sym}] ATR 계산 데이터 부족 → scale=1.0 유지")
            self._ratios[sym] = 1.0
            return

        atr_series  = self._compute_atr(bars)
        current_atr = float(atr_series.iloc[-1])
        avg_atr     = float(atr_series.iloc[-self._LOOKBACK:].mean())
        raw_ratio   = current_atr / avg_atr if avg_atr > 0 else 1.0
        ratio       = float(np.clip(raw_ratio, self._SCALE_MIN, self._SCALE_MAX))

        self._ratios[sym] = ratio
        log.info(
            f"[{sym}] ATR scale={ratio:.3f}  "
            f"current={current_atr:.4f}  avg30={avg_atr:.4f}  "
            f"→ stop={STOP_LOSS_PCT * ratio * 100:.3f}%  "
            f"partial={PARTIAL_PCT * ratio * 100:.3f}%"
        )

    def scale(self, sym: str) -> float:
        return self._ratios.get(sym, 1.0)

    def stop_pct(self, sym: str) -> float:
        return STOP_LOSS_PCT * self.scale(sym)

    def partial_pct(self, sym: str) -> float:
        return PARTIAL_PCT * self.scale(sym)

    def _fetch_daily_bars(self, sym: str) -> "pd.DataFrame | None":
        try:
            n_fetch = (self._ATR_PERIOD + self._LOOKBACK) * 2  # 주말·휴장 여유분
            end     = datetime.now(ET).date()
            start   = end - timedelta(days=n_fetch)
            req     = StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            )
            raw = self._client.get_stock_bars(req).df
            if isinstance(raw.index, pd.MultiIndex):
                raw = raw.loc[sym]
            return raw.tail(self._ATR_PERIOD + self._LOOKBACK)
        except Exception as e:
            log.warning(f"[{sym}] ATR 데이터 조회 실패: {e}")
            return None

    @staticmethod
    def _compute_atr(bars: pd.DataFrame) -> pd.Series:
        high, low, close = bars["high"], bars["low"], bars["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=VolatilityScaler._ATR_PERIOD, adjust=False).mean()



# ── 주문 실행 (Alpaca Stock API) ──────────────────────────────────────────────

class OrderExecutor:
    def __init__(self, trading_client: TradingClient):
        self.client = trading_client

    def buy(self, sym: str, qty: int) -> bool:
        try:
            req = MarketOrderRequest(
                symbol=sym, qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            order = self.client.submit_order(req)
            log.info(f"BUY 주문 제출: {sym} {qty}주  order_id={order.id}")
            return True
        except Exception as e:
            log.error(f"BUY 주문 실패 [{sym}]: {e}")
            return False

    def sell(self, sym: str, qty: int) -> bool:
        try:
            req = MarketOrderRequest(
                symbol=sym, qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = self.client.submit_order(req)
            log.info(f"SELL 주문 제출: {sym} {qty}주  order_id={order.id}")
            return True
        except Exception as e:
            log.error(f"SELL 주문 실패 [{sym}]: {e}")
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
        """해당 종목의 미체결 주문을 전량 취소. 취소된 건수 반환."""
        try:
            orders = self.client.get_orders(GetOrdersRequest(
                status=QueryOrderStatus.OPEN,
                symbols=[sym],
            ))
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

    def get_last_sell_fill_info(self, sym: str) -> tuple[float, int]:
        """가장 최근 체결된 SELL 주문의 (평균체결가, 체결수량) 반환. 실패 시 (0.0, 0)."""
        try:
            today_et = datetime.now(ET).date()
            today_start = ET.localize(
                datetime.combine(today_et, dtime(0, 0, 0))
            ).astimezone(timezone.utc)
            orders = self.client.get_orders(GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                symbols=[sym],
                limit=10,
                after=today_start,
            ))
            for order in orders:
                if (str(order.side).lower() in ("sell", "orderside.sell")
                        and order.filled_avg_price is not None
                        and order.filled_qty is not None):
                    return float(order.filled_avg_price), int(float(order.filled_qty))
            return 0.0, 0
        except Exception as e:
            log.warning(f"[{sym}] SELL 체결 정보 조회 실패: {e}")
            return 0.0, 0

    def get_position_qty(self, sym: str) -> int:
        try:
            pos = self.client.get_open_position(sym)
            return int(float(pos.qty))
        except Exception:
            return 0

    def get_avg_entry_price(self, sym: str) -> float:
        try:
            pos = self.client.get_open_position(sym)
            return float(pos.avg_entry_price)
        except Exception:
            return 0.0

    def get_current_price(self, sym: str) -> float:
        try:
            pos = self.client.get_open_position(sym)
            return float(pos.current_price)
        except Exception:
            return 0.0

    def get_position_snapshot(self, sym: str) -> tuple[int, float]:
        """(qty, avg_entry_price)를 단일 API 호출로 조회. 실패 시 (0, 0.0)."""
        try:
            pos = self.client.get_open_position(sym)
            return int(float(pos.qty)), float(pos.avg_entry_price)
        except Exception:
            return 0, 0.0

    def get_account_equity(self) -> float:
        try:
            return float(self.client.get_account().equity)
        except Exception as e:
            log.warning(f"계좌 equity 조회 실패: {e}")
            return 0.0

    def get_account_cash(self) -> float:
        try:
            return float(self.client.get_account().cash)
        except Exception as e:
            log.warning(f"계좌 cash 조회 실패: {e}")
            return 0.0

    def get_account_snapshot(self) -> tuple[float, float]:
        """(equity, cash)를 단일 API 호출로 조회. 실패 시 (0.0, 0.0)."""
        try:
            acct = self.client.get_account()
            return float(acct.equity), float(acct.cash)
        except Exception as e:
            log.warning(f"계좌 정보 조회 실패: {e}")
            return 0.0, 0.0

    def get_day_orders(self, day: date) -> list:
        """ET 기준 당일 체결된 모든 주문 반환 (체결 완료 주문만)."""
        try:
            day_start = ET.localize(datetime.combine(day, dtime(0,  0,  0))).astimezone(timezone.utc)
            day_end   = ET.localize(datetime.combine(day, dtime(23, 59, 59))).astimezone(timezone.utc)
            orders = self.client.get_orders(GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                after=day_start,
                until=day_end,
                limit=500,
            ))
            return [o for o in orders
                    if o.filled_avg_price is not None and float(o.filled_qty or 0) > 0]
        except Exception as e:
            log.warning(f"당일 주문 조회 실패: {e}")
            return []


# ── 메인 트레이더 ─────────────────────────────────────────────────────────────

class SPYQQQLiveTrader:
    def __init__(self):
        self.engines     = {sym: SignalEngine()     for sym in INSTRUMENTS}
        self.states      = {sym: PositionState(sym) for sym in INSTRUMENTS}
        self.vol_scaler  = VolatilityScaler()
        self.executor    = OrderExecutor(
            TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True))
        self.stream      = None
        self._last_processed_bar_ts: dict[str, Optional[datetime]] = {
            sym: None for sym in INSTRUMENTS}
        self._trade_locks = {sym: asyncio.Lock() for sym in INSTRUMENTS}
        self._running            = True
        self._current_date:      Optional[date] = None
        self._daily_buy_notional = 0.0   # 당일 누적 매수 금액 (슬리피지 포함)
        self._initial_equity     = 0.0   # 당일 시작 equity (일별 손익 계산용)
        self._eod_summary_done   = False # 당일 EOD 요약 출력 여부

        self._load_state()
        self._sync_with_broker()

        # 시작 시 즉시 ATR 로드 (장 중 재시작 포함)
        log.info("ATR 스케일 초기 로드 중...")
        for sym in INSTRUMENTS:
            self.vol_scaler.refresh(sym)

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
                # 당일 데이터만 복구 (익일은 한도 리셋)
                self._daily_buy_notional = data.get("daily_buy_notional", 0.0)
                self._current_date = today
            for sym in INSTRUMENTS:
                if sym in data:
                    self.states[sym].from_dict(data[sym])
            log.info(f"상태 복구: 일일누적=${self._daily_buy_notional:,.0f}")
            for sym in INSTRUMENTS:
                st = self.states[sym]
                log.info(f"  [{sym}] direction={st.direction}  qty={st.qty}주")
        except Exception as e:
            log.warning(f"상태 파일 로드 실패 ({e}) — 초기화로 시작")

    def _sync_with_broker(self):
        """시작 시 Alpaca 실제 포지션과 상태 파일 자동 대조."""
        equity, cash = self.executor.get_account_snapshot()
        log.info(f"계좌 현황: equity=${equity:,.2f}  cash=${cash:,.2f}")

        for sym in INSTRUMENTS:
            actual_qty = self.executor.get_position_qty(sym)
            st         = self.states[sym]

            if actual_qty > 0 and st.direction == "flat":
                log.warning(f"[{sym}] 실제 포지션 {actual_qty}주 있음 — 상태 flat 불일치. "
                            f"수동 확인 필요.")
            elif actual_qty == 0 and st.direction == "long":
                log.warning(f"[{sym}] 상태는 long이나 실제 포지션 없음 — 상태 초기화.")
                st.reset()
            elif actual_qty > 0 and st.direction == "long" and actual_qty != st.qty:
                log.info(f"[{sym}] 포지션 수량 보정: {st.qty} → {actual_qty}주")
                st.qty = actual_qty

        self._save_state()

    # ── 세션/EOD 헬퍼 ─────────────────────────────────────────────────────────

    def _in_session(self, now: datetime) -> bool:
        now_et = now.astimezone(ET).time()
        return SESSION_OPEN <= now_et < SESSION_CLOSE

    def _is_eod(self, now: datetime) -> bool:
        now_et = now.astimezone(ET).time()
        return now_et >= SESSION_CLOSE

    def _is_eod_tight(self, now: datetime) -> bool:
        now_et = now.astimezone(ET).time()
        return now_et >= EOD_TIGHT_TIME

    def _is_eod_reduce(self, now: datetime) -> bool:
        return now.astimezone(ET).time() >= EOD_REDUCE_TIME

    async def _maybe_reset_day(self, now: datetime):
        today = now.astimezone(ET).date()
        if self._current_date == today:
            return
        self._current_date       = today
        self._daily_buy_notional = 0.0
        self._eod_summary_done   = False
        # 동기 네트워크 호출을 스레드로 위임 — 이벤트 루프 블로킹 방지
        self._initial_equity = await asyncio.to_thread(self.executor.get_account_equity)
        log.info(f"=== 새 거래일: {today} | 일일 한도 초기화 | "
                 f"시작 equity=${self._initial_equity:,.2f} ===")
        # ATR 갱신(동기 HTTP)도 스레드로 위임, EOD 플래그 초기화
        for sym in INSTRUMENTS:
            await asyncio.to_thread(self.vol_scaler.refresh, sym)
            self.states[sym].eod_reduced = False

    async def _print_daily_summary(self, now: datetime):
        """장 마감 후 일별 손익 요약 출력 + 캔들 차트 PNG 자동 생성."""
        final_equity = await asyncio.to_thread(self.executor.get_account_equity)
        today        = now.astimezone(ET).date()
        pnl          = final_equity - self._initial_equity
        pnl_pct      = pnl / self._initial_equity * 100 if self._initial_equity > 0 else 0.0
        sign         = "✅" if pnl >= 0 else "🔴"
        log.info("=" * 65)
        log.info(f"{sign} [{today}] 일별 손익 요약")
        log.info(f"  시작 equity : ${self._initial_equity:>12,.2f}")
        log.info(f"  최종 equity : ${final_equity:>12,.2f}")
        log.info(f"  당일 PnL    : ${pnl:>+12,.2f}  ({pnl_pct:+.3f}%)")
        log.info("=" * 65)

        # ── EOD 사후 정산 → 캔들 차트 생성 (백그라운드, 순서 보장) ───────────
        chart_sym = "QQQ" if "QQQ" in INSTRUMENTS else INSTRUMENTS[0]
        date_str  = today.strftime("%Y-%m-%d")
        yyyymmdd  = today.strftime("%Y%m%d")
        out_path  = LOG_DIR / f"trades_{yyyymmdd}_candle.png"
        asyncio.create_task(self._eod_tasks(chart_sym, date_str, out_path, today))

    async def _reconcile_csv(self, today: date):
        """장 마감 후 Alpaca 실제 체결 기준으로 당일 CSV를 재작성."""
        log.info("[reconcile] Alpaca 실제 체결 기준 CSV 정산 시작...")
        orders = await asyncio.to_thread(self.executor.get_day_orders, today)
        if not orders:
            log.warning("[reconcile] 당일 주문 없음 — 정산 건너뜀")
            return
        if not TRADE_CSV.exists():
            log.warning("[reconcile] CSV 파일 없음 — 정산 건너뜀")
            return

        df = pd.read_csv(TRADE_CSV)
        if df.empty:
            return

        df["timestamp_et"] = pd.to_datetime(df["timestamp_et"])
        today_mask = df["timestamp_et"].dt.date == today
        if not today_mask.any():
            log.warning(f"[reconcile] {today} 거래 행 없음 — 정산 건너뜀")
            return

        # 종목별 BUY / SELL 주문 목록 (체결 시각 오름차순)
        _min_dt = datetime.min.replace(tzinfo=timezone.utc)
        sym_buys  = {}
        sym_sells = {}
        for sym in INSTRUMENTS:
            sym_buys[sym] = sorted(
                [o for o in orders
                 if o.symbol == sym and str(o.side).lower() in ("buy", "orderside.buy")],
                key=lambda o: o.filled_at or _min_dt,
            )
            sym_sells[sym] = sorted(
                [o for o in orders
                 if o.symbol == sym and str(o.side).lower() in ("sell", "orderside.sell")],
                key=lambda o: o.filled_at or _min_dt,
            )

        buy_idx  = {sym: 0 for sym in INSTRUMENTS}
        sell_idx = {sym: 0 for sym in INSTRUMENTS}
        entry_px = {sym: 0.0 for sym in INSTRUMENTS}
        matched  = 0

        for i, row in df[today_mask].iterrows():
            sym    = row["symbol"]
            action = str(row["action"])

            if action == "ENTER":
                blist = sym_buys.get(sym, [])
                idx   = buy_idx.get(sym, 0)
                if idx < len(blist):
                    o = blist[idx]
                    df.at[i, "price"]   = float(o.filled_avg_price)
                    df.at[i, "qty"]     = int(float(o.filled_qty))
                    df.at[i, "pnl_usd"] = 0.0
                    entry_px[sym] = float(o.filled_avg_price)
                    buy_idx[sym]  = idx + 1
                    matched += 1
                else:
                    log.warning(f"[reconcile] {sym} ENTER 매칭 실패 (row {i})")

            elif action.startswith("EXIT") or action == "PARTIAL_EXIT":
                slist = sym_sells.get(sym, [])
                idx   = sell_idx.get(sym, 0)
                if idx < len(slist):
                    o        = slist[idx]
                    exit_px  = float(o.filled_avg_price)
                    exit_qty = int(float(o.filled_qty))
                    pnl      = (exit_px - entry_px[sym]) * exit_qty if entry_px[sym] > 0 else 0.0
                    df.at[i, "price"]   = exit_px
                    df.at[i, "qty"]     = exit_qty
                    df.at[i, "pnl_usd"] = round(pnl, 2)
                    sell_idx[sym] = idx + 1
                    if action.startswith("EXIT"):
                        entry_px[sym] = 0.0
                    matched += 1
                else:
                    log.warning(f"[reconcile] {sym} EXIT 매칭 실패 (row {i})")

        # 정산 결과 요약 로그 (timestamp 문자열 변환 전에 계산해야 함)
        exits = df[today_mask & (
            df["action"].str.startswith("EXIT") | (df["action"] == "PARTIAL_EXIT")
        )]
        total_pnl  = exits["pnl_usd"].astype(float).sum()
        n_trades   = len(exits)
        win_trades = (exits["pnl_usd"].astype(float) > 0).sum()
        win_rate   = win_trades / n_trades * 100 if n_trades > 0 else 0.0

        # timestamp 문자열 복원 후 재작성
        df["timestamp_et"] = df["timestamp_et"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df.to_csv(TRADE_CSV, index=False, float_format="%.4f")

        log.info(f"[reconcile] ✅ 정산 완료: {matched}건 수정 | "
                 f"총 PnL=${total_pnl:+.2f} | 거래 {n_trades}건 | 승률={win_rate:.0f}%")

    async def _eod_tasks(self, sym: str, date_str: str, out_path: Path, today: date):
        """EOD: CSV 정산 완료 후 캔들 차트 생성 (순서 보장)."""
        await self._reconcile_csv(today)
        await self._generate_chart(sym, date_str, out_path)

    async def _generate_chart(self, sym: str, date_str: str, out_path: Path):
        """캔들 차트 PNG를 백그라운드 스레드에서 생성."""
        try:
            log.info(f"  [chart] {sym} {date_str} PNG 생성 중...")
            from analysis.live_chart import draw_candle_chart
            await asyncio.to_thread(
                draw_candle_chart,
                TRADE_CSV,
                date_str,
                out_path,
                ALPACA_API_KEY,
                ALPACA_SECRET_KEY,
                sym,
            )
        except Exception as e:
            log.warning(f"  [chart] PNG 생성 실패: {e}")

    def _can_enter(self, notional: float, cash: float) -> tuple[bool, str]:
        """일일 매매 한도 체크 (슬리피지 포함)."""
        spread_cost = notional * (SPREAD_BPS / 10_000)
        total       = notional + spread_cost
        limit       = cash * DAILY_TURNOVER_LIMIT
        if self._daily_buy_notional + total > limit:
            return False, (
                f"일일 한도 초과: "
                f"${self._daily_buy_notional + total:,.0f} > "
                f"cash×{DAILY_TURNOVER_LIMIT:.0%}=${limit:,.0f}"
            )
        return True, ""

    # ── WebSocket 핸들러 ──────────────────────────────────────────────────────

    async def _on_quote(self, quote):
        sym = getattr(quote, "symbol", None)
        if sym not in self.engines:
            return
        try:
            self.engines[sym].on_quote(
                bid_px=float(quote.bid_price or 0),
                ask_px=float(quote.ask_price or 0),
                bid_sz=float(quote.bid_size  or 0),
                ask_sz=float(quote.ask_size  or 0),
                ts_sec=quote.timestamp.timestamp(),
            )
            await self._maybe_process_bar(sym)
        except Exception as e:
            log.warning(f"[{sym}] quote 처리 오류: {e}")

    async def _on_trade(self, trade):
        sym = getattr(trade, "symbol", None)
        if sym not in self.engines:
            return
        try:
            self.engines[sym].on_trade(
                price=float(trade.price or 0),
                size=float(trade.size  or 0),
                ts_sec=trade.timestamp.timestamp(),
            )
        except Exception as e:
            log.warning(f"[{sym}] trade 처리 오류: {e}")

    # ── bar 처리 ───────────────────────────────────────────────────────────────

    async def _maybe_process_bar(self, sym: str):
        """새 1분 bar가 완성되었을 때만 신호 처리 (종목별 독립)."""
        bar_ts = self.engines[sym].latest_bar_ts
        if bar_ts is None or bar_ts == self._last_processed_bar_ts[sym]:
            return

        self._last_processed_bar_ts[sym] = bar_ts
        now = datetime.now(tz=timezone.utc)
        await self._maybe_reset_day(now)

        st     = self.states[sym]
        engine = self.engines[sym]

        def _mid_px() -> float:
            bid, ask = engine._last_bid, engine._last_ask
            return ((bid + ask) / 2
                    if np.isfinite(bid) and np.isfinite(ask) and bid > 0
                    else 0.0)

        # ── 15:45 ET: 신규 진입 차단 + 보유 포지션 50% 선제청산 (1회) ──────
        if self._is_eod_reduce(now) and not st.eod_reduced:
            st.eod_reduced = True
            st.pending_sig = None
            if st.direction == "long" and st.qty > 0:
                px = _mid_px()
                log.info(f"[{sym}] 15:45 EOD 선제청산 50%  현재가=${px:.2f}")
                await self._partial_close(sym, px)
            self._save_state()

        # ── 15:55 ET: 잔여 포지션 전량 강제청산 ─────────────────────────────
        if self._is_eod(now) and st.direction == "long":
            px = _mid_px()
            await self._exit(sym, "EOD 강제청산", 0.0, px, now)
            self._save_state()

        # ── 모든 종목 flat → 일별 요약 1회 출력 ──────────────────────────────
        if (self._is_eod(now)
                and not self._eod_summary_done
                and all(self.states[s].direction == "flat" for s in INSTRUMENTS)):
            await self._print_daily_summary(now)
            self._eod_summary_done = True
            return

        if not self._in_session(now):
            st.pending_sig = None
            return

        sig    = self.engines[sym].latest_signal
        now_et = now.astimezone(ET)
        log.info(f"[{sym}] bar {now_et.strftime('%H:%M ET')}  "
                 f"signal={sig:+.4f}  position={st.direction}  "
                 f"atr_scale={self.vol_scaler.scale(sym):.3f}")

        await self._step(sym, sig, now)
        self._save_state()

    # ── 전략 로직 (종목별 독립) ───────────────────────────────────────────────

    async def _step(self, sym: str, sig: float, now: datetime):
        st  = self.states[sym]
        eod = self._is_eod_tight(now)   # 14:55 ET 이후 기준 강화

        # ATR 스케일 적용 동적 파라미터 (EOD tight 시 추가 절반)
        _scale       = self.vol_scaler.scale(sym)
        stop_pct     = self.vol_scaler.stop_pct(sym)    / 2 if eod else self.vol_scaler.stop_pct(sym)
        partial_pct  = self.vol_scaler.partial_pct(sym) / 2 if eod else self.vol_scaler.partial_pct(sym)
        eod_tag      = " [EOD⚡]" if eod else ""

        # Step 1: 이전 bar에서 예약된 진입 실행 (t+1 open 진입)
        just_entered = False
        if st.pending_sig is not None and st.direction == "flat":
            await self._enter(sym, st.pending_sig, now)
            just_entered = True
        st.pending_sig = None

        # Step 2: 포지션 평가 및 청산 판단
        # 진입 직후 bar에서는 모든 청산 판단 skip — 동일 bar BUY+SELL = wash trade 방지
        if just_entered:
            return

        if st.direction == "long" and st.qty > 0:
            engine = self.engines[sym]
            bid, ask = engine._last_bid, engine._last_ask
            if np.isfinite(bid) and np.isfinite(ask) and bid > 0:
                px_now = (bid + ask) / 2
            else:
                px_now = await asyncio.to_thread(self.executor.get_current_price, sym)
            if px_now <= 0:
                log.warning(f"[{sym}] 현재가 조회 실패 — 청산 판단 스킵")
                return

            pnl_abs = (px_now - st.entry_px) * st.qty
            pnl_pct = pnl_abs / (st.entry_px * st.qty)
            st.pnl_peak = max(st.pnl_peak, pnl_abs)

            log.info(f"  [{sym}] {st.qty}주 @ ${st.entry_px:.2f}  "
                     f"현재 ${px_now:.2f}  PnL={pnl_pct:+.3%}  "
                     f"stop={stop_pct:.3%}  partial={partial_pct:.3%}{eod_tag}")

            if pnl_pct <= -stop_pct:
                await self._exit(sym, f"손절{eod_tag}", sig, px_now, now)
                return

            # ── 로컬저점 손절: 진입 전 캡처된 로컬저점 이탈 시 시장가 청산 ──
            if st.local_min_stop > 0 and px_now <= st.local_min_stop:
                log.info(f"  [{sym}] 🔴 로컬저점 손절: "
                         f"현재가 ${px_now:.2f} ≤ 진입전저점 ${st.local_min_stop:.2f}")
                await self._exit(sym, "로컬저점손절", sig, px_now, now)
                st.last_conv_exit = now
                return

            if sig < CONVICTION_LONG_EXIT:
                await self._exit(sym, "신호소멸", sig, px_now, now)
                st.last_conv_exit = now
                return

            if not st.partial_done and pnl_pct >= partial_pct:
                await self._partial_close(sym, px_now)
                return

            if st.partial_done and st.pnl_peak > 0:
                drawdown = (st.pnl_peak - pnl_abs) / st.pnl_peak
                if drawdown >= TRAIL_PULLBACK:
                    await self._exit(sym, "트레일링", sig, px_now, now)
                    return

        # Step 3: 진입 신호 예약
        # 15:45(EOD_REDUCE_TIME) 이후는 신규 진입 완전 차단
        if (st.direction == "flat"
                and not self._is_eod_reduce(now)
                and not self._is_eod(now)):
            if sig >= MIN_CONV and st.switch_ok(now) and st.conv_ok(now):
                log.info(f"  [{sym}] 진입 예약: signal={sig:+.4f}  "
                         f"atr_scale={_scale:.3f} → 다음 bar open 진입")
                st.pending_sig = sig

    # ── 진입/청산 실행 ────────────────────────────────────────────────────────

    async def _enter(self, sym: str, sig: float, now: datetime):
        async with self._trade_locks[sym]:
            st     = self.states[sym]
            engine = self.engines[sym]

            bid, ask = engine._last_bid, engine._last_ask
            if not (np.isfinite(bid) and np.isfinite(ask) and bid > 0):
                log.warning(f"[{sym}] 진입 실패: 호가 없음")
                return

            mid_px = (bid + ask) / 2

            # 진입 전 미체결 주문 정리 — wash trade 후 잔류 SELL 주문이 qty를 오염시키는 문제 방지
            await asyncio.to_thread(self.executor.cancel_open_orders, sym)

            equity, cash = await asyncio.to_thread(self.executor.get_account_snapshot)
            if equity <= 0:
                log.warning(f"[{sym}] 진입 실패: 계좌 조회 실패")
                return

            notional = equity * KELLY
            qty      = int(notional / mid_px)
            log.info(f"  [{sym}] 진입 계산: equity=${equity:,.2f}  mid=${mid_px:.2f}  "
                     f"notional=${notional:.2f}  qty_계산={qty}주  "
                     f"cash=${cash:,.2f}  한도=${cash*DAILY_TURNOVER_LIMIT:,.0f}")
            if qty < 1:
                log.warning(f"[{sym}] 진입 실패: qty={qty}주 (최소 1주 미만)")
                return

            # 일일 한도 체크 (슬리피지 포함)
            can, reason = self._can_enter(qty * mid_px, cash)
            if not can:
                log.info(f"  [{sym}] 진입 차단 — {reason}")
                return

            ok = await asyncio.to_thread(self.executor.buy, sym, qty)
            if not ok:
                return

            # 평균 체결가 조회 (포지션 API 업데이트 대기)
            await asyncio.sleep(2.0)
            actual_qty, actual_entry = await asyncio.to_thread(
                self.executor.get_position_snapshot, sym)
            fill_px = actual_entry if actual_entry > 0 else mid_px

            # 수량 결정:
            #   buy(qty) 가 True를 반환했으므로 주문은 제출됨.
            #   actual_qty < qty 는 포지션 API 지연(2초 내 미반영)일 가능성이 높으므로
            #   주문 수량(qty)을 신뢰. actual_qty > qty 이면 잔류 포지션 의심 → 실제값 사용.
            if actual_qty <= 0:
                final_qty = qty
                log.warning(f"  [{sym}] 포지션 API 미반영 (actual=0) — 주문 수량 {qty}주 사용")
            elif actual_qty < qty:
                final_qty = qty
                log.warning(f"  [{sym}] API 지연: 브로커={actual_qty}주 < 주문={qty}주 → {qty}주 기준 유지")
            elif actual_qty > qty:
                final_qty = actual_qty
                log.warning(f"  [{sym}] 잔류 포지션 의심: 브로커={actual_qty}주 > 주문={qty}주 → {actual_qty}주 사용")
            else:
                final_qty = qty  # actual_qty == qty, 정상

            # 일일 누적 매수 금액 갱신 (슬리피지 포함)
            spread_cost = final_qty * fill_px * (SPREAD_BPS / 10_000)
            self._daily_buy_notional += final_qty * fill_px + spread_cost

            # 진입 전 가장 최근 로컬저점 → 포지션 고정 손절선으로 저장
            # 단, 진입가 대비 0.3% 이상 아래일 때만 활성화 (너무 가까우면 노이즈에 즉시 청산)
            raw_lmin = self.engines[sym].latest_local_min
            if raw_lmin > 0 and (fill_px - raw_lmin) / fill_px >= MIN_LOCAL_STOP_DIST:
                local_min_stop = raw_lmin
            else:
                local_min_stop = 0.0

            st.direction      = "long"
            st.entry_px       = fill_px
            st.qty            = final_qty
            st.pnl_peak       = 0.0
            st.partial_done   = False
            st.last_switch    = now
            st.local_min_stop = local_min_stop

            slippage_bps = (fill_px - mid_px) / mid_px * 10_000
            lmin_str     = f"${local_min_stop:.2f}" if local_min_stop > 0 else "N/A"
            log.info(f"  [{sym}] ✅ ENTER: {final_qty}주  "
                     f"mid=${mid_px:.2f} → 체결${fill_px:.2f}  "
                     f"슬리피지={slippage_bps:+.2f}bp  signal={sig:+.4f}  "
                     f"로컬저점손절={lmin_str}  "
                     f"일일누적=${self._daily_buy_notional:,.0f}"
                     f"/{cash * DAILY_TURNOVER_LIMIT:,.0f}")
            await asyncio.to_thread(
                log_trade_csv, sym, "ENTER", fill_px, final_qty,
                0.0, slippage_bps, sig, equity)

    async def _exit(self, sym: str, reason: str, sig: float,
                    px_now: float, now: datetime):
        async with self._trade_locks[sym]:
            st = self.states[sym]
            if st.qty <= 0:
                return

            ok = await asyncio.to_thread(self.executor.close_position, sym)
            if not ok:
                # 청산 실패 시 미체결 주문 정리 후 재시도 1회 (wash trade 거부 대응)
                await asyncio.sleep(1.0)
                await asyncio.to_thread(self.executor.cancel_open_orders, sym)
                ok = await asyncio.to_thread(self.executor.close_position, sym)
                if not ok:
                    log.warning(f"  [{sym}] 청산 재시도도 실패 — 다음 bar에서 재평가")
                    return

            # 포지션 완전 청산 대기 (최대 10초 폴링)
            await asyncio.sleep(0.5)
            for _ in range(19):                # 추가 최대 9.5초 (19 × 0.5s)
                remaining = await asyncio.to_thread(self.executor.get_position_qty, sym)
                if remaining == 0:
                    break
                await asyncio.sleep(0.5)

            # 실제 체결가 & 체결수량 & equity 조회 (포지션 정산 완료 후)
            actual_exit_px, actual_exit_qty = await asyncio.to_thread(
                self.executor.get_last_sell_fill_info, sym)
            fill_px   = actual_exit_px  if actual_exit_px  > 0 else px_now
            close_qty = actual_exit_qty if actual_exit_qty > 0 else st.qty
            equity    = await asyncio.to_thread(self.executor.get_account_equity)

            gross    = (fill_px - st.entry_px) * close_qty if st.entry_px > 0 else 0.0
            slippage = (fill_px - px_now) / px_now * 10_000 if px_now > 0 else 0.0
            sign     = '✅' if gross >= 0 else '🔴'
            log.info(f"  [{sym}] {sign} {reason}: "
                     f"{close_qty}주 @ mid${px_now:.2f} → 실체결${fill_px:.2f}  "
                     f"PnL=${gross:+.2f}  슬리피지={slippage:+.2f}bp  "
                     f"portfolio=${equity:,.2f}")
            await asyncio.to_thread(
                log_trade_csv, sym, f"EXIT_{reason}", fill_px, close_qty,
                gross, slippage, sig, equity)
            st.reset()

    async def _partial_close(self, sym: str, px_now: float):
        async with self._trade_locks[sym]:
            st = self.states[sym]

            # Alpaca 실제 잔량 기준으로 절반 청산
            actual_qty = await asyncio.to_thread(self.executor.get_position_qty, sym)
            if actual_qty <= 0:
                log.warning(f"[{sym}] 부분익절 실패: 실제 포지션 없음")
                return

            close_qty = max(1, actual_qty // 2)
            ok = await asyncio.to_thread(self.executor.sell, sym, close_qty)
            if not ok:
                return

            st.qty          = actual_qty - close_qty
            st.partial_done = True

            await asyncio.sleep(0.5)
            actual_fill_px, _ = await asyncio.to_thread(
                self.executor.get_last_sell_fill_info, sym)
            fill_px = actual_fill_px if actual_fill_px > 0 else px_now
            gross   = (fill_px - st.entry_px) * close_qty
            equity  = await asyncio.to_thread(self.executor.get_account_equity)
            log.info(f"  [{sym}] 📉 부분익절: {close_qty}주 @ 실체결${fill_px:.2f}  "
                     f"PnL=${gross:+.2f}  잔여={st.qty}주")
            await asyncio.to_thread(
                log_trade_csv, sym, "PARTIAL_EXIT", fill_px, close_qty,
                gross, 0.0, 0.0, equity)

    # ── 실행 진입점 ────────────────────────────────────────────────────────────

    async def _heartbeat(self):
        """60초마다 상태 로그 출력. EOD 이후 일별 요약 1회 출력."""
        while self._running:
            await asyncio.sleep(60)
            cash = await asyncio.to_thread(self.executor.get_account_cash)
            now  = datetime.now(tz=timezone.utc)
            now_et = now.astimezone(ET).strftime("%H:%M ET")
            positions = " | ".join(
                f"{sym}: {self.states[sym].direction}({self.states[sym].qty}주)"
                for sym in INSTRUMENTS
            )
            log.info(f"[heartbeat] {now_et}  {positions}  "
                     f"일일누적=${self._daily_buy_notional:,.0f}"
                     f"/{cash * DAILY_TURNOVER_LIMIT:,.0f}")

            # EOD 이후 모든 포지션 flat이면 일별 요약 1회 출력
            if (self._is_eod(now)
                    and not self._eod_summary_done
                    and all(self.states[s].direction == "flat" for s in INSTRUMENTS)):
                await self._print_daily_summary(now)
                self._eod_summary_done = True

    def _on_shutdown(self, *_):
        log.info("종료 신호 수신 — 안전 종료 중...")
        self._running = False
        self._save_state()
        log.info("상태 저장 완료")
        sys.exit(0)

    def run(self):
        equity, cash = self.executor.get_account_snapshot()
        now_et = datetime.now(ET)
        self._initial_equity = equity

        log.info("=" * 65)
        log.info("SPY/QQQ Live Trader 시작")
        log.info(f"종목     : {', '.join(INSTRUMENTS)}")
        log.info(f"Kelly    : {KELLY*100:.1f}% / 진입  "
                 f"(최대 20회 × {KELLY*100:.1f}% = {KELLY*20*100:.0f}%)")
        log.info(f"Stop     : -{STOP_LOSS_PCT*100:.1f}% × ATR_scale  "
                 f"(EOD 타이트 추가 ÷2)")
        log.info(f"Partial  : +{PARTIAL_PCT*100:.1f}% × ATR_scale  "
                 f"(EOD 타이트 추가 ÷2)")
        for sym in INSTRUMENTS:
            s = self.vol_scaler.scale(sym)
            log.info(f"  [{sym}] ATR scale={s:.3f}  "
                     f"→ stop={STOP_LOSS_PCT*s*100:.3f}%  "
                     f"partial={PARTIAL_PCT*s*100:.3f}%")
        log.info(f"Trail    : 고점 대비 {TRAIL_PULLBACK*100:.0f}% 되돌림")
        log.info(f"신호소멸 : 합성 < {CONVICTION_LONG_EXIT}")
        log.info(f"세션     : {SESSION_OPEN.strftime('%H:%M')}~"
                 f"{SESSION_CLOSE.strftime('%H:%M')} ET")
        log.info(f"EOD 타이트: {EOD_TIGHT_TIME.strftime('%H:%M')} ET 이후 기준 절반")
        log.info(f"EOD 청산  : {EOD_REDUCE_TIME.strftime('%H:%M')} ET 50% 선제청산 "
                 f"→ {SESSION_CLOSE.strftime('%H:%M')} ET 잔여 전량 청산")
        log.info(f"일일 한도: cash × {DAILY_TURNOVER_LIMIT*100:.0f}% "
                 f"= ${cash * DAILY_TURNOVER_LIMIT:,.0f}  "
                 f"(당일 누적 ${self._daily_buy_notional:,.0f})")
        log.info(f"계좌     : equity=${equity:,.2f}  cash=${cash:,.2f}")
        log.info(f"현재시각 : {now_et.strftime('%Y-%m-%d %H:%M ET')}")
        log.info("=" * 65)

        signal.signal(signal.SIGINT,  self._on_shutdown)
        signal.signal(signal.SIGTERM, self._on_shutdown)

        log.info(f"스트림 구독 시작: {', '.join(INSTRUMENTS)}")

        async def _run():
            hb_task = asyncio.create_task(self._heartbeat())
            reconnect_delay = 5
            while self._running:
                try:
                    self.stream = StockDataStream(
                        ALPACA_API_KEY, ALPACA_SECRET_KEY,
                        feed=DataFeed.SIP,
                    )
                    for sym in INSTRUMENTS:
                        self.stream.subscribe_quotes(self._on_quote, sym)
                        self.stream.subscribe_trades(self._on_trade, sym)
                    reconnect_delay = 5   # 성공 시 딜레이 초기화
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
    import sys
    ans = sys.argv[1].strip().upper()
    INSTRUMENTS.extend(["SPY", "QQQ"] if ans == "BOTH" else [ans])
    print(f"  선택된 종목: {', '.join(INSTRUMENTS)}\n")

    trader = SPYQQQLiveTrader()
    trader.run()
