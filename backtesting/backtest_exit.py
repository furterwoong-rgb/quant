#!/usr/bin/env python3
"""
backtest_exit.py — Exit Strategy Backtester
────────────────────────────────────────────
OBI+TFI 신호로 진입하고, 다양한 청산 전략을 비교합니다.

Usage:
    python backtest_exit.py --date 2026-05-18
    python backtest_exit.py --date 2026-05-18 --sym QQQ
    python backtest_exit.py --date 2026-05-18 --sweep        # trailing 파라미터 스윕
    python backtest_exit.py --start 2026-05-12 --end 2026-05-18 --sweep
"""

import argparse
import pickle
import sys
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (StockBarsRequest, StockQuotesRequest,
                                   StockTradesRequest)
from alpaca.data.timeframe import TimeFrame
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY

ET        = pytz.timezone("America/New_York")
CACHE_DIR = Path("logs/tick_cache")
LOG_DIR   = Path("logs")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── 전략 고정 파라미터 (live_trader 와 동일) ──────────────────────────────────
OBI_ALPHA   = 0.08
TFI_ALPHA   = 0.15
MIN_CONV    = 0.20
MAX_CONV    = 0.30
STOP_LOSS_PCT      = 0.005   # -0.5% 손절
PARTIAL_PCT        = 0.004   # +0.4% 일부청산
SIGNAL_EXIT_THR    = -0.10   # 신호소멸 임계값
SWITCH_COOLDOWN    = 15      # 분
CONVICTION_COOLDOWN = 5      # 분
KELLY       = 0.10
PORTFOLIO   = 100_000.0

SESSION_OPEN  = datetime.strptime("09:45", "%H:%M").time()
SESSION_CLOSE = datetime.strptime("15:55", "%H:%M").time()

# ── 청산 전략 정의 ────────────────────────────────────────────────────────────

@dataclass
class ExitConfig:
    """청산 전략 파라미터."""
    name: str                        # 표시 이름

    # 신호 청산
    signal_exit_thr: float = -0.10   # signal < 이 값이면 청산

    # 달러 기반 trailing stop
    trailing_usd: float = 0.0        # 0 = 비활성. 고점 대비 이만큼 하락 시 청산
    trail_activate_usd: float = 0.0  # trailing 활성화 최소 수익 ($). 0 = 즉시

    # 손절 (모든 전략에 항상 적용)
    stop_loss_pct: float = STOP_LOSS_PCT

    def __str__(self):
        parts = [self.name]
        if self.trailing_usd > 0:
            parts.append(f"trail=${self.trailing_usd:.2f}")
            if self.trail_activate_usd > 0:
                parts.append(f"activate=${self.trail_activate_usd:.2f}")
        return "  ".join(parts)


SWEEP_CONFIGS = [
    ExitConfig("baseline",    trailing_usd=0.00),
    ExitConfig("trail $0.30", trailing_usd=0.30),
    ExitConfig("trail $0.50", trailing_usd=0.50),
    ExitConfig("trail $0.75", trailing_usd=0.75),
    ExitConfig("trail $1.00", trailing_usd=1.00),
    ExitConfig("trail $1.50", trailing_usd=1.50),
    ExitConfig("trail $0.50 (act $5)", trailing_usd=0.50, trail_activate_usd=5.0),
    ExitConfig("trail $0.75 (act $5)", trailing_usd=0.75, trail_activate_usd=5.0),
]


# ── 데이터 로더 ───────────────────────────────────────────────────────────────

def _date_range_utc(d: date):
    s_et = ET.localize(datetime(d.year, d.month, d.day, 9, 30))
    e_et = ET.localize(datetime(d.year, d.month, d.day, 16, 0))
    return s_et.astimezone(pytz.utc), e_et.astimezone(pytz.utc)


def load_ticks(sym: str, d: date, client) -> tuple[pd.DataFrame, pd.DataFrame]:
    ds = d.isoformat()
    cache_t = CACHE_DIR / f"{sym}_{ds}_trades.pkl"
    cache_q = CACHE_DIR / f"{sym}_{ds}_quotes.pkl"
    s_utc, e_utc = _date_range_utc(d)

    def _get(cache, fetch_fn):
        if cache.exists():
            with open(cache, "rb") as f:
                return pickle.load(f)
        label = cache.stem.split("_")[-1]
        for attempt in range(1, 4):
            try:
                print(f"  downloading {sym} {ds} {label}"
                      f"{f' (retry {attempt})' if attempt > 1 else ''}...",
                      end=" ", flush=True)
                df = fetch_fn()
                if isinstance(df.index, pd.MultiIndex):
                    df = df.xs(sym, level="symbol")
                df.index = pd.DatetimeIndex(df.index).tz_convert(ET)
                with open(cache, "wb") as f:
                    pickle.dump(df, f)
                print(f"ok ({len(df):,} rows)")
                return df
            except Exception as e:
                print(f"failed: {e}")
                if attempt == 3:
                    raise
                import time; time.sleep(5 * attempt)

    trades = _get(cache_t, lambda: client.get_stock_trades(
        StockTradesRequest(symbol_or_symbols=sym, start=s_utc, end=e_utc, feed="sip")).df)
    quotes = _get(cache_q, lambda: client.get_stock_quotes(
        StockQuotesRequest(symbol_or_symbols=sym, start=s_utc, end=e_utc, feed="sip")).df)
    return trades, quotes


def load_bars(sym: str, d: date, client) -> pd.DataFrame:
    ds = d.isoformat()
    cache = CACHE_DIR / f"{sym}_{ds}_bars.pkl"
    s_utc, e_utc = _date_range_utc(d)
    if not cache.exists():
        print(f"  downloading {sym} {ds} bars...", end=" ", flush=True)
        df = client.get_stock_bars(
            StockBarsRequest(symbol_or_symbols=sym, start=s_utc, end=e_utc,
                             timeframe=TimeFrame.Minute, feed="sip")).df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(sym, level="symbol")
        with open(cache, "wb") as f:
            pickle.dump(df, f)
        print(f"ok ({len(df)} rows)")
    with open(cache, "rb") as f:
        df = pickle.load(f)
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(sym, level="symbol")
    df.index = pd.DatetimeIndex(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize(ET)
    else:
        df.index = df.index.tz_convert(ET)
    return df[df.index.date == d].between_time("09:30", "16:00")


# ── OBI / TFI 신호 계산 ───────────────────────────────────────────────────────

def compute_signal(d: date, trades: pd.DataFrame, quotes: pd.DataFrame) -> pd.Series:
    idx = pd.date_range(
        ET.localize(datetime(d.year, d.month, d.day, 9, 30)),
        ET.localize(datetime(d.year, d.month, d.day, 16, 0)),
        freq="1min")

    # OBI
    q = quotes[["bid_size", "ask_size"]].copy()
    q = q[(q["bid_size"] > 0) & (q["ask_size"] > 0)]
    if not q.empty:
        raw_obi = ((q["bid_size"] - q["ask_size"]) /
                   (q["bid_size"] + q["ask_size"]))
        obi = (raw_obi.ewm(alpha=OBI_ALPHA, adjust=False).mean()
               .resample("1min").last().ffill())
    else:
        obi = pd.Series(0.0, index=idx)

    # TFI (Lee-Ready)
    q2 = quotes[["bid_price", "ask_price"]].rename(
        columns={"bid_price": "bid", "ask_price": "ask"})
    t2 = trades[["price", "size"]].copy()
    if not t2.empty and not q2.empty:
        merged = pd.concat([q2, t2]).sort_index()
        merged["bid"] = merged["bid"].ffill()
        merged["ask"] = merged["ask"].ffill()
        trd = merged[merged["price"].notna()].copy()
        trd["dir"] = np.where(trd["price"] >= trd["ask"],  1,
                    np.where(trd["price"] <= trd["bid"], -1, 0))
        trd["bvol"] = np.where(trd["dir"] ==  1, trd["size"], 0.0)
        trd["svol"] = np.where(trd["dir"] == -1, trd["size"], 0.0)
        buy_r  = trd["bvol"].rolling("5min").sum()
        sell_r = trd["svol"].rolling("5min").sum()
        total  = buy_r + sell_r
        raw_tfi = ((buy_r - sell_r) / total.replace(0, np.nan)).fillna(0.0)
        tfi = (raw_tfi.ewm(alpha=TFI_ALPHA, adjust=False).mean()
               .resample("1min").last().ffill())
    else:
        tfi = pd.Series(0.0, index=idx)

    def align(s):
        return s.reindex(idx, method="ffill").fillna(0.0)

    return (align(obi) * 0.35 + align(tfi) * 0.65).clip(-1, 1)


# ── 시뮬레이션 ────────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    entry_time:  datetime
    entry_price: float
    qty:         int
    signal_in:   float
    exit_time:   Optional[datetime] = None
    exit_price:  Optional[float]    = None
    exit_reason: str                = ""
    pnl:         float              = 0.0


def run_sim(sym: str, d: date, signal: pd.Series,
            bars: pd.DataFrame, cfg: ExitConfig) -> list[TradeRecord]:
    """
    단일 날짜 시뮬레이션.
    Returns list of completed TradeRecord.
    """
    portfolio = PORTFOLIO
    trades: list[TradeRecord] = []

    direction    = "flat"
    pending_sig  = None
    last_switch  = None
    last_conv_exit = None
    current: Optional[TradeRecord] = None
    price_peak   = 0.0   # trailing stop 추적용 (bar close 기준)

    all_ts = signal.index

    for ts in all_ts:
        t = ts.time()
        if not (SESSION_OPEN <= t < SESSION_CLOSE):
            pending_sig = None
            continue

        is_eod = t >= datetime.strptime("14:55", "%H:%M").time()

        # 현재 bar 가격
        bar_row = bars[bars.index <= ts]
        if bar_row.empty:
            continue
        bar = bar_row.iloc[-1]
        close_px = float(bar["close"])
        open_px  = float(bar["open"])
        sig      = float(signal.get(ts, 0.0))

        # ── Step 1: t+1 진입 실행 ─────────────────────────────────────────────
        if pending_sig is not None and direction == "flat" and not is_eod:
            qty = max(1, int(portfolio * KELLY / open_px))
            current = TradeRecord(
                entry_time=ts, entry_price=open_px,
                qty=qty, signal_in=pending_sig,
            )
            direction  = "long"
            price_peak = open_px
        pending_sig = None

        # ── Step 2: 포지션 평가 및 청산 ──────────────────────────────────────
        if direction == "long" and current is not None:
            price_peak = max(price_peak, close_px)
            pnl_abs    = (close_px - current.entry_price) * current.qty
            pnl_pct    = pnl_abs / (current.entry_price * current.qty)

            exit_reason = None

            # 손절 (-0.5%)
            if pnl_pct <= -cfg.stop_loss_pct:
                exit_reason = "STOP_LOSS"

            # 신호소멸
            elif sig < cfg.signal_exit_thr:
                exit_reason = "SIGNAL_EXIT"

            # Trailing stop (달러 기반)
            elif cfg.trailing_usd > 0:
                activated = (cfg.trail_activate_usd == 0 or
                             (price_peak - current.entry_price) * current.qty
                             >= cfg.trail_activate_usd)
                if activated and (price_peak - close_px) >= cfg.trailing_usd:
                    exit_reason = "TRAIL_EXIT"

            # EOD 강제 청산
            if t >= SESSION_CLOSE or (is_eod and ts == all_ts[-1]):
                exit_reason = exit_reason or "EOD"

            if exit_reason:
                current.exit_time   = ts
                current.exit_price  = close_px
                current.exit_reason = exit_reason
                current.pnl         = pnl_abs
                portfolio          += pnl_abs
                trades.append(current)
                current    = None
                direction  = "flat"
                price_peak = 0.0
                if exit_reason == "SIGNAL_EXIT":
                    last_conv_exit = ts

        # ── Step 3: 진입 신호 예약 ────────────────────────────────────────────
        if direction == "flat" and not is_eod:
            sw_ok = (last_switch is None or
                     (ts - last_switch).total_seconds() >= SWITCH_COOLDOWN * 60)
            cv_ok = (last_conv_exit is None or
                     (ts - last_conv_exit).total_seconds() >= CONVICTION_COOLDOWN * 60)
            if MIN_CONV <= sig < MAX_CONV and sw_ok and cv_ok:
                pending_sig  = sig
                last_switch  = ts

    # EOD 미청산 강제 청산
    if direction == "long" and current is not None:
        last_ts  = all_ts[-1]
        bar_row  = bars[bars.index <= last_ts]
        close_px = float(bar_row.iloc[-1]["close"]) if not bar_row.empty else current.entry_price
        pnl_abs  = (close_px - current.entry_price) * current.qty
        current.exit_time   = last_ts
        current.exit_price  = close_px
        current.exit_reason = "EOD"
        current.pnl         = pnl_abs
        trades.append(current)

    return trades


# ── 결과 집계 ─────────────────────────────────────────────────────────────────

def summarize(trades: list[TradeRecord]) -> dict:
    if not trades:
        return dict(n=0, total=0.0, wins=0, losses=0, win_rate=0.0,
                    avg_win=0.0, avg_loss=0.0, best=0.0, worst=0.0)
    pnls  = [t.pnl for t in trades]
    wins  = [p for p in pnls if p > 0]
    losses= [p for p in pnls if p <= 0]
    return dict(
        n        = len(trades),
        total    = sum(pnls),
        wins     = len(wins),
        losses   = len(losses),
        win_rate = len(wins) / len(trades) * 100,
        avg_win  = np.mean(wins)   if wins   else 0.0,
        avg_loss = np.mean(losses) if losses else 0.0,
        best     = max(pnls),
        worst    = min(pnls),
    )


# ── 차트 ─────────────────────────────────────────────────────────────────────

EXIT_COLORS = {
    "STOP_LOSS":   "#D50000",
    "SIGNAL_EXIT": "#FF6D00",
    "TRAIL_EXIT":  "#1565C0",
    "EOD":         "#757575",
}


def plot_single_day(sym: str, d: date, bars: pd.DataFrame,
                    signal: pd.Series,
                    results: dict[str, list[TradeRecord]],
                    save_path: Path):
    """기준 vs trailing 전략 비교 차트 (단일 날짜)."""
    configs_to_plot = list(results.keys())
    n = len(configs_to_plot)

    fig = plt.figure(figsize=(18, 4 + 3 * n))
    fig.suptitle(f"{sym}  {d}  Exit Strategy Comparison",
                 fontsize=13, fontweight="bold")

    outer = gridspec.GridSpec(2, 1, height_ratios=[3, 1.2], hspace=0.35)
    price_gs  = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0])
    cumPnl_gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1])

    ax_price = fig.add_subplot(price_gs[0])
    ax_cum   = fig.add_subplot(cumPnl_gs[0])

    session_bars = bars.between_time("09:45", "15:55")
    ax_price.plot(session_bars.index, session_bars["close"],
                  color="#78909C", lw=1.0, alpha=0.85, label=f"{sym} close")
    ax_price.set_ylabel("Price ($)", fontsize=10)
    ax_price.grid(alpha=0.2)

    palette = ["#1565C0", "#E65100", "#2E7D32", "#6A1B9A",
               "#00838F", "#AD1457", "#F57F17", "#37474F"]

    for ci, cfg_name in enumerate(configs_to_plot):
        trades = results[cfg_name]
        color  = palette[ci % len(palette)]

        # 진입/청산 마커
        for tr in trades:
            ax_price.scatter(tr.entry_time, tr.entry_price,
                             marker="^", color=color, s=100,
                             zorder=5, edgecolors="white", linewidths=0.5,
                             alpha=0.8)
            ec = EXIT_COLORS.get(tr.exit_reason, color)
            ax_price.scatter(tr.exit_time, tr.exit_price,
                             marker="v", color=ec, s=100,
                             zorder=5, edgecolors="white", linewidths=0.5,
                             alpha=0.8)
            # PnL 레이블 (baseline만)
            if ci == 0:
                ax_price.annotate(
                    f"${tr.pnl:+.0f}",
                    xy=(tr.exit_time, tr.exit_price),
                    xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=7,
                    color="#2E7D32" if tr.pnl >= 0 else "#C62828",
                    fontweight="bold")

        # 누적 PnL
        if trades:
            cum_t   = [tr.exit_time for tr in trades]
            cum_pnl = np.cumsum([tr.pnl for tr in trades]).tolist()
            s = summarize(trades)
            label = f"{cfg_name}  total=${s['total']:+.2f}  ({s['wins']}W/{s['losses']}L)"
            ax_cum.step(cum_t, cum_pnl, where="post",
                        color=color, lw=1.8, label=label)

    ax_cum.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax_cum.set_ylabel("Cum. PnL ($)", fontsize=9)
    ax_cum.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax_cum.grid(alpha=0.2)
    ax_cum.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ET))
    ax_cum.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 30)))
    fig.autofmt_xdate(rotation=0, ha="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved: {save_path}")
    plt.close(fig)


def plot_sweep_summary(sweep_results: dict[str, list[float]],
                       dates: list[str], save_path: Path):
    """날짜 × 전략 히트맵 + 전략별 누적 PnL."""
    cfg_names = list(sweep_results.keys())
    totals    = {c: sum(sweep_results[c]) for c in cfg_names}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Exit Strategy Sweep — Summary", fontsize=12, fontweight="bold")

    # 히트맵
    matrix = np.array([sweep_results[c] for c in cfg_names])
    im = ax1.imshow(matrix, aspect="auto",
                    cmap="RdYlGn", vmin=-100, vmax=100)
    ax1.set_xticks(range(len(dates)))
    ax1.set_xticklabels([d[5:] for d in dates], rotation=45, ha="right", fontsize=8)
    ax1.set_yticks(range(len(cfg_names)))
    ax1.set_yticklabels(cfg_names, fontsize=8)
    for i, c in enumerate(cfg_names):
        for j, v in enumerate(sweep_results[c]):
            ax1.text(j, i, f"${v:+.0f}", ha="center", va="center",
                     fontsize=7, color="black")
    plt.colorbar(im, ax=ax1, label="Daily PnL ($)")
    ax1.set_title("PnL by Date × Strategy", fontsize=10)

    # 총합 바차트
    colors = ["#2E7D32" if totals[c] >= 0 else "#C62828" for c in cfg_names]
    bars_h = ax2.barh(cfg_names, [totals[c] for c in cfg_names], color=colors, alpha=0.8)
    ax2.axvline(0, color="black", lw=0.8)
    for bar, (c, v) in zip(bars_h, totals.items()):
        ax2.text(v + (2 if v >= 0 else -2), bar.get_y() + bar.get_height() / 2,
                 f"${v:+.2f}", va="center", ha="left" if v >= 0 else "right",
                 fontsize=8)
    ax2.set_xlabel("Total PnL ($)", fontsize=9)
    ax2.set_title("Total PnL by Strategy", fontsize=10)
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Sweep chart saved: {save_path}")
    plt.close(fig)


# ── 메인 ─────────────────────────────────────────────────────────────────────

def print_table(sym: str, d: date, results: dict[str, list[TradeRecord]]):
    print(f"\n{'='*72}")
    print(f"  {sym}  {d}  Exit Strategy Comparison")
    print(f"{'='*72}")
    print(f"  {'Strategy':<28} {'Trades':>6} {'Total':>9} "
          f"{'WinRate':>8} {'AvgWin':>8} {'AvgLoss':>9} {'Best':>8} {'Worst':>8}")
    print(f"  {'-'*72}")
    for cfg_name, trades in results.items():
        s = summarize(trades)
        print(f"  {cfg_name:<28} {s['n']:>6} ${s['total']:>+8.2f} "
              f"{s['win_rate']:>7.0f}% ${s['avg_win']:>+7.2f} "
              f"${s['avg_loss']:>+8.2f} ${s['best']:>+7.2f} ${s['worst']:>+7.2f}")

    # 개별 거래 내역 (baseline)
    baseline_key = next(iter(results))
    baseline_trades = results[baseline_key]
    if baseline_trades:
        print(f"\n  [Baseline trades]")
        print(f"  {'#':<3} {'Entry':>5} {'Entry$':>8} {'Exit':>5} "
              f"{'Exit$':>8} {'PnL':>8} {'Reason'}")
        print(f"  {'-'*58}")
        for i, tr in enumerate(baseline_trades, 1):
            print(f"  {i:<3} "
                  f"{tr.entry_time.strftime('%H:%M'):>5} "
                  f"${tr.entry_price:>7.2f} "
                  f"{tr.exit_time.strftime('%H:%M'):>5} "
                  f"${tr.exit_price:>7.2f} "
                  f"${tr.pnl:>+7.2f}  {tr.exit_reason}")
    print()


def run_date(sym: str, d: date, client, configs: list[ExitConfig],
             do_plot: bool = True) -> dict[str, list[TradeRecord]]:
    print(f"\n--- {sym} {d} ---")
    trades_df, quotes_df = load_ticks(sym, d, client)
    bars_df              = load_bars(sym, d, client)
    signal               = compute_signal(d, trades_df, quotes_df)

    results = {}
    for cfg in configs:
        results[cfg.name] = run_sim(sym, d, signal, bars_df, cfg)

    print_table(sym, d, results)

    if do_plot:
        out = LOG_DIR / f"exit_strategy_{sym}_{d}.png"
        plot_single_day(sym, d, bars_df, signal, results, out)

    return results


def main():
    parser = argparse.ArgumentParser(description="Exit Strategy Backtester")
    parser.add_argument("--sym",   default="QQQ", help="Symbol (default: QQQ)")
    parser.add_argument("--date",  help="Single date YYYY-MM-DD")
    parser.add_argument("--start", help="Start date for range")
    parser.add_argument("--end",   help="End date for range")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep all ExitConfigs in SWEEP_CONFIGS")
    parser.add_argument("--trailing", type=float, default=0.50,
                        help="Trailing stop in $ (single run, default: 0.50)")
    parser.add_argument("--no-chart", action="store_true",
                        help="Skip chart generation")
    args = parser.parse_args()

    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    # 날짜 목록
    if args.date:
        dates = [date.fromisoformat(args.date)]
    elif args.start and args.end:
        s, e = date.fromisoformat(args.start), date.fromisoformat(args.end)
        dates = [s + timedelta(days=i) for i in range((e - s).days + 1)
                 if (s + timedelta(days=i)).weekday() < 5]
    else:
        dates = [date.today()]

    # 전략 목록
    configs = (SWEEP_CONFIGS if args.sweep
               else [ExitConfig("baseline",    trailing_usd=0.0),
                     ExitConfig(f"trail ${args.trailing:.2f}",
                                trailing_usd=args.trailing)])

    # 실행
    sweep_pnls: dict[str, list[float]] = {c.name: [] for c in configs}
    for d in dates:
        try:
            results = run_date(args.sym, d, client, configs,
                               do_plot=(not args.no_chart))
            for cfg in configs:
                s = summarize(results[cfg.name])
                sweep_pnls[cfg.name].append(s["total"])
        except Exception as e:
            print(f"  ERROR {d}: {e}")
            for cfg in configs:
                sweep_pnls[cfg.name].append(0.0)

    # 다중 날짜 스윕 요약
    if len(dates) > 1:
        print(f"\n{'='*55}")
        print(f"  SWEEP SUMMARY  ({args.sym}  {dates[0]} ~ {dates[-1]})")
        print(f"{'='*55}")
        print(f"  {'Strategy':<28} {'Total':>9}  {'Days'}")
        print(f"  {'-'*50}")
        for cfg in configs:
            total = sum(sweep_pnls[cfg.name])
            days  = "  ".join(f"${v:+.0f}" for v in sweep_pnls[cfg.name])
            print(f"  {cfg.name:<28} ${total:>+8.2f}  [{days}]")

        if not args.no_chart:
            date_strs = [d.isoformat() for d in dates]
            plot_sweep_summary(
                sweep_pnls, date_strs,
                LOG_DIR / f"exit_sweep_{args.sym}_{dates[0]}_{dates[-1]}.png")


if __name__ == "__main__":
    main()
