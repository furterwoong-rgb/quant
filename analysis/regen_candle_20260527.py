#!/usr/bin/env python3
"""
2026-05-27 candle PNG 재생성 — 원본 디자인(4-panel dark) 복원
──────────────────────────────────────────────────────────────
Alpaca 실체결 기준으로 qty=3 / 올바른 fill price / 정확한 PnL 반영.
저장: logs/2026/05/27/trades_20260527_candle.png
"""

import os, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from matplotlib import gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY

ET       = ZoneInfo("America/New_York")
SYM      = "QQQ"
DATE_STR = "2026-05-27"
OUT = Path(__file__).parent.parent / "logs" / "2026" / "05" / "27" / "trades_20260527_candle.png"

# ── 폰트 ──────────────────────────────────────────────────────────────────────
for _cand in ("AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic"):
    _match = next((f.fname for f in fm.fontManager.ttflist if _cand in f.name), None)
    if _match:
        plt.rcParams["font.family"] = fm.FontProperties(fname=_match).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

# ── 색상 팔레트 ────────────────────────────────────────────────────────────────
C = dict(
    bg        = "#0d1117",
    ax        = "#161b22",
    grid      = "#21262d",
    border    = "#30363d",
    text      = "#8b949e",
    text_hi   = "#e6edf3",
    green     = "#26a69a",
    red       = "#ef5350",
    win_fill  = "#1a3c2a",   # 이긴 거래 구간 배경
    loss_fill = "#3c1a1a",   # 진 거래 구간 배경
    vwap      = "#f0b429",
    ma5       = "#ff6b6b",
    ma10      = "#4d9fff",
    ma20      = "#ffd700",
    entry_mk  = "#ffffff",
    exit_win  = "#00e676",
    exit_loss = "#ff5252",
    sig_pos   = "#26a69a",
    sig_neg   = "#ef5350",
    port_line = "#58a6ff",
    port_fill = "#1c2d42",
    thresh    = "#484f58",
)

# ── Alpaca 실체결 기준 보정 데이터 (qty=3 전부) ────────────────────────────────
CORRECTED = [
    ("2026-05-27 10:13:01", "ENTER",          728.5267, 3,  0.00,  0.2045),
    ("2026-05-27 10:13:02", "EXIT_신호소멸", 728.44,   3, -0.26, -0.2449),
    ("2026-05-27 10:35:01", "ENTER",          727.12,   3,  0.00,  0.2397),
    ("2026-05-27 10:38:59", "EXIT_신호소멸", 728.16,   3,  3.12, -0.1957),
    ("2026-05-27 10:46:01", "ENTER",          728.1633, 3,  0.00,  0.2154),
    ("2026-05-27 10:46:59", "EXIT_신호소멸", 728.665,  3,  1.50, -0.2035),
    ("2026-05-27 11:47:03", "ENTER",          725.96,   3,  0.00,  0.4519),
    ("2026-05-27 11:50:01", "EXIT_신호소멸", 726.00,   3,  0.12, -0.2264),
    ("2026-05-27 12:14:03", "ENTER",          728.16,   3,  0.00,  0.2441),
    ("2026-05-27 12:14:04", "EXIT_신호소멸", 728.12,   3, -0.12, -0.1967),
    ("2026-05-27 12:32:03", "ENTER",          728.19,   3,  0.00,  0.2194),
    ("2026-05-27 12:33:01", "EXIT_신호소멸", 728.39,   3,  0.60, -0.1276),
    ("2026-05-27 12:47:03", "ENTER",          729.4667, 3,  0.00,  0.3782),
    ("2026-05-27 12:47:04", "EXIT_신호소멸", 729.4067, 3, -0.18, -0.1277),
    ("2026-05-27 12:54:03", "ENTER",          728.59,   3,  0.00,  0.3203),
    ("2026-05-27 12:59:01", "EXIT_신호소멸", 728.0933, 3, -1.49, -0.1438),
    ("2026-05-27 13:17:03", "ENTER",          726.7333, 3,  0.00,  0.2215),
    ("2026-05-27 13:22:01", "EXIT_신호소멸", 727.2233, 3,  1.47, -0.1302),
    ("2026-05-27 13:33:03", "ENTER",          727.5233, 3,  0.00,  0.2488),
    ("2026-05-27 13:36:01", "EXIT_신호소멸", 727.79,   3,  0.80, -0.1770),
    ("2026-05-27 13:49:03", "ENTER",          727.1867, 3,  0.00,  0.2832),
    ("2026-05-27 13:54:01", "EXIT_신호소멸", 727.9233, 3,  2.21, -0.1040),
    ("2026-05-27 14:00:03", "ENTER",          727.78,   3,  0.00,  0.2709),
    ("2026-05-27 14:01:01", "EXIT_신호소멸", 727.53,   3, -0.75, -0.1392),
    ("2026-05-27 14:16:03", "ENTER",          728.3467, 3,  0.00,  0.6269),
    ("2026-05-27 14:20:01", "EXIT_신호소멸", 728.59,   3,  0.73, -0.2685),
    ("2026-05-27 14:34:03", "ENTER",          729.66,   3,  0.00,  0.3790),
    ("2026-05-27 14:35:01", "EXIT_신호소멸", 729.6233, 3, -0.11, -0.2119),
    ("2026-05-27 14:49:03", "ENTER",          729.58,   3,  0.00,  0.2341),
    ("2026-05-27 15:00:01", "EXIT_신호소멸", 729.38,   3, -0.60, -0.1457),
    ("2026-05-27 15:26:03", "ENTER",          729.2833, 3,  0.00,  0.2181),
    ("2026-05-27 15:36:01", "EXIT_신호소멸", 729.5033, 3,  0.66, -0.3259),
]


# ─────────────────────────────────────────────────────────────────────────────

def fetch_bars() -> pd.DataFrame:
    print("  Alpaca 1분봉 로딩...")
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    d   = datetime.strptime(DATE_STR, "%Y-%m-%d")
    req = StockBarsRequest(
        symbol_or_symbols=SYM,
        timeframe=TimeFrame.Minute,
        start=datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET),
        end=datetime(d.year, d.month, d.day, 16, 0, tzinfo=ET),
        feed="sip",
    )
    bars = client.get_stock_bars(req).df.xs(SYM, level="symbol")
    bars.index = bars.index.tz_convert(ET)
    print(f"  {len(bars)}개 bar 완료")
    return bars


def build_vwap(bars: pd.DataFrame) -> pd.Series:
    tp  = (bars["high"] + bars["low"] + bars["close"]) / 3
    vol = bars["volume"].replace(0, np.nan)
    return (tp * vol).cumsum() / vol.cumsum()


def load_trades() -> pd.DataFrame:
    rows = []
    for ts, action, price, qty, pnl, sig in CORRECTED:
        rows.append({
            "timestamp_et": pd.Timestamp(ts, tz=ET),
            "action": action,
            "price": float(price),
            "qty": int(qty),
            "pnl_usd": float(pnl),
            "signal": float(sig),
        })
    df = pd.DataFrame(rows)
    # portfolio: CSV에서 가져온 값 (마지막 EXIT의 portfolio 값)
    portfolios = [
        99135.20, 99135.29, 99138.91, 99138.83, 99140.23,
        99140.23, 99140.40, 99140.35, 99140.20, 99140.23,
        99140.92, 99140.83, 99140.75, 99140.65, 99139.20,
        99139.16, 99140.61, 99140.63, 99141.49, 99141.43,
        99143.75, 99143.64, 99142.95, 99142.89, 99143.62,
        99143.62, 99143.53, 99143.51, 99142.73, 99142.91,
        99143.62, 99143.62,
    ]
    df["portfolio"] = portfolios
    return df


def _ax_style(ax):
    ax.set_facecolor(C["ax"])
    ax.tick_params(colors=C["text"], labelsize=8)
    ax.spines[:].set_color(C["border"])
    ax.grid(alpha=0.3, color=C["grid"], linewidth=0.5)


def _xfmt(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ET))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 30)))


# ─────────────────────────────────────────────────────────────────────────────

def plot(bars: pd.DataFrame, trades: pd.DataFrame):
    vwap   = build_vwap(bars)
    enters = trades[trades["action"] == "ENTER"].copy()
    exits  = trades[trades["action"].str.startswith("EXIT")].copy()

    # 거래 쌍 (entry_time, exit_time, pnl)
    pairs = []
    e_idx = 0
    x_idx = 0
    while e_idx < len(enters) and x_idx < len(exits):
        en = enters.iloc[e_idx]
        ex = exits.iloc[x_idx]
        pairs.append((en["timestamp_et"], ex["timestamp_et"], ex["pnl_usd"]))
        e_idx += 1; x_idx += 1

    # ── 레이아웃 ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 14), facecolor=C["bg"])
    fig.suptitle(
        f"{SYM}  OBI+TFI Live Trader  —  {DATE_STR}  (1-min)  "
        f"[Alpaca 실체결 보정]",
        color=C["text_hi"], fontsize=12, fontweight="bold", y=0.995
    )
    gs = gridspec.GridSpec(4, 1, hspace=0.08,
                           height_ratios=[5, 1.2, 1.8, 1.8])
    ax1 = fig.add_subplot(gs[0])          # candle
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # volume
    ax3 = fig.add_subplot(gs[2], sharex=ax1)  # signal
    ax4 = fig.add_subplot(gs[3], sharex=ax1)  # portfolio

    for ax in (ax1, ax2, ax3, ax4):
        _ax_style(ax)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)

    # ── 거래 구간 배경 (candle + signal 패널) ─────────────────────────────────
    for en_t, ex_t, pnl in pairs:
        fill_c = C["win_fill"] if pnl >= 0 else C["loss_fill"]
        for ax in (ax1, ax3):
            ax.axvspan(en_t, ex_t, color=fill_c, alpha=0.45, zorder=1)

    # ── 캔들스틱 ──────────────────────────────────────────────────────────────
    W = 0.00042   # 1분봉 body 너비
    for ts, row in bars.iterrows():
        up    = row["close"] >= row["open"]
        clr   = C["green"] if up else C["red"]
        lo    = min(row["open"], row["close"])
        hi_b  = max(row["open"], row["close"])
        # wick
        ax1.plot([ts, ts], [row["low"], row["high"]],
                 color=clr, linewidth=0.6, zorder=3)
        # body
        ax1.add_patch(mpatches.Rectangle(
            (mdates.date2num(ts) - W / 2, lo),
            W, max(hi_b - lo, 0.02),
            facecolor=clr, edgecolor="none", zorder=4
        ))

    # VWAP
    ax1.plot(bars.index, vwap,
             color=C["vwap"], linewidth=1.1, label="VWAP", zorder=5, alpha=0.9)

    # MA
    ax1.plot(bars.index, bars["close"].rolling(5).mean(),
             color=C["ma5"],  linewidth=0.8, label="MA5",  zorder=5, alpha=0.75)
    ax1.plot(bars.index, bars["close"].rolling(20).mean(),
             color=C["ma20"], linewidth=0.8, label="MA20", zorder=5, alpha=0.75)

    # 세션 구분선
    for hm, lbl, clr in [
        ("09:45", "진입시작", "#ffeb3b"),
        ("14:55", "EOD tight", "#ff9800"),
        ("15:45", "EOD reduce", "#ff5722"),
        ("15:55", "세션종료",  "#f44336"),
    ]:
        t = pd.Timestamp(f"{DATE_STR} {hm}:00", tz=ET)
        if bars.index[0] <= t <= bars.index[-1]:
            for ax in (ax1, ax3):
                ax.axvline(t, color=clr, linewidth=0.7,
                           linestyle="--", alpha=0.5, zorder=2)
            ax1.text(t, ax1.get_ylim()[1] if ax1.get_ylim()[1] != 1 else bars["high"].max(),
                     f" {lbl}", color=clr, fontsize=7, va="top", zorder=6)

    # 진입/청산 마커
    ax1.scatter(enters["timestamp_et"], enters["price"],
                marker="^", color=C["entry_mk"], s=80, zorder=7,
                edgecolors=C["text"], linewidths=0.5, label="Entry")
    win_exits  = exits[exits["pnl_usd"] >= 0]
    loss_exits = exits[exits["pnl_usd"] <  0]
    ax1.scatter(win_exits["timestamp_et"],  win_exits["price"],
                marker="v", color=C["exit_win"],  s=80, zorder=7,
                edgecolors="#00a050", linewidths=0.5, label="Exit (win)")
    ax1.scatter(loss_exits["timestamp_et"], loss_exits["price"],
                marker="v", color=C["exit_loss"], s=80, zorder=7,
                edgecolors="#b00020", linewidths=0.5, label="Exit (loss)")

    # PnL 레이블
    for _, row in exits.iterrows():
        clr = C["exit_win"] if row["pnl_usd"] >= 0 else C["exit_loss"]
        ax1.annotate(
            f"${row['pnl_usd']:+.2f}",
            xy=(row["timestamp_et"], row["price"]),
            xytext=(0, -14), textcoords="offset points",
            ha="center", fontsize=7, color=clr, fontweight="bold", zorder=8
        )

    # 통계 박스
    total_pnl = exits["pnl_usd"].sum()
    n_win  = (exits["pnl_usd"] > 0).sum()
    n_tot  = len(exits)
    wins_s = exits.loc[exits["pnl_usd"] > 0, "pnl_usd"]
    loss_s = exits.loc[exits["pnl_usd"] < 0, "pnl_usd"]
    pf     = wins_s.sum() / abs(loss_s.sum()) if loss_s.sum() != 0 else float("inf")
    stat_clr = C["exit_win"] if total_pnl >= 0 else C["exit_loss"]
    ax1.text(
        0.005, 0.995,
        f"Daily PnL: ${total_pnl:+.2f}  |  "
        f"{n_win}W {n_tot-n_win}L  WR {n_win/n_tot*100:.0f}%  |  "
        f"PF {pf:.2f}  |  "
        f"AvgW ${wins_s.mean():+.2f}  AvgL ${loss_s.mean():+.2f}",
        transform=ax1.transAxes, ha="left", va="top",
        fontsize=8.5, color=stat_clr,
        bbox=dict(boxstyle="round,pad=0.35",
                  fc=C["ax"], ec=stat_clr, alpha=0.9),
        zorder=9
    )

    ax1.set_ylabel("QQQ Price ($)", color=C["text"], fontsize=9)

    # 범례
    legend_elems = [
        Line2D([0],[0], color=C["vwap"],     lw=1.2, label="VWAP"),
        Line2D([0],[0], color=C["ma5"],      lw=1.0, label="MA5"),
        Line2D([0],[0], color=C["ma20"],     lw=1.0, label="MA20"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor=C["entry_mk"],
               markersize=7, lw=0, label="Entry"),
        Line2D([0],[0], marker="v", color="w", markerfacecolor=C["exit_win"],
               markersize=7, lw=0, label="Exit (win)"),
        Line2D([0],[0], marker="v", color="w", markerfacecolor=C["exit_loss"],
               markersize=7, lw=0, label="Exit (loss)"),
        mpatches.Patch(facecolor=C["win_fill"],  alpha=0.7, label="Win span"),
        mpatches.Patch(facecolor=C["loss_fill"], alpha=0.7, label="Loss span"),
    ]
    ax1.legend(handles=legend_elems, loc="upper right",
               facecolor=C["ax"], edgecolor=C["border"],
               labelcolor=C["text_hi"], fontsize=7.5, ncol=4)

    # ── 거래량 ────────────────────────────────────────────────────────────────
    for ts, row in bars.iterrows():
        clr = C["green"] if row["close"] >= row["open"] else C["red"]
        ax2.bar(ts, row["volume"], width=pd.Timedelta(seconds=50),
                color=clr, alpha=0.65, linewidth=0)
    ax2.set_ylabel("Volume", color=C["text"], fontsize=8)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))

    # ── OBI+TFI 시그널 (거래 시점 값만 표시) ─────────────────────────────────
    # 진입 threshold 선
    MIN_CONV = 0.20
    ax3.axhline( MIN_CONV, color=C["green"], linewidth=0.8,
                 linestyle="--", alpha=0.7, label=f"Entry threshold (+{MIN_CONV})")
    ax3.axhline(-MIN_CONV, color=C["red"],   linewidth=0.8,
                 linestyle="--", alpha=0.7, label=f"Exit threshold (-{MIN_CONV})")
    ax3.axhline(0, color=C["thresh"], linewidth=0.6, alpha=0.5)

    # 신호 scatter (ENTER/EXIT 시점)
    for _, row in enters.iterrows():
        clr = C["sig_pos"] if row["signal"] >= 0 else C["sig_neg"]
        ax3.scatter(row["timestamp_et"], row["signal"],
                    marker="^", color=C["entry_mk"], s=60, zorder=5,
                    edgecolors=C["text"], linewidths=0.5)
    for _, row in exits.iterrows():
        clr = C["exit_win"] if row["pnl_usd"] >= 0 else C["exit_loss"]
        ax3.scatter(row["timestamp_et"], row["signal"],
                    marker="v", color=clr, s=60, zorder=5,
                    edgecolors="none")

    # 신호값 stem 막대
    for _, row in trades.iterrows():
        clr = C["sig_pos"] if row["signal"] >= 0 else C["sig_neg"]
        ax3.plot([row["timestamp_et"], row["timestamp_et"]],
                 [0, row["signal"]],
                 color=clr, linewidth=1.0, alpha=0.6, zorder=3)

    ax3.set_ylabel("OBI+TFI Signal", color=C["text"], fontsize=8)
    ax3.set_ylim(-0.6, 0.6)
    ax3.legend(loc="lower right", facecolor=C["ax"],
               edgecolor=C["border"], labelcolor=C["text_hi"], fontsize=7)

    # ── 포트폴리오 에쿼티 ─────────────────────────────────────────────────────
    # EXIT 시점의 portfolio 값으로 에쿼티 커브 구성
    port_ts  = exits["timestamp_et"].tolist()
    port_val = exits["portfolio"].tolist()

    # 시작점 추가 (첫 ENTER 이전)
    initial = enters.iloc[0]["portfolio"]
    t_start = enters.iloc[0]["timestamp_et"]
    port_ts  = [t_start] + port_ts
    port_val = [initial] + port_val

    ax4.plot(port_ts, port_val,
             color=C["port_line"], linewidth=1.3, zorder=4, label="Portfolio Equity")
    ax4.fill_between(port_ts, initial, port_val,
                     where=[v >= initial for v in port_val],
                     alpha=0.20, color="#2ea043", zorder=3)
    ax4.fill_between(port_ts, initial, port_val,
                     where=[v < initial for v in port_val],
                     alpha=0.20, color="#f85149", zorder=3)
    ax4.axhline(initial, color=C["thresh"], linewidth=0.8, linestyle="--", alpha=0.7)
    ax4.scatter(port_ts[1:], port_val[1:], color=C["port_line"], s=20, zorder=5)

    daily_pnl = port_val[-1] - initial
    ax4.set_ylabel("Equity ($)", color=C["text"], fontsize=8)
    ax4.set_title(f"Portfolio Equity  (Daily PnL: ${daily_pnl:+.2f})",
                  color=C["text"], fontsize=8, pad=2)
    ax4.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # x축 (마지막 패널만)
    _xfmt(ax4)
    fig.autofmt_xdate(rotation=0, ha="center")
    for ax in (ax1, ax2, ax3, ax4):
        ax.tick_params(axis="x", colors=C["text"])

    # ── 저장 ──────────────────────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.993])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=180, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"  PNG 저장: {OUT}")


if __name__ == "__main__":
    print("=" * 60)
    print("  2026-05-27 candle PNG 재생성 (원본 디자인, Alpaca 보정)")
    print("=" * 60)
    bars   = fetch_bars()
    trades = load_trades()
    exits  = trades[trades["action"].str.startswith("EXIT")]
    print(f"  16거래  Total PnL: ${exits['pnl_usd'].sum():+.2f}")
    plot(bars, trades)
    print("완료.")
