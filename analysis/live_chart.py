"""
live_chart.py — 일별 캔들 차트 자동 생성
─────────────────────────────────────────
spy_qqq_live_trader.py 의 _print_daily_summary() 에서 호출.
CSV를 읽어 Alpaca 1분봉 + 4-panel 다크 차트를 생성한다.

Usage (외부 호출):
    from analysis.live_chart import draw_candle_chart
    draw_candle_chart(csv_path, date_str, out_path)
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

# ── 색상 ──────────────────────────────────────────────────────────────────────
C = dict(
    bg        = "#0d1117",
    ax        = "#161b22",
    grid      = "#21262d",
    border    = "#30363d",
    text      = "#8b949e",
    text_hi   = "#e6edf3",
    green     = "#26a69a",
    red       = "#ef5350",
    win_fill  = "#1a3c2a",
    loss_fill = "#3c1a1a",
    vwap      = "#f0b429",
    ma5       = "#ff6b6b",
    ma20      = "#ffd700",
    ma60      = "#a78bfa",
    entry_mk  = "#ffffff",
    exit_win  = "#00e676",
    exit_loss = "#ff5252",
    exit_lmin = "#ff9800",
    port_line = "#58a6ff",
    thresh    = "#484f58",
)


def _setup_font():
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    for cand in ("AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic"):
        m = next((f.fname for f in fm.fontManager.ttflist if cand in f.name), None)
        if m:
            plt.rcParams["font.family"] = fm.FontProperties(fname=m).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def _fetch_bars(sym: str, date_str: str, api_key: str, secret_key: str) -> pd.DataFrame:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    d = datetime.strptime(date_str, "%Y-%m-%d")
    client = StockHistoricalDataClient(api_key, secret_key)
    req = StockBarsRequest(
        symbol_or_symbols=sym,
        timeframe=TimeFrame.Minute,
        start=datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET),
        end=datetime(d.year, d.month, d.day, 16, 0, tzinfo=ET),
        feed="sip",
    )
    bars = client.get_stock_bars(req).df
    if bars.empty:
        raise ValueError(f"{sym} bar 없음 ({date_str})")
    bars = bars.xs(sym, level="symbol")
    bars.index = bars.index.tz_convert(ET)
    return bars


def _build_vwap(bars: pd.DataFrame) -> pd.Series:
    tp  = (bars["high"] + bars["low"] + bars["close"]) / 3
    vol = bars["volume"].replace(0, np.nan)
    return (tp * vol).cumsum() / vol.cumsum()


def _ax_style(ax):
    ax.set_facecolor(C["ax"])
    ax.tick_params(colors=C["text"], labelsize=8)
    ax.spines[:].set_color(C["border"])
    ax.grid(alpha=0.3, color=C["grid"], linewidth=0.5)


def draw_candle_chart(
    csv_path: str | Path,
    date_str: str,
    out_path: str | Path,
    api_key: str,
    secret_key: str,
    sym: str = "QQQ",
) -> None:
    """
    Parameters
    ----------
    csv_path   : logs/YYYY/MM/DD/spy_qqq_trades.csv
    date_str   : 'YYYY-MM-DD'  (ET 기준 거래일)
    out_path   : 저장할 PNG 경로
    api_key    : Alpaca API key
    secret_key : Alpaca secret key
    sym        : 'QQQ' or 'SPY'
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    from matplotlib import gridspec

    _setup_font()

    # ── 데이터 로드 ────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path, parse_dates=["timestamp_et"])
    df["timestamp_et"] = pd.to_datetime(df["timestamp_et"]).dt.tz_localize(ET)
    df = df[df["symbol"] == sym].copy()

    if df.empty:
        print(f"  [chart] {sym} 거래 없음 — 차트 생략")
        return

    bars = _fetch_bars(sym, date_str, api_key, secret_key)
    vwap = _build_vwap(bars)

    enters = df[df["action"] == "ENTER"].copy()
    exits  = df[df["action"].str.startswith("EXIT")].copy()

    if exits.empty:
        print(f"  [chart] EXIT 거래 없음 — 차트 생략")
        return

    # 거래 쌍 (entry_time, exit_time, pnl)
    pairs = list(zip(
        enters["timestamp_et"].tolist(),
        exits["timestamp_et"].tolist(),
        exits["pnl_usd"].tolist(),
    ))

    # ── 레이아웃 ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 15), facecolor=C["bg"])
    fig.suptitle(
        f"{sym}  OBI+TFI Live Trader  —  {date_str}  (1-min)",
        color=C["text_hi"], fontsize=12, fontweight="bold", y=0.998,
    )
    gs = gridspec.GridSpec(5, 1, hspace=0.10,
                           height_ratios=[2.0, 5, 1.2, 1.8, 1.8])
    ax0 = fig.add_subplot(gs[0])   # 메트릭 카드
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    ax3 = fig.add_subplot(gs[3], sharex=ax1)
    ax4 = fig.add_subplot(gs[4], sharex=ax1)

    ax0.set_facecolor(C["bg"])
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.axis("off")
    for ax in (ax1, ax2, ax3, ax4):
        _ax_style(ax)
    for ax in (ax1, ax2, ax3):
        plt.setp(ax.get_xticklabels(), visible=False)

    # ── 거래 구간 배경 ─────────────────────────────────────────────────────────
    for en_t, ex_t, pnl in pairs:
        fill = C["win_fill"] if pnl >= 0 else C["loss_fill"]
        for ax in (ax1, ax3):
            ax.axvspan(en_t, ex_t, color=fill, alpha=0.45, zorder=1)

    # ── 캔들스틱 ──────────────────────────────────────────────────────────────
    W = 0.00042
    for ts, row in bars.iterrows():
        up  = row["close"] >= row["open"]
        clr = C["green"] if up else C["red"]
        ax1.plot([ts, ts], [row["low"], row["high"]],
                 color=clr, linewidth=0.6, zorder=3)
        ax1.add_patch(mpatches.Rectangle(
            (mdates.date2num(ts) - W / 2, min(row["open"], row["close"])),
            W, max(abs(row["close"] - row["open"]), 0.02),
            facecolor=clr, edgecolor="none", zorder=4,
        ))

    # VWAP + MA
    ax1.plot(bars.index, vwap,
             color=C["vwap"],  lw=1.1, label="VWAP",  zorder=5, alpha=0.9)
    ax1.plot(bars.index, bars["close"].rolling(5).mean(),
             color=C["ma5"],   lw=0.8, label="MA5",   zorder=5, alpha=0.75)
    ax1.plot(bars.index, bars["close"].rolling(20).mean(),
             color=C["ma20"],  lw=0.8, label="MA20",  zorder=5, alpha=0.75)
    ax1.plot(bars.index, bars["close"].rolling(60).mean(),
             color=C["ma60"],  lw=0.9, label="MA60",  zorder=5, alpha=0.75)

    # 세션 구분선
    for hm, lbl, clr in [
        ("09:45", "진입시작",   "#ffeb3b"),
        ("14:55", "EOD tight",  "#ff9800"),
        ("15:45", "EOD reduce", "#ff5722"),
        ("15:55", "세션종료",   "#f44336"),
    ]:
        t = pd.Timestamp(f"{date_str} {hm}:00", tz=ET)
        if not bars.empty and bars.index[0] <= t <= bars.index[-1]:
            for ax in (ax1, ax3):
                ax.axvline(t, color=clr, lw=0.7, ls="--", alpha=0.5, zorder=2)
            ymax = bars["high"].max()
            ax1.text(t, ymax, f" {lbl}", color=clr, fontsize=7, va="top", zorder=6)

    # 진입/청산 마커
    lme = exits[exits["action"] == "EXIT_로컬저점손절"]
    we  = exits[(exits["pnl_usd"] >= 0) & (exits["action"] != "EXIT_로컬저점손절")]
    le  = exits[(exits["pnl_usd"] <  0) & (exits["action"] != "EXIT_로컬저점손절")]
    ax1.scatter(enters["timestamp_et"], enters["price"],
                marker="^", color=C["entry_mk"], s=80, zorder=7,
                edgecolors=C["text"], linewidths=0.5, label="Entry")
    ax1.scatter(we["timestamp_et"], we["price"],
                marker="v", color=C["exit_win"],  s=80, zorder=7,
                edgecolors="#00a050", linewidths=0.5, label="Exit (win)")
    ax1.scatter(le["timestamp_et"], le["price"],
                marker="v", color=C["exit_loss"], s=80, zorder=7,
                edgecolors="#b00020", linewidths=0.5, label="Exit (loss)")
    if not lme.empty:
        ax1.scatter(lme["timestamp_et"], lme["price"],
                    marker="X", color=C["exit_lmin"], s=100, zorder=7,
                    edgecolors="#b35900", linewidths=0.5, label="Exit (local min)")

    # PnL 레이블
    for _, row in exits.iterrows():
        if row["action"] == "EXIT_로컬저점손절":
            clr = C["exit_lmin"]
        elif row["pnl_usd"] >= 0:
            clr = C["exit_win"]
        else:
            clr = C["exit_loss"]
        ax1.annotate(
            f"${row['pnl_usd']:+.2f}",
            xy=(row["timestamp_et"], row["price"]),
            xytext=(0, -14), textcoords="offset points",
            ha="center", fontsize=7, color=clr, fontweight="bold", zorder=8,
        )

    # ── 메트릭 카드 ───────────────────────────────────────────────────────────
    total_pnl = exits["pnl_usd"].astype(float).sum()
    n_win     = (exits["pnl_usd"].astype(float) > 0).sum()
    n_tot     = len(exits)
    w_ser     = exits.loc[exits["pnl_usd"].astype(float) > 0, "pnl_usd"].astype(float)
    l_ser     = exits.loc[exits["pnl_usd"].astype(float) < 0, "pnl_usd"].astype(float)
    wr_pct    = n_win / n_tot * 100 if n_tot > 0 else 0.0
    avg_w     = w_ser.mean() if not w_ser.empty else 0.0
    avg_l     = l_ser.mean() if not l_ser.empty else 0.0
    max_w     = w_ser.max()  if not w_ser.empty else 0.0
    max_l     = l_ser.min()  if not l_ser.empty else 0.0

    pf = w_ser.sum() / abs(l_ser.sum()) if l_ser.sum() != 0 else float("inf")
    pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"

    total_buy_val  = (enters["price"].astype(float) * enters["qty"].astype(float)).sum()
    total_buy_qty  = enters["qty"].astype(float).sum()
    total_sell_val = (exits["price"].astype(float) * exits["qty"].astype(float)).sum()
    total_sell_qty = exits["qty"].astype(float).sum()
    avg_buy_px     = total_buy_val  / total_buy_qty  if total_buy_qty  > 0 else 0.0
    avg_sell_px    = total_sell_val / total_sell_qty if total_sell_qty > 0 else 0.0
    avg_qty        = enters["qty"].astype(float).mean() if not enters.empty else 1.0
    position_size  = avg_buy_px * avg_qty
    profit_rate    = total_pnl / position_size * 100 if position_size > 0 else 0.0

    # 자산 현황
    yesterday_equity = enters["portfolio"].astype(float).iloc[0]  if not enters.empty else 0.0
    today_equity     = exits["portfolio"].astype(float).iloc[-1]   if not exits.empty  else 0.0
    pr1 = total_pnl / today_equity    * 100 if today_equity    > 0 else 0.0
    pr2 = total_pnl / total_buy_val   * 100 if total_buy_val   > 0 else 0.0
    pr3 = profit_rate  # PnL / (avg_buy_px × avg_qty) × 100

    pnl_clr = C["exit_win"] if total_pnl >= 0 else C["exit_loss"]

    # ── 상단 3칸: 자산·매수총액·포지션단위 수익률 ──────────────────────────────
    top_cards = [
        (
            "Total Assets",
            f"${today_equity:,.2f}",
            f"(${yesterday_equity:,.2f}  {total_pnl:+.2f})",
            f"PR1 = PnL / Total Assets × 100  =  {pr1:+.4f}%",
            pnl_clr,
        ),
        (
            "Total Buy Price",
            f"${total_buy_val:,.2f}",
            f"{int(total_buy_qty)}주  ×  avg ${avg_buy_px:.2f}",
            f"PR2 = PnL / Total Buy Price × 100  =  {pr2:+.4f}%",
            pnl_clr,
        ),
        (
            "Position-Unit PR",
            f"{pr3:+.4f}%",
            f"avg {avg_qty:.0f}주  ×  ${avg_buy_px:.2f}  =  ${position_size:,.2f}",
            f"PR3 = PnL / (AvgBuyPx × AvgQty) × 100",
            pnl_clr,
        ),
    ]

    top_gap = 0.035
    top_cw  = (1.0 - top_gap * 4) / 3
    for k, (tlbl, tval, tsub1, tsub2, tclr) in enumerate(top_cards):
        tx0 = top_gap + k * (top_cw + top_gap)
        ax0.add_patch(mpatches.FancyBboxPatch(
            (tx0, 0.62), top_cw, 0.35,
            boxstyle="round,pad=0.01",
            facecolor=C["ax"], edgecolor=C["border"], linewidth=0.8,
            zorder=2,
        ))
        ax0.text(tx0 + top_cw / 2, 0.93, tlbl,
                 ha="center", va="center", fontsize=8, color=C["text"], zorder=3)
        ax0.text(tx0 + top_cw / 2, 0.83, tval,
                 ha="center", va="center", fontsize=11, color=tclr,
                 fontweight="bold", zorder=3)
        ax0.text(tx0 + top_cw / 2, 0.74, tsub1,
                 ha="center", va="center", fontsize=7, color=C["text"], zorder=3)
        ax0.text(tx0 + top_cw / 2, 0.65, tsub2,
                 ha="center", va="center", fontsize=6.5, color=C["text"], alpha=0.75,
                 style="italic", zorder=3)

    # ── 하단 9칸: 성과 지표 카드 ──────────────────────────────────────────────
    # (label, value, value_color, sub_description)
    cards = [
        ("Daily PnL",    f"${total_pnl:+.2f}",       pnl_clr,                                               ""),
        ("W / L",        f"{n_win}W  {n_tot-n_win}L", C["text_hi"],                                          ""),
        ("Win Rate",     f"{wr_pct:.0f}%",             C["exit_win"] if wr_pct >= 50 else C["exit_loss"],     ""),
        ("Profit Rate",  f"{pr3:+.4f}%",               pnl_clr,                                               "PnL / (AvgBuyPx×AvgQty)"),
        ("PF",           pf_str,                        C["exit_win"] if pf >= 1.0 else C["exit_loss"],        "Gross profit / Gross loss"),
        ("Avg W",        f"${avg_w:+.2f}",             C["exit_win"],                                         ""),
        ("Avg L",        f"${avg_l:+.2f}",             C["exit_loss"],                                        ""),
        ("Max W",        f"${max_w:+.2f}",             C["exit_win"],                                         ""),
        ("Max L",        f"${max_l:+.2f}",             C["exit_lmin"],                                        ""),
    ]

    n_c = len(cards)
    gap = 0.010
    cw  = (1.0 - gap * (n_c + 1)) / n_c

    for k, (lbl, val, val_clr, sub) in enumerate(cards):
        x0 = gap + k * (cw + gap)
        ax0.add_patch(mpatches.FancyBboxPatch(
            (x0, 0.03), cw, 0.40,
            boxstyle="round,pad=0.01",
            facecolor=C["ax"], edgecolor=C["border"], linewidth=0.8,
            zorder=2,
        ))
        ax0.text(x0 + cw / 2, 0.40, lbl,
                 ha="center", va="center", fontsize=7.5, color=C["text"], zorder=3)
        ax0.text(x0 + cw / 2, 0.19, val,
                 ha="center", va="center", fontsize=10,
                 color=val_clr, fontweight="bold", zorder=3)
        if sub:
            ax0.text(x0 + cw / 2, 0.06, sub,
                     ha="center", va="center", fontsize=5,
                     color=C["text"], alpha=0.7, zorder=3)

    ax1.set_ylabel("QQQ Price ($)", color=C["text"], fontsize=9)
    legend_elems = [
        Line2D([0],[0], color=C["vwap"], lw=1.2, label="VWAP"),
        Line2D([0],[0], color=C["ma5"],  lw=1.0, label="MA5"),
        Line2D([0],[0], color=C["ma20"], lw=1.0, label="MA20"),
        Line2D([0],[0], color=C["ma60"], lw=1.0, label="MA60"),
        Line2D([0],[0], marker="^", color="w",
               markerfacecolor=C["entry_mk"], markersize=7, lw=0, label="Entry"),
        Line2D([0],[0], marker="v", color="w",
               markerfacecolor=C["exit_win"],  markersize=7, lw=0, label="Exit (win)"),
        Line2D([0],[0], marker="v", color="w",
               markerfacecolor=C["exit_loss"], markersize=7, lw=0, label="Exit (loss)"),
        Line2D([0],[0], marker="X", color="w",
               markerfacecolor=C["exit_lmin"], markersize=7, lw=0, label="Exit (local min)"),
        mpatches.Patch(facecolor=C["win_fill"],  alpha=0.7, label="Win span"),
        mpatches.Patch(facecolor=C["loss_fill"], alpha=0.7, label="Loss span"),
    ]
    ax1.legend(handles=legend_elems, loc="lower right",
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

    # ── OBI+TFI 시그널 ─────────────────────────────────────────────────────────
    MIN_CONV = 0.20
    ax3.axhline( MIN_CONV, color=C["green"], lw=0.8, ls="--", alpha=0.7,
                label=f"+{MIN_CONV} (entry)")
    ax3.axhline(-MIN_CONV, color=C["red"],   lw=0.8, ls="--", alpha=0.7,
                label=f"-{MIN_CONV} (exit)")
    ax3.axhline(0, color=C["thresh"], lw=0.6, alpha=0.5)

    if "signal" in df.columns:
        for _, row in df.iterrows():
            clr = C["green"] if row["signal"] >= 0 else C["red"]
            ax3.plot([row["timestamp_et"], row["timestamp_et"]],
                     [0, row["signal"]],
                     color=clr, lw=1.0, alpha=0.6, zorder=3)
        ax3.scatter(enters["timestamp_et"], enters["signal"],
                    marker="^", color=C["entry_mk"], s=55, zorder=5,
                    edgecolors=C["text"], linewidths=0.5)
        ax3.scatter(we["timestamp_et"], we["signal"],
                    marker="v", color=C["exit_win"],  s=55, zorder=5)
        ax3.scatter(le["timestamp_et"], le["signal"],
                    marker="v", color=C["exit_loss"], s=55, zorder=5)
        if not lme.empty:
            ax3.scatter(lme["timestamp_et"], lme["signal"],
                        marker="X", color=C["exit_lmin"], s=70, zorder=5)

    ax3.set_ylabel("OBI+TFI Signal", color=C["text"], fontsize=8)
    ax3.set_ylim(-0.65, 0.65)
    ax3.legend(loc="lower right", facecolor=C["ax"],
               edgecolor=C["border"], labelcolor=C["text_hi"], fontsize=7.5)

    # ── 포트폴리오 에쿼티 ─────────────────────────────────────────────────────
    if "portfolio" in exits.columns and exits["portfolio"].notna().any():
        initial = enters.iloc[0]["portfolio"] if "portfolio" in enters.columns else exits["portfolio"].iloc[0]
        port_ts  = [enters.iloc[0]["timestamp_et"]] + exits["timestamp_et"].tolist()
        port_val = [initial] + exits["portfolio"].tolist()

        ax4.plot(port_ts, port_val, color=C["port_line"], lw=1.3, zorder=4)
        ax4.fill_between(port_ts, initial, port_val,
                         where=[v >= initial for v in port_val],
                         alpha=0.20, color="#2ea043", zorder=3)
        ax4.fill_between(port_ts, initial, port_val,
                         where=[v < initial for v in port_val],
                         alpha=0.20, color="#f85149", zorder=3)
        ax4.axhline(initial, color=C["thresh"], lw=0.8, ls="--", alpha=0.7)
        ax4.scatter(port_ts[1:], port_val[1:], color=C["port_line"], s=20, zorder=5)

        daily_equity_pnl = port_val[-1] - initial
        ax4.set_title(f"Portfolio Equity  (Daily PnL: ${daily_equity_pnl:+.2f})",
                      color=C["text"], fontsize=8, pad=2)
    else:
        # portfolio 컬럼 없으면 누적 pnl_usd로 대체
        cum_pnl = exits["pnl_usd"].cumsum()
        ax4.plot(exits["timestamp_et"], cum_pnl, color=C["port_line"], lw=1.3, zorder=4)
        ax4.axhline(0, color=C["thresh"], lw=0.8, ls="--", alpha=0.7)
        ax4.set_title(f"Cumulative PnL  (Total: ${cum_pnl.iloc[-1]:+.2f})",
                      color=C["text"], fontsize=8, pad=2)

    ax4.set_ylabel("Equity ($)", color=C["text"], fontsize=8)
    ax4.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # x축
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ET))
    ax4.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 30)))
    fig.autofmt_xdate(rotation=0, ha="center")

    # ── 저장 ──────────────────────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.993])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  [chart] PNG 저장 완료: {out_path}")
