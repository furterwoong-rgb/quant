"""
proto_bar_probmap.py — 1분봉 전체 기반 P(next bar up | sig, atr_scale)

진입 여부와 무관하게 모든 1분봉에 대해:
  x: sig        = OBI×0.35 + TFI×0.65  (현재 봉 마감 시점)
  y: atr_scale  = ATR14 / avg30  (당일 고정)
  z: P(up)      = P(다음 봉 close > 현재 봉 close)

실행:
    python algo_lab/proto_bar_probmap.py
    python algo_lab/proto_bar_probmap.py QQQ 2025-04-03 2026-05-19
"""
import sys
from pathlib import Path
from datetime import time as dtime

ROOT   = Path(__file__).parent.parent
ALGO   = Path(__file__).parent
BT_DIR = ROOT / "backtesting" / "engine"

for p in (str(ROOT), str(ALGO), str(BT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import numpy as np
import pandas as pd
import pytz
from scipy.stats import gaussian_kde

import backtest_spy_qqq as bt
bt.CACHE_DIR = ROOT / "logs" / "tick_cache"

ET           = pytz.timezone("America/New_York")
SESSION_OPEN  = dtime(9, 45)
SESSION_CLOSE = dtime(15, 54)   # 마지막 봉(15:55)은 next bar 없으므로 제외


# ── 1. 전체 봉 데이터 수집 ─────────────────────────────────────────────────────

def collect_bars(sym: str, start: str, end: str) -> pd.DataFrame:
    """
    모든 날짜의 1분봉에 대해 (sig, atr_scale, next_up) 수집.
    next_up = 1 if next_close > this_close, else 0.
    """
    avail = sorted(
        p.name.split(f"{sym}_")[1].split("_quotes")[0]
        for p in bt.CACHE_DIR.glob(f"{sym}_*_quotes.pkl")
    )
    days = [d for d in avail if start <= d <= end]
    print(f"  사용 날짜: {len(days)}일  ({days[0]} ~ {days[-1]})")

    rows = []
    for date_str in days:
        # ATR 스케일 (당일 고정값)
        atr_scale = bt.get_atr_scale(sym, date_str)

        # 신호 시계열 (분봉 단위)
        try:
            sig_series = bt.compute_signals(sym, date_str)
        except Exception as e:
            print(f"  [{date_str}] 신호 계산 실패: {e}")
            continue

        # bars (close 가격)
        try:
            _, _, bars = bt.load_day(sym, date_str)
        except Exception as e:
            print(f"  [{date_str}] bars 로드 실패: {e}")
            continue

        # sig와 bars를 같은 인덱스로 정렬
        bars_et   = bars.copy()
        bars_et.index = bars_et.index.tz_convert(ET)

        sig_aligned = sig_series.reindex(bars_et.index, method="ffill").fillna(0.0)

        closes = bars_et["close"].values
        sigs   = sig_aligned.values
        times  = bars_et.index

        for i in range(len(closes) - 1):   # 마지막 봉은 next 없으므로 제외
            t = times[i].time()
            if not (SESSION_OPEN <= t <= SESSION_CLOSE):
                continue

            next_up = 1 if closes[i + 1] > closes[i] else 0
            rows.append({
                "date":      date_str,
                "time":      str(t)[:5],
                "sig":       round(float(sigs[i]), 4),
                "atr_scale": round(float(atr_scale), 4),
                "this_close": round(float(closes[i]), 4),
                "next_close": round(float(closes[i + 1]), 4),
                "next_up":   next_up,
            })

        print(f"  [{date_str}]  ATR={atr_scale:.3f}  봉={len(rows)}건 누적")

    df = pd.DataFrame(rows)
    print(f"\n  총 {len(df)}개 봉  (상승 {df['next_up'].sum()} / 하락 {(df['next_up']==0).sum()})")
    return df


# ── 2. KDE 학습 ────────────────────────────────────────────────────────────────

def fit_kde(df: pd.DataFrame):
    ups   = df[df["next_up"] == 1]
    downs = df[df["next_up"] == 0]

    X_up   = ups[["sig", "atr_scale"]].values.T
    X_down = downs[["sig", "atr_scale"]].values.T

    kde_up   = gaussian_kde(X_up,   bw_method=0.35)
    kde_down = gaussian_kde(X_down, bw_method=0.35)

    n_up   = len(ups)
    n_down = len(downs)
    return kde_up, kde_down, n_up, n_down


# ── 3. 시각화 ──────────────────────────────────────────────────────────────────

def plot(df: pd.DataFrame, kde_up, kde_down, n_up: int, n_down: int):
    n    = n_up + n_down
    base = n_up / n   # 전체 상승 비율

    # 격자 생성 (데이터 실제 범위 기반)
    sig_min, sig_max = df["sig"].min() - 0.05, df["sig"].max() + 0.05
    atr_min, atr_max = df["atr_scale"].min() - 0.05, df["atr_scale"].max() + 0.05

    sig_grid = np.linspace(sig_min, sig_max, 80)
    atr_grid = np.linspace(atr_min, atr_max, 60)
    SIG, ATR = np.meshgrid(sig_grid, atr_grid)
    pts      = np.vstack([SIG.ravel(), ATR.ravel()])

    # Bayesian: P(up|x) = n_up×kde_up(x) / (n_up×kde_up(x) + n_down×kde_down(x))
    W    = kde_up(pts)   * n_up
    L    = kde_down(pts) * n_down
    TOT  = W + L
    PROB = np.where(TOT > 0, W / TOT, base).reshape(SIG.shape)

    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor("#0d1117")

    # ── 왼쪽: 3D surface ────────────────────────────────────────────────────
    ax3d = fig.add_subplot(121, projection="3d")
    ax3d.set_facecolor("#0d1117")
    surf = ax3d.plot_surface(SIG, ATR, PROB,
                              cmap="RdYlGn", alpha=0.90, linewidth=0)
    ax3d.set_xlabel("Signal (OBI+TFI)", color="#8b949e", labelpad=8)
    ax3d.set_ylabel("ATR Scale",         color="#8b949e", labelpad=8)
    ax3d.set_zlabel("P(next bar up)",    color="#8b949e", labelpad=8)
    ax3d.set_zlim(0, 1)
    ax3d.tick_params(colors="#8b949e")
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    cb = fig.colorbar(surf, ax=ax3d, shrink=0.45, pad=0.1)
    cb.ax.yaxis.set_tick_params(color="#8b949e")
    ax3d.set_title(
        f"P(next bar up | sig, atr_scale)\n"
        f"N={n:,}봉  ↑{n_up:,} / ↓{n_down:,}  base={base:.1%}",
        color="white", fontsize=10,
    )

    # ── 오른쪽: 2D heatmap ──────────────────────────────────────────────────
    ax2d = fig.add_subplot(122)
    ax2d.set_facecolor("#0d1117")
    im   = ax2d.contourf(SIG, ATR, PROB, levels=25, cmap="RdYlGn",
                          vmin=0.3, vmax=0.7)   # 0.5 중심으로 ±0.2 강조
    cb2  = fig.colorbar(im, ax=ax2d, label="P(next bar up)")
    cb2.ax.yaxis.set_tick_params(color="#8b949e")

    # 0.5 등고선 (기준선)
    ax2d.contour(SIG, ATR, PROB, levels=[0.5],
                 colors="white", linewidths=1.2, linestyles="--")

    # MIN_CONV 기준선
    ax2d.axvline(0.20, color="cyan", ls=":", lw=1.0, alpha=0.7,
                 label="MIN_CONV=0.20")

    # 실제 데이터 분포 (밀도로 표현, scatter 는 너무 많아 kernel density contour 사용)
    from scipy.stats import gaussian_kde as gkde
    xy  = df[["sig", "atr_scale"]].values.T
    kde_all = gkde(xy)
    xx, yy  = np.meshgrid(np.linspace(sig_min, sig_max, 60),
                           np.linspace(atr_min, atr_max, 50))
    zz  = kde_all(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax2d.contour(xx, yy, zz, levels=5,
                 colors="white", linewidths=0.6, alpha=0.4)

    ax2d.axvline(0, color="gray", ls=":", lw=0.7, alpha=0.5)
    ax2d.set_xlabel("Signal (OBI+TFI)", color="#8b949e")
    ax2d.set_ylabel("ATR Scale",         color="#8b949e")
    ax2d.tick_params(colors="#8b949e")
    ax2d.legend(fontsize=8, facecolor="#161b22", labelcolor="white")
    ax2d.set_title(
        "P(next bar up) — 흰 점선: P=0.5  흰 실선: 데이터 밀도",
        color="white", fontsize=10,
    )

    plt.suptitle(
        "Bar-level Probability Map  (모든 1분봉 기반)",
        color="white", fontsize=13,
    )
    plt.tight_layout()

    out_png = ALGO / "archive" / "bar_probmap.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  PNG 저장: {out_png}")
    plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args  = sys.argv[1:]
    sym   = args[0] if len(args) > 0 else "QQQ"
    start = args[1] if len(args) > 1 else "2025-04-03"
    end   = args[2] if len(args) > 2 else "2026-05-19"

    print(f"\n{'='*55}")
    print(f"  Bar-level ProbMap  |  {sym}  |  {start} ~ {end}")
    print(f"{'='*55}\n")

    print("[1/3] 1분봉 데이터 수집 중...")
    df = collect_bars(sym, start, end)

    print("\n[2/3] KDE 학습 중...")
    kde_up, kde_down, n_up, n_down = fit_kde(df)
    print(f"  완료  (상승 KDE N={n_up:,} / 하락 KDE N={n_down:,})")

    print("\n[3/3] 시각화 중...")
    plot(df, kde_up, kde_down, n_up, n_down)
