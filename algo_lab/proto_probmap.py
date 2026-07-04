"""
proto_probmap.py — 백테스트 → ProbMap 3D 시각화 프로토타입

백테스트 실행 → ENTER/EXIT 쌍 추출 → signals.parquet 생성 → 3D plot

실행:
    python algo_lab/proto_probmap.py              # QQQ 전체 날짜
    python algo_lab/proto_probmap.py QQQ 2026-05-01 2026-05-19
    python algo_lab/proto_probmap.py QQQ 2026-05-01 2026-05-19 --no-clear
"""
import re
import sys
from pathlib import Path

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent          # quant_trading/
ALGO    = Path(__file__).parent                 # algo_lab/
BT_DIR  = ROOT / "backtesting" / "engine"

for p in (str(ROOT), str(ALGO), str(BT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── matplotlib backend: Agg(백테스터) 보다 먼저 설정 ──────────────────────────
import matplotlib
matplotlib.use("MacOSX")          # Spyder/터미널 팝업 창
import matplotlib.pyplot as plt

# ── 백테스트 엔진 import + CACHE_DIR 패치 ─────────────────────────────────────
import backtest_spy_qqq as bt
bt.CACHE_DIR = ROOT / "logs" / "tick_cache"    # 실제 경로로 교정

import numpy as np
import pandas as pd

from signal_archiver import (
    record_entry, record_exit, load_archive,
    ARCHIVE_DIR, ARCHIVE_PATH,
)
# signals.parquet 언급 제거 (CSV로 변경됨)
from prob_map import ProbMap, MIN_SAMPLES


# ── 1. 백테스트 실행 ───────────────────────────────────────────────────────────

def run(sym: str, start: str, end: str) -> pd.DataFrame:
    """run_backtest 래퍼: 결과 DataFrame 반환."""
    # _plot()이 Agg로 저장하려 하므로 임시 패치
    orig_plot = bt._plot
    bt._plot = lambda *a, **kw: None   # 백테스터 차트 출력 억제
    try:
        df = bt.run_backtest(sym=sym, start=start, end=end)
    finally:
        bt._plot = orig_plot
    return df


# ── 2. ENTER/EXIT 쌍 추출 → signals.parquet ───────────────────────────────────

def _parse_atr(extra: str) -> float:
    """extra 문자열에서 atr 값 파싱. 예: 'atr=1.230 stop=...' → 1.23"""
    m = re.search(r"atr=([0-9.]+)", str(extra))
    return float(m.group(1)) if m else 1.0


def backtest_to_archive(df: pd.DataFrame, clear_first: bool = True) -> int:
    """
    백테스트 결과 DataFrame → signals.parquet 변환.
    ENTER 이후 첫 EXIT(PARTIAL_EXIT 제외)를 쌍으로 매칭.
    clear_first=True: 기존 아카이브 삭제 후 재작성.
    """
    if clear_first and ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()
        print("  기존 signals.parquet 삭제")

    enters  = df[df["action"] == "ENTER"].copy()
    exits   = df[~df["action"].isin(["ENTER", "PARTIAL_EXIT"])].copy()

    added = 0
    import pytz
    ET = pytz.timezone("America/New_York")

    for sym in df["symbol"].unique():
        e_rows = enters[enters["symbol"] == sym].reset_index(drop=True)
        x_rows = exits[exits["symbol"] == sym].reset_index(drop=True)

        for _, enter in e_rows.iterrows():
            later = x_rows[x_rows["timestamp"] > enter["timestamp"]]
            if later.empty:
                continue
            exit_row  = later.iloc[0]
            atr_scale = _parse_atr(enter.get("extra", ""))

            ts = enter["timestamp"]
            if ts.tzinfo is None:
                ts = ET.localize(ts)

            tid = record_entry(
                sym=sym,
                sig=float(enter["signal"]),
                atr_scale=atr_scale,
                entry_px=float(enter["price"]),
                now=ts,
            )
            record_exit(
                trade_id=tid,
                exit_px=float(exit_row["price"]),
                exit_reason=exit_row["action"].replace("EXIT_", ""),
                holding_bars=0,
            )
            added += 1

    return added


# ── 3. ProbMap 학습 + 3D plot ──────────────────────────────────────────────────

def fit_and_plot(df_archive: pd.DataFrame):
    n      = len(df_archive)
    n_win  = int(df_archive["result"].sum())
    n_loss = n - n_win
    wr     = n_win / n * 100 if n > 0 else 0

    print(f"\n  학습 데이터: {n}건  (승 {n_win} / 패 {n_loss})  승률={wr:.1f}%")

    pm = ProbMap()
    pm.fit(df_archive)
    pm.save()

    # ── 3D 지형도 ──────────────────────────────────────────────────────────────
    sig_grid = np.linspace(-1.0, 1.0,  60)
    atr_grid = np.linspace( 0.6, 2.0,  50)
    SIG, ATR = np.meshgrid(sig_grid, atr_grid)
    pts      = np.vstack([SIG.ravel(), ATR.ravel()])

    W   = pm._kde_win(pts)  * pm._n_win
    L   = (pm._kde_loss(pts) * pm._n_loss
           if pm._kde_loss is not None else np.zeros_like(W))
    TOT = W + L
    PROB = np.where(TOT > 0, W / TOT, 0.5).reshape(SIG.shape)

    fig = plt.figure(figsize=(14, 7))

    # ── 왼쪽: 3D surface ────────────────────────────────────────────────────
    ax3d = fig.add_subplot(121, projection="3d")
    surf = ax3d.plot_surface(SIG, ATR, PROB,
                              cmap="RdYlGn", alpha=0.88, linewidth=0)
    ax3d.set_xlabel("Signal (OBI+TFI)",  labelpad=8)
    ax3d.set_ylabel("ATR Scale",          labelpad=8)
    ax3d.set_zlabel("P(win)",             labelpad=8)
    ax3d.set_zlim(0, 1)
    ax3d.set_title(
        f"P(win | sig, atr_scale)\n"
        f"N={n}  W={n_win} / L={n_loss}  WR={wr:.0f}%",
        fontsize=10,
    )
    fig.colorbar(surf, ax=ax3d, shrink=0.45, pad=0.1, label="P(win)")

    # ── 오른쪽: 2D heatmap (위에서 내려다보기) ──────────────────────────────
    ax2d = fig.add_subplot(122)
    im   = ax2d.contourf(SIG, ATR, PROB, levels=20, cmap="RdYlGn")
    fig.colorbar(im, ax=ax2d, label="P(win)")

    # 실제 거래 scatter
    wins   = df_archive[df_archive["result"] == 1.0]
    losses = df_archive[df_archive["result"] == 0.0]
    ax2d.scatter(wins["sig"],   wins["atr_scale"],
                 c="lime",  s=25, alpha=0.7, label=f"Win({n_win})", zorder=5)
    ax2d.scatter(losses["sig"], losses["atr_scale"],
                 c="red",   s=25, alpha=0.7, label=f"Loss({n_loss})", zorder=5)
    ax2d.axvline(0.20, color="white", ls="--", lw=1, label="MIN_CONV=0.20")
    ax2d.axvline(0,    color="gray",  ls=":",  lw=0.8)
    ax2d.set_xlabel("Signal (OBI+TFI)")
    ax2d.set_ylabel("ATR Scale")
    ax2d.set_title("P(win) Heatmap + 실거래 산점도", fontsize=10)
    ax2d.legend(fontsize=8, loc="upper left")

    plt.suptitle("ProbMap Prototype  —  Backtest Data", fontsize=12, y=1.01)
    plt.tight_layout()

    # PNG 저장 (백업용 — 팝업이 안 열려도 확인 가능)
    out_png = ARCHIVE_DIR / "prob_map_prototype.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\n  PNG 저장: {out_png}")

    plt.show()
    return pm


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args      = sys.argv[1:]
    sym       = args[0] if len(args) > 0 else "QQQ"
    start     = args[1] if len(args) > 1 else "2025-04-03"
    end       = args[2] if len(args) > 2 else "2026-05-19"
    no_clear  = "--no-clear" in args

    print(f"\n{'='*55}")
    print(f"  ProbMap 프로토타입  |  {sym}  |  {start} ~ {end}")
    print(f"{'='*55}")

    # Step 1: 백테스트
    print("\n[1/3] 백테스트 실행 중...")
    df_trades = run(sym, start, end)
    print(f"  총 거래 레코드: {len(df_trades)}행")

    # Step 2: 아카이브 변환
    print("\n[2/3] signals.parquet 생성 중...")
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    n_added = backtest_to_archive(df_trades, clear_first=not no_clear)
    df_arch  = load_archive()
    print(f"  추가: {n_added}건  |  아카이브 총계: {len(df_arch)}건")

    if len(df_arch) < MIN_SAMPLES:
        print(f"\n⚠️  학습 데이터 부족: {len(df_arch)}건 < {MIN_SAMPLES}건")
        sys.exit(1)

    # Step 3: ProbMap 학습 + plot
    print("\n[3/3] ProbMap 학습 + 3D 시각화...")
    fit_and_plot(df_arch)
