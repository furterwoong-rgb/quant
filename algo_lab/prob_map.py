"""
prob_map.py — 2D KDE 조건부 확률 지형도

x축: sig       [-1, 1]    OBI+TFI 합성신호
y축: atr_scale [0.6, 2.0] ATR 변동성 스케일
z축: P(win)    [0, 1]     KDE 추정 승률

Bayesian KDE:
  P(win|x) = n_win × kde_win(x) / (n_win × kde_win(x) + n_loss × kde_loss(x))
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ARCHIVE_DIR = Path(__file__).parent / "archive"
PKL_PATH    = ARCHIVE_DIR / "prob_map.pkl"

MIN_SAMPLES = 30   # KDE 학습 최소 건수


class ProbMap:
    def __init__(self):
        self._kde_win:  gaussian_kde | None = None
        self._kde_loss: gaussian_kde | None = None
        self._n_win:    int = 0
        self._n_loss:   int = 0
        self._n:        int = 0

    # ── 학습 ────────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame):
        """
        load_archive() 결과로 학습. 최소 MIN_SAMPLES 건 필요.
        승 / 패 각 KDE를 별도 추정 → Bayes 결합으로 P(win|x) 계산.
        """
        if len(df) < MIN_SAMPLES:
            raise ValueError(
                f"최소 {MIN_SAMPLES}건 필요 (현재 {len(df)}건). "
                "데이터 수집 후 재실행하세요."
            )

        wins   = df[df["result"] == 1.0]
        losses = df[df["result"] == 0.0]

        self._n      = len(df)
        self._n_win  = len(wins)
        self._n_loss = len(losses)

        if self._n_win >= 2:
            X_win = wins[["sig", "atr_scale"]].values.T
            self._kde_win = gaussian_kde(X_win)
        if self._n_loss >= 2:
            X_loss = losses[["sig", "atr_scale"]].values.T
            self._kde_loss = gaussian_kde(X_loss)

    # ── 조회 ────────────────────────────────────────────────────────────────────

    def query(self, sig: float, atr_scale: float) -> float:
        """
        P(win | sig, atr_scale) 반환.
        미학습 / 데이터 부족 시 0.5 (무정보 prior).
        """
        if self._kde_win is None or self._kde_loss is None:
            return 0.5

        pt = np.array([[float(sig)], [float(atr_scale)]])

        w = float(self._kde_win(pt))  * self._n_win
        l = float(self._kde_loss(pt)) * self._n_loss
        total = w + l

        return 0.5 if total <= 0 else float(np.clip(w / total, 0.0, 1.0))

    def query_win_loss_ratio(self, sig: float, atr_scale: float,
                              df: pd.DataFrame) -> float:
        """
        손익비 b = avg_win_pct / avg_loss_pct
        sig ±0.2 / atr_scale ±0.3 범위 유사 거래 기준.
        유사 거래 5건 미만이면 전체 평균 사용.
        """
        if df.empty:
            return 1.0

        mask = (
            df["sig"].between(sig - 0.2, sig + 0.2) &
            df["atr_scale"].between(atr_scale - 0.3, atr_scale + 0.3)
        )
        subset = df[mask] if mask.sum() >= 5 else df

        wins   = subset.loc[subset["pnl_pct"] > 0, "pnl_pct"]
        losses = subset.loc[subset["pnl_pct"] < 0, "pnl_pct"].abs()

        avg_loss = float(losses.mean()) if not losses.empty else 0.0
        avg_win  = float(wins.mean())   if not wins.empty  else 0.0

        if avg_loss <= 0:
            return 1.5   # 손실 없으면 보수적 1.5 반환
        b = avg_win / avg_loss if avg_win > 0 else 0.5
        return float(np.clip(b, 0.1, 10.0))

    # ── 시각화 ──────────────────────────────────────────────────────────────────

    def plot3d(self, show: bool = True):
        """P(win) 3D 지형도. matplotlib 필요."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        if self._kde_win is None:
            print("⚠️  미학습 상태. fit() 먼저 호출하세요.")
            return

        sig_grid = np.linspace(-1.0,  1.0, 60)
        atr_grid = np.linspace( 0.6,  2.0, 50)
        SIG, ATR = np.meshgrid(sig_grid, atr_grid)
        pts      = np.vstack([SIG.ravel(), ATR.ravel()])

        W    = self._kde_win(pts)  * self._n_win
        L    = (self._kde_loss(pts) * self._n_loss
                if self._kde_loss is not None else np.zeros_like(W))
        TOT  = W + L
        PROB = np.where(TOT > 0, W / TOT, 0.5).reshape(SIG.shape)

        fig  = plt.figure(figsize=(13, 7))
        ax   = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(SIG, ATR, PROB,
                               cmap="RdYlGn", alpha=0.88, linewidth=0)
        fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1, label="P(win)")
        ax.set_xlabel("Signal (OBI+TFI)")
        ax.set_ylabel("ATR Scale")
        ax.set_zlabel("P(win)")
        ax.set_zlim(0, 1)
        ax.set_title(
            f"Probability Map  N={self._n}  "
            f"(W={self._n_win} / L={self._n_loss})"
        )
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    # ── 저장 / 로드 ─────────────────────────────────────────────────────────────

    def save(self, path: Path = PKL_PATH):
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"ProbMap 저장: {path}  (N={self._n})")

    @staticmethod
    def load(path: Path = PKL_PATH) -> "ProbMap":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"ProbMap 로드: {path}  (N={obj._n})")
        return obj


# ── 단독 실행 테스트 ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from signal_archiver import load_archive

    df = load_archive()
    print(f"아카이브 데이터: {len(df)}건")

    if len(df) < MIN_SAMPLES:
        print(f"⚠️  학습 데이터 부족 ({len(df)}/{MIN_SAMPLES}건)")
        print("   prior 모드 테스트: query() → 0.5")
        pm = ProbMap()
        print(f"   P(win|sig=0.4, atr=1.0) = {pm.query(0.4, 1.0):.3f}")
    else:
        pm = ProbMap()
        pm.fit(df)
        pm.save()
        print(f"\nP(win) 쿼리 샘플:")
        for sig, atr in [(0.3, 1.0), (0.5, 0.8), (0.6, 1.5), (-0.1, 1.0)]:
            p = pm.query(sig, atr)
            b = pm.query_win_loss_ratio(sig, atr, df)
            print(f"  sig={sig:+.1f}  atr={atr:.1f}  P(win)={p:.3f}  b={b:.2f}")
        pm.plot3d()
