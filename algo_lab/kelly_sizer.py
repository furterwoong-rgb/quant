"""
kelly_sizer.py — Dynamic Kelly 포지션 사이저

ProbMap + 아카이브 기반 적응형 Kelly fraction 계산.

단계별 동작:
  N < 30  : prior 모드 — f* = 0.025 고정
  N ≥ 30  : KDE 학습 시작
  N ≥ 100 : Dynamic Kelly 본격 운영

Kelly 공식:
  f*  = (b × p − (1−p)) / b
  f   = f* × 0.5           (half-Kelly)
  f   = clip(f, 0.01, 0.05)
  f ≤ 0 → 진입 차단 (notional=0 반환)
"""
from pathlib import Path

import numpy as np

KELLY_MIN  = 0.01    # half-Kelly 하한 (equity의 1%)
KELLY_MAX  = 0.05    # half-Kelly 상한 (equity의 5%)
HALF_KELLY = 0.5
PRIOR_F    = 0.025   # N<30 고정값

_ALGO_LAB = Path(__file__).parent


class KellySizer:
    def __init__(self):
        self._pm   = None   # ProbMap instance (None = prior 모드)
        self._df   = None   # 완료 거래 DataFrame
        self._n    = 0
        self.refresh()

    # ── 재학습 ──────────────────────────────────────────────────────────────────

    def refresh(self):
        """
        아카이브 재로드 + ProbMap 재학습.
        장 시작 전 1회 호출 권장.
        """
        import sys
        sys.path.insert(0, str(_ALGO_LAB))
        from signal_archiver import load_archive
        from prob_map import ProbMap, PKL_PATH, MIN_SAMPLES

        self._df = load_archive()
        self._n  = len(self._df)

        if self._n >= MIN_SAMPLES:
            pm = ProbMap()
            try:
                pm.fit(self._df)
                pm.save()
                self._pm = pm
            except Exception as e:
                print(f"[KellySizer] ProbMap 학습 실패: {e}")
                # 저장된 pkl 폴백
                if PKL_PATH.exists():
                    self._pm = ProbMap.load(PKL_PATH)
                else:
                    self._pm = None
        elif PKL_PATH.exists():
            # N < MIN_SAMPLES지만 이전 모델이 있으면 재사용
            from prob_map import ProbMap
            self._pm = ProbMap.load(PKL_PATH)
        else:
            self._pm = None

        mode = (
            f"prior(N={self._n}<30)"   if self._n < 30  else
            f"KDE-early(N={self._n})"  if self._n < 100 else
            f"dynamic(N={self._n})"
        )
        print(f"[KellySizer] refresh 완료 — 모드: {mode}")

    # ── 포지션 계산 ─────────────────────────────────────────────────────────────

    def notional(self, equity: float, sig: float,
                 atr_scale: float) -> tuple[float, dict]:
        """
        진입 notional 금액 계산.

        Returns:
            (notional_usd, info)
            notional_usd=0.0 이면 진입 차단.
            info: {p, b, raw_f, f, rule}

        사용 예 (원본 트레이더 _enter()):
            notional, info = sizer.notional(equity, sig, atr_scale)
            if notional <= 0:
                log.info(f"[{sym}] Kelly 진입 차단: {info}")
                return
            qty = int(notional / mid_px)
        """
        from prob_map import MIN_SAMPLES

        # ── prior 모드 ──────────────────────────────────────────────────────
        if self._n < MIN_SAMPLES or self._pm is None:
            return equity * PRIOR_F, {
                "p": 0.5, "b": 1.0,
                "raw_f": PRIOR_F, "f": PRIOR_F,
                "rule": f"prior(N={self._n})",
            }

        # ── Dynamic Kelly ───────────────────────────────────────────────────
        p = self._pm.query(sig, atr_scale)
        b = self._pm.query_win_loss_ratio(sig, atr_scale, self._df)

        # f* = (b*p - (1-p)) / b
        raw_f = (b * p - (1.0 - p)) / b if b > 0 else 0.0

        if raw_f <= 0:
            return 0.0, {
                "p": round(p, 4), "b": round(b, 4),
                "raw_f": round(raw_f, 6), "f": 0.0,
                "rule": "no_edge",
            }

        f = float(np.clip(raw_f * HALF_KELLY, KELLY_MIN, KELLY_MAX))
        rule = (
            "dynamic"   if self._n >= 100 else
            "kde_early"
        )
        return equity * f, {
            "p":     round(p,     4),
            "b":     round(b,     4),
            "raw_f": round(raw_f, 6),
            "f":     round(f,     6),
            "rule":  f"{rule}(N={self._n})",
        }

    # ── 진단 ────────────────────────────────────────────────────────────────────

    def summary(self):
        """현재 상태 출력."""
        from prob_map import MIN_SAMPLES
        mode = (
            "prior"     if self._n < MIN_SAMPLES or self._pm is None else
            "KDE-early" if self._n < 100 else
            "dynamic"
        )
        print(f"KellySizer 상태:")
        print(f"  완료 거래 N = {self._n}")
        print(f"  모드        = {mode}")
        if self._n > 0 and self._df is not None:
            wr  = float(self._df["result"].mean())
            avg = float(self._df["pnl_pct"].mean() * 100)
            print(f"  승률        = {wr:.1%}  ({self._n}건)")
            print(f"  평균 PnL    = {avg:+.3f}%")


# ── 단독 실행 테스트 ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    sizer  = KellySizer()
    equity = 100_000.0

    sizer.summary()
    print()

    test_cases = [
        ( 0.30, 1.0),
        ( 0.50, 0.8),
        ( 0.60, 1.5),
        ( 0.25, 1.2),
        (-0.05, 1.0),
    ]
    print(f"{'sig':>6}  {'atr':>5}  {'notional':>12}  {'p':>6}  {'b':>5}  {'f':>8}  rule")
    print("-" * 65)
    for sig, atr in test_cases:
        n, info = sizer.notional(equity, sig, atr)
        print(
            f"{sig:+6.2f}  {atr:5.1f}  ${n:>11,.0f}  "
            f"{info['p']:6.3f}  {info['b']:5.2f}  "
            f"{info['f']:8.5f}  {info['rule']}"
        )
