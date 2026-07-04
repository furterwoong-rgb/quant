#!/usr/bin/env python3
"""
Signal Logger — 백테스트 실행 시 signal_log.csv 자동 생성  (v2)
───────────────────────────────────────────────────────────────────────────────
GPT 피드백 반영 사항 (v2):
  1. threshold 기본값 0.0 → 모든 bar 저장 (selection bias 제거)
     → is_signal 컬럼(True/False)으로 Tier 3에서 자체 필터링
  2. inline baseline 컬럼 제거 (bar_idx-10 방식은 local regime 편향)
     → Tier 3에서 non-signal rows(is_signal=False)를 baseline으로 사용
  3. cooldown_bars 파라미터 추가 (중첩 신호 dedup)
     → 신호 발생 후 cooldown_bars 내 중복 기록 방지
  4. 방법 A(틱 연동) 우선 권고 강화
     → 방법 B(bar 근사)는 candle shape feature에 가까우며 microstructure 훼손

생성되는 signal_log.csv 컬럼 (v2):
  signal_time, symbol, signal_score, obi_raw, tfi_raw, is_signal,
  ret_1m, ret_3m, ret_5m, ret_10m

  ※ baseline_ret_* 컬럼 없음 — Tier 3에서 is_signal=False rows로 대체

방법 A (권장): 백테스터 내부 연동
  from validation.signal_logger import BacktestSignalLogger
  logger = BacktestSignalLogger(sym="SPY")
  # 매 bar 신호 계산 후:
  logger.on_bar(ts, signal, obi_raw, tfi_raw, bars_df, bar_idx)
  # 완료 후:
  logger.save("./data/signal_log.csv")

방법 B (근사, standalone): 1분봉 bars만 있을 때
  python signal_logger.py \\
    --data-path ./data/spy_qqq_1min.csv \\
    --symbol    SPY \\
    --out       ./data/signal_log.csv \\
    --save-threshold 0.0       # 0.0 = 전체 저장 (권장)
    --signal-threshold 0.25    # is_signal=True 기준
    --cooldown 5               # 5분 cooldown
  ⚠️ 방법 B의 OBI/TFI는 candle shape feature 근사이며 실제 microstructure 신호가 아님.
     진짜 edge 검증에는 반드시 방법 A를 사용할 것.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OBI_ALPHA        = 0.08
TFI_ALPHA        = 0.15
SIGNAL_THRESHOLD = 0.25   # is_signal=True 기준 (저장 임계값과 별개)
FWD_HORIZONS     = [1, 3, 5, 10]


# ══════════════════════════════════════════════════════════════════════════════
# 방법 A: 백테스터 내부 연동용 클래스
# ══════════════════════════════════════════════════════════════════════════════

class BacktestSignalLogger:
    """
    백테스터 내부에서 호출.

    변경 (v2):
      - threshold 기본값 0.0 (모든 bar 저장)
      - is_signal 컬럼 추가 (signal_threshold 기준)
      - baseline 컬럼 제거 (Tier 3에서 non-signal rows로 대체)
      - cooldown_bars: 연속 신호 중복 기록 방지

    사용 예시 (backtest_spy_qqq.py 내부):
        logger = BacktestSignalLogger(sym="SPY")

        # 매 bar 신호 계산 후 (매매 룰 전 위치에 배치):
        logger.on_bar(
            ts=bar_ts,
            signal=sig_value,
            obi_raw=obi_raw_value,
            tfi_raw=tfi_raw_value,
            bars_df=bars_df,
            bar_idx=current_bar_index,
        )

        # 백테스트 완료 후:
        logger.save("./data/signal_log.csv")
    """

    def __init__(
        self,
        sym: str = "SPY",
        save_threshold: float = 0.0,    # 이 값 이상인 bar만 저장 (0.0 = 전부)
        signal_threshold: float = SIGNAL_THRESHOLD,  # is_signal=True 기준
        cooldown_bars: int = 5,         # 신호 발생 후 cooldown (bars 단위)
    ):
        self.sym              = sym
        self.save_threshold   = save_threshold
        self.signal_threshold = signal_threshold
        self.cooldown_bars    = cooldown_bars
        self._rows: list[dict] = []
        self._last_signal_bar: int = -cooldown_bars  # cooldown 추적

    def on_bar(
        self,
        ts: pd.Timestamp,
        signal: float,
        obi_raw: float,
        tfi_raw: float,
        bars_df: pd.DataFrame,
        bar_idx: int,
    ) -> None:
        """
        매 bar 호출. save_threshold 이상이고 cooldown 통과한 bar만 기록.
        is_signal: signal >= signal_threshold (Tier 3에서 분류 기준)
        """
        if signal < self.save_threshold:
            return

        is_signal = signal >= self.signal_threshold

        # cooldown: is_signal=True 신호만 cooldown 적용 (non-signal은 항상 저장)
        if is_signal:
            if bar_idx - self._last_signal_bar < self.cooldown_bars:
                return
            self._last_signal_bar = bar_idx

        fwd_rets = _compute_forward_returns(bars_df, bar_idx)

        self._rows.append({
            "signal_time":  ts,
            "symbol":       self.sym,
            "signal_score": round(float(signal), 6),
            "obi_raw":      round(float(obi_raw), 6),
            "tfi_raw":      round(float(tfi_raw), 6),
            "is_signal":    is_signal,
            **fwd_rets,
        })

    # 하위 호환성: 구버전 on_signal 인터페이스 유지
    def on_signal(self, ts, signal, obi_raw, tfi_raw, bars_df, bar_idx):
        self.on_bar(ts, signal, obi_raw, tfi_raw, bars_df, bar_idx)

    def save(self, out_path: str) -> pd.DataFrame:
        df = pd.DataFrame(self._rows)
        if len(df) == 0:
            print(f"  [signal_logger] 기록된 row 없음 "
                  f"(save_threshold={self.save_threshold})")
            return df
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        n_sig  = int(df["is_signal"].sum())
        n_base = int((~df["is_signal"]).sum())
        print(
            f"  [signal_logger] 저장: {out_path}  "
            f"(total={len(df):,}  signal={n_sig:,}  non-signal={n_base:,})"
        )
        return df

    def clear(self):
        self._rows.clear()
        self._last_signal_bar = -self.cooldown_bars


# ══════════════════════════════════════════════════════════════════════════════
# Forward Return 계산 (baseline 제거됨 v2)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_forward_returns(bars_df: pd.DataFrame, bar_idx: int) -> dict:
    """bar_idx 기준 이후 1/3/5/10분 수익률 (순수 price forward return)."""
    result = {}
    if bar_idx >= len(bars_df):
        return {f"ret_{h}m": np.nan for h in FWD_HORIZONS}

    entry_close = float(bars_df["close"].iloc[bar_idx])
    if not np.isfinite(entry_close) or entry_close <= 0:
        return {f"ret_{h}m": np.nan for h in FWD_HORIZONS}

    for h in FWD_HORIZONS:
        fwd_idx = bar_idx + h
        if fwd_idx < len(bars_df):
            fwd_close = float(bars_df["close"].iloc[fwd_idx])
            result[f"ret_{h}m"] = round((fwd_close - entry_close) / entry_close, 8)
        else:
            result[f"ret_{h}m"] = np.nan
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 방법 B: 1분봉 bars → 근사 신호 생성 (standalone)
# ══════════════════════════════════════════════════════════════════════════════

def generate_from_bars(
    bars_df: pd.DataFrame,
    symbol: str          = "SPY",
    save_threshold: float = 0.0,
    signal_threshold: float = SIGNAL_THRESHOLD,
    obi_alpha: float     = OBI_ALPHA,
    tfi_alpha: float     = TFI_ALPHA,
    cooldown_bars: int   = 5,
) -> pd.DataFrame:
    """
    1분봉 OHLCV에서 OBI/TFI 근사값 계산 → signal_log DataFrame 반환.

    ⚠️ 이 방법의 OBI/TFI는 tick 기반이 아닌 candle shape feature 근사:
       OBI 근사: (close - open) / (high - low + ε)
       TFI 근사: 2*(close - low) / (high - low + ε) - 1  [Williams %R 기반]
    실제 edge 검증에는 BacktestSignalLogger(방법 A)를 사용할 것.

    v2 변경:
      - save_threshold=0.0 기본값 (모든 bar 저장)
      - is_signal 컬럼 추가
      - baseline 컬럼 제거
      - cooldown_bars dedup 적용
    """
    df = bars_df.copy()

    if "timestamp" in df.columns:
        df["ts"] = pd.to_datetime(df["timestamp"])
    elif hasattr(df.index, "hour"):
        df["ts"] = df.index
    else:
        df["ts"] = pd.to_datetime(df.index)

    hl_range = (df["high"] - df["low"]).clip(lower=1e-8)
    obi_raw  = (df["close"] - df["open"]) / hl_range
    tfi_raw  = 2.0 * (df["close"] - df["low"]) / hl_range - 1.0

    obi_ewm = obi_raw.ewm(alpha=obi_alpha, adjust=False).mean()
    tfi_ewm = tfi_raw.ewm(alpha=tfi_alpha, adjust=False).mean()
    signal  = (obi_ewm * 0.35 + tfi_ewm * 0.65).clip(-1, 1)

    rows: list[dict] = []
    last_signal_bar  = -cooldown_bars

    for i in range(len(df)):
        sig = float(signal.iloc[i])
        if sig < save_threshold:
            continue

        is_sig = sig >= signal_threshold

        if is_sig:
            if i - last_signal_bar < cooldown_bars:
                continue
            last_signal_bar = i

        fwd = _compute_forward_returns(df, i)
        rows.append({
            "signal_time":  df["ts"].iloc[i],
            "symbol":       symbol,
            "signal_score": round(sig, 6),
            "obi_raw":      round(float(obi_raw.iloc[i]), 6),
            "tfi_raw":      round(float(tfi_raw.iloc[i]), 6),
            "is_signal":    is_sig,
            **fwd,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# CLI (방법 B standalone 실행)
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Signal Logger v2 — bars → signal_log.csv",
        epilog=(
            "⚠️  방법 B(bar 근사)는 candle shape feature 근사입니다.\n"
            "    진짜 microstructure OBI/TFI 검증에는 방법 A(BacktestSignalLogger)를 사용하세요."
        ),
    )
    p.add_argument("--data-path",         required=True,  help="1분봉 OHLCV CSV")
    p.add_argument("--symbol",            default="SPY")
    p.add_argument("--save-threshold",    type=float, default=0.0,
                   help="저장 최소 signal score (0.0 = 전부 저장, 권장)")
    p.add_argument("--signal-threshold",  type=float, default=SIGNAL_THRESHOLD,
                   help="is_signal=True 기준 (기본 0.25)")
    p.add_argument("--obi-alpha",         type=float, default=OBI_ALPHA)
    p.add_argument("--tfi-alpha",         type=float, default=TFI_ALPHA)
    p.add_argument("--cooldown",          type=int,   default=5,
                   help="신호 간 cooldown (bars 수, 기본 5)")
    p.add_argument("--out",               default="./data/signal_log.csv")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    print(f"bars 로드: {args.data_path}")
    bars = pd.read_csv(args.data_path)
    if "timestamp" not in bars.columns:
        bars = bars.rename(columns={bars.columns[0]: "timestamp"})

    print(f"  rows: {len(bars):,}")
    print(f"  신호 계산 (OBI α={args.obi_alpha}, TFI α={args.tfi_alpha})")
    print(f"  save_threshold={args.save_threshold}  "
          f"signal_threshold={args.signal_threshold}  "
          f"cooldown={args.cooldown} bars")
    print("  ⚠️  bar 근사 모드 — microstructure 신호 아님")

    sig_df = generate_from_bars(
        bars,
        symbol           = args.symbol,
        save_threshold   = args.save_threshold,
        signal_threshold = args.signal_threshold,
        obi_alpha        = args.obi_alpha,
        tfi_alpha        = args.tfi_alpha,
        cooldown_bars    = args.cooldown,
    )

    if len(sig_df) == 0:
        print("  경고: 저장된 row 없음 — save_threshold를 낮춰 보세요.")
        sys.exit(0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sig_df.to_csv(args.out, index=False)

    n_sig  = int(sig_df["is_signal"].sum())
    n_base = int((~sig_df["is_signal"]).sum())
    print(f"\n  완료: {args.out}")
    print(f"  total={len(sig_df):,}  is_signal=True: {n_sig:,}  "
          f"is_signal=False (baseline용): {n_base:,}")
    print(f"  기간: {sig_df['signal_time'].min()} ~ {sig_df['signal_time'].max()}")
    print(f"  score 범위: {sig_df['signal_score'].min():.4f} ~ "
          f"{sig_df['signal_score'].max():.4f}")
