#!/usr/bin/env python3
"""
Tier 3 — 신호 정보력 검증 (Signal Validity Test)
───────────────────────────────────────────────────────────────────────────────
OBI+TFI 합성신호가 실제로 미래 수익률 정보를 담고 있는지 직접 검증.
트레이딩 룰과 무관하게 신호 자체의 예측력을 측정.

실행:
  python tier3_signal.py \
    --signal-log ./data/signal_log.csv \
    --data-path   ./data/spy_qqq_1min.csv \
    --out-dir     ./validation_results/20260529_120000 \
    --symbol      SPY \
    --block-size  10

판정 기준 (2가지 모두 충족):
  ① Permutation test p < 0.05 (primary — "랜덤 신호였어도 이 결과 나오나?")
  ② Spearman 단조성 r > 0 AND p < 0.10 (N≥50 ME bin 기준)

변경 이력:
  v2: GPT 피드백 반영
    - block_bootstrap → 진짜 null 검증이 아님 → permutation_test 추가 (primary)
    - all(a<=b) strict monotonicity → Spearman + ME bins 으로 교체
    - nested bins → mutually exclusive bins (ME_BINS) 추가
      (분포 시각화는 nested 유지, 단조성 검증은 ME bins 사용)
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

# ── Nested bins (분포 시각화·PF 비교용) ──────────────────────────────────────
SIGNAL_BINS = [
    ("전체 (>0.25)",    0.25),
    ("상위 50% (>0.30)", 0.30),
    ("상위 25% (>0.35)", 0.35),
    ("상위 10% (>0.40)", 0.40),
]

# ── Mutually Exclusive bins (단조성 검증용) ──────────────────────────────────
ME_BINS = [
    ("0.25~0.30", 0.25, 0.30),
    ("0.30~0.35", 0.30, 0.35),
    ("0.35~0.40", 0.35, 0.40),
    (">0.40",     0.40, 1.01),
]
ME_MIDS = [0.275, 0.325, 0.375, 0.42]   # 각 bin 대표값 (Spearman x축)

FWD_HORIZONS_MIN = [1, 3, 5, 10]        # bars 모드 (분)
FWD_HORIZONS_SEC = [1, 5, 10, 30, 60]  # tick 모드 (초)
FWD_HORIZONS     = FWD_HORIZONS_MIN    # 런타임에 load_signal_log에서 덮어씀
_UNIT            = "m"                 # "m" 또는 "s", load_signal_log에서 결정
MIN_SAMPLE       = 50   # N < MIN_SAMPLE → 결과 신뢰 불가, 판정 제외

def _col(h: int) -> str:
    """현재 모드에 맞는 forward return 컬럼명 반환."""
    return f"ret_{h}{_UNIT}"

def _base_col(h: int) -> str:
    return f"baseline_ret_{h}{_UNIT}"

def _primary_h() -> int:
    """주 판정에 사용할 기준 horizon (분 모드: 1m, 초 모드: 5s)."""
    return FWD_HORIZONS[1] if _UNIT == "s" else FWD_HORIZONS[0]

def _h_label(h: int) -> str:
    return f"+{h}{'초' if _UNIT == 's' else '분'}"


# ══════════════════════════════════════════════════════════════════════════════
# 데이터 로드
# ══════════════════════════════════════════════════════════════════════════════

def load_signal_log(path: str) -> pd.DataFrame:
    """
    signal_log.csv 로드. v1/v2 포맷 모두 지원.

    v2 포맷 (권장):
      signal_time, symbol, signal_score, obi_raw, tfi_raw, is_signal,
      ret_1m, ret_3m, ret_5m, ret_10m
      → baseline = is_signal=False rows (Tier 3 내부에서 자동 분리)

    v1 포맷 (하위 호환):
      signal_time, signal_score, ret_1m..ret_10m,
      baseline_ret_1m..baseline_ret_10m
      → baseline 컬럼을 그대로 사용
    """
    df = pd.read_csv(path, parse_dates=["signal_time"])
    df = df.sort_values("signal_time").reset_index(drop=True)

    # tick 모드 미리 감지 후 필수 컬럼 검사
    if "ret_1s" in df.columns:
        required_core = ["signal_score", "ret_1s", "ret_5s", "ret_10s", "ret_30s", "ret_60s"]
    else:
        required_core = ["signal_score", "ret_1m", "ret_3m", "ret_5m", "ret_10m"]
    missing = [c for c in required_core if c not in df.columns]
    if missing:
        raise ValueError(f"signal_log.csv 누락 컬럼: {missing}")

    # v2: is_signal 컬럼 없으면 signal_score ≥ 0.25 기준으로 자동 생성
    if "is_signal" not in df.columns:
        if all(f"baseline_ret_{h}m" in df.columns for h in [1, 3, 5, 10]):
            # v1 포맷: baseline 컬럼이 이미 있음
            df["is_signal"] = True   # v1에서는 모든 row가 signal
        else:
            df["is_signal"] = df["signal_score"] >= 0.25

    df["is_signal"] = df["is_signal"].astype(bool)

    # tick 모드 감지: ret_1s 컬럼 존재 여부
    global FWD_HORIZONS, _UNIT
    if "ret_1s" in df.columns:
        FWD_HORIZONS = FWD_HORIZONS_SEC
        _UNIT = "s"
        has_baseline = all(f"baseline_ret_{h}s" in df.columns for h in FWD_HORIZONS_SEC)
        if not has_baseline:
            df = _inject_baseline_from_nonsignal(df, unit="s")
    else:
        FWD_HORIZONS = FWD_HORIZONS_MIN
        _UNIT = "m"
        has_baseline = all(f"baseline_ret_{h}m" in df.columns for h in [1, 3, 5, 10])
        if not has_baseline:
            df = _inject_baseline_from_nonsignal(df, unit="m")

    return df


def _inject_baseline_from_nonsignal(df: pd.DataFrame, unit: str = "m") -> pd.DataFrame:
    """
    v2 포맷용: is_signal=False rows의 ret 분포를 baseline_ret 컬럼으로 주입.

    방법:
      non-signal rows를 랜덤 샘플링하여 signal rows와 1:1 매핑.
      permutation test가 primary이므로 이 baseline은 시각화용.
    """
    horizons = FWD_HORIZONS_SEC if unit == "s" else FWD_HORIZONS_MIN
    non_sig  = df[~df["is_signal"]].copy()

    if len(non_sig) == 0:
        for h in horizons:
            df[f"baseline_ret_{h}{unit}"] = np.nan
        return df

    rng_local = np.random.default_rng(42)

    for h in horizons:
        col = f"ret_{h}{unit}"
        base_vals = non_sig[col].dropna().values if col in non_sig.columns else np.array([])
        if len(base_vals) == 0:
            df[f"baseline_ret_{h}{unit}"] = np.nan
            continue
        sampled = rng_local.choice(base_vals, size=len(df), replace=True)
        df[f"baseline_ret_{h}{unit}"] = sampled

    return df


def load_bars(path: str, symbol: str = None) -> pd.DataFrame:
    """
    1분봉 OHLCV CSV 로드.
    컬럼: timestamp, open, high, low, close, volume  (+ 선택: symbol)
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    if symbol and "symbol" in df.columns:
        df = df[df["symbol"] == symbol].reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 1. Forward Return 분포 분석
# ══════════════════════════════════════════════════════════════════════════════

def analyze_forward_return_distribution(
    sig_df: pd.DataFrame,
    out_dir: Path,
) -> dict:
    """
    신호 구간별 forward return 분포 분석.
    히스토그램 4종 저장: 각 horizon(1/3/5/10m)별로 모든 bin을 오버레이.
    """
    results = {}
    n_h   = len(FWD_HORIZONS)
    n_cols = 2
    n_rows = (n_h + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    if n_rows == 1:
        axes = [axes]   # 1행이면 리스트로 통일
    fig.patch.set_facecolor("#0d1117")

    for ax_idx, h in enumerate(FWD_HORIZONS):
        ax = axes[ax_idx // n_cols][ax_idx % n_cols]
        ax.set_facecolor("#161b22")
        col = _col(h)

        # Baseline 분포
        baseline_col = _base_col(h)
        baseline_vals = sig_df[baseline_col].dropna().values
        ax.hist(baseline_vals, bins=60, alpha=0.4, color="#8b949e",
                label=f"Baseline (N={len(baseline_vals):,})", density=True)

        bin_results = []
        colors = ["#58a6ff", "#3fb950", "#f0883e", "#ff7b72"]
        for i, (label, thresh) in enumerate(SIGNAL_BINS):
            subset = sig_df[sig_df["signal_score"] > thresh][col].dropna()
            n = len(subset)
            note = " *(표본 부족 — 참고용)" if n < MIN_SAMPLE else ""
            ax.hist(subset.values, bins=60, alpha=0.5, color=colors[i],
                    label=f"{label}: N={n:,}{note}", density=True)
            bin_results.append({
                "label":    label,
                "threshold": thresh,
                "N":        n,
                "mean_ret": float(subset.mean()) if n > 0 else None,
                "std_ret":  float(subset.std())  if n > 1 else None,
                "note":     "표본 부족 — 참고용" if n < MIN_SAMPLE else "",
            })

        ax.axvline(0, color="#f85149", linewidth=1, linestyle="--")
        ax.set_title(f"Forward Return {_h_label(h)}", color="#e6edf3", fontsize=11)
        ax.set_xlabel("Return", color="#8b949e")
        ax.tick_params(colors="#8b949e")
        ax.legend(fontsize=7, facecolor="#161b22", labelcolor="#e6edf3")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        results[_col(h)] = bin_results

    # 남은 빈 서브플롯 숨기기
    for ax_idx in range(n_h, n_rows * n_cols):
        axes[ax_idx // n_cols][ax_idx % n_cols].set_visible(False)

    fig.suptitle("Forward Return Distribution by Signal Strength",
                 color="#e6edf3", fontsize=13, y=1.01)
    plt.tight_layout()
    out_path = out_dir / "tier3_fwd_return_dist.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [저장] {out_path.name}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 2. Signal-Return 단조성 검증 (Mutually Exclusive bins + Spearman)
# ══════════════════════════════════════════════════════════════════════════════

def test_signal_monotonicity(
    sig_df: pd.DataFrame,
    horizon: int,
    out_dir: Path,
) -> dict:
    """
    Mutually Exclusive bins (0.25~0.30, 0.30~0.35, 0.35~0.40, >0.40) 사용.

    판정 기준 (둘 중 하나 충족):
      A) strict: 연속 증가 (노이즈 내성 낮음, 보조)
      B) Spearman: r > 0 AND p < 0.10 (주 판정)
    두 기준 중 B 기준으로 PASS/FAIL 결정.

    변경 이유:
      이전 nested bins(>0.25, >0.30, ...)은 모두 중복 포함(not independent).
      실제 noisy market에서 완벽 단조 증가는 잘 나오지 않으므로
      전체 추세를 보는 Spearman이 더 robust.
    """
    col = _col(horizon)

    means, ns, mids = [], [], []
    for label, lo, hi in ME_BINS:
        subset = sig_df[
            (sig_df["signal_score"] > lo) & (sig_df["signal_score"] <= hi)
        ][col].dropna()
        means.append(float(subset.mean()) if len(subset) > 0 else np.nan)
        ns.append(len(subset))

    # 유효 bin 추출 (N ≥ MIN_SAMPLE)
    valid_items = [
        (ME_MIDS[i], means[i])
        for i in range(len(ME_BINS))
        if ns[i] >= MIN_SAMPLE and not np.isnan(means[i])
    ]
    valid_mids  = [v[0] for v in valid_items]
    valid_means = [v[1] for v in valid_items]

    # A) strict monotonicity
    strict_mono = all(
        valid_means[i] <= valid_means[i + 1]
        for i in range(len(valid_means) - 1)
    ) if len(valid_means) >= 2 else False

    # B) Spearman rank correlation
    spearman_r, spearman_p = (np.nan, np.nan)
    if len(valid_mids) >= 3:
        spearman_r, spearman_p = stats.spearmanr(valid_mids, valid_means)
    spearman_r = float(spearman_r) if not np.isnan(spearman_r) else 0.0
    spearman_p = float(spearman_p) if not np.isnan(spearman_p) else 1.0

    # 주 판정: Spearman r>0 AND p<0.10
    is_monotonic = (spearman_r > 0) and (spearman_p < 0.10)

    # 시각화 (2패널: ME bins 막대 + Spearman scatter)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0d1117")

    bin_labels_str = [b[0] for b in ME_BINS]

    # 왼쪽: ME bins 막대
    ax1 = axes[0]
    ax1.set_facecolor("#161b22")
    colors = ["#58a6ff" if n >= MIN_SAMPLE else "#8b949e" for n in ns]
    bar_vals = [m if not np.isnan(m) else 0 for m in means]
    bars = ax1.bar(bin_labels_str, bar_vals, color=colors, alpha=0.85)
    for bar, n, m in zip(bars, ns, means):
        if np.isnan(m):
            continue
        note = "" if n >= MIN_SAMPLE else "\n(표본부족)"
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 m + 0.0001,
                 f"N={n:,}{note}", ha="center", va="bottom",
                 color="#e6edf3", fontsize=8)
    ax1.axhline(0, color="#f85149", linewidth=0.8, linestyle="--")
    mono_flag = "strict ✓" if strict_mono else "strict ✗"
    ax1.set_title(f"ME Bins ({_h_label(horizon)})  [{mono_flag}]",
                  color="#e6edf3", fontsize=10)
    ax1.set_xlabel("Signal Score Bin (ME)", color="#8b949e")
    ax1.set_ylabel("Avg Forward Return", color="#8b949e")
    ax1.tick_params(colors="#8b949e")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#30363d")

    # 오른쪽: Spearman scatter
    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    if len(valid_mids) >= 2:
        ax2.scatter(valid_mids, valid_means, color="#3fb950", s=80, zorder=5)
        # 추세선
        fit = np.polyfit(valid_mids, valid_means, 1)
        x_line = np.linspace(min(valid_mids), max(valid_mids), 100)
        ax2.plot(x_line, np.polyval(fit, x_line),
                 color="#58a6ff", linewidth=1.5, linestyle="--", alpha=0.8)
        for x, y, n in zip(valid_mids, valid_means, [ns[i] for i in range(len(ME_BINS)) if ns[i] >= MIN_SAMPLE]):
            ax2.annotate(f"N={n:,}", (x, y), textcoords="offset points",
                         xytext=(5, 5), color="#8b949e", fontsize=7)
    ax2.axhline(0, color="#f85149", linewidth=0.8, linestyle="--")

    spearman_flag = "✓" if is_monotonic else "✗"
    r_str = f"{spearman_r:.3f}"
    p_str = f"{spearman_p:.3f}"
    ax2.set_title(
        f"Spearman r={r_str}  p={p_str}  →  {'MONOTONIC' if is_monotonic else 'NO TREND'} {spearman_flag}",
        color="#e6edf3", fontsize=10,
    )
    ax2.set_xlabel("Signal Score (bin midpoint)", color="#8b949e")
    ax2.set_ylabel("Avg Forward Return", color="#8b949e")
    ax2.tick_params(colors="#8b949e")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#30363d")

    verdict_str = "PASS" if is_monotonic else "FAIL"
    fig.suptitle(f"Signal Monotonicity {_h_label(horizon)} — {verdict_str}",
                 color="#e6edf3", fontsize=12)
    plt.tight_layout()
    out_path = out_dir / f"tier3_monotonicity_{horizon}{_UNIT}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [저장] {out_path.name}")

    return {
        "horizon":         horizon,
        "me_bin_labels":   [b[0] for b in ME_BINS],
        "me_avg_returns":  [round(m, 6) if not np.isnan(m) else None for m in means],
        "me_sample_sizes": ns,
        "strict_monotonic": strict_mono,
        "spearman_r":      round(spearman_r, 4),
        "spearman_p":      round(spearman_p, 4),
        "is_monotonic":    is_monotonic,   # 주 판정 (Spearman 기준)
        "valid_bins_used": len(valid_items),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. No-trade Baseline 비교
# ══════════════════════════════════════════════════════════════════════════════

def no_trade_baseline_comparison(
    sig_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    horizon: int,
    out_dir: Path,
) -> dict:
    """
    상위 10%, 5%, 1% signal만 거래했을 때의 PF vs 전체 거래 PF 비교.
    """
    col = _col(horizon)

    def pf_from_rets(rets: pd.Series) -> float:
        wins   = rets[rets > 0].sum()
        losses = rets[rets < 0].abs().sum()
        return float(wins / losses) if losses > 0 else float("inf")

    base_thresh = 0.25
    base_df     = sig_df[sig_df["signal_score"] > base_thresh][col].dropna()
    base_pf     = pf_from_rets(base_df)
    base_n      = len(base_df)

    thresholds = [
        ("전체 (>0.25)",   np.percentile(sig_df["signal_score"], 0)  if len(sig_df) > 0 else 0.25),
        ("상위 50% (>p50)", np.percentile(sig_df["signal_score"], 50)),
        ("상위 25% (>p75)", np.percentile(sig_df["signal_score"], 75)),
        ("상위 10% (>p90)", np.percentile(sig_df["signal_score"], 90)),
        ("상위 5% (>p95)",  np.percentile(sig_df["signal_score"], 95)),
        ("상위 1% (>p99)",  np.percentile(sig_df["signal_score"], 99)),
    ]
    # percentile 대신 signal_score threshold 0.25 기준 내 상위 분위
    scored = sig_df[sig_df["signal_score"] > base_thresh]["signal_score"]

    rows = []
    for label, val_thresh in thresholds:
        subset = sig_df[sig_df["signal_score"] > val_thresh][col].dropna()
        n      = len(subset)
        pf_val = pf_from_rets(subset)
        reliable = n >= 30
        rows.append({
            "label":    label,
            "threshold": round(val_thresh, 4),
            "N":        n,
            "PF":       round(pf_val, 3) if pf_val != float("inf") else None,
            "note":     "" if reliable else "N<30 — 신뢰 불가",
        })

    # 시각화
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    labels  = [r["label"] for r in rows]
    pf_vals = [r["PF"] or 0 for r in rows]
    ns      = [r["N"] for r in rows]
    colors  = ["#3fb950" if n >= 30 else "#8b949e" for n in ns]

    bars = ax.bar(range(len(labels)), pf_vals, color=colors, alpha=0.85)
    ax.axhline(1.0, color="#f85149", linewidth=1, linestyle="--", label="PF=1.0")
    for i, (bar, n, pf_v) in enumerate(zip(bars, ns, pf_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                pf_v + 0.01,
                f"N={n:,}", ha="center", va="bottom",
                color="#e6edf3", fontsize=7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8, color="#8b949e")
    ax.set_title(f"Profit Factor by Signal Selectivity ({_h_label(horizon)})",
                 color="#e6edf3", fontsize=11)
    ax.set_ylabel("Profit Factor", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    plt.tight_layout()
    out_path = out_dir / f"tier3_selectivity_pf_{horizon}{_UNIT}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [저장] {out_path.name}")

    return {"horizon": horizon, "rows": rows}


# ══════════════════════════════════════════════════════════════════════════════
# 4. 통계적 유의성 검정
# ══════════════════════════════════════════════════════════════════════════════

def permutation_test(
    sig_df: pd.DataFrame,
    horizon: int,
    n_permutations: int = 10_000,
) -> dict:
    """
    Permutation test (primary): "랜덤 신호였어도 이 결과가 나왔는가?"

    H0: signal_score와 ret_{h}m 사이에 관계 없음
        (signal을 랜덤 타이밍에 배치해도 같은 mean diff가 나온다)
    H1: signal > 0.25 구간의 평균 return > 나머지 구간 (단측)

    방법:
      ret_{h}m 값을 고정하고, signal_score를 셔플하여
      "signal > 0.25" 구간이 바뀔 때마다 mean diff 계산.
      → 실제 관측값이 null 분포 상위 몇 %인지로 p값 산출.

    block_bootstrap와의 차이:
      block_bootstrap: 두 분포를 각각 resampling → 두 분포가 다른지 검정
      permutation:     신호 타이밍을 무작위화 → 신호 자체에 정보력 있는지 검정
                       (진짜 null hypothesis)
    """
    col = _col(horizon)
    valid_df = sig_df.dropna(subset=[col]).copy()

    if len(valid_df) < MIN_SAMPLE * 2:
        return {
            "horizon":        horizon,
            "p_value_perm":   None,
            "obs_diff":       None,
            "n_signal":       None,
            "n_non_signal":   None,
            "note":           "표본 부족 — 검정 불가",
        }

    ret_vals = valid_df[col].values

    # v2: is_signal 컬럼 우선 사용, 없으면 signal_score > 0.25
    if "is_signal" in valid_df.columns:
        sig_mask = valid_df["is_signal"].values.astype(bool)
    else:
        sig_mask = valid_df["signal_score"].values > 0.25

    n_sig  = int(sig_mask.sum())
    n_non  = int((~sig_mask).sum())

    if n_sig < MIN_SAMPLE or n_non < MIN_SAMPLE:
        return {
            "horizon":       horizon,
            "p_value_perm":  None,
            "obs_diff":      None,
            "n_signal":      n_sig,
            "n_non_signal":  n_non,
            "note":          f"signal({n_sig}) 또는 non-signal({n_non}) 표본 부족",
        }

    obs_diff = float(ret_vals[sig_mask].mean() - ret_vals[~sig_mask].mean())

    # Permutation: ret 값 고정, signal mask를 랜덤 셔플
    perm_diffs = np.empty(n_permutations)
    for i in range(n_permutations):
        perm_mask   = np.zeros(len(ret_vals), dtype=bool)
        chosen_idx  = rng.choice(len(ret_vals), size=n_sig, replace=False)
        perm_mask[chosen_idx] = True
        perm_diffs[i] = ret_vals[perm_mask].mean() - ret_vals[~perm_mask].mean()

    p_perm = float(np.mean(perm_diffs >= obs_diff))

    return {
        "horizon":       horizon,
        "n_signal":      n_sig,
        "n_non_signal":  n_non,
        "obs_diff":      round(obs_diff, 6),
        "p_value_perm":  round(p_perm, 4),
        "significant":   p_perm < 0.05,
        "n_permutations": n_permutations,
    }


def block_bootstrap_test(
    sig_df: pd.DataFrame,
    horizon: int,
    n_bootstrap: int = 10_000,
    block_size: int = 10,
) -> dict:
    """
    Block bootstrap (intraday autocorrelation 보정).
    signal 구간 vs baseline 구간 forward return 차이를 검정.

    H0: signal 구간의 평균 return = baseline 구간의 평균 return
    H1: signal 구간 > baseline 구간 (단측)
    """
    col          = _col(horizon)
    base_col     = _base_col(horizon)
    sig_vals     = sig_df[sig_df["signal_score"] > 0.25][col].dropna().values
    baseline_vals = sig_df[base_col].dropna().values

    if len(sig_vals) < MIN_SAMPLE or len(baseline_vals) < MIN_SAMPLE:
        return {
            "horizon":       horizon,
            "p_value_block": None,
            "p_value_ttest": None,
            "cohens_d":      None,
            "obs_diff":      None,
            "note":          "표본 부족 — 검정 불가",
        }

    obs_diff = float(sig_vals.mean() - baseline_vals.mean())

    # Block bootstrap
    def bootstrap_mean(arr: np.ndarray) -> float:
        n      = len(arr)
        starts = rng.integers(0, max(1, n - block_size + 1),
                              size=n // block_size + 1)
        blocks = [arr[s: s + block_size] for s in starts]
        sample = np.concatenate(blocks)[:n]
        return float(sample.mean())

    null_diffs = np.array([
        bootstrap_mean(sig_vals) - bootstrap_mean(baseline_vals)
        for _ in range(n_bootstrap)
    ])
    # 단측 검정: p = P(null_diff >= obs_diff)
    p_block = float(np.mean(null_diffs >= obs_diff))

    # t-test (보조)
    t_stat, p_ttest = stats.ttest_ind(sig_vals, baseline_vals,
                                       equal_var=False, alternative="greater")
    p_ttest = float(p_ttest)

    # Cohen's d
    pooled_std = float(np.sqrt(
        (sig_vals.std() ** 2 + baseline_vals.std() ** 2) / 2
    ))
    cohens_d = float(obs_diff / pooled_std) if pooled_std > 0 else 0.0

    return {
        "horizon":       horizon,
        "n_signal":      int(len(sig_vals)),
        "n_baseline":    int(len(baseline_vals)),
        "obs_diff":      round(obs_diff, 6),
        "p_value_block": round(p_block, 4),
        "p_value_ttest": round(p_ttest, 4),
        "cohens_d":      round(cohens_d, 4),
        "significant":   p_block < 0.05,
        "block_size":    block_size,
        "n_bootstrap":   n_bootstrap,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. Signal-Return Functional Form (GPT 권장: inverted-U / exhaustion 확인)
# ══════════════════════════════════════════════════════════════════════════════

def plot_signal_return_curve(
    sig_df: pd.DataFrame,
    out_dir: Path,
    n_quantiles: int = 20,
) -> dict:
    """
    signal_score 전체를 n_quantiles 등분위로 나눠
    각 분위 평균 return을 horizon별로 플롯.

    목적:
      - monotonic(trend-following) 관계인가?
      - inverted-U(moderate = best)?
      - exhaustion(extreme = reversal)?
    중립 non-signal rows도 포함해 전체 score 범위를 커버.
    """
    all_df = sig_df.copy()
    # 분위 라벨 부여 (signal_score 기준, 중복 bin edge는 first로)
    try:
        all_df["score_q"] = pd.qcut(
            all_df["signal_score"], q=n_quantiles, labels=False, duplicates="drop"
        )
    except ValueError:
        # 데이터 부족시 n_quantiles 낮춤
        n_quantiles = 10
        all_df["score_q"] = pd.qcut(
            all_df["signal_score"], q=n_quantiles, labels=False, duplicates="drop"
        )

    q_mids = (
        all_df.groupby("score_q")["signal_score"]
        .mean()
        .to_dict()
    )

    n_h    = len(FWD_HORIZONS)
    n_cols = 2
    n_rows = (n_h + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    if n_rows == 1:
        axes = [axes]
    fig.patch.set_facecolor("#0d1117")

    curve_data = {}
    for ax_idx, h in enumerate(FWD_HORIZONS):
        ax  = axes[ax_idx // n_cols][ax_idx % n_cols]
        ax.set_facecolor("#161b22")
        col = _col(h)

        grp = all_df.dropna(subset=[col]).groupby("score_q")
        xs  = []
        ys  = []
        ns  = []
        for q, g in grp:
            xs.append(q_mids.get(q, q))
            ys.append(float(g[col].mean()))
            ns.append(len(g))

        xs, ys, ns = map(list, zip(*sorted(zip(xs, ys, ns)))) if xs else ([], [], [])

        # 선 + 점
        ax.plot(xs, ys, color="#58a6ff", linewidth=1.5, zorder=3)
        colors_pt = ["#3fb950" if y > 0 else "#f85149" for y in ys]
        ax.scatter(xs, ys, c=colors_pt, s=40, zorder=4)
        ax.axhline(0, color="#f85149", linewidth=0.8, linestyle="--")
        ax.axvline(0.25, color="#f0883e", linewidth=1, linestyle=":",
                   label="threshold=0.25")

        # 이동 평균 (스무딩)
        if len(ys) >= 5:
            from scipy.ndimage import uniform_filter1d
            ys_smooth = uniform_filter1d(ys, size=3)
            ax.plot(xs, ys_smooth, color="#e6edf3", linewidth=1,
                    linestyle="--", alpha=0.6, label="smoothed")

        ax.set_title(f"Signal Score → Avg Return {_h_label(h)}",
                     color="#e6edf3", fontsize=10)
        ax.set_xlabel("Signal Score (quantile midpoint)", color="#8b949e")
        ax.set_ylabel("Avg Forward Return", color="#8b949e")
        ax.tick_params(colors="#8b949e")
        ax.legend(fontsize=7, facecolor="#161b22", labelcolor="#e6edf3")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        curve_data[_col(h)] = {"xs": xs, "ys": ys, "ns": ns}

    for ax_idx in range(n_h, n_rows * n_cols):
        axes[ax_idx // n_cols][ax_idx % n_cols].set_visible(False)

    fig.suptitle(
        f"Signal-Return Functional Form ({n_quantiles}-quantile, "
        f"orange: threshold=0.25)",
        color="#e6edf3", fontsize=12,
    )
    plt.tight_layout()
    out_path = out_dir / "tier3_signal_return_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [저장] {out_path.name}")

    return curve_data


# ══════════════════════════════════════════════════════════════════════════════
# 6. 진입 구간 한정 Permutation Test (Gemini 권장)
# ══════════════════════════════════════════════════════════════════════════════

def zone_permutation_test(
    sig_df: pd.DataFrame,
    horizon: int,
    zone_lo: float = 0.20,
    zone_hi: float = 0.35,
    n_permutations: int = 10_000,
) -> dict:
    """
    진입 구간(zone_lo < signal ≤ zone_hi) vs non-signal(signal ≤ 0.25) 한정 검정.

    목적:
      전체 is_signal=True를 쓰면 과매수 극단값이 섞여 신호 왜곡 가능.
      실제 진입 타점 구간만 떼어내 permutation test 재실행.

    H0: zone 구간 return = non-signal return (랜덤과 동일)
    H1: zone 구간 > non-signal (단측)
    """
    col = _col(horizon)

    zone_df    = sig_df[
        (sig_df["signal_score"] > zone_lo) &
        (sig_df["signal_score"] <= zone_hi)
    ].dropna(subset=[col])
    nonsig_df  = sig_df[~sig_df["is_signal"]].dropna(subset=[col])

    n_zone   = len(zone_df)
    n_nonsig = len(nonsig_df)

    if n_zone < MIN_SAMPLE or n_nonsig < MIN_SAMPLE:
        return {
            "horizon":      horizon,
            "zone":         f"{zone_lo}~{zone_hi}",
            "p_value_perm": None,
            "obs_diff":     None,
            "n_zone":       n_zone,
            "n_nonsig":     n_nonsig,
            "note":         f"표본 부족 (zone={n_zone}, nonsig={n_nonsig})",
        }

    # zone + nonsig 합쳐서 permutation
    combined  = np.concatenate([zone_df[col].values, nonsig_df[col].values])
    zone_mask = np.zeros(len(combined), dtype=bool)
    zone_mask[:n_zone] = True

    obs_diff  = float(combined[zone_mask].mean() - combined[~zone_mask].mean())

    perm_diffs = np.empty(n_permutations)
    for i in range(n_permutations):
        perm_m   = np.zeros(len(combined), dtype=bool)
        chosen   = rng.choice(len(combined), size=n_zone, replace=False)
        perm_m[chosen] = True
        perm_diffs[i]  = combined[perm_m].mean() - combined[~perm_m].mean()

    p_perm = float(np.mean(perm_diffs >= obs_diff))

    return {
        "horizon":      horizon,
        "zone":         f"{zone_lo}~{zone_hi}",
        "n_zone":       n_zone,
        "n_nonsig":     n_nonsig,
        "obs_diff":     round(obs_diff, 6),
        "zone_mean":    round(float(zone_df[col].mean()), 6),
        "nonsig_mean":  round(float(nonsig_df[col].mean()), 6),
        "p_value_perm": round(p_perm, 4),
        "significant":  p_perm < 0.05,
        "n_permutations": n_permutations,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 메인 Tier 3 실행
# ══════════════════════════════════════════════════════════════════════════════

def run_tier3(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("TIER 3 — Signal Validity Test")
    print("=" * 60)

    # 신호 로그 로드
    sig_df = load_signal_log(args.signal_log)
    print(f"  신호 로그: {len(sig_df):,} rows  "
          f"({sig_df['signal_time'].min().date()} ~ "
          f"{sig_df['signal_time'].max().date()})")

    # ── ① Forward Return 분포 ────────────────────────────────────────────────
    print("\n  [①] Forward Return 분포 분석...")
    fwd_results = analyze_forward_return_distribution(sig_df, out_dir)

    # 표본 수 콘솔 출력
    ph = _primary_h()
    print(f"\n  {'bin':<20} {'threshold':>10} {'N':>8}")
    print("  " + "-" * 42)
    for item in fwd_results.get(_col(ph), []):
        note = "  ← 표본 부족" if item["note"] else ""
        print(f"  {item['label']:<20} {item['threshold']:>10.2f} "
              f"{item['N']:>8,}{note}")

    # ── ② 단조성 검증 (primary horizon 기준 메인, 나머지 보조) ────────────────
    print(f"\n  [②] Signal-Return 단조성 검증 ({_h_label(_primary_h())})...")
    mono_results = {}
    for h in FWD_HORIZONS:
        res = test_signal_monotonicity(sig_df, h, out_dir)
        mono_results[_col(h)] = res
        verdict = "PASS" if res["is_monotonic"] else "FAIL"
        print(f"       {_h_label(h)}: {verdict}  "
              f"(유효 bin {res['valid_bins_used']}개 기준)")

    mono_pass = mono_results[_col(_primary_h())]["is_monotonic"]

    # ── ③ No-trade Baseline 비교 ─────────────────────────────────────────────
    print("\n  [③] No-trade Baseline 비교...")
    selectivity_results = {}
    sel_horizons = FWD_HORIZONS[:2]   # 처음 두 horizon만 (분: 1m+3m, 초: 1s+5s)
    for h in sel_horizons:
        res = no_trade_baseline_comparison(sig_df, None, h, out_dir)
        selectivity_results[_col(h)] = res
        for r in res["rows"]:
            note = f"  [{r['note']}]" if r["note"] else ""
            pf_str = f"{r['PF']:.3f}" if r["PF"] is not None else "inf"
            print(f"       {r['label']:<22} N={r['N']:>5,}  PF={pf_str}{note}")

    # ── ④ 통계적 유의성 검정 ─────────────────────────────────────────────────
    print(f"\n  [④-A] Permutation Test (n={args.n_bootstrap:,}, primary)...")
    perm_results = {}
    main_p_perm  = None
    for h in FWD_HORIZONS:
        res = permutation_test(sig_df, h, n_permutations=args.n_bootstrap)
        perm_results[_col(h)] = res
        if res["p_value_perm"] is not None:
            sig_flag = "★" if res["significant"] else " "
            print(f"       {_h_label(h)}: p_perm={res['p_value_perm']:.4f} {sig_flag}"
                  f"  obs_diff={res['obs_diff']:.6f}"
                  f"  N_sig={res['n_signal']:,}")
        else:
            print(f"       {_h_label(h)}: {res['note']}")
        if h == _primary_h():
            main_p_perm = res["p_value_perm"]

    print(f"\n  [④-B] Block Bootstrap (n={args.n_bootstrap:,}, "
          f"block_size={args.block_size}{'초' if _UNIT=='s' else '분'}, 보조)...")
    stat_results = {}
    main_p_block = None
    for h in FWD_HORIZONS:
        res = block_bootstrap_test(
            sig_df, h,
            n_bootstrap=args.n_bootstrap,
            block_size=args.block_size,
        )
        stat_results[_col(h)] = res
        if res["p_value_block"] is not None:
            sig_flag = "★" if res["significant"] else " "
            print(f"       {_h_label(h)}: p_block={res['p_value_block']:.4f} {sig_flag}"
                  f"  p_ttest={res['p_value_ttest']:.4f}"
                  f"  Cohen's d={res['cohens_d']:.4f}")
        else:
            print(f"       {_h_label(h)}: {res['note']}")
        if h == _primary_h():
            main_p_block = res["p_value_block"]

    # ── ⑤ Signal-Return Functional Form ────────────────────────────────────────
    print("\n  [⑤] Signal-Return 함수형태 (quantile curve)...")
    curve_results = plot_signal_return_curve(sig_df, out_dir)
    # 간략 콘솔: primary horizon의 마지막 4분위 vs 첫 4분위
    ph_curve = curve_results.get(_col(_primary_h()), {})
    xs_c = ph_curve.get("xs", [])
    ys_c = ph_curve.get("ys", [])
    if len(xs_c) >= 4:
        q_low  = ys_c[1]   # 하위 10% 근방
        q_high = ys_c[-2]  # 상위 10% 근방
        print(f"       {_h_label(_primary_h())} 하위10분위 avg={q_low:.6f}"
              f"  상위10분위 avg={q_high:.6f}"
              f"  {'↓ exhaustion 패턴' if q_high < q_low else '↑ continuation 패턴'}")

    # ── ⑥ 진입 구간 한정 Permutation Test ──────────────────────────────────────
    print(f"\n  [⑥] 진입 구간 한정 Permutation (zone=0.20~0.35, "
          f"n={args.n_bootstrap:,})...")
    zone_results = {}
    main_p_zone  = None
    for h in FWD_HORIZONS:
        res = zone_permutation_test(
            sig_df, h,
            zone_lo=0.20, zone_hi=0.35,
            n_permutations=args.n_bootstrap,
        )
        zone_results[_col(h)] = res
        if res["p_value_perm"] is not None:
            sig_flag = "★" if res["significant"] else " "
            print(f"       {_h_label(h)}: p_perm={res['p_value_perm']:.4f} {sig_flag}"
                  f"  obs_diff={res['obs_diff']:.6f}"
                  f"  zone_mean={res['zone_mean']:.6f}"
                  f"  N_zone={res['n_zone']:,}")
        else:
            print(f"       {_h_label(h)}: {res.get('note', '')}")
        if h == _primary_h():
            main_p_zone = res["p_value_perm"]

    zone_pass = main_p_zone is not None and main_p_zone < 0.05

    # 판정: Permutation(primary) + Spearman 단조성
    perm_pass  = main_p_perm  is not None and main_p_perm  < 0.05
    stat_pass  = main_p_block is not None and main_p_block < 0.05  # 보조
    tier3_pass = mono_pass and perm_pass
    verdict    = "PASS" if tier3_pass else "FAIL"

    print("\n  ────────────────────────────────────────")
    spearman_r = mono_results[_col(_primary_h())].get("spearman_r", 0)
    spearman_p = mono_results[_col(_primary_h())].get("spearman_p", 1)
    print(f"  Spearman 단조성 ({_h_label(_primary_h())}): {'PASS' if mono_pass else 'FAIL'}"
          f"  (r={spearman_r:.3f}, p={spearman_p:.3f})")
    print(f"  Permutation test (전체):  {'PASS (p<0.05)' if perm_pass else 'FAIL'}"
          f"  (p={main_p_perm:.4f})" if main_p_perm is not None
          else f"  Permutation test (전체):  {'PASS' if perm_pass else 'FAIL'}")
    print(f"  Permutation test (진입구간 0.20~0.35): "
          f"{'PASS (p<0.05)' if zone_pass else 'FAIL'}"
          f"  (p={main_p_zone:.4f})" if main_p_zone is not None
          else f"  Permutation test (진입구간): {'PASS' if zone_pass else 'FAIL'}")
    print(f"  Block Bootstrap (보조): {'PASS' if stat_pass else 'FAIL'}"
          f"  (p={main_p_block:.4f})" if main_p_block is not None
          else f"  Block Bootstrap (보조): {'PASS' if stat_pass else 'FAIL'}")
    print(f"\n  TIER 3 최종:           {verdict}")
    print("  ────────────────────────────────────────")

    ph_key = _col(_primary_h())
    ph_lbl = f"{_primary_h()}{_UNIT}"
    report = {
        "tier": 3,
        "result":                            verdict,
        f"monotonic_{ph_lbl}":               mono_pass,
        f"spearman_r_{ph_lbl}":              mono_results[ph_key].get("spearman_r"),
        f"spearman_p_{ph_lbl}":              mono_results[ph_key].get("spearman_p"),
        f"perm_p_value_{ph_lbl}":            main_p_perm,
        f"block_p_value_{ph_lbl}":           main_p_block,
        f"perm_significant_{ph_lbl}":          perm_pass,
        f"block_significant_{ph_lbl}":        stat_pass,
        f"zone_perm_significant_{ph_lbl}":    zone_pass,
        f"zone_perm_p_{ph_lbl}":              main_p_zone,
        "fwd_return_dist":      fwd_results,
        "monotonicity":         mono_results,
        "selectivity":          selectivity_results,
        "permutation_tests":    perm_results,
        "zone_permutation_tests": zone_results,
        "signal_return_curve":  curve_results,
        "statistical_tests":    stat_results,
        "params": {
            "n_bootstrap":  args.n_bootstrap,
            "block_size":   args.block_size,
            "min_sample":   MIN_SAMPLE,
            "signal_bins":  [{"label": l, "threshold": t} for l, t in SIGNAL_BINS],
            "me_bins":      [{"label": b[0], "lo": b[1], "hi": b[2]} for b in ME_BINS],
            "pass_criteria": {
                "primary":   "permutation p<0.05 AND Spearman r>0, p<0.10",
                "secondary": "block_bootstrap p<0.05 (참고용)",
            },
        },
    }

    out_json = out_dir / "tier3_signal_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  [저장] {out_json.name}")

    return report


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tier 3 — Signal Validity Test")
    p.add_argument("--signal-log",    required=True,  help="signal_log.csv 경로")
    p.add_argument("--data-path",     default=None,   help="1분봉 OHLCV CSV (선택)")
    p.add_argument("--symbol",        default="SPY",  help="종목 필터")
    p.add_argument("--out-dir",       default="./validation_results/tmp")
    p.add_argument("--n-bootstrap",   type=int, default=10_000)
    p.add_argument("--block-size",    type=int, default=10,
                   help="Block bootstrap 블록 크기 (분)")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    result = run_tier3(args)
    print(f"\n최종 판정: {result['result']}")
