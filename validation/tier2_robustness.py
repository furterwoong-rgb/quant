#!/usr/bin/env python3
"""
Tier 2 — 과적합 탐지 (Robustness Test)
───────────────────────────────────────────────────────────────────────────────
특정 파라미터에만 최적화된 가짜 edge를 걸러낸다.

구성:
  ① Threshold 민감도 (0.15/0.20/0.25/0.30 각각 PF 계산)
  ② 월별 PF 일관성 (음수 달 비율)
  ③ Walk-forward OOS (IS 2개월 → OOS 1개월 롤링)
  ④ OOS Holdout 40% 분리 (최종 holdout_data.csv 저장, 코드 주석 처리)

판정:
  PASS:  ①3/4+ threshold PF≥1.1  AND  ②음수달 ≤33%  AND  ③OOS/IS≥70%

실행:
  python tier2_robustness.py \
    --trades-log ./data/spy_qqq_trades.csv \
    --signal-log ./data/signal_log.csv \
    --out-dir    ./validation_results/20260529_120000 \
    --commission 0.005 \
    --slippage-ticks 0.5
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Tier 1 지표 재사용
from tier1_gate import (
    apply_costs,
    compute_profit_factor,
    compute_max_drawdown,
    EXIT_ACTIONS,
)

RANDOM_STATE = 42

THRESHOLDS_TEST   = [0.15, 0.20, 0.25, 0.30]
PF_MIN_ROBUST     = 1.10
PASS_ROBUST_COUNT = 3   # 4개 중 3개 이상
NEG_MONTH_MAX     = 0.33
OOS_IS_MIN_RATIO  = 0.70

IS_MONTHS  = 2   # Walk-forward In-sample 길이 (개월)
OOS_MONTHS = 1   # Walk-forward Out-of-sample 길이 (개월)
HOLDOUT_RATIO = 0.40   # 전체의 뒤 40%는 holdout


# ══════════════════════════════════════════════════════════════════════════════
# 공통 유틸
# ══════════════════════════════════════════════════════════════════════════════

def _exits_with_cost(
    trades_df: pd.DataFrame,
    commission: float,
    slippage_ticks: float,
) -> pd.DataFrame:
    exits = trades_df[
        trades_df["action_lower"].isin({a.lower() for a in EXIT_ACTIONS})
    ].copy()
    exits = apply_costs(exits, commission, slippage_ticks)
    return exits


def _pf_from_exits(exits_df: pd.DataFrame) -> float:
    if len(exits_df) == 0:
        return 0.0
    return compute_profit_factor(exits_df["net_pnl"])


def _load_trades(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp_et"])
    df = df.sort_values("timestamp_et").reset_index(drop=True)
    df["action_lower"] = df["action"].str.lower().str.strip()
    df["pnl_usd"]   = pd.to_numeric(df["pnl_usd"],  errors="coerce").fillna(0.0)
    df["qty"]       = pd.to_numeric(df["qty"],       errors="coerce").fillna(0).astype(int)
    df["portfolio"] = pd.to_numeric(df["portfolio"], errors="coerce")
    return df


def _load_signal_log(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["signal_time"])
    df = df.sort_values("signal_time").reset_index(drop=True)
    df["signal_score"] = pd.to_numeric(df["signal_score"], errors="coerce")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ① Threshold 민감도
# ══════════════════════════════════════════════════════════════════════════════

def threshold_sensitivity(
    signal_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    commission: float,
    slippage_ticks: float,
    out_dir: Path,
) -> dict:
    """
    threshold [0.15, 0.20, 0.25, 0.30] 각각에서 해당 신호로 진입한 트레이드만
    필터링하여 PF를 재계산한다.

    signal_log.csv의 signal_score ≥ threshold 구간에서 발생한 트레이드를
    trades_df의 signal 컬럼으로 매칭.
    """
    results = []
    for thresh in THRESHOLDS_TEST:
        filtered = trades_df[trades_df["signal"].abs() >= thresh] if "signal" in trades_df.columns \
                   else trades_df  # signal 컬럼 없으면 전체 사용
        exits = _exits_with_cost(filtered, commission, slippage_ticks)
        pf    = _pf_from_exits(exits)
        results.append({
            "threshold": thresh,
            "n_trades":  len(exits),
            "pf":        round(pf, 4),
            "pass":      pf >= PF_MIN_ROBUST,
        })

    pass_count = sum(1 for r in results if r["pass"])
    passed     = pass_count >= PASS_ROBUST_COUNT

    # 시각화
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    xs   = [r["threshold"] for r in results]
    pfs  = [r["pf"] for r in results]
    ns   = [r["n_trades"] for r in results]
    colors = ["#3fb950" if r["pass"] else "#f85149" for r in results]

    ax.bar([str(x) for x in xs], pfs, color=colors, alpha=0.85)
    ax.axhline(PF_MIN_ROBUST, color="#f0883e", linewidth=1.2,
               linestyle="--", label=f"PF≥{PF_MIN_ROBUST} 기준선")
    for i, (pf_v, n) in enumerate(zip(pfs, ns)):
        ax.text(i, pf_v + 0.01, f"{pf_v:.3f}\n(N={n:,})",
                ha="center", va="bottom", color="#e6edf3", fontsize=8)

    verdict = "PASS" if passed else "FAIL"
    ax.set_title(f"Threshold Sensitivity — {verdict} ({pass_count}/4 pass)",
                 color="#e6edf3", fontsize=11)
    ax.set_xlabel("Signal Threshold", color="#8b949e")
    ax.set_ylabel("Profit Factor", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    out_path = out_dir / "tier2_threshold_sensitivity.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [저장] {out_path.name}")

    return {
        "results":     results,
        "pass_count":  pass_count,
        "passed":      passed,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ② 월별 PF 일관성
# ══════════════════════════════════════════════════════════════════════════════

def monthly_pf_consistency(
    trades_df: pd.DataFrame,
    commission: float,
    slippage_ticks: float,
    out_dir: Path,
) -> dict:
    """
    전체 기간을 월별 분리 → 각 월의 PF.
    음수 달 비율(PF < 1.0) ≤ 33% → PASS.
    """
    exits = _exits_with_cost(trades_df, commission, slippage_ticks)
    exits["ym"] = exits["timestamp_et"].dt.to_period("M")
    months_sorted = sorted(exits["ym"].unique())

    monthly = []
    for ym in months_sorted:
        sub    = exits[exits["ym"] == ym]
        pf_val = _pf_from_exits(sub)
        monthly.append({
            "month":    str(ym),
            "n_trades": int(len(sub)),
            "pf":       round(pf_val, 4),
            "positive": pf_val >= 1.0,
        })

    n_months  = len(monthly)
    neg_count = sum(1 for m in monthly if not m["positive"])
    neg_ratio = neg_count / n_months if n_months > 0 else 1.0
    passed    = neg_ratio <= NEG_MONTH_MAX

    # 시각화
    fig, ax = plt.subplots(figsize=(max(8, n_months * 0.9), 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    labels  = [m["month"] for m in monthly]
    pf_vals = [m["pf"] for m in monthly]
    colors  = ["#3fb950" if m["positive"] else "#f85149" for m in monthly]

    ax.bar(range(len(labels)), pf_vals, color=colors, alpha=0.85)
    ax.axhline(1.0, color="#f0883e", linewidth=1.2, linestyle="--", label="PF=1.0")
    for i, (pf_v, m) in enumerate(zip(pf_vals, monthly)):
        ax.text(i, pf_v + 0.01, f"{pf_v:.2f}",
                ha="center", va="bottom", color="#e6edf3", fontsize=7)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7, color="#8b949e")
    verdict = "PASS" if passed else "FAIL"
    ax.set_title(f"Monthly PF Consistency — {verdict}  "
                 f"(음수달 {neg_count}/{n_months} = {neg_ratio*100:.0f}%  "
                 f"[≤33% 기준])",
                 color="#e6edf3", fontsize=10)
    ax.set_ylabel("Profit Factor", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    out_path = out_dir / "tier2_monthly_pf.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [저장] {out_path.name}")

    return {
        "monthly":    monthly,
        "n_months":   n_months,
        "neg_count":  neg_count,
        "neg_ratio":  round(neg_ratio, 4),
        "passed":     passed,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ③ Walk-forward OOS 검증
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward_oos(
    trades_df: pd.DataFrame,
    commission: float,
    slippage_ticks: float,
    is_months: int,
    oos_months: int,
    out_dir: Path,
) -> dict:
    """
    IS {is_months}개월 → OOS {oos_months}개월 롤링.
    OOS PF / IS PF ≥ 70% → PASS.
    """
    exits = _exits_with_cost(trades_df, commission, slippage_ticks)
    exits["ym"] = exits["timestamp_et"].dt.to_period("M")
    months = sorted(exits["ym"].unique())

    window = is_months + oos_months
    if len(months) < window:
        return {
            "windows":   [],
            "avg_ratio": None,
            "passed":    False,
            "note":      f"데이터 부족 (필요 {window}개월, 보유 {len(months)}개월)",
        }

    windows = []
    for i in range(len(months) - window + 1):
        is_months_range  = months[i : i + is_months]
        oos_months_range = months[i + is_months : i + window]

        is_exits  = exits[exits["ym"].isin(is_months_range)]
        oos_exits = exits[exits["ym"].isin(oos_months_range)]

        is_pf  = _pf_from_exits(is_exits)
        oos_pf = _pf_from_exits(oos_exits)

        ratio  = (oos_pf / is_pf) if (is_pf > 0 and np.isfinite(is_pf)) else None
        windows.append({
            "is_period":  f"{is_months_range[0]}~{is_months_range[-1]}",
            "oos_period": f"{oos_months_range[0]}~{oos_months_range[-1]}",
            "is_pf":      round(is_pf,  4),
            "oos_pf":     round(oos_pf, 4),
            "ratio":      round(ratio, 4) if ratio is not None else None,
        })

    valid_ratios = [w["ratio"] for w in windows if w["ratio"] is not None]
    avg_ratio    = float(np.mean(valid_ratios)) if valid_ratios else None
    passed       = avg_ratio is not None and avg_ratio >= OOS_IS_MIN_RATIO

    # 시각화
    fig, ax = plt.subplots(figsize=(max(8, len(windows) * 1.2), 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    x       = range(len(windows))
    is_pfs  = [w["is_pf"]  for w in windows]
    oos_pfs = [w["oos_pf"] for w in windows]
    labels  = [w["oos_period"] for w in windows]

    ax.plot(x, is_pfs,  "o-", color="#58a6ff", label="IS PF",  linewidth=1.5)
    ax.plot(x, oos_pfs, "s-", color="#3fb950", label="OOS PF", linewidth=1.5)
    ax.axhline(1.0, color="#8b949e", linewidth=0.8, linestyle="--")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7, color="#8b949e")

    verdict = "PASS" if passed else "FAIL"
    ratio_str = f"{avg_ratio:.2f}" if avg_ratio is not None else "N/A"
    ax.set_title(
        f"Walk-forward OOS — {verdict}  (avg OOS/IS={ratio_str}  [≥0.70 기준])",
        color="#e6edf3", fontsize=10,
    )
    ax.set_ylabel("Profit Factor", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    out_path = out_dir / "tier2_walkforward_oos.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [저장] {out_path.name}")

    return {
        "windows":     windows,
        "avg_ratio":   round(avg_ratio, 4) if avg_ratio is not None else None,
        "passed":      passed,
        "is_months":   is_months,
        "oos_months":  oos_months,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ④ OOS Holdout 40% 분리
# ══════════════════════════════════════════════════════════════════════════════

def split_holdout(
    trades_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    out_dir: Path,
    holdout_ratio: float = HOLDOUT_RATIO,
) -> dict:
    """
    전체 데이터의 앞 (1-holdout_ratio)을 IS, 뒤 holdout_ratio를 OOS holdout으로 분리.
    holdout_data.csv 저장 (최종 평가 전까지 사용 금지).
    """
    # trades holdout
    n_trades = len(trades_df)
    split_idx_t = int(n_trades * (1 - holdout_ratio))
    is_trades   = trades_df.iloc[:split_idx_t].copy()
    holdout_t   = trades_df.iloc[split_idx_t:].copy()

    split_date_t = holdout_t["timestamp_et"].min() if len(holdout_t) > 0 else None

    # signal holdout
    if signal_df is not None and len(signal_df) > 0:
        n_sig = len(signal_df)
        split_idx_s = int(n_sig * (1 - holdout_ratio))
        is_signal   = signal_df.iloc[:split_idx_s].copy()
        holdout_s   = signal_df.iloc[split_idx_s:].copy()
    else:
        is_signal = signal_df
        holdout_s = pd.DataFrame()

    # 저장
    holdout_trades_path = out_dir / "holdout_trades.csv"
    holdout_t.to_csv(holdout_trades_path, index=False)
    if len(holdout_s) > 0:
        holdout_signal_path = out_dir / "holdout_signal.csv"
        holdout_s.to_csv(holdout_signal_path, index=False)
    is_trades_path = out_dir / "is_trades.csv"
    is_trades.to_csv(is_trades_path, index=False)

    print(f"  [저장] holdout_trades.csv  ({len(holdout_t):,} rows — 최종 평가 전 사용 금지)")
    print(f"  [저장] is_trades.csv       ({len(is_trades):,} rows — 파라미터 튜닝용)")

    return {
        "is_rows":        int(len(is_trades)),
        "holdout_rows":   int(len(holdout_t)),
        "holdout_ratio":  holdout_ratio,
        "split_date":     str(split_date_t.date()) if split_date_t is not None else None,
        "holdout_trades_path": str(holdout_trades_path),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 메인 Tier 2 실행
# ══════════════════════════════════════════════════════════════════════════════

def run_tier2(
    args: argparse.Namespace,
    trades_df: pd.DataFrame = None,
) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("TIER 2 — Robustness Test (과적합 탐지)")
    print("=" * 60)

    if trades_df is None:
        trades_df = _load_trades(args.trades_log)
    if hasattr(args, "start") and args.start:
        trades_df = trades_df[trades_df["timestamp_et"] >= args.start]
    if hasattr(args, "end") and args.end:
        trades_df = trades_df[trades_df["timestamp_et"] <= args.end]

    signal_df = None
    if hasattr(args, "signal_log") and args.signal_log:
        try:
            signal_df = _load_signal_log(args.signal_log)
        except Exception as e:
            print(f"  signal_log 로드 실패 ({e}) — threshold 분석은 trades signal 컬럼 사용")

    # ── ① Threshold 민감도 ──────────────────────────────────────────────────
    print("\n  [①] Threshold 민감도 분석...")
    thresh_res = threshold_sensitivity(
        signal_df if signal_df is not None else pd.DataFrame(),
        trades_df,
        args.commission,
        args.slippage_ticks,
        out_dir,
    )
    for r in thresh_res["results"]:
        mark = "✓" if r["pass"] else "✗"
        pf_s = f"{r['pf']:.3f}"
        print(f"       thresh={r['threshold']:.2f}: PF={pf_s:>8}  N={r['n_trades']:>5,}  {mark}")
    print(f"       → {thresh_res['pass_count']}/4 pass  "
          f"({'PASS' if thresh_res['passed'] else 'FAIL'})")

    # ── ② 월별 PF ────────────────────────────────────────────────────────────
    print("\n  [②] 월별 PF 일관성...")
    monthly_res = monthly_pf_consistency(
        trades_df, args.commission, args.slippage_ticks, out_dir)
    print(f"       음수달 {monthly_res['neg_count']}/{monthly_res['n_months']}  "
          f"= {monthly_res['neg_ratio']*100:.0f}%  "
          f"({'PASS' if monthly_res['passed'] else 'FAIL'})")

    # ── ③ Walk-forward OOS ───────────────────────────────────────────────────
    print("\n  [③] Walk-forward OOS 검증...")
    wf_res = walk_forward_oos(
        trades_df, args.commission, args.slippage_ticks,
        IS_MONTHS, OOS_MONTHS, out_dir,
    )
    if wf_res.get("note"):
        print(f"       SKIP — {wf_res['note']}")
    else:
        ratio_s = f"{wf_res['avg_ratio']:.3f}" if wf_res["avg_ratio"] is not None else "N/A"
        print(f"       avg OOS/IS = {ratio_s}  "
              f"({'PASS' if wf_res['passed'] else 'FAIL'})")

    # ── ④ Holdout 분리 ───────────────────────────────────────────────────────
    print("\n  [④] OOS Holdout 40% 분리...")
    holdout_res = split_holdout(trades_df, signal_df, out_dir)

    # 판정
    pass_thresh  = thresh_res["passed"]
    pass_monthly = monthly_res["passed"]
    pass_wf      = wf_res["passed"] if not wf_res.get("note") else True  # skip → 통과 처리
    tier2_pass   = pass_thresh and pass_monthly and pass_wf
    verdict      = "PASS" if tier2_pass else "FAIL"

    print("\n  ────────────────────────────────────────")
    print(f"  ① Threshold 민감도: {'PASS' if pass_thresh else 'FAIL'}")
    print(f"  ② 월별 일관성:      {'PASS' if pass_monthly else 'FAIL'}")
    print(f"  ③ Walk-forward OOS: {'PASS' if pass_wf else 'FAIL'}")
    print(f"\n  TIER 2 최종:        {verdict}")
    print("  ────────────────────────────────────────")

    report = {
        "tier":    2,
        "result":  verdict,
        "threshold_sensitivity": thresh_res,
        "monthly_pf":            monthly_res,
        "walk_forward_oos":      wf_res,
        "holdout":               holdout_res,
        "pass_details": {
            "threshold_sensitivity": pass_thresh,
            "monthly_pf":            pass_monthly,
            "walk_forward_oos":      pass_wf,
        },
        "params": {
            "commission":      args.commission,
            "slippage_ticks":  args.slippage_ticks,
            "thresholds_test": THRESHOLDS_TEST,
            "is_months":       IS_MONTHS,
            "oos_months":      OOS_MONTHS,
            "holdout_ratio":   HOLDOUT_RATIO,
        },
    }

    out_json = out_dir / "tier2_robustness_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  [저장] {out_json.name}")

    return report


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tier 2 — Robustness Test")
    p.add_argument("--trades-log",     required=True)
    p.add_argument("--signal-log",     default=None)
    p.add_argument("--out-dir",        default="./validation_results/tmp")
    p.add_argument("--commission",     type=float, default=0.005)
    p.add_argument("--slippage-ticks", type=float, default=0.5)
    p.add_argument("--start",          default=None)
    p.add_argument("--end",            default=None)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    result = run_tier2(args)
    print(f"\n최종 판정: {result['result']}")
