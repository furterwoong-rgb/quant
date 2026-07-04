#!/usr/bin/env python3
"""
OBI+TFI Algorithm Edge Validation Pipeline — 메인 실행
───────────────────────────────────────────────────────────────────────────────
4-Tier 검증을 순차 실행하고 종합 리포트(validation_summary.json)를 생성한다.

실행 순서:
  Tier 3 (신호 정보력) → Tier 1 (최소 통과) → Tier 2 (과적합) → Tier 4 (Regime)

판정 기준:
  "전략 재설계 필요"              → Tier 1 FAIL
  "완전한 착각은 아님"             → Tier 1+2 PASS
  "신호에 정보력 있음 — paper trading 시작 허용"  → Tier 1+2+3 PASS
  "실전 투입 고려 가능"            → 전체 PASS

실행 예시:
  python run_validation.py \\
    --data-path    ./data/spy_qqq_1min.csv \\
    --signal-log   ./data/signal_log.csv \\
    --trades-log   ./data/spy_qqq_trades.csv \\
    --vix-path     ./data/vix.csv \\
    --start        2026-01-01 \\
    --end          2026-05-29 \\
    --commission   0.005 \\
    --slippage-ticks 0.5 \\
    --threshold    0.25

주의:
  validation_results/{timestamp}/holdout_trades.csv 는
  파라미터 튜닝 완료 전까지 절대 열지 말 것.
  (Holdout — 최종 실행 전까지 주석 유지)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# ── Tier 모듈 import ───────────────────────────────────────────────────────────
# run_validation.py 와 같은 디렉터리에서 실행 가정
import os
sys.path.insert(0, os.path.dirname(__file__))

from tier1_gate      import run_tier1, _build_parser as _t1_parser, load_trades
from tier2_robustness import run_tier2
from tier3_signal    import run_tier3
from tier4_regime    import run_tier4


# ── Overall verdict 결정 ──────────────────────────────────────────────────────

def determine_overall_verdict(
    t1_pass: bool,
    t2_pass: bool,
    t3_pass: bool,
    t4_pass: bool,
) -> str:
    if not t1_pass:
        return "전략 재설계 필요"
    if t1_pass and t2_pass and t3_pass and t4_pass:
        return "실전 투입 고려 가능"
    if t1_pass and t2_pass and t3_pass:
        return "신호에 정보력 있음 — paper trading 시작 허용"
    if t1_pass and t2_pass:
        return "완전한 착각은 아님"
    return "완전한 착각은 아님 (Tier 2 추가 개선 필요)"


# ── 종합 리포트 시각화 ─────────────────────────────────────────────────────────

def plot_summary(results: dict, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")
        ax.axis("off")

        tier_data = [
            ("Tier 1\nGate Test",      results.get("tier1", {}).get("result", "?")),
            ("Tier 2\nRobustness",     results.get("tier2", {}).get("result", "?")),
            ("Tier 3\nSignal Validity", results.get("tier3", {}).get("result", "?")),
            ("Tier 4\nRegime",         results.get("tier4", {}).get("result", "?")),
        ]
        colors_map = {"PASS": "#3fb950", "FAIL": "#f85149", "SKIP": "#8b949e", "?": "#8b949e"}

        for i, (label, verdict) in enumerate(tier_data):
            color = colors_map.get(verdict, "#8b949e")
            rect  = mpatches.FancyBboxPatch(
                (0.05 + i * 0.23, 0.35), 0.18, 0.30,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor="#30363d", alpha=0.85,
                transform=ax.transAxes,
            )
            ax.add_patch(rect)
            ax.text(0.14 + i * 0.23, 0.50, verdict,
                    ha="center", va="center", transform=ax.transAxes,
                    color="#0d1117", fontsize=14, fontweight="bold")
            ax.text(0.14 + i * 0.23, 0.30, label,
                    ha="center", va="top", transform=ax.transAxes,
                    color="#8b949e", fontsize=9)

        overall = results.get("overall_verdict", "")
        ax.text(0.5, 0.85, overall,
                ha="center", va="center", transform=ax.transAxes,
                color="#e6edf3", fontsize=13, fontweight="bold",
                bbox=dict(facecolor="#161b22", edgecolor="#30363d",
                          boxstyle="round,pad=0.5"))

        period = results.get("data_period", "")
        ax.text(0.5, 0.12, f"검증 기간: {period}  |  실행일: {results.get('run_date', '')}",
                ha="center", va="center", transform=ax.transAxes,
                color="#8b949e", fontsize=8)

        out_path = out_dir / "validation_summary.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"  [저장] {out_path.name}")
    except Exception as e:
        print(f"  summary chart 저장 실패: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 메인 파이프라인
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(args: argparse.Namespace) -> dict:
    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / run_ts
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(out_dir)

    print("\n" + "█" * 60)
    print("  OBI+TFI Algorithm Edge Validation Pipeline")
    print(f"  실행 시각: {run_ts}")
    print(f"  기간:      {args.start} ~ {args.end}")
    print(f"  결과 경로: {out_dir}")
    print("█" * 60)

    # ── trades 사전 로드 (각 Tier에서 재사용) ────────────────────────────────
    trades_df = None
    if args.trades_log:
        try:
            trades_df = load_trades(args.trades_log)
            if args.start:
                trades_df = trades_df[trades_df["timestamp_et"] >= args.start]
            if args.end:
                trades_df = trades_df[trades_df["timestamp_et"] <= args.end]
            print(f"\n  trades 로드: {len(trades_df):,} rows")
        except Exception as e:
            print(f"\n  trades 로드 실패: {e}")

    tier_results: dict = {}

    # ── Tier 3: 신호 정보력 ──────────────────────────────────────────────────
    t3_result = None
    if args.signal_log:
        try:
            t3_result = run_tier3(args)
            tier_results["tier3"] = t3_result
        except Exception as e:
            print(f"\n  Tier 3 실패: {e}")
            tier_results["tier3"] = {"result": "ERROR", "error": str(e)}
    else:
        print("\n  [SKIP] Tier 3 — --signal-log 미제공")
        tier_results["tier3"] = {"result": "SKIP", "note": "signal_log 미제공"}

    # ── Tier 1: Gate Test ────────────────────────────────────────────────────
    t1_result = None
    if trades_df is not None:
        try:
            t1_result = run_tier1(args, trades_df=trades_df)
            tier_results["tier1"] = t1_result
        except Exception as e:
            print(f"\n  Tier 1 실패: {e}")
            tier_results["tier1"] = {"result": "ERROR", "error": str(e)}
    else:
        print("\n  [SKIP] Tier 1 — --trades-log 미제공")
        tier_results["tier1"] = {"result": "SKIP", "note": "trades_log 미제공"}

    # ── Tier 2: Robustness ───────────────────────────────────────────────────
    if trades_df is not None:
        try:
            t2_result = run_tier2(args, trades_df=trades_df)
            tier_results["tier2"] = t2_result
        except Exception as e:
            print(f"\n  Tier 2 실패: {e}")
            tier_results["tier2"] = {"result": "ERROR", "error": str(e)}
    else:
        print("\n  [SKIP] Tier 2 — trades_log 미제공")
        tier_results["tier2"] = {"result": "SKIP", "note": "trades_log 미제공"}

    # ── Tier 4: Regime ───────────────────────────────────────────────────────
    if trades_df is not None:
        try:
            t4_result = run_tier4(args, trades_df=trades_df)
            tier_results["tier4"] = t4_result
        except Exception as e:
            print(f"\n  Tier 4 실패: {e}")
            tier_results["tier4"] = {"result": "ERROR", "error": str(e)}
    else:
        print("\n  [SKIP] Tier 4 — trades_log 미제공")
        tier_results["tier4"] = {"result": "SKIP", "note": "trades_log 미제공"}

    # ── Overall Verdict ──────────────────────────────────────────────────────
    def _passed(key: str) -> bool:
        return tier_results.get(key, {}).get("result") == "PASS"

    overall = determine_overall_verdict(
        t1_pass=_passed("tier1"),
        t2_pass=_passed("tier2"),
        t3_pass=_passed("tier3"),
        t4_pass=_passed("tier4"),
    )

    # ── 종합 summary JSON ────────────────────────────────────────────────────
    def _pf(key: str):
        r = tier_results.get(key, {})
        return r.get("profit_factor") or r.get("pf")

    summary = {
        "run_date":        run_ts,
        "data_period":     f"{args.start} ~ {args.end}",
        "tier1": {
            "result":   tier_results.get("tier1", {}).get("result", "SKIP"),
            "pf":       _pf("tier1"),
            "trades":   tier_results.get("tier1", {}).get("n_trades"),
            "mdd":      tier_results.get("tier1", {}).get("mdd"),
        },
        "tier2": {
            "result":            tier_results.get("tier2", {}).get("result", "SKIP"),
            "robust_thresholds": tier_results.get("tier2", {}).get(
                "threshold_sensitivity", {}).get("pass_count"),
            "neg_months":        tier_results.get("tier2", {}).get(
                "monthly_pf", {}).get("neg_count"),
        },
        "tier3": {
            "result":       tier_results.get("tier3", {}).get("result", "SKIP"),
            "p_perm":       tier_results.get("tier3", {}).get("perm_p_value_1m"),
            "p_block":      tier_results.get("tier3", {}).get("block_p_value_1m"),
            "spearman_r":   tier_results.get("tier3", {}).get("spearman_r_1m"),
            "monotonic":    tier_results.get("tier3", {}).get("monotonic_1m"),
        },
        "tier4": {
            "result":         tier_results.get("tier4", {}).get("result", "SKIP"),
            "low_vol_pf":     tier_results.get("tier4", {}).get(
                "vix_regime", {}).get("low_vol_pf"),
            "event_pf_drop":  tier_results.get("tier4", {}).get(
                "event_removal", {}).get("pf_drop"),
        },
        "overall_verdict": overall,
        "params": {
            "commission":      args.commission,
            "slippage_ticks":  args.slippage_ticks,
            "threshold":       getattr(args, "threshold", 0.25),
            "start":           args.start,
            "end":             args.end,
        },
    }

    # 종합 출력
    print("\n" + "═" * 60)
    print("  ★ VALIDATION SUMMARY ★")
    print("═" * 60)
    print(f"  Tier 1 (Gate):    {tier_results.get('tier1', {}).get('result', '?'):>6}"
          f"  PF={summary['tier1']['pf'] or '?'}")
    print(f"  Tier 2 (Robust):  {tier_results.get('tier2', {}).get('result', '?'):>6}"
          f"  robust_thresh={summary['tier2']['robust_thresholds'] or '?'}/4")
    print(f"  Tier 3 (Signal):  {tier_results.get('tier3', {}).get('result', '?'):>6}"
          f"  p_perm={summary['tier3']['p_perm'] or '?'}  "
          f"Spearman r={summary['tier3']['spearman_r'] or '?'}")
    print(f"  Tier 4 (Regime):  {tier_results.get('tier4', {}).get('result', '?'):>6}"
          f"  low_vol_pf={summary['tier4']['low_vol_pf'] or '?'}")
    print(f"\n  ▶ 최종 판정: {overall}")
    print("═" * 60)

    # 시각화
    plot_summary(summary, out_dir)

    # JSON 저장
    out_json = out_dir / "validation_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  [저장] {out_json}")

    # 파라미터 스냅샷 (재현용)
    params_path = out_dir / "run_params.json"
    with open(params_path, "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"  [저장] {params_path.name}")

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OBI+TFI Algorithm Edge Validation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python run_validation.py \\
    --data-path    ./data/spy_qqq_1min.csv \\
    --signal-log   ./data/signal_log.csv \\
    --trades-log   ./data/spy_qqq_trades.csv \\
    --start 2026-01-01 --end 2026-05-29

  # Tier 1+2 only (signal_log 없이)
  python run_validation.py --trades-log ./data/trades.csv --start 2026-01-01 --end 2026-05-29
        """,
    )

    # 데이터
    p.add_argument("--data-path",      default=None,  help="1분봉 OHLCV CSV")
    p.add_argument("--signal-log",     default=None,  help="signal_log.csv")
    p.add_argument("--trades-log",     default=None,  help="trades CSV")
    p.add_argument("--vix-path",       default=None,  help="VIX CSV (없으면 yfinance)")

    # 기간
    p.add_argument("--start",          default=None)
    p.add_argument("--end",            default=None)

    # 비용 모델
    p.add_argument("--commission",     type=float, default=0.005)
    p.add_argument("--slippage-ticks", type=float, default=0.5)

    # 전략 파라미터
    p.add_argument("--threshold",      type=float, default=0.25)

    # Tier 3
    p.add_argument("--n-bootstrap",   type=int, default=10_000)
    p.add_argument("--block-size",    type=int, default=10)

    # Tier 4
    p.add_argument("--event-periods", default=None,
                   help="이벤트 기간 (쉼표 구분, ex: '2026-01-01,2026-03-31')")

    # 초기 자본
    p.add_argument("--initial-capital", type=float, default=100_000.0)

    # 출력
    p.add_argument("--out-dir",        default="./validation_results")

    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if not any([args.trades_log, args.signal_log]):
        print("오류: --trades-log 또는 --signal-log 중 하나 이상 제공해야 합니다.")
        sys.exit(1)

    summary = run_pipeline(args)
    verdict = summary.get("overall_verdict", "알 수 없음")
    print(f"\n실행 완료 → {verdict}")
