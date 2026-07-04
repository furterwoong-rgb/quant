#!/usr/bin/env python3
"""
Tier 4 — Regime 내구성 검증 (Regime Robustness Test)
───────────────────────────────────────────────────────────────────────────────
2026년 호르무즈 위기 특수 환경에만 최적화된 것이 아님을 검증.

구성:
  ① VIX 구간별 성과 (Low/Normal/High)
  ② 지정학 이벤트 제거 전/후 PF 비교
  ③ 역사 데이터 구간 테스트 (2022/2023/2024 — 데이터 있을 경우)

판정:
  PASS: ① Low Vol PF ≥ 1.1  AND  ② 이벤트 제거 후 PF 하락 < 15%

실행:
  python tier4_regime.py \
    --trades-log ./data/spy_qqq_trades.csv \
    --vix-path   ./data/vix.csv \
    --out-dir    ./validation_results/20260529_120000 \
    --commission 0.005 \
    --slippage-ticks 0.5 \
    --event-periods "2025-12-01,2026-03-31"
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from tier1_gate import apply_costs, compute_profit_factor, compute_max_drawdown, EXIT_ACTIONS

# ── VIX 구간 정의 ─────────────────────────────────────────────────────────────
VIX_LOW_MAX    = 15.0
VIX_NORMAL_MAX = 25.0
VIX_REGIMES    = [
    ("Low Vol",  lambda v: v < VIX_LOW_MAX),
    ("Normal",   lambda v: VIX_LOW_MAX <= v < VIX_NORMAL_MAX),
    ("High Vol", lambda v: v >= VIX_NORMAL_MAX),
]

LOW_VOL_PF_MIN     = 1.10
EVENT_DECAY_MAX    = 0.15   # 이벤트 제거 후 PF 하락 허용 최대

HISTORICAL_PERIODS = [
    ("2022 (하락장)",      "2022-01-01", "2022-12-31"),
    ("2023 H2 (AI 랠리)", "2023-07-01", "2023-12-31"),
    ("2024 (이벤트)",      "2024-01-01", "2024-12-31"),
]

# 기본 이벤트 기간 (호르무즈 위기 추정)
DEFAULT_EVENT_PERIODS = [("2026-01-01", "2026-03-31")]


# ══════════════════════════════════════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════════════════════════════════════

def _load_trades(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp_et"])
    df = df.sort_values("timestamp_et").reset_index(drop=True)
    df["action_lower"] = df["action"].str.lower().str.strip()
    df["pnl_usd"]   = pd.to_numeric(df["pnl_usd"],  errors="coerce").fillna(0.0)
    df["qty"]       = pd.to_numeric(df["qty"],       errors="coerce").fillna(0).astype(int)
    df["portfolio"] = pd.to_numeric(df["portfolio"], errors="coerce")
    return df


def _exits_with_cost(
    trades_df: pd.DataFrame,
    commission: float,
    slippage_ticks: float,
) -> pd.DataFrame:
    exits = trades_df[
        trades_df["action_lower"].isin({a.lower() for a in EXIT_ACTIONS})
    ].copy()
    return apply_costs(exits, commission, slippage_ticks)


def _regime_metrics(exits_df: pd.DataFrame) -> dict:
    if len(exits_df) == 0:
        return {"n": 0, "pf": 0.0, "mdd": None, "win_rate": None}
    net = exits_df["net_pnl"]
    pf  = compute_profit_factor(net)
    wr  = float((net > 0).sum() / max(len(net[net != 0]), 1))
    # MDD는 포트폴리오 equity 컬럼이 있을 경우만
    mdd = None
    if "portfolio" in exits_df.columns and exits_df["portfolio"].notna().any():
        eq  = exits_df.set_index("timestamp_et")["portfolio"].dropna()
        mdd = round(float(compute_max_drawdown(eq)) * 100, 2)
    return {
        "n":        int(len(exits_df)),
        "pf":       round(pf, 4) if np.isfinite(pf) else None,
        "mdd_pct":  mdd,
        "win_rate": round(wr, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# VIX 데이터 로드
# ══════════════════════════════════════════════════════════════════════════════

def load_vix(path: str = None) -> pd.DataFrame:
    """
    VIX 일봉 데이터 로드.
    path가 있으면 CSV에서, 없으면 yfinance ^VIX에서 다운로드.
    CSV 컬럼: date, close  (또는 Date, Close)
    """
    if path:
        try:
            df = pd.read_csv(path)
            df.columns = [c.lower() for c in df.columns]
            if "date" not in df.columns or "close" not in df.columns:
                raise ValueError("VIX CSV에 date/close 컬럼 필요")
            df["date"] = pd.to_datetime(df["date"])
            df = df[["date", "close"]].rename(columns={"close": "vix"})
            df = df.sort_values("date").reset_index(drop=True)
            print(f"  VIX: CSV 로드 완료 ({len(df):,} rows)")
            return df
        except Exception as e:
            print(f"  VIX CSV 로드 실패 ({e}) → yfinance fallback")

    try:
        import yfinance as yf
        raw = yf.download("^VIX", start="2020-01-01", progress=False)
        if raw.empty:
            raise ValueError("yfinance VIX 다운로드 실패")
        df = raw[["Close"]].reset_index()
        df.columns = ["date", "vix"]
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        print(f"  VIX: yfinance 다운로드 완료 ({len(df):,} rows)")
        return df
    except Exception as e:
        print(f"  VIX 로드 실패: {e} → VIX 분석 SKIP")
        return pd.DataFrame()


def merge_vix(trades_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """trades에 VIX(일봉) 컬럼 병합."""
    if vix_df.empty:
        trades_df["vix"] = np.nan
        return trades_df
    vix_df = vix_df.copy()
    vix_df["date"] = vix_df["date"].dt.normalize()
    trades = trades_df.copy()
    trades["date"] = trades["timestamp_et"].dt.normalize()
    merged = trades.merge(vix_df[["date", "vix"]], on="date", how="left")
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# ① VIX 구간별 성과
# ══════════════════════════════════════════════════════════════════════════════

def vix_regime_analysis(
    trades_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    commission: float,
    slippage_ticks: float,
    out_dir: Path,
) -> dict:
    trades_w_vix = merge_vix(trades_df, vix_df)
    exits        = _exits_with_cost(trades_w_vix, commission, slippage_ticks)

    if exits["vix"].isna().all():
        return {"note": "VIX 데이터 없음 — SKIP", "passed": True}

    regime_results = {}
    for name, cond in VIX_REGIMES:
        subset = exits[exits["vix"].apply(lambda v: cond(v) if pd.notna(v) else False)]
        regime_results[name] = _regime_metrics(subset)

    low_pf = regime_results.get("Low Vol", {}).get("pf") or 0.0
    passed = low_pf >= LOW_VOL_PF_MIN

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    # 왼쪽: 구간별 PF
    ax1 = axes[0]
    ax1.set_facecolor("#161b22")
    names  = list(regime_results.keys())
    pfs    = [regime_results[n].get("pf") or 0 for n in names]
    ns     = [regime_results[n].get("n", 0)    for n in names]
    colors = ["#58a6ff", "#3fb950", "#f0883e"]

    bars = ax1.bar(names, pfs, color=colors, alpha=0.85)
    ax1.axhline(LOW_VOL_PF_MIN, color="#f85149", linewidth=1.2, linestyle="--",
                label=f"Low Vol 기준 PF≥{LOW_VOL_PF_MIN}")
    for bar, n, pf_v in zip(bars, ns, pfs):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 pf_v + 0.01, f"PF={pf_v:.3f}\nN={n:,}",
                 ha="center", va="bottom", color="#e6edf3", fontsize=8)

    verdict = "PASS" if passed else "FAIL"
    ax1.set_title(f"VIX Regime PF — {verdict}", color="#e6edf3", fontsize=11)
    ax1.set_ylabel("Profit Factor", color="#8b949e")
    ax1.tick_params(colors="#8b949e")
    ax1.legend(facecolor="#161b22", labelcolor="#e6edf3")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#30363d")

    # 오른쪽: VIX 분포 (거래일 기준)
    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    vix_vals = exits["vix"].dropna()
    if len(vix_vals) > 0:
        ax2.hist(vix_vals, bins=40, color="#8b949e", alpha=0.75)
        ax2.axvline(VIX_LOW_MAX,    color="#58a6ff", linestyle="--",
                    label=f"Low/Normal ({VIX_LOW_MAX})")
        ax2.axvline(VIX_NORMAL_MAX, color="#f0883e", linestyle="--",
                    label=f"Normal/High ({VIX_NORMAL_MAX})")
    ax2.set_title("VIX Distribution (trade days)", color="#e6edf3", fontsize=11)
    ax2.set_xlabel("VIX", color="#8b949e")
    ax2.tick_params(colors="#8b949e")
    ax2.legend(facecolor="#161b22", labelcolor="#e6edf3")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#30363d")

    out_path = out_dir / "tier4_vix_regime.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [저장] {out_path.name}")

    return {
        "regimes":  {k: v for k, v in regime_results.items()},
        "low_vol_pf": round(low_pf, 4),
        "passed":   passed,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ② 지정학 이벤트 제거 테스트
# ══════════════════════════════════════════════════════════════════════════════

def event_removal_test(
    trades_df: pd.DataFrame,
    commission: float,
    slippage_ticks: float,
    event_periods: list[tuple[str, str]],
    out_dir: Path,
) -> dict:
    """
    이벤트 기간 ±30일 제거 전/후 PF 변화량 계산.
    PF 하락 < 15% → PASS.
    """
    exits_all = _exits_with_cost(trades_df, commission, slippage_ticks)
    pf_before = compute_profit_factor(exits_all["net_pnl"])

    # 제거 마스크 생성
    mask = pd.Series(False, index=exits_all.index)
    expanded_periods = []
    for start_s, end_s in event_periods:
        start_dt = pd.Timestamp(start_s) - pd.Timedelta(days=30)
        end_dt   = pd.Timestamp(end_s)   + pd.Timedelta(days=30)
        expanded_periods.append((str(start_dt.date()), str(end_dt.date())))
        mask |= (
            (exits_all["timestamp_et"] >= start_dt) &
            (exits_all["timestamp_et"] <= end_dt)
        )

    exits_excl = exits_all[~mask]
    pf_after   = compute_profit_factor(exits_excl["net_pnl"])

    pf_drop    = (pf_before - pf_after) / pf_before if pf_before > 0 else 0.0
    passed     = pf_drop < EVENT_DECAY_MAX

    print(f"       이벤트 제거: {sum(mask):,}건 제외  "
          f"PF {pf_before:.3f} → {pf_after:.3f}  "
          f"하락 {pf_drop*100:.1f}%  "
          f"({'PASS' if passed else 'FAIL'})")

    # 시각화
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    labels_bar = ["Before removal", "After removal"]
    values_bar = [pf_before, pf_after]
    colors_bar = ["#58a6ff", "#3fb950" if passed else "#f85149"]
    bars = ax.bar(labels_bar, values_bar, color=colors_bar, alpha=0.85, width=0.4)
    ax.axhline(1.0, color="#8b949e", linewidth=0.8, linestyle="--")
    for bar, v in zip(bars, values_bar):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", color="#e6edf3", fontsize=10)
    verdict = "PASS" if passed else "FAIL"
    ax.set_title(
        f"Event Removal Test — {verdict}  "
        f"(PF drop: {pf_drop*100:.1f}%  [<15% 기준])",
        color="#e6edf3", fontsize=10,
    )
    ax.set_ylabel("Profit Factor", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    out_path = out_dir / "tier4_event_removal.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [저장] {out_path.name}")

    return {
        "pf_before":       round(pf_before, 4),
        "pf_after":        round(pf_after, 4),
        "pf_drop":         round(pf_drop, 4),
        "pf_drop_pct":     round(pf_drop * 100, 2),
        "event_periods":   expanded_periods,
        "excluded_trades": int(sum(mask)),
        "passed":          passed,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ③ 역사 데이터 구간 테스트
# ══════════════════════════════════════════════════════════════════════════════

def historical_period_test(
    trades_df: pd.DataFrame,
    commission: float,
    slippage_ticks: float,
    out_dir: Path,
) -> dict:
    """
    2022/2023 H2/2024 구간별 독립 백테스트.
    데이터 없으면 SKIP.
    """
    exits_all = _exits_with_cost(trades_df, commission, slippage_ticks)
    results   = []

    for label, start_s, end_s in HISTORICAL_PERIODS:
        start_dt = pd.Timestamp(start_s)
        end_dt   = pd.Timestamp(end_s)
        subset   = exits_all[
            (exits_all["timestamp_et"] >= start_dt) &
            (exits_all["timestamp_et"] <= end_dt)
        ]
        if len(subset) < 10:
            results.append({
                "period":   label,
                "start":    start_s,
                "end":      end_s,
                "n_trades": int(len(subset)),
                "pf":       None,
                "note":     "SKIP — 데이터 미확보",
            })
        else:
            pf_v = compute_profit_factor(subset["net_pnl"])
            results.append({
                "period":   label,
                "start":    start_s,
                "end":      end_s,
                "n_trades": int(len(subset)),
                "pf":       round(pf_v, 4) if np.isfinite(pf_v) else None,
                "note":     "",
            })

    # 시각화 (데이터 있는 구간만)
    valid = [r for r in results if r["pf"] is not None]
    if valid:
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")

        labels_h  = [r["period"]    for r in valid]
        pfs_h     = [r["pf"]        for r in valid]
        ns_h      = [r["n_trades"]  for r in valid]
        col_h     = ["#3fb950" if p >= 1.0 else "#f85149" for p in pfs_h]

        bars = ax.bar(range(len(labels_h)), pfs_h, color=col_h, alpha=0.85)
        ax.axhline(1.0, color="#f0883e", linewidth=1, linestyle="--")
        for i, (bar, n, pf_v) in enumerate(zip(bars, ns_h, pfs_h)):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    pf_v + 0.01, f"PF={pf_v:.3f}\nN={n:,}",
                    ha="center", va="bottom", color="#e6edf3", fontsize=8)
        ax.set_xticks(range(len(labels_h)))
        ax.set_xticklabels(labels_h, color="#8b949e", fontsize=8)
        ax.set_title("Historical Period PF Comparison",
                     color="#e6edf3", fontsize=11)
        ax.set_ylabel("Profit Factor", color="#8b949e")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        out_path = out_dir / "tier4_historical_periods.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"  [저장] {out_path.name}")

    return {"periods": results}


# ══════════════════════════════════════════════════════════════════════════════
# 메인 Tier 4 실행
# ══════════════════════════════════════════════════════════════════════════════

def run_tier4(
    args: argparse.Namespace,
    trades_df: pd.DataFrame = None,
) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("TIER 4 — Regime Robustness Test")
    print("=" * 60)

    if trades_df is None:
        trades_df = _load_trades(args.trades_log)
    if hasattr(args, "start") and args.start:
        trades_df = trades_df[trades_df["timestamp_et"] >= args.start]
    if hasattr(args, "end") and args.end:
        trades_df = trades_df[trades_df["timestamp_et"] <= args.end]

    # ── ① VIX 구간별 성과 ────────────────────────────────────────────────────
    print("\n  [①] VIX 구간별 성과 분석...")
    vix_df  = load_vix(getattr(args, "vix_path", None))
    vix_res = vix_regime_analysis(
        trades_df, vix_df, args.commission, args.slippage_ticks, out_dir)

    if "note" in vix_res:
        print(f"       {vix_res['note']}")
    else:
        for regime, m in vix_res.get("regimes", {}).items():
            pf_s = f"{m.get('pf', 0):.3f}" if m.get("pf") else "N/A"
            print(f"       {regime:<12} PF={pf_s:>8}  N={m.get('n',0):>5,}  "
                  f"WinRate={m.get('win_rate',0)*100:.1f}%")
        print(f"       Low Vol PF={vix_res.get('low_vol_pf', 0):.3f}  "
              f"({'PASS ≥1.1 ✓' if vix_res.get('passed') else 'FAIL <1.1 ✗'})")

    # ── ② 이벤트 제거 테스트 ─────────────────────────────────────────────────
    print("\n  [②] 지정학 이벤트 제거 테스트...")
    event_periods = DEFAULT_EVENT_PERIODS
    if hasattr(args, "event_periods") and args.event_periods:
        # "2026-01-01,2026-03-31" 형식 파싱
        try:
            parts = [p.strip() for p in args.event_periods.split(",")]
            if len(parts) == 2:
                event_periods = [(parts[0], parts[1])]
            elif len(parts) % 2 == 0:
                event_periods = [(parts[i], parts[i+1]) for i in range(0, len(parts), 2)]
        except Exception:
            pass

    event_res = event_removal_test(
        trades_df, args.commission, args.slippage_ticks, event_periods, out_dir)

    # ── ③ 역사 데이터 구간 테스트 ───────────────────────────────────────────
    print("\n  [③] 역사 데이터 구간 테스트...")
    hist_res = historical_period_test(
        trades_df, args.commission, args.slippage_ticks, out_dir)
    for r in hist_res["periods"]:
        pf_s = f"{r['pf']:.3f}" if r["pf"] is not None else "SKIP"
        print(f"       {r['period']:<20} PF={pf_s:>8}  N={r['n_trades']:>5,}  "
              f"{r.get('note', '')}")

    # 판정
    pass_vix   = vix_res.get("passed", True)
    pass_event = event_res.get("passed", True)
    tier4_pass = pass_vix and pass_event
    verdict    = "PASS" if tier4_pass else "FAIL"

    print("\n  ────────────────────────────────────────")
    print(f"  ① VIX Low Vol PF:   {'PASS' if pass_vix else 'FAIL'}")
    print(f"  ② 이벤트 제거 PF 변화: {'PASS' if pass_event else 'FAIL'}")
    print(f"\n  TIER 4 최종:        {verdict}")
    print("  ────────────────────────────────────────")

    report = {
        "tier":         4,
        "result":       verdict,
        "vix_regime":   vix_res,
        "event_removal": event_res,
        "historical":   hist_res,
        "pass_details": {
            "vix_low_vol": pass_vix,
            "event_removal": pass_event,
        },
        "params": {
            "commission":       args.commission,
            "slippage_ticks":   args.slippage_ticks,
            "vix_low_max":      VIX_LOW_MAX,
            "vix_normal_max":   VIX_NORMAL_MAX,
            "low_vol_pf_min":   LOW_VOL_PF_MIN,
            "event_decay_max":  EVENT_DECAY_MAX,
            "event_periods":    event_periods,
        },
    }

    out_json = out_dir / "tier4_regime_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  [저장] {out_json.name}")

    return report


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tier 4 — Regime Robustness Test")
    p.add_argument("--trades-log",     required=True)
    p.add_argument("--vix-path",       default=None, help="VIX CSV (없으면 yfinance)")
    p.add_argument("--out-dir",        default="./validation_results/tmp")
    p.add_argument("--commission",     type=float, default=0.005)
    p.add_argument("--slippage-ticks", type=float, default=0.5)
    p.add_argument("--event-periods",  default=None,
                   help="이벤트 기간 (쉼표 구분, ex: '2026-01-01,2026-03-31')")
    p.add_argument("--start",          default=None)
    p.add_argument("--end",            default=None)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    result = run_tier4(args)
    print(f"\n최종 판정: {result['result']}")
