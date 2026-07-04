#!/usr/bin/env python3
"""
Tier 1 — 최소 통과 조건 (Gate Test)
───────────────────────────────────────────────────────────────────────────────
수수료·슬리피지 반영 후에도 edge가 존재하는지 확인.

비용 모델:
  - 수수료: $0.005/주 (편도)  → 왕복 $0.01/주
  - 슬리피지: 0.5 tick (=$0.005/주) 편도 → 왕복 $0.01/주
  - 합산 왕복 비용: ($0.005 + $0.005) × 2 = $0.02/주

판정 기준:
  PASS: PF ≥ 1.25  AND  트레이드 수 ≥ 300  AND  MDD ≤ 20%
  FAIL: 위 조건 하나라도 미달

실행:
  python tier1_gate.py \
    --trades-log ./data/spy_qqq_trades.csv \
    --out-dir    ./validation_results/20260529_120000 \
    --commission 0.005 \
    --slippage-ticks 0.5

입력 trades CSV 컬럼:
  timestamp_et, symbol, action, price, qty, pnl_usd, slippage_bps, signal, portfolio
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

warnings.filterwarnings("ignore")

# ── 판정 기준 ─────────────────────────────────────────────────────────────────
PASS_PF          = 1.25
PASS_MIN_TRADES  = 300
PASS_MAX_MDD     = 0.20   # 20%

# 청산 action 집합 (pnl_usd가 의미있는 행)
EXIT_ACTIONS = {
    # 영문 (백테스터/구버전)
    "SELL", "STOP", "PARTIAL_SELL", "TRAIL_STOP",
    "CONV_EXIT", "EOD_REDUCE", "EOD_CLOSE",
    "partial", "stop", "conv_exit", "trail", "eod_reduce", "eod_close", "sell",
    # 한글 (라이브 트레이더 실제 CSV)
    "EXIT_신호소멸", "EXIT_로컬저점손절", "EXIT_손절", "EXIT_부분익절",
    "EXIT_트레일링", "EXIT_EOD_REDUCE", "EXIT_EOD_CLOSE",
    "exit_신호소멸", "exit_로컬저점손절", "exit_손절", "exit_부분익절",
    "exit_트레일링", "exit_eod_reduce", "exit_eod_close",
}


# ══════════════════════════════════════════════════════════════════════════════
# 데이터 로드
# ══════════════════════════════════════════════════════════════════════════════

def load_trades(path: str) -> pd.DataFrame:
    """
    trades CSV 로드 및 정제.
    timestamp_et 파싱, action 소문자 정규화.
    """
    df = pd.read_csv(path, parse_dates=["timestamp_et"])
    df = df.sort_values("timestamp_et").reset_index(drop=True)
    df["action_lower"] = df["action"].str.lower().str.strip()
    df["pnl_usd"]  = pd.to_numeric(df["pnl_usd"],  errors="coerce").fillna(0.0)
    df["qty"]      = pd.to_numeric(df["qty"],       errors="coerce").fillna(0).astype(int)
    df["portfolio"] = pd.to_numeric(df["portfolio"], errors="coerce")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 비용 적용
# ══════════════════════════════════════════════════════════════════════════════

def apply_costs(
    exits_df: pd.DataFrame,
    commission: float,
    slippage_ticks: float,
    tick_size: float = 0.01,
) -> pd.DataFrame:
    """
    청산 행에 왕복 수수료+슬리피지를 적용해 net_pnl 컬럼 추가.
    왕복 비용 = 2 × (commission + slippage_ticks × tick_size) × qty
    """
    round_trip_cost_per_share = 2 * (commission + slippage_ticks * tick_size)
    exits_df = exits_df.copy()
    exits_df["cost"] = round_trip_cost_per_share * exits_df["qty"]
    exits_df["net_pnl"] = exits_df["pnl_usd"] - exits_df["cost"]
    return exits_df


# ══════════════════════════════════════════════════════════════════════════════
# 지표 계산
# ══════════════════════════════════════════════════════════════════════════════

def compute_profit_factor(net_pnls: pd.Series) -> float:
    wins   = net_pnls[net_pnls > 0].sum()
    losses = net_pnls[net_pnls < 0].abs().sum()
    if losses == 0:
        return float("inf")
    return float(wins / losses)


def compute_win_rate(net_pnls: pd.Series) -> float:
    completed = net_pnls[net_pnls != 0]
    if len(completed) == 0:
        return 0.0
    return float((completed > 0).sum() / len(completed))


def compute_avg_win_loss(net_pnls: pd.Series) -> tuple[float, float]:
    wins   = net_pnls[net_pnls > 0]
    losses = net_pnls[net_pnls < 0]
    avg_win  = float(wins.mean())  if len(wins)   > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    return avg_win, avg_loss


def compute_max_drawdown(portfolio: pd.Series) -> float:
    """포트폴리오 equity 시계열에서 MDD(%) 계산."""
    portfolio = portfolio.dropna()
    if len(portfolio) < 2:
        return 0.0
    peak    = portfolio.expanding().max()
    drawdown = (portfolio - peak) / peak
    return float(drawdown.min())  # 음수


def build_equity_curve(
    df: pd.DataFrame,
    initial_capital: float,
    commission: float,
    slippage_ticks: float,
) -> pd.Series:
    """
    portfolio 컬럼이 있으면 그것을 사용.
    없으면 initial_capital + cumsum(net_pnl) 로 근사 재구성.
    """
    if "portfolio" in df.columns and df["portfolio"].notna().any():
        portfolio = df[["timestamp_et", "portfolio"]].dropna(subset=["portfolio"])
        portfolio = portfolio.set_index("timestamp_et")["portfolio"]
        return portfolio

    # 청산 행의 net_pnl을 누적
    exits = df[df["action_lower"].isin({a.lower() for a in EXIT_ACTIONS})].copy()
    if len(exits) == 0:
        return pd.Series([initial_capital], name="portfolio")
    exits = apply_costs(exits, commission, slippage_ticks)
    curve = initial_capital + exits.set_index("timestamp_et")["net_pnl"].cumsum()
    return curve


# ══════════════════════════════════════════════════════════════════════════════
# 리포트 시각화
# ══════════════════════════════════════════════════════════════════════════════

def plot_tier1_report(
    equity_curve: pd.Series,
    exits_df: pd.DataFrame,
    metrics: dict,
    out_dir: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    # 왼쪽: Equity Curve
    ax1 = axes[0]
    ax1.set_facecolor("#161b22")
    ax1.plot(equity_curve.index, equity_curve.values,
             color="#58a6ff", linewidth=1.2)
    ax1.fill_between(equity_curve.index, equity_curve.min(),
                     equity_curve.values, alpha=0.15, color="#58a6ff")
    ax1.axhline(equity_curve.iloc[0], color="#8b949e",
                linewidth=0.8, linestyle="--")
    verdict = metrics.get("result", "?")
    ax1.set_title(f"Equity Curve  [Tier 1: {verdict}]",
                  color="#e6edf3", fontsize=11)
    ax1.set_xlabel("Date", color="#8b949e")
    ax1.set_ylabel("Portfolio ($)", color="#8b949e")
    ax1.tick_params(colors="#8b949e")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#30363d")

    # 오른쪽: PnL 분포 히스토그램
    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    net_pnls = exits_df["net_pnl"].dropna() if "net_pnl" in exits_df.columns else pd.Series()
    if len(net_pnls) > 0:
        wins   = net_pnls[net_pnls > 0]
        losses = net_pnls[net_pnls <= 0]
        ax2.hist(wins.values,   bins=40, color="#3fb950", alpha=0.75, label="Win")
        ax2.hist(losses.values, bins=40, color="#f85149", alpha=0.75, label="Loss")
        ax2.axvline(0, color="#e6edf3", linewidth=0.8, linestyle="--")
    ax2.set_title("Net PnL Distribution (cost-adjusted)", color="#e6edf3", fontsize=11)
    ax2.set_xlabel("Net PnL ($)", color="#8b949e")
    ax2.tick_params(colors="#8b949e")
    ax2.legend(facecolor="#161b22", labelcolor="#e6edf3")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#30363d")

    # Stats annotation
    stats_text = (
        f"PF:        {metrics.get('profit_factor', 0):.3f}  "
        f"({'≥1.25 ✓' if metrics.get('profit_factor', 0) >= PASS_PF else '<1.25 ✗'})\n"
        f"Trades:    {metrics.get('n_trades', 0):,}  "
        f"({'≥300 ✓' if metrics.get('n_trades', 0) >= PASS_MIN_TRADES else '<300 ✗'})\n"
        f"MDD:       {metrics.get('mdd_pct', 0):.1f}%  "
        f"({'≤20% ✓' if abs(metrics.get('mdd_pct', 0)) <= 20 else '>20% ✗'})\n"
        f"Win Rate:  {metrics.get('win_rate', 0)*100:.1f}%\n"
        f"Avg Win:   ${metrics.get('avg_win', 0):.2f}\n"
        f"Avg Loss:  ${metrics.get('avg_loss', 0):.2f}"
    )
    ax1.text(0.02, 0.97, stats_text, transform=ax1.transAxes,
             va="top", fontsize=8, color="#e6edf3",
             fontfamily="monospace",
             bbox=dict(facecolor="#0d1117", edgecolor="#30363d", alpha=0.85))

    plt.tight_layout()
    out_path = out_dir / "tier1_gate_report.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [저장] {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 메인 Tier 1 실행
# ══════════════════════════════════════════════════════════════════════════════

def run_tier1(
    args: argparse.Namespace,
    trades_df: pd.DataFrame = None,
) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("TIER 1 — Gate Test (최소 통과 조건)")
    print("=" * 60)

    # trades 로드
    if trades_df is None:
        trades_df = load_trades(args.trades_log)
    print(f"  trades rows: {len(trades_df):,}")

    # 날짜 필터 (args에 start/end가 있으면 적용)
    if hasattr(args, "start") and args.start:
        trades_df = trades_df[trades_df["timestamp_et"] >= args.start]
    if hasattr(args, "end") and args.end:
        trades_df = trades_df[trades_df["timestamp_et"] <= args.end]

    # 청산 행 추출
    exits_df = trades_df[
        trades_df["action_lower"].isin({a.lower() for a in EXIT_ACTIONS})
    ].copy()
    exits_df = apply_costs(exits_df, args.commission, args.slippage_ticks)
    n_trades = len(exits_df)

    print(f"  청산 이벤트:  {n_trades:,}개")
    print(f"  비용 모델:    수수료 ${args.commission:.4f}/주 "
          f"+ 슬리피지 {args.slippage_ticks} tick → "
          f"왕복 ${2*(args.commission + args.slippage_ticks*0.01):.4f}/주")

    # 지표 계산
    net_pnls    = exits_df["net_pnl"]
    pf          = compute_profit_factor(net_pnls)
    win_rate    = compute_win_rate(net_pnls)
    avg_win, avg_loss = compute_avg_win_loss(net_pnls)

    initial_cap = getattr(args, "initial_capital", 100_000.0)
    equity_curve = build_equity_curve(
        trades_df, initial_cap, args.commission, args.slippage_ticks)
    mdd = compute_max_drawdown(equity_curve)
    mdd_pct = mdd * 100

    # 판정
    pass_pf     = pf     >= PASS_PF
    pass_trades = n_trades >= PASS_MIN_TRADES
    pass_mdd    = abs(mdd) <= PASS_MAX_MDD
    tier1_pass  = pass_pf and pass_trades and pass_mdd
    verdict     = "PASS" if tier1_pass else "FAIL"

    # 콘솔 출력
    print("\n  ────────────────────────────────────────")
    pf_str = f"{pf:.3f}" if pf != float("inf") else "inf"
    print(f"  Profit Factor: {pf_str:>8}  "
          f"({'≥1.25 ✓' if pass_pf else '<1.25 ✗'})")
    print(f"  Trades:        {n_trades:>8,}  "
          f"({'≥300 ✓' if pass_trades else '<300 ✗'})")
    print(f"  MDD:           {mdd_pct:>7.1f}%  "
          f"({'≤20% ✓' if pass_mdd else '>20% ✗'})")
    print(f"  Win Rate:      {win_rate*100:>7.1f}%")
    print(f"  Avg Win:       ${avg_win:>7.2f}")
    print(f"  Avg Loss:      ${avg_loss:>7.2f}")
    print(f"\n  TIER 1 최종:   {verdict}")
    print("  ────────────────────────────────────────")

    metrics = {
        "result":        verdict,
        "profit_factor": round(pf, 4) if pf != float("inf") else None,
        "n_trades":      int(n_trades),
        "mdd":           round(mdd, 4),
        "mdd_pct":       round(mdd_pct, 2),
        "win_rate":      round(win_rate, 4),
        "avg_win":       round(avg_win, 4),
        "avg_loss":      round(avg_loss, 4),
        "total_net_pnl": round(float(net_pnls.sum()), 2),
        "pass_pf":       pass_pf,
        "pass_trades":   pass_trades,
        "pass_mdd":      pass_mdd,
        "params": {
            "commission":      args.commission,
            "slippage_ticks":  args.slippage_ticks,
            "pass_criteria": {
                "pf_min":       PASS_PF,
                "trades_min":   PASS_MIN_TRADES,
                "mdd_max_pct":  PASS_MAX_MDD * 100,
            },
        },
    }

    # 시각화
    plot_tier1_report(equity_curve, exits_df, metrics, out_dir)

    # JSON 저장
    out_json = out_dir / "tier1_gate_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
    print(f"  [저장] {out_json.name}")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tier 1 — Gate Test")
    p.add_argument("--trades-log",      required=True, help="trades CSV 경로")
    p.add_argument("--out-dir",         default="./validation_results/tmp")
    p.add_argument("--commission",      type=float, default=0.005,
                   help="편도 수수료 ($/주)")
    p.add_argument("--slippage-ticks",  type=float, default=0.5,
                   help="편도 슬리피지 (tick 수)")
    p.add_argument("--initial-capital", type=float, default=100_000.0)
    p.add_argument("--start",           default=None)
    p.add_argument("--end",             default=None)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    result = run_tier1(args)
    print(f"\n최종 판정: {result['result']}")
