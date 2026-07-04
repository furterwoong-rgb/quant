#!/usr/bin/env python3
"""
BTC 매매 성과 분석 — btc_trades.csv 기반
실행: python analyze_trades.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("logs/btc_trades.csv")

if not CSV_PATH.exists():
    print(f"파일 없음: {CSV_PATH}")
    exit(1)

df = pd.read_csv(CSV_PATH)
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
df = df.sort_values("timestamp_utc").reset_index(drop=True)

exits = df[df["action"].str.contains("EXIT")]
enters = df[df["action"] == "ENTER"]

total  = len(exits)
wins   = len(exits[exits["pnl_usd"] > 0])
losses = len(exits[exits["pnl_usd"] <= 0])
wr     = wins / total * 100 if total > 0 else 0
total_pnl     = exits["pnl_usd"].sum()
avg_win       = exits[exits["pnl_usd"] > 0]["pnl_usd"].mean() if wins else 0
avg_loss      = exits[exits["pnl_usd"] <= 0]["pnl_usd"].mean() if losses else 0
avg_slippage  = enters["slippage_bps"].mean() if not enters.empty else 0

print("=" * 50)
print("  BTC 매매 성과 요약")
print("=" * 50)
print(f"  총 청산 횟수  : {total}회  ({wins}승 / {losses}패)")
print(f"  승률          : {wr:.1f}%")
print(f"  누적 PnL      : ${total_pnl:+,.2f}")
print(f"  평균 수익 (W) : ${avg_win:+,.2f}")
print(f"  평균 손실 (L) : ${avg_loss:+,.2f}")
print(f"  평균 슬리피지 : {avg_slippage:.2f} bps")
print("=" * 50)

# ── 차트 ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle("BTC Trader — 성과 분석", fontsize=14, fontweight="bold")

# 1. 누적 포트폴리오
ax1 = axes[0]
ax1.plot(df["timestamp_utc"], df["portfolio"], color="#1565C0", lw=1.5)
ax1.set_ylabel("Portfolio ($)")
ax1.set_title("누적 포트폴리오 가치")
ax1.grid(True, alpha=0.3)

# 2. 거래별 PnL 바 차트
ax2 = axes[1]
colors = ["#00C853" if p > 0 else "#D50000" for p in exits["pnl_usd"]]
ax2.bar(range(len(exits)), exits["pnl_usd"].values, color=colors, alpha=0.8)
ax2.axhline(0, color="black", lw=0.8)
ax2.set_ylabel("PnL ($)")
ax2.set_title("거래별 손익")
ax2.grid(True, alpha=0.3, axis="y")

# 3. 슬리피지 분포
ax3 = axes[2]
if not enters.empty and enters["slippage_bps"].notna().any():
    ax3.hist(enters["slippage_bps"].dropna(), bins=20, color="#7B1FA2", alpha=0.7)
    ax3.axvline(avg_slippage, color="red", ls="--", lw=1.2,
                label=f"평균 {avg_slippage:.2f}bp")
    ax3.legend()
ax3.set_xlabel("Slippage (bps)")
ax3.set_ylabel("빈도")
ax3.set_title("진입 슬리피지 분포")
ax3.grid(True, alpha=0.3)

plt.tight_layout()
out = Path("logs/btc_analysis.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\n차트 저장: {out}")
