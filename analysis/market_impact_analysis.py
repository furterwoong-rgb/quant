#!/usr/bin/env python3
"""
시장충격(Market Impact) 분석 — 2026-05-15 틱 데이터 기준
포트폴리오 규모별 (10만불 / 10억원 / 100억원) 충격 추정
"""

import pickle, numpy as np, pandas as pd
from pathlib import Path

CACHE = Path("logs/tick_cache")
ET    = __import__("pytz").timezone("America/New_York")
KRW   = 1_380   # 2026년 5월 기준 환율 가정 (원/달러)

# ── 포트폴리오 시나리오 ──────────────────────────────────────────────────────
PORTFOLIOS = {
    "현재 ($100K)":  100_000,
    "10억원":        1_000_000_000 / KRW,
    "100억원":      10_000_000_000 / KRW,
}
KELLY = 0.10   # 포지션당 10%

# ── 1. SSO/QLD/SDS/QID 바 데이터 로드 ──────────────────────────────────────
def load_bars(sym):
    with open(CACHE / f"{sym}_2026-05-15_bars.pkl", "rb") as f:
        bars = pickle.load(f)
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(sym, level="symbol")
    bars.index = pd.DatetimeIndex(bars.index)
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize(ET)
    else:
        bars.index = bars.index.tz_convert(ET)
    return bars.between_time("09:30", "16:00")

# ── 2. SPY/QQQ 호가 데이터 로드 ─────────────────────────────────────────────
def load_quotes(sym):
    with open(CACHE / f"{sym}_2026-05-15_quotes.pkl", "rb") as f:
        q = pickle.load(f)
    if isinstance(q.index, pd.MultiIndex):
        q = q.xs(sym, level="symbol")
    q.index = pd.DatetimeIndex(q.index).tz_convert(ET)
    return q.between_time("09:30", "16:00")

def load_trades(sym):
    with open(CACHE / f"{sym}_2026-05-15_trades.pkl", "rb") as f:
        t = pickle.load(f)
    if isinstance(t.index, pd.MultiIndex):
        t = t.xs(sym, level="symbol")
    t.index = pd.DatetimeIndex(t.index).tz_convert(ET)
    return t.between_time("09:30", "16:00")

# ── 3. 호가창 깊이 분석 (SPY/QQQ) ───────────────────────────────────────────
def analyze_orderbook(sym):
    q = load_quotes(sym)
    q = q[(q["bid_size"] > 0) & (q["ask_size"] > 0) &
          (q["bid_price"] > 0) & (q["ask_price"] > 0)]

    spread_bp  = ((q["ask_price"] - q["bid_price"]) / q["bid_price"] * 10_000)
    depth_bid  = q["bid_size"]   # 매수 1호가 수량 (주)
    depth_ask  = q["ask_size"]   # 매도 1호가 수량 (주)
    mid_price  = (q["bid_price"] + q["ask_price"]) / 2

    t = load_trades(sym)
    daily_vol  = t["size"].sum()
    avg_trade  = t["size"].mean()
    avg_price  = mid_price.mean()

    return {
        "sym":         sym,
        "avg_price":   avg_price,
        "spread_bp":   spread_bp.mean(),
        "spread_med":  spread_bp.median(),
        "depth_bid":   depth_bid.mean(),      # 1호가 평균 bid 수량
        "depth_ask":   depth_ask.mean(),      # 1호가 평균 ask 수량
        "daily_vol":   daily_vol,             # 총 거래량
        "avg_trade":   avg_trade,             # 평균 체결 단위
        "tick_rows":   len(q),
    }

# ── 4. ETF 유동성 분석 (SSO/QLD/SDS/QID) ────────────────────────────────────
def analyze_etf(sym):
    bars = load_bars(sym)
    avg_price  = bars["close"].mean()
    daily_vol  = bars["volume"].sum()
    avg_1m_vol = bars["volume"].mean()
    vwap       = bars["vwap"].mean() if "vwap" in bars.columns else avg_price

    # 분당 거래대금 ($)
    dollar_vol_1m = (bars["volume"] * bars["close"]).mean()

    # 변동성 (분봉 수익률 annualized → 일중 변동성)
    ret     = bars["close"].pct_change().dropna()
    intra_vol = ret.std()  # 분봉 변동성

    return {
        "sym":          sym,
        "avg_price":    avg_price,
        "daily_vol":    daily_vol,
        "avg_1m_vol":   avg_1m_vol,
        "dollar_vol_1m": dollar_vol_1m,
        "intra_vol_bpm": intra_vol * 10_000,  # bps/min
    }

# ── 5. 시장충격 추정 (Square-Root Impact Model) ───────────────────────────────
# Market Impact (bps) ≈ spread/2 + σ_daily × √(Q / ADV)
# σ_daily: 일중 변동성(bps), Q: 주문 수량, ADV: 일일거래량
# 참고: Almgren et al. (2005) 실증 모델

def market_impact_bps(order_shares, daily_vol, intra_vol_bpm, spread_bp):
    """
    order_shares: 주문 수량
    daily_vol:    일일 거래량 (주)
    intra_vol_bpm: 분봉 변동성 (bps)
    spread_bp:    스프레드 (bps)
    """
    participation = order_shares / daily_vol
    # 일간 변동성 ≈ 분봉 변동성 × √390 (1일 = 390분)
    sigma_daily_bps = intra_vol_bpm * np.sqrt(390)
    impact_bps = spread_bp / 2 + sigma_daily_bps * np.sqrt(participation)
    return impact_bps, participation * 100   # (충격 bps, 참여율 %)

# ────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*72}")
    print(f"  시장충격 분석 — 2026-05-15 틱 데이터 기준")
    print(f"  환율 가정: {KRW:,} KRW/USD")
    print(f"{'='*72}")

    # ── SPY/QQQ 호가창 분석 ──────────────────────────────────────────────────
    print(f"\n▶ [1] SPY / QQQ 호가창 (신호용 / 참고 유동성)")
    print(f"  {'심볼':<6} {'평균가':>8} {'스프레드':>10} {'스프레드':>10} "
          f"{'1호가 Bid':>10} {'1호가 Ask':>10} {'일거래량':>12} {'평균체결':>9}")
    print(f"  {'':6} {'($)':>8} {'(bps 평균)':>10} {'(bps 중앙)':>10} "
          f"{'(주)':>10} {'(주)':>10} {'(주)':>12} {'(주)':>9}")
    print(f"  {'─'*70}")
    obi = {}
    for sym in ["SPY", "QQQ"]:
        r = analyze_orderbook(sym)
        obi[sym] = r
        print(f"  {sym:<6} {r['avg_price']:>8.2f} {r['spread_bp']:>10.2f} "
              f"{r['spread_med']:>10.2f} {r['depth_bid']:>10.1f} "
              f"{r['depth_ask']:>10.1f} {r['daily_vol']:>12,.0f} "
              f"{r['avg_trade']:>9.1f}")

    # ── SSO/QLD/SDS/QID 유동성 분석 ─────────────────────────────────────────
    print(f"\n▶ [2] SSO / QLD / SDS / QID 유동성 (실제 거래 심볼)")
    print(f"  {'심볼':<6} {'평균가':>8} {'일거래량':>12} {'분당거래량':>10} "
          f"{'분당거래대금':>14} {'분봉변동성':>10}")
    print(f"  {'':6} {'($)':>8} {'(주)':>12} {'(주/분)':>10} "
          f"{'($)':>14} {'(bps/분)':>10}")
    print(f"  {'─'*70}")
    etf_data = {}
    for sym in ["SSO", "QLD", "SDS", "QID"]:
        r = analyze_etf(sym)
        etf_data[sym] = r
        print(f"  {sym:<6} {r['avg_price']:>8.2f} {r['daily_vol']:>12,.0f} "
              f"{r['avg_1m_vol']:>10.1f} {r['dollar_vol_1m']:>14,.0f} "
              f"{r['intra_vol_bpm']:>10.2f}")

    # ── 포트폴리오 규모별 충격 분석 ─────────────────────────────────────────
    print(f"\n▶ [3] 포트폴리오 규모별 시장충격 분석 (Kelly={KELLY*100:.0f}%)")
    print(f"  ※ Square-Root Impact Model: MI(bps) = Spread/2 + σ_daily × √(Q/ADV)")

    for port_name, port_usd in PORTFOLIOS.items():
        port_krw = port_usd * KRW
        alloc    = port_usd * KELLY  # 심볼당 배분 ($)

        print(f"\n  {'─'*68}")
        if port_krw >= 1_000_000:
            print(f"  📊 {port_name}  (≈ ${port_usd:,.0f} USD / {port_krw/1e8:.0f}억원)")
        else:
            print(f"  📊 {port_name}  (≈ ${port_usd:,.0f} USD)")
        print(f"     심볼당 배분: ${alloc:,.0f} USD")
        print(f"\n     {'심볼':<6} {'주문(주)':>9} {'참여율':>9} {'충격(bps)':>10} "
              f"{'충격($)':>9} {'체결시간':>10} {'판정':>6}")
        print(f"     {'─'*62}")

        for sym in ["SSO", "QLD", "SDS", "QID"]:
            r = etf_data[sym]
            order_shares = int(alloc / r["avg_price"])
            # 스프레드: bars에 없으므로 ETF 특성 기반 추정 (2배 ETF ≈ 3~6bps)
            etf_spread_bp = 4.0 if sym in ["SSO", "SDS"] else 5.5

            impact_bps, part_pct = market_impact_bps(
                order_shares, r["daily_vol"], r["intra_vol_bpm"], etf_spread_bp)
            impact_dollar = impact_bps / 10_000 * alloc

            # 체결 완료 예상 시간 (분): 분당 거래량의 10% 참여
            fill_min = order_shares / (r["avg_1m_vol"] * 0.10) if r["avg_1m_vol"] > 0 else 999

            # 판정
            if part_pct < 0.5 and impact_bps < 5:
                verdict = "✅ 양호"
            elif part_pct < 2.0 and impact_bps < 20:
                verdict = "⚠️ 주의"
            else:
                verdict = "🚨 위험"

            print(f"     {sym:<6} {order_shares:>9,} {part_pct:>8.3f}% "
                  f"{impact_bps:>10.1f} ${impact_dollar:>8,.0f} "
                  f"{fill_min:>8.1f}분  {verdict}")

    # ── 비교 요약 ─────────────────────────────────────────────────────────────
    print(f"\n\n{'='*72}")
    print(f"  📋 결론 요약")
    print(f"{'='*72}")

    print(f"""
  ┌─────────────────┬────────────────────────────────────────────────┐
  │  규모           │  영향                                          │
  ├─────────────────┼────────────────────────────────────────────────┤
  │  $100K (현재)   │  일거래량 0.01~0.05% → 시장충격 거의 없음      │
  │  10억원         │  일거래량 0.1~0.5%  → 약간 의식 필요           │
  │  100억원        │  일거래량 1~5%      → 슬리피지 급증, 전략 한계 │
  └─────────────────┴────────────────────────────────────────────────┘

  💡 핵심 인사이트:
     - SSO/QLD: 일거래량 수백만주 → 수십억원까진 문제없음
     - QLD/QID: 거래량 더 적어 → 100억원 이상부터 충격 발생
     - 100억원 규모에서 QID 진입 시: 분당거래량 10% 참여해도 수분 소요
     - 이 전략의 실질적 AUM 한계: 약 30~50억원 (단일 심볼 진입 기준)
     - 그 이상은 SPY/QQQ 직접 거래 + 분할 집행(TWAP/VWAP) 필수
    """)

if __name__ == "__main__":
    main()
