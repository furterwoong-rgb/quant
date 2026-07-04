"""
gen_all_charts.py — 전체 날짜 PNG 일괄 재생성 스크립트
"""
import sys
sys.path.insert(0, "/Users/woongyeol/quant_trading")

from pathlib import Path
from analysis.live_chart import draw_candle_chart
from config.settings import ALPACA_API_KEY as API_KEY, ALPACA_SECRET_KEY as SECRET_KEY

DATES = [
    ("2026-05-18", "18"),
    ("2026-05-19", "19"),
    ("2026-05-20", "20"),
    ("2026-05-21", "21"),
    ("2026-05-22", "22"),
    ("2026-05-26", "26"),
    ("2026-05-27", "27"),
    ("2026-05-28", "28"),
]

BASE = Path("/Users/woongyeol/quant_trading/logs/2026/05")

for date_str, day in DATES:
    csv_path = BASE / day / "spy_qqq_trades.csv"
    if not csv_path.exists():
        print(f"  [skip] {date_str} — CSV 없음")
        continue
    # 원본 위치에 덮어쓰기
    out_path = BASE / day / f"trades_2026{day.zfill(2).replace(day, '05' + day)}_candle.png"
    # 날짜 포맷: trades_20260518_candle.png
    yyyymmdd = date_str.replace("-", "")
    out_path = BASE / day / f"trades_{yyyymmdd}_candle.png"
    print(f"\n=== {date_str} → {out_path.name} ===")
    try:
        draw_candle_chart(
            csv_path  = csv_path,
            date_str  = date_str,
            out_path  = out_path,
            api_key   = API_KEY,
            secret_key= SECRET_KEY,
            sym       = "QQQ",
        )
    except Exception as e:
        print(f"  [ERROR] {e}")

print("\n완료!")
