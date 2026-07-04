"""
data_manager.py — PNG / CSV 파일 스캔
결과 PNG : logs/YYYY/MM/DD/trades_YYYYMMDD_candle.png
거래 CSV : logs/YYYY/MM/DD/spy_qqq_trades.csv
백테스트 : backtesting/results/*.png  +  logs/backtest_*.png
"""
import glob
import os
from pathlib import Path

LOGS_DIR   = "/Users/woongyeol/quant_trading/logs"
BT_RES_DIR = "/Users/woongyeol/quant_trading/backtesting/results"


# ── 날짜 목록 ──────────────────────────────────────────────────────────────

def get_trade_dates() -> list[str]:
    """
    logs/YYYY/MM/DD/ 구조에서 날짜 폴더를 읽어 YYYYMMDD 문자열 목록 반환 (최신순).
    """
    pattern = os.path.join(LOGS_DIR, "*", "*", "*")
    dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    dates = []
    for d in dirs:
        rel = d.replace(LOGS_DIR + os.sep, "").replace(LOGS_DIR + "/", "")
        parts = rel.replace("\\", "/").split("/")
        if len(parts) == 3:
            try:
                yyyymmdd = parts[0] + parts[1] + parts[2]
                if len(yyyymmdd) == 8 and yyyymmdd.isdigit():
                    dates.append(yyyymmdd)
            except Exception:
                pass
    return sorted(set(dates), reverse=True)


# ── 결과 경로 ──────────────────────────────────────────────────────────────

def get_candle_png(yyyymmdd: str) -> str:
    y, m, d = yyyymmdd[:4], yyyymmdd[4:6], yyyymmdd[6:]
    return os.path.join(LOGS_DIR, y, m, d, f"trades_{yyyymmdd}_candle.png")


def get_trades_csv(yyyymmdd: str) -> str:
    y, m, d = yyyymmdd[:4], yyyymmdd[4:6], yyyymmdd[6:]
    return os.path.join(LOGS_DIR, y, m, d, "spy_qqq_trades.csv")


# ── 백테스트 결과 이미지 ───────────────────────────────────────────────────

def get_backtest_pngs() -> list[dict]:
    """
    backtesting/results/*.png  +  logs/backtest_*.png
    mtime 내림차순 정렬, 중복 파일명 제거.
    """
    seen: set[str] = set()
    results: list[dict] = []

    sources = [
        (BT_RES_DIR, "*.png"),
        (LOGS_DIR,   "backtest_*.png"),
    ]
    for directory, pattern in sources:
        for f in glob.glob(os.path.join(directory, pattern)):
            name = os.path.basename(f)
            if name not in seen:
                seen.add(name)
                results.append({
                    "filename": name,
                    "path":     f,
                    "mtime":    os.path.getmtime(f),
                })

    return sorted(results, key=lambda x: x["mtime"], reverse=True)
