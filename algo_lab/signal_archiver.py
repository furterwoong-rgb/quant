"""
signal_archiver.py — 매매 신호 + 결과 아카이버

진입/청산 기록을 algo_lab/archive/signals.parquet에 누적 저장.
원본 트레이더에서 import해 사용하는 최소 침습 설계.
"""
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz

ET = pytz.timezone("America/New_York")

ARCHIVE_DIR  = Path(__file__).parent / "archive"
ARCHIVE_PATH = ARCHIVE_DIR / "signals.csv"

_COLUMNS = [
    "trade_id", "timestamp_et", "symbol", "sig", "atr_scale",
    "entry_px", "exit_px", "exit_reason", "pnl_pct", "holding_bars", "result",
]


# ── 내부 헬퍼 ──────────────────────────────────────────────────────────────────

def _load_raw() -> pd.DataFrame:
    if ARCHIVE_PATH.exists():
        return pd.read_csv(ARCHIVE_PATH, dtype={"trade_id": str})
    return pd.DataFrame(columns=_COLUMNS)


def _save(df: pd.DataFrame):
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ARCHIVE_PATH, index=False)


# ── 공개 API ───────────────────────────────────────────────────────────────────

def record_entry(sym: str, sig: float, atr_scale: float,
                 entry_px: float, now: datetime) -> str:
    """
    진입 이벤트 기록. 8자리 trade_id 반환.

    원본 트레이더 _enter() 끝에서 호출:
        trade_id = record_entry(sym, sig, atr_scale=..., entry_px=fill_px, now=now)
    """
    trade_id = str(uuid.uuid4())[:8]
    now_et   = now.astimezone(ET).strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "trade_id":     trade_id,
        "timestamp_et": now_et,
        "symbol":       sym,
        "sig":          round(float(sig), 4),
        "atr_scale":    round(float(atr_scale), 4),
        "entry_px":     round(float(entry_px), 4),
        "exit_px":      float("nan"),
        "exit_reason":  None,
        "pnl_pct":      float("nan"),
        "holding_bars": 0,
        "result":       float("nan"),
    }
    df = _load_raw()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save(df)
    return trade_id


def record_exit(trade_id: str, exit_px: float,
                exit_reason: str, holding_bars: int):
    """
    청산 이벤트 기록. trade_id로 기존 행 업데이트.

    원본 트레이더 _exit()에서 st.reset() 직전 호출:
        record_exit(st.trade_id, exit_px=fill_px,
                    exit_reason=reason, holding_bars=st.bar_count)
    """
    df   = _load_raw()
    mask = df["trade_id"] == trade_id
    if not mask.any():
        return

    entry_px = float(df.loc[mask, "entry_px"].iloc[0])
    pnl_pct  = (exit_px - entry_px) / entry_px if entry_px > 0 else 0.0
    result   = 1.0 if pnl_pct > 0 else 0.0

    df.loc[mask, "exit_px"]      = round(float(exit_px), 4)
    df.loc[mask, "exit_reason"]  = exit_reason
    df.loc[mask, "pnl_pct"]      = round(pnl_pct, 6)
    df.loc[mask, "holding_bars"] = int(holding_bars)
    df.loc[mask, "result"]       = result
    _save(df)


def load_archive() -> pd.DataFrame:
    """완료된 거래만 반환 (result가 NaN이 아닌 행)."""
    df = _load_raw()
    return df[df["result"].notna()].reset_index(drop=True)


def export_csv(date_str: str | None = None) -> Path:
    """
    아카이브를 CSV로 내보내기.
    date_str='YYYY-MM-DD' 지정 시 해당 날짜만, None이면 전체.
    반환: 저장된 CSV 경로
    """
    df = _load_raw()
    if date_str:
        df = df[df["timestamp_et"].str.startswith(date_str)]
    out = ARCHIVE_DIR / f"{date_str or 'all'}_signals.csv"
    df.to_csv(out, index=False)
    return out


# ── 단독 실행 테스트 ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from datetime import timezone

    now = datetime.now(timezone.utc)

    print("=== signal_archiver 단독 테스트 ===")

    # 진입 2건
    tid1 = record_entry("SPY", sig=0.45, atr_scale=1.1,
                         entry_px=550.00, now=now)
    tid2 = record_entry("QQQ", sig=0.31, atr_scale=0.9,
                         entry_px=470.50, now=now)
    print(f"  진입: SPY trade_id={tid1}")
    print(f"  진입: QQQ trade_id={tid2}")

    # 청산
    record_exit(tid1, exit_px=554.40, exit_reason="트레일링",   holding_bars=6)
    record_exit(tid2, exit_px=468.20, exit_reason="손절",       holding_bars=3)

    df = load_archive()
    print(f"\n완료 거래: {len(df)}건")
    print(df[["trade_id", "symbol", "sig", "pnl_pct", "result", "exit_reason"]]
          .to_string(index=False))
