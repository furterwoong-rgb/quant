"""
download_tick_cache.py — Alpaca tick 데이터 병렬 다운로더

병목: HTTP/JSON 페이지네이션 (~13,000 records/sec, 플랜 무관)
해결: 여러 날을 동시에 받아 전체 시간 단축

예상 시간 (QQQ 기준):
  workers=5 : ~104분
  workers=10: ~ 52분

실행:
    python algo_lab/download_tick_cache.py --start 2026-01-02 --end 2026-05-28
    python algo_lab/download_tick_cache.py --start 2026-01-02 --end 2026-05-28 --workers 10
    python algo_lab/download_tick_cache.py --dry-run
"""
import argparse
import pickle
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytz

ROOT      = Path(__file__).parent.parent
CACHE_DIR = ROOT / "logs" / "tick_cache"
sys.path.insert(0, str(ROOT))

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (StockBarsRequest, StockQuotesRequest,
                                   StockTradesRequest)
from alpaca.data.timeframe import TimeFrame
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY

ET = pytz.timezone("America/New_York")

# ── 공유 상태 (thread-safe) ────────────────────────────────────────────────────
_lock         = threading.Lock()
_completed    = 0
_skipped      = 0
_failed_days: list[str] = []


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def _bar(pct: float, width: int = 25) -> str:
    filled = int(width * pct)
    return "█" * filled + "░" * (width - filled)

def _fmt_sec(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s   = divmod(rem, 60)
    if h:   return f"{h}h {m:02d}m {s:02d}s"
    if m:   return f"{m}m {s:02d}s"
    return f"{s}s"

def _print(msg: str):
    with _lock:
        print(msg, flush=True)


# ── 거래일 목록 ────────────────────────────────────────────────────────────────

def get_trading_days(start: str, end: str) -> list[str]:
    return pd.bdate_range(start, end).strftime("%Y-%m-%d").tolist()

def already_cached(sym: str, ds: str) -> bool:
    return all(
        (CACHE_DIR / f"{sym}_{ds}_{tag}.pkl").exists()
        for tag in ("quotes", "trades", "bars")
    )


# ── 단일 파일 fetch + save ─────────────────────────────────────────────────────

def _date_range_utc(ds: str):
    d = datetime.strptime(ds, "%Y-%m-%d").date()
    s = ET.localize(datetime(d.year, d.month, d.day, 9, 30)).astimezone(pytz.utc)
    e = ET.localize(datetime(d.year, d.month, d.day, 16, 0)).astimezone(pytz.utc)
    return s, e

def _fetch_one(client, sym: str, ds: str, tag: str) -> tuple[str, int, float]:
    """
    한 종류 파일 다운로드 → pkl 저장.
    Returns: (tag, n_rows, elapsed_sec)
    """
    cache = CACHE_DIR / f"{sym}_{ds}_{tag}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            df = pickle.load(f)
        return tag, len(df), 0.0

    s_utc, e_utc = _date_range_utc(ds)

    fetch_map = {
        "quotes": lambda: client.get_stock_quotes(
            StockQuotesRequest(symbol_or_symbols=sym, start=s_utc, end=e_utc, feed="sip")).df,
        "trades": lambda: client.get_stock_trades(
            StockTradesRequest(symbol_or_symbols=sym, start=s_utc, end=e_utc, feed="sip")).df,
        "bars":   lambda: client.get_stock_bars(
            StockBarsRequest(symbol_or_symbols=sym, start=s_utc, end=e_utc,
                             timeframe=TimeFrame.Minute, feed="sip")).df,
    }

    for attempt in range(1, 4):
        try:
            t0 = time.time()
            df = fetch_map[tag]()
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(sym, level="symbol")
            df.index = pd.DatetimeIndex(df.index).tz_convert(ET)
            elapsed = time.time() - t0

            if len(df) == 0:
                return tag, 0, elapsed

            with open(cache, "wb") as f:
                pickle.dump(df, f)
            return tag, len(df), elapsed

        except Exception as e:
            if attempt < 3:
                time.sleep(10 * attempt)
            else:
                raise RuntimeError(f"{ds} {tag} 실패: {e}") from e


# ── 하루치 다운로드 (quotes + trades + bars 동시) ─────────────────────────────

def download_day(client, sym: str, ds: str,
                 day_idx: int, total: int,
                 wall_start: float) -> bool:
    """
    한 날짜의 quotes/trades/bars를 스레드 풀로 동시 다운로드.
    Returns: True=데이터 있음, False=휴장일
    """
    t0 = time.time()
    _print(f"  ┌ [{day_idx:3d}/{total}] {ds}  시작")

    # quotes / trades / bars 동시 실행
    results = {}
    errors  = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(_fetch_one, client, sym, ds, tag): tag
                for tag in ("quotes", "trades", "bars")}
        for fut in as_completed(futs):
            tag = futs[fut]
            try:
                t, n, sec = fut.result()
                results[t] = (n, sec)
                mb = (CACHE_DIR / f"{sym}_{ds}_{tag}.pkl").stat().st_size / 1_048_576 \
                     if (CACHE_DIR / f"{sym}_{ds}_{tag}.pkl").exists() else 0
                cached_mark = "(캐시)" if sec == 0.0 else f"{sec:.0f}s"
                _print(f"  │   {tag:<7} {n:>9,}행  {mb:>5.0f}MB  {cached_mark}")
            except Exception as e:
                errors.append(str(e))
                _print(f"  │   {tag:<7} ❌ {e}")

    day_sec   = time.time() - t0
    is_empty  = results.get("quotes", (0,))[0] == 0

    # 휴장일 빈 파일 정리
    if is_empty:
        for tag in ("quotes", "trades", "bars"):
            p = CACHE_DIR / f"{sym}_{ds}_{tag}.pkl"
            if p.exists() and p.stat().st_size < 2000:
                p.unlink()

    global _completed, _skipped
    with _lock:
        if is_empty:
            _skipped += 1
        else:
            _completed += 1
        comp, skip = _completed, _skipped

    # 전체 진행률 출력
    done_days = comp + skip
    pct       = done_days / total
    elapsed   = time.time() - wall_start
    avg       = elapsed / done_days if done_days > 0 else 0
    eta       = avg * (total - done_days)

    status = "휴장" if is_empty else f"완료 {day_sec:.0f}s"
    _print(
        f"  └ [{day_idx:3d}/{total}] {ds}  {status}\n"
        f"    전체 [{_bar(pct)}] {pct:5.1%}  "
        f"완료 {comp} / 휴장 {skip} / 실패 {len(_failed_days)}  "
        f"경과 {_fmt_sec(elapsed)}  ETA {_fmt_sec(eta)}"
    )
    return not is_empty


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym",     default="QQQ")
    parser.add_argument("--start",   default=None)
    parser.add_argument("--end",     default=None)
    parser.add_argument("--days",    type=int, default=60)
    parser.add_argument("--workers", type=int, default=5,
                        help="동시 다운로드 날짜 수 (기본 5, 최대 권장 10)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    end_str = args.end or (date.today() - timedelta(days=1)).isoformat()
    if args.start:
        start_str = args.start
    else:
        end_dt    = datetime.strptime(end_str, "%Y-%m-%d").date()
        start_str = (end_dt - timedelta(days=int(args.days * 1.5))).isoformat()

    all_days = get_trading_days(start_str, end_str)
    todo     = [d for d in all_days if not already_cached(args.sym, d)]
    if not args.start and len(todo) > args.days:
        todo = todo[-args.days:]

    cached_days = sorted(
        p.name.split(f"{args.sym}_")[1].split("_quotes")[0]
        for p in CACHE_DIR.glob(f"{args.sym}_*_quotes.pkl")
    )

    # 시간 예측 (QQQ quotes ~340s/일 기준)
    est_min = len(todo) * 340 / args.workers / 60

    print(f"\n{'='*60}")
    print(f"  Tick Cache 병렬 다운로더  |  {args.sym}")
    print(f"{'='*60}")
    print(f"  이미 캐시  : {len(cached_days)}일")
    print(f"  다운로드   : {len(todo)}일  ({todo[0] if todo else '-'} ~ {todo[-1] if todo else '-'})")
    print(f"  병렬 workers: {args.workers}개")
    print(f"  예상 용량  : ~{len(todo) * 0.34:.1f}GB")
    print(f"  예상 시간  : ~{est_min:.0f}분  ({len(todo)}일 ÷ {args.workers} workers × 340s/일)")
    print(f"  RAM 사용   : ~{args.workers * 0.34:.1f}GB (16GB 중)")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("[ dry-run — 실제 다운로드 없음 ]")
        for d in todo:
            print(f"  {d}")
        return

    if not todo:
        print("모든 날짜 캐시 완료.")
        return

    # 병렬 다운로드 실행
    # StockHistoricalDataClient는 thread-safe (REST client)
    # workers별로 client 인스턴스 분리 (연결 충돌 방지)
    clients = [
        StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        for _ in range(args.workers)
    ]

    wall_start = time.time()

    def _worker(idx_ds):
        idx, ds = idx_ds
        client  = clients[idx % args.workers]
        try:
            return download_day(client, args.sym, ds,
                                idx + 1, len(todo), wall_start)
        except Exception as e:
            with _lock:
                _failed_days.append(ds)
            _print(f"  ✗ {ds} 전체 실패: {e}")
            return False

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(_worker, enumerate(todo)))

    # 최종 요약
    total_elapsed = time.time() - wall_start
    total_cached  = len(list(CACHE_DIR.glob(f"{args.sym}_*_quotes.pkl")))
    cache_gb      = sum(p.stat().st_size for p in CACHE_DIR.glob("*.pkl")) / 1_073_741_824

    print(f"\n{'='*60}")
    print(f"  완료  총 {_fmt_sec(total_elapsed)}")
    print(f"  다운로드 {_completed}일  |  휴장 {_skipped}일  |  실패 {len(_failed_days)}일")
    if _failed_days:
        print(f"  실패 날짜: {_failed_days}")
    print(f"  총 캐시: {total_cached}일  |  용량: {cache_gb:.2f}GB")
    print(f"\n  ProbMap 재학습:")
    first = cached_days[0] if cached_days else (todo[0] if todo else "")
    print(f"  python algo_lab/proto_bar_probmap.py {args.sym} {first} {end_str}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
