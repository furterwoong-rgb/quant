"""
scheduler.py — KST 기준 자동 매매 스케줄러
22:28 자동 시작 (BOTH), 05:00 자동 종료.
"""
import asyncio
import os
import sys

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

KST = pytz.timezone("Asia/Seoul")
ET = pytz.timezone("America/New_York")

# Daily QQQ option-state snapshot (greeks + OI) — builds gamma history for backtesting.
SNAPSHOT_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "analysis", "snapshot_option_state.py",
)

_scheduler: AsyncIOScheduler | None = None
_enabled:   bool = True
_start_fn   = None
_stop_fn    = None


def setup_scheduler(start_fn, stop_fn) -> None:
    global _scheduler, _start_fn, _stop_fn
    _start_fn = start_fn
    _stop_fn  = stop_fn

    _scheduler = AsyncIOScheduler(timezone=KST)

    async def _auto_start():
        await start_fn("BOTH")

    async def _auto_stop():
        await stop_fn()

    _scheduler.add_job(
        _auto_start,
        CronTrigger(day_of_week="mon-fri", hour=22, minute=28, timezone=KST),
        id="auto_start",
        replace_existing=True,
    )
    _scheduler.add_job(
        _auto_stop,
        CronTrigger(day_of_week="tue-sat", hour=5, minute=0, timezone=KST),
        id="auto_stop",
        replace_existing=True,
    )

    async def _snapshot_option_state():
        """Capture QQQ greeks+OI after the US close. Subprocess => no thread, never blocks the loop."""
        log_dir = os.path.join(os.path.dirname(SNAPSHOT_SCRIPT), os.pardir, "logs", "option_cache")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "snapshot.log"), "ab") as logf:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, SNAPSHOT_SCRIPT, "--force", stdout=logf, stderr=logf,
            )
            await proc.wait()

    _scheduler.add_job(
        _snapshot_option_state,
        CronTrigger(day_of_week="mon-fri", hour=16, minute=5, timezone=ET),
        id="option_snapshot",
        replace_existing=True,
    )
    _scheduler.start()


def toggle(enabled: bool) -> None:
    global _enabled
    _enabled = enabled
    if _scheduler is None:
        return
    for job_id in ("auto_start", "auto_stop"):
        if enabled:
            _scheduler.resume_job(job_id)
        else:
            _scheduler.pause_job(job_id)


def get_info() -> dict:
    if _scheduler is None:
        return {"enabled": _enabled, "next_start": "-", "next_stop": "-"}

    jobs = {j.id: j for j in _scheduler.get_jobs()}

    def fmt(job_id: str) -> str:
        j = jobs.get(job_id)
        if j and j.next_run_time:
            return j.next_run_time.astimezone(KST).strftime("%Y-%m-%d %H:%M KST")
        return "-"

    return {
        "enabled":    _enabled,
        "next_start": fmt("auto_start"),
        "next_stop":  fmt("auto_stop"),
    }
