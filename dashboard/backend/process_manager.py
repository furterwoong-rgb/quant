"""
process_manager.py — 라이브 트레이딩 프로세스 관리 + SSE 로그 스트리밍
caffeinate -i 로 맥 수면 방지 포함.
"""
import asyncio
import os
import signal

PYTHON   = "/opt/miniconda3/envs/quant-env/bin/python"
SCRIPT   = "spy_qqq_live_trader.py"
WORK_DIR = "/Users/woongyeol/quant_trading"

_process:    asyncio.subprocess.Process | None = None
_status:     str  = "IDLE"   # IDLE | RUNNING | STOPPED
_log_buffer: list = []
_ticker:     str  = "BOTH"


async def start_trading(ticker: str = "BOTH") -> None:
    global _process, _status, _log_buffer, _ticker
    if _status == "RUNNING":
        return
    _ticker     = ticker
    _log_buffer = []
    _process = await asyncio.create_subprocess_exec(
        "caffeinate", "-i", PYTHON, "-u", SCRIPT, ticker,
        cwd=WORK_DIR,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,  # 독립 프로세스 그룹 → killpg로 caffeinate+python 동시 종료
    )
    _status = "RUNNING"
    asyncio.create_task(_read_logs())


async def stop_trading() -> None:
    global _process, _status
    if _process and _status == "RUNNING":
        try:
            # caffeinate이 부모이므로 프로세스 그룹 전체에 SIGTERM 전달
            pgid = os.getpgid(_process.pid)
            os.killpg(pgid, signal.SIGTERM)
            await asyncio.wait_for(_process.wait(), timeout=15)
        except (ProcessLookupError, asyncio.TimeoutError, PermissionError):
            try:
                # 그룹 kill 실패 시 직접 SIGKILL
                pgid = os.getpgid(_process.pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    _process.kill()
                except ProcessLookupError:
                    pass
            try:
                await asyncio.wait_for(_process.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass
    _status = "STOPPED"


async def _read_logs() -> None:
    global _log_buffer, _status
    try:
        async for line in _process.stdout:
            decoded = line.decode("utf-8", errors="replace").rstrip()
            _log_buffer.append(decoded)
            if len(_log_buffer) > 1000:
                _log_buffer.pop(0)
    except Exception:
        pass
    finally:
        _status = "STOPPED"


async def stream_logs():
    """SSE 제너레이터 — 각 클라이언트가 독립적으로 소비."""
    last = 0
    try:
        while True:
            if last < len(_log_buffer):
                for line in _log_buffer[last:]:
                    yield f"data: {line}\n\n"
                last = len(_log_buffer)
            if _status != "RUNNING" and last >= len(_log_buffer):
                yield "data: [STREAM END]\n\n"
                break
            yield ": heartbeat\n\n"
            await asyncio.sleep(0.1)
    except GeneratorExit:
        pass


def get_status() -> dict:
    return {"status": _status, "ticker": _ticker}
