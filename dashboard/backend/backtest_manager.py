"""
backtest_manager.py — SQLite 버전 관리 + 임시 파일 실행 + SSE 로그 스트리밍
"""
import asyncio
import os
import sqlite3
import tempfile
import uuid
from datetime import datetime

PYTHON   = "/opt/miniconda3/envs/quant-env/bin/python"
WORK_DIR = "/Users/woongyeol/quant_trading"
DB_PATH  = "/Users/woongyeol/quant_trading/dashboard/trading.db"

# sys.path 헤더 — 백테스트 코드가 quant_trading 모듈을 import 할 수 있게
_SYS_PATH_HEADER = f"""\
import sys as _bt_sys
_bt_sys.path.insert(0, r"{WORK_DIR}")
del _bt_sys

"""

# ── DB 초기화 ──────────────────────────────────────────────────────────────

def init_db() -> None:
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS backtest_versions (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            description TEXT,
            created_at  TEXT NOT NULL,
            code        TEXT NOT NULL
        )
    """)
    con.commit()
    con.close()


# ── 버전 CRUD ──────────────────────────────────────────────────────────────

def load_versions() -> list:
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT id, name, description, created_at FROM backtest_versions ORDER BY created_at DESC"
    ).fetchall()
    con.close()
    return [
        {"id": r[0], "name": r[1], "description": r[2], "created_at": r[3]}
        for r in rows
    ]


def get_version_code(vid: str) -> str:
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT code FROM backtest_versions WHERE id=?", (vid,)
    ).fetchone()
    con.close()
    return row[0] if row else ""


def save_version(name: str, description: str, code: str) -> str:
    vid = str(uuid.uuid4())
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO backtest_versions VALUES (?,?,?,?,?)",
        (vid, name, description or "", datetime.now().isoformat(timespec="seconds"), code),
    )
    con.commit()
    con.close()
    return vid


def delete_version(vid: str) -> None:
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM backtest_versions WHERE id=?", (vid,))
    con.commit()
    con.close()


# ── 실행 상태 ──────────────────────────────────────────────────────────────

_bt_process:    asyncio.subprocess.Process | None = None
_bt_status:     str  = "IDLE"   # IDLE | RUNNING | DONE
_bt_log_buffer: list = []


async def run_backtest(code: str) -> None:
    global _bt_process, _bt_status, _bt_log_buffer
    _bt_log_buffer = []
    _bt_status     = "RUNNING"

    tmp = tempfile.NamedTemporaryFile(
        suffix=".py", delete=False, dir="/tmp",
        mode="w", encoding="utf-8",
    )
    tmp.write(_SYS_PATH_HEADER + code)
    tmp.close()

    _bt_process = await asyncio.create_subprocess_exec(
        PYTHON, tmp.name,
        cwd=WORK_DIR,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    asyncio.create_task(_read_bt_logs(tmp.name))


async def _read_bt_logs(tmp_path: str) -> None:
    global _bt_status, _bt_log_buffer
    try:
        async for line in _bt_process.stdout:
            _bt_log_buffer.append(line.decode("utf-8", errors="replace").rstrip())
        await _bt_process.wait()
    except Exception as e:
        _bt_log_buffer.append(f"[ERROR] {e}")
    finally:
        _bt_status = "DONE"
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


async def stream_bt_logs():
    """SSE 제너레이터."""
    last = 0
    try:
        while True:
            if last < len(_bt_log_buffer):
                for line in _bt_log_buffer[last:]:
                    yield f"data: {line}\n\n"
                last = len(_bt_log_buffer)
            if _bt_status in ("DONE", "IDLE") and last >= len(_bt_log_buffer):
                yield "data: [STREAM END]\n\n"
                break
            yield ": heartbeat\n\n"
            await asyncio.sleep(0.1)
    except GeneratorExit:
        pass


def get_bt_status() -> str:
    return _bt_status
