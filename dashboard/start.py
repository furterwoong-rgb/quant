"""
Trading Dashboard — 진입점
Usage: python start.py
"""
import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser

sys.path.insert(0, os.path.dirname(__file__))

import uvicorn


def free_port(port: int) -> None:
    """해당 포트를 사용 중인 프로세스를 강제 종료."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True
        )
        pids = result.stdout.strip().split()
        for pid in pids:
            if pid:
                os.kill(int(pid), signal.SIGKILL)
                print(f"  ↳ 기존 프로세스 종료 (PID {pid})")
        if pids:
            time.sleep(0.5)
    except Exception:
        pass

URL = "http://127.0.0.1:8765"


def open_browser():
    time.sleep(1.5)
    webbrowser.open(URL)


if __name__ == "__main__":
    free_port(8765)
    print("=" * 50)
    print("  Trading Dashboard")
    print(f"  URL: {URL}")
    print("  Ctrl+C 로 종료")
    print("=" * 50)
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
        app_dir=os.path.dirname(__file__),
        log_level="info",
    )
