"""
main.py — FastAPI 앱 + 모든 라우터
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import backend.backtest_manager as bm
import backend.data_manager as dm
import backend.process_manager as pm
import backend.scheduler as sc

FRONTEND = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")

SSE_HEADERS = {
    "Cache-Control":    "no-cache",
    "X-Accel-Buffering": "no",
    "Connection":        "keep-alive",
}


# ── 앱 수명 주기 ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    bm.init_db()
    sc.setup_scheduler(pm.start_trading, pm.stop_trading)
    yield


app = FastAPI(title="Trading Dashboard", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════
# 프론트엔드
# ══════════════════════════════════════════════════════════════════════════

@app.get("/")
def index():
    return FileResponse(FRONTEND)


# ══════════════════════════════════════════════════════════════════════════
# 라이브 트레이딩
# ══════════════════════════════════════════════════════════════════════════

class StartBody(BaseModel):
    ticker: str = "BOTH"


@app.post("/api/trading/start")
async def trading_start(body: StartBody):
    await pm.start_trading(body.ticker)
    return {"ok": True}


@app.post("/api/trading/stop")
async def trading_stop():
    await pm.stop_trading()
    return {"ok": True}


@app.get("/api/trading/status")
def trading_status():
    return {**pm.get_status(), "schedule": sc.get_info()}


@app.get("/api/trading/logs/stream")
async def trading_logs():
    return StreamingResponse(
        pm.stream_logs(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.post("/api/trading/schedule/toggle")
async def schedule_toggle(body: dict):
    sc.toggle(bool(body.get("enabled", True)))
    return {"ok": True, "schedule": sc.get_info()}


# ══════════════════════════════════════════════════════════════════════════
# 결과
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/results")
def results_list():
    return {"dates": dm.get_trade_dates()}


@app.get("/api/results/{date}/image")
def results_image(date: str):
    path = dm.get_candle_png(date)
    if not os.path.exists(path):
        return {"error": "not found"}
    return FileResponse(path, media_type="image/png")


@app.get("/api/results/{date}/csv")
def results_csv(date: str):
    path = dm.get_trades_csv(date)
    if not os.path.exists(path):
        return {"error": "not found"}
    return FileResponse(
        path,
        media_type="text/csv",
        filename=f"trades_{date}.csv",
        headers={"Content-Disposition": f'attachment; filename="trades_{date}.csv"'},
    )


# ══════════════════════════════════════════════════════════════════════════
# 백테스트 버전 관리
# ══════════════════════════════════════════════════════════════════════════

class VersionBody(BaseModel):
    name: str
    description: str = ""
    code: str


class RunBody(BaseModel):
    code: str


@app.get("/api/backtest/versions")
def bt_versions():
    return {"versions": bm.load_versions()}


@app.post("/api/backtest/versions")
def bt_save(body: VersionBody):
    vid = bm.save_version(body.name, body.description, body.code)
    return {"ok": True, "id": vid}


@app.get("/api/backtest/versions/{vid}")
def bt_get(vid: str):
    code = bm.get_version_code(vid)
    if not code:
        return {"error": "not found"}
    return {"code": code}


@app.delete("/api/backtest/versions/{vid}")
def bt_delete(vid: str):
    bm.delete_version(vid)
    return {"ok": True}


# ══════════════════════════════════════════════════════════════════════════
# 백테스트 실행
# ══════════════════════════════════════════════════════════════════════════

@app.post("/api/backtest/run")
async def bt_run(body: RunBody):
    await bm.run_backtest(body.code)
    return {"ok": True}


@app.get("/api/backtest/logs/stream")
async def bt_logs():
    return StreamingResponse(
        bm.stream_bt_logs(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.get("/api/backtest/status")
def bt_status():
    return {"status": bm.get_bt_status()}


# ══════════════════════════════════════════════════════════════════════════
# 백테스트 결과
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/backtest/results")
def bt_results():
    files = dm.get_backtest_pngs()
    # path 필드는 클라이언트에 노출하지 않음
    return {"files": [{"filename": f["filename"], "mtime": f["mtime"]} for f in files]}


@app.get("/api/backtest/results/{filename}")
def bt_result_image(filename: str):
    for item in dm.get_backtest_pngs():
        if item["filename"] == filename:
            return FileResponse(item["path"], media_type="image/png")
    return {"error": "not found"}
