#!/usr/bin/env python3
"""
memory_report.py — 현재 메모리 상황을 Claude 붙여넣기용으로 출력
사용법: python3 analysis/memory_report.py
"""

import subprocess
import psutil
import datetime
import pytz

KST = pytz.timezone("Asia/Seoul")
now = datetime.datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")


def bytes_to_gb(b):
    return round(b / 1024**3, 2)


def bytes_to_mb(b):
    return round(b / 1024**2, 1)


# ── 1. 시스템 메모리 ────────────────────────────────────────────
vm = psutil.virtual_memory()
swap = psutil.swap_memory()

total_gb  = bytes_to_gb(vm.total)
used_gb   = bytes_to_gb(vm.used)
avail_gb  = bytes_to_gb(vm.available)
used_pct  = vm.percent
swap_used = bytes_to_gb(swap.used)
swap_total= bytes_to_gb(swap.total)

# ── 2. macOS memory pressure (vm_stat) ────────────────────────
try:
    vm_stat = subprocess.check_output(["vm_stat"], text=True)
    page_size = 16384  # M-series: 16KB
    lines = {l.split(":")[0].strip(): int(l.split(":")[1].strip().rstrip("."))
             for l in vm_stat.strip().split("\n")[1:] if ":" in l}
    wired    = bytes_to_gb(lines.get("Pages wired down", 0) * page_size)
    active   = bytes_to_gb(lines.get("Pages active", 0) * page_size)
    inactive = bytes_to_gb(lines.get("Pages inactive", 0) * page_size)
    compressed = bytes_to_gb(lines.get("Pages occupied by compressor", 0) * page_size)
except Exception:
    wired = active = inactive = compressed = 0.0

# ── 3. memory_pressure 커맨드 ──────────────────────────────────
try:
    pressure_raw = subprocess.check_output(
        ["memory_pressure"], text=True, stderr=subprocess.DEVNULL
    ).strip().split("\n")
    pressure_line = next((l for l in pressure_raw if "System memory pressure" in l), "")
    pressure_level = pressure_line.split(":")[-1].strip() if pressure_line else "Unknown"
except Exception:
    pressure_level = "Unknown"

# ── 4. 상위 프로세스 (메모리 기준) ────────────────────────────
procs = []
for p in psutil.process_iter(["pid", "name", "memory_info", "cpu_percent", "username"]):
    try:
        mi = p.info["memory_info"]
        if mi and mi.rss > 50 * 1024**2:  # 50MB 이상만
            procs.append({
                "pid":  p.info["pid"],
                "name": p.info["name"][:28],
                "rss":  bytes_to_mb(mi.rss),
                "cpu":  p.info["cpu_percent"],
            })
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

procs.sort(key=lambda x: x["rss"], reverse=True)
top_procs = procs[:15]

# ── 5. Python 프로세스 상세 ────────────────────────────────────
py_procs = []
for p in psutil.process_iter(["pid", "name", "memory_info", "cpu_percent", "cmdline"]):
    try:
        if "python" in p.info["name"].lower():
            mi = p.info["memory_info"]
            cmd = " ".join(p.info.get("cmdline") or [])[-60:]
            py_procs.append({
                "pid":  p.info["pid"],
                "rss":  bytes_to_mb(mi.rss),
                "cpu":  p.info["cpu_percent"],
                "cmd":  cmd,
            })
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

py_procs.sort(key=lambda x: x["rss"], reverse=True)

# ── 출력 ──────────────────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════════════════╗
║         Memory Report — {now}        ║
╚══════════════════════════════════════════════════════╝

[시스템 메모리]
  총 RAM       : {total_gb} GB
  사용 중      : {used_gb} GB  ({used_pct}%)
  사용 가능    : {avail_gb} GB
  Swap 사용    : {swap_used} / {swap_total} GB

[메모리 세부]
  Wired        : {wired} GB  (커널/항상 점유)
  Active       : {active} GB
  Inactive     : {inactive} GB
  Compressed   : {compressed} GB

[macOS Memory Pressure]
  → {pressure_level}

[상위 15 프로세스 (RSS 기준)]
  {"PID":>7}  {"Name":<28}  {"RSS(MB)":>8}  {"CPU%":>5}""")

for p in top_procs:
    print(f"  {p['pid']:>7}  {p['name']:<28}  {p['rss']:>8.1f}  {p['cpu']:>5.1f}%")

print(f"""
[Python 프로세스]
  {"PID":>7}  {"RSS(MB)":>8}  {"CPU%":>5}  Command""")
for p in py_procs:
    print(f"  {p['pid']:>7}  {p['rss']:>8.1f}  {p['cpu']:>5.1f}%  ...{p['cmd']}")

print("""
──────────────────────────────────────────────────────
위 내용을 Claude 채팅에 붙여넣으면 됩니다.
""")
