# algo_lab — Dynamic Kelly 시스템

원본 `spy_qqq_live_trader.py`를 **수정하지 않고** 신호 아카이빙 + Dynamic Kelly를 붙이는 애드온 레이어.

---

## 파일 구조

```
algo_lab/
├── signal_archiver.py   진입/청산 기록 → parquet 누적
├── prob_map.py          2D KDE P(win|sig, atr_scale)
├── kelly_sizer.py       Dynamic Kelly 포지션 사이저
├── README.md            (이 파일)
└── archive/             자동 생성
    ├── signals.parquet  누적 아카이브
    ├── YYYY-MM-DD_signals.csv  (export_csv 호출 시 생성)
    └── prob_map.pkl     학습된 ProbMap
```

---

## 단계별 운영 모드

| N (완료 거래 수) | 모드 | Kelly f |
|---|---|---|
| 0 ~ 29 | prior | 0.025 고정 (원본과 동일) |
| 30 ~ 99 | KDE-early | Dynamic (half-Kelly, clip 0.01~0.05) |
| 100+ | dynamic | Dynamic (full 운영) |

---

## 원본 트레이더 통합 (4군데 최소 침습)

### A. `PositionState.__init__()` 끝에 추가

```python
self.trade_id:  str = ""
self.bar_count: int = 0
```

### B. `_enter()` — `log_trade_csv` 호출 직후

```python
from algo_lab.signal_archiver import record_entry
st.trade_id  = record_entry(
    sym, sig,
    atr_scale=self.vol_scaler.scale(sym),
    entry_px=fill_px, now=now,
)
st.bar_count = 0
```

### C. `_exit()` — `st.reset()` 직전

```python
from algo_lab.signal_archiver import record_exit
record_exit(st.trade_id, exit_px=fill_px,
            exit_reason=reason, holding_bars=st.bar_count)
```

### D. `_maybe_process_bar()` — `_step()` 호출 전

```python
if self.states[sym].direction == "long":
    self.states[sym].bar_count += 1
```

---

## Kelly 통합 (선택, 데이터 30건 후)

원본 `_enter()` 안의 한 줄 교체:

```python
# 변경 전
notional = equity * KELLY

# 변경 후
from algo_lab.kelly_sizer import KellySizer
# (트레이더 __init__에서 self.kelly_sizer = KellySizer() 생성)
notional, _info = self.kelly_sizer.notional(equity, sig, self.vol_scaler.scale(sym))
if notional <= 0:
    log.info(f"  [{sym}] Kelly 진입 차단: {_info}")
    return
```

장 시작 전 `_maybe_reset_day()` 내에서 `self.kelly_sizer.refresh()` 호출.

---

## 단독 실행 / 테스트

```bash
# 환경 활성화
conda activate quant-env

# 각 모듈 단독 테스트 (quant_trading/ 디렉토리에서 실행)
python algo_lab/signal_archiver.py
python algo_lab/prob_map.py
python algo_lab/kelly_sizer.py
```

---

## 작업 로드맵

- [x] signal_archiver.py
- [x] prob_map.py
- [x] kelly_sizer.py
- [ ] 원본 트레이더 A~D hook 통합
- [ ] 실전 데이터 30건 수집
- [ ] prob_map 첫 학습 + 3D 지형도 확인
- [ ] Kelly 통합 → Dynamic 모드 전환
- [ ] 백테스트: Dynamic Kelly vs 고정 Kelly 비교
