# 모델 전환 Control Primitive 안전 범위

상태: **PARTIAL / production 비활성**

이 문서는 `cogni_demo/model_switch.py`가 현재 보장하는 제어 평면의 범위와 아직
보장하지 않는 범위를 구분한다. 이 primitive는 GPU나 실제 모델을 직접 다루지 않으며,
제품의 모델 선택 UI·서버 엔드포인트에도 연결되어 있지 않다.

## 현재 보장하는 항목

- 이전 worker의 unload acknowledgement와 독립 memory-release probe가 모두 성공하기
  전에는 이전 모델이나 후보 모델을 다시 생성하지 않는다. stop 예외 또는 시간 초과는
  즉시 safe mode로 끝난다.
- admission은 단순한 `open=True`가 아니다. 모든 요청은 `AtomicAdmissionGate`에서 현재
  slot generation, binding, worker generation, GPU lease id/epoch를 다시 검증한 뒤 정확한
  runtime snapshot을 pin한다.
- gate close와 request pin 획득은 한쪽만 승리한다. 전환 drain은 닫힌 gate에서 이전
  slot generation의 모든 request pin이 반환될 때까지 기다린다.
- factory는 side-effect-free `prepare`로 정리 handle을 먼저 반환한 뒤 `materialize`한다.
  미완료 경로에는 `abort` 증거를, 성공적인 runtime 이전에는 `dispose` 증거를 요구한다.
- runtime slot CAS는 injected runtime/lease callback을 slot lock 밖에서 실행한다. 중첩
  publication이 발생하면 외부 CAS가 이를 덮어쓰지 않고 실패한다.
- safe mode 판정은 maintenance callback의 자기 보고만 신뢰하지 않는다. 별도의 원자 gate
  readback으로 admission 차단 상태를 다시 확인한다.

## 명시적인 한계

- `cooperative_only = true`: runtime stop, healthcheck, memory probe, maintenance callback 같은
  in-process port가 timeout을 무시하고 반환하지 않으면 Python 제어 평면이 강제 종료할 수
  없다. snapshot은 이 한계를 항상 노출한다.
- `crash_journal = false`: unload, CAS, admission 재설정 사이에 프로세스가 종료됐을 때
  재시작 복구를 위한 durable transaction journal은 아직 없다.
- `product_wiring = false`: CogniBoard 모델 선택 API, AgentManager request lifecycle,
  resident supervisor와 아직 연결되지 않았다.

위 세 항목 때문에 `production_ready = false`이며 `production_enable=True`는 항상
`MODEL_SWITCH_PRODUCTION_UNAVAILABLE`로 거부된다. 이 제한은 capability 선언을 바꿔
우회할 수 없다.

## production 전환을 위한 잔여 조건

1. 각 blocking port를 종료·회수 가능한 별도 worker process로 격리하고 전체 transaction
   wall-clock deadline을 강제한다.
2. 단계별 intent/evidence를 fsync하는 crash journal과 재시작 reconciliation을 구현한다.
3. 실제 request entry가 모두 generation pin API를 통하도록 제품 경로를 연결한다.
4. GPU 장애주입, process kill, 전원 복구, OOM 경계 검증을 별도 승인 환경에서 통과한다.

