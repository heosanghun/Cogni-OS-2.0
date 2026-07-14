# Cogni-OS 2.0 Genesis v0.3.0 구현·검증 보고서

- 기준일: 2026-07-12
- 소스 버전: `0.3.0`
- 대상 백본: 로컬 dense Gemma 4 E4B
- 개발 실측 장치: NVIDIA RTX 5090 Laptop GPU
- 목표 장치: NVIDIA RTX 4090 24GB
- 제품 기본 모드: 폐쇄망, 단일 CUDA owner, `proposal_only`
- 릴리스 증거 상태: **CURRENT-SCOPE GATES PASS — v0.3.0 artifact 동결 완료**

## 1. 현재 판정 — 구현 완료, 현재-scope 검증 통과

Phase 1~11의 **안전한 소프트웨어 경로**는 하나의 제품 조립체로 연결됐다.
사용자는 로컬 Gemma와 대화하고, Fact-book에 근거한 자기 설명을 받고, CTS/DEQ
canary가 인과적으로 반영된 답변을 생성하며, 허용된 로컬 작업은 immutable typed
plan으로 실행할 수 있다. System 2.5의 야간 transaction, System 3 후보 lifecycle,
System 4 세션/PCAS, 연구 전용 AFlow, 증거 기반 Self-Harness proposal ledger도
주·야간 배타 제어 아래 연결됐다.

그러나 다음 표현은 사용하지 않는다.

- “모든 연구 모듈이 학습 완료되어 답변 품질을 향상한다.”
- “RTX 4090에서 16.7GiB가 인증됐다.”
- “Self-Harness가 현재 소스를 자동 수정·승격한다.”
- “전체 Gemma decoder가 수학적으로 전역 수축한다.”
- “무한 추론/O(1) 전체 메모리/AGI가 증명됐다.”

v0.3.0의 정확한 판정은 **Phase 1~11 구현 완료, 제품 권한은 증거에 따라
`authoritative/canary/gated/advisory/night_only/research/proposal_only`로 제한**이다.
외부 학습·독립 검증·목표 하드웨어 증거가 없는 기능은 의도적으로 승격하지 않았다.

현재 소스 트리에 대한 최종 전체 회귀, 자동화 20턴 완결성 시험, 통합 GPU runtime
실측, System 4 stress 실측은 모두 현재 source/model/config/device scope에서
통과했다. 단, 개발 장치 RTX 5090 Laptop GPU의 결과는 목표 RTX 4090 인증이 아니다.
최종 release artifact는 동결했으며 byte 단위 검증값은 `SHA256SUMS.txt`에 기록한다.
현재-scope 원시·구조화 증거는 `release/evidence/`의 20턴 대화, GPU CTS runtime,
System 4 stress, 소스 회귀 JSON 4개에 함께 보존한다.

## 2. 증거와 권한 모델

서비스가 `READY`인 것, Python 클래스가 존재하는 것, 단위 테스트가 통과한 것,
답변에 영향을 줄 권한이 있는 것은 서로 다른 사실이다.

| 분류 | 의미 |
|---|---|
| `measured` | 정확한 모델·코드·config·device scope의 원시 관측 |
| `verified` | 그 scope에서 재현된 정책·schema·소프트웨어 불변조건 |
| `target` | 합격 목표이며 결과가 아님 |
| `plan` | 향후 계획/사업계획이며 현재 기능이 아님 |

`EvidenceRecordV1`은 canonical JSON과 content-derived ID를 사용한다. 모든 증거는
모델, 코드, runtime config, 장치 digest에 묶이며 하나라도 바뀌면 과거 실측은
stale이 된다. 외부 `FactBookSnapshotStore`는 유효한 claim만 포함한 immutable
snapshot과 원자적 last-known-good pointer를 관리한다. source tree 안의 문서나 UI
숫자만으로 `VERIFIED`를 만들 수 없다.

## 3. Phase 1~11 구현 상태

| Phase | 구현·통합 결과 | 현재 권한 | 아직 필요한 증거 |
|---:|---|---|---|
| 1 대화 무결성 | manifest 기반 Fact-book, 정확한 모델/파라미터 설명, 반복·역할 누출·미완성·false-complete 차단, 1회 bounded 응답 복구와 정적 안전 응답 | Gemma `authoritative` | 자동화 20/20 `PASS`; 독립 외부 human-labelled 20턴 시험은 별도 필요 |
| 2 Lifecycle/GPU lease | 단일 authority, IPC v3 job/epoch/deadline/artifact/session binding, stale/late/cross-session 거부 | `verified` software invariant | 장시간 목표 장치 fault/endurance |
| 3 Gemma-DEQ 인과 결합 | frozen Gemma feature→bounded equilibrium reasoner→non-zero causal logits bias | CTS/DEQ `canary` | 학습 DEQ/Wproj와 held-out ablation |
| 4 CTS V2 | fixed 301 arena, rank-16 history, bounded retrieval, policy/critic 분리, failed edge·MAC/ACT telemetry | `canary` | 학습 artifact와 품질 비열등/이득 |
| 5 System 1.5 | trained-checkpoint, AQ/OOD, quality/norm/session/TTL gate와 CTS fallback | `gated` | 실제 학습 FWP checkpoint |
| 6 System 2.5 | empirical Fisher, FP-EWC, C-FIRE, generation checkpoint transaction | `night_only` | 3 seed 이상 BWT/FWT |
| 7 System 4 | 28-agent 구성, tensor-only swarm, topology 인증, global norm/spectral, session, PCAS stress | `advisory` | production PCAS calibration·품질 ablation |
| 8 System 3 | 8-slot/top-k=2, z-independent router, 후보 전용 학습, held-out/Fisher/canary, quarantine/rollback | `advisory` | 학습 expert와 독립 verifier |
| 9 로컬 작업 | immutable TaskPlan, exact risk tier, single-use capability, T0 read/output artifact write, T1 fixed-test primitive, T2 proposal staging, T3 거부 | bounded policy `verified` | T1은 제품 기본 차단이며 신뢰 호스트 명시 opt-in에서만 사용 |
| 10 AFlow/ADAS | 6개 typed operator, sealed evaluator/policy/suite, held-in/out 반복, lineage/replay, bounded archive | `research` | 실제 attested evaluator/held-out 결과 |
| 11 Self-Harness | 성공+실패 영속 증거, 엄격하고 bounded한 재시작 hydration, exact signature, K≥3 후보, primary evidence, immutable surface, negative archive | `proposal_only` | 실제 corpus 99% capture·95% cluster precision, Phase 12 sandbox |

## 4. 핵심 안전 불변조건

### VRAM과 탐색

- 16.7GiB는 allocated VRAM의 hard postcondition이다.
- CTS는 depth가 아니라 고정 arena/node capacity와 solver rank로 작업 tensor를
  할당한다.
- KV cache를 사용하지 않는다.
- 이는 solver/search working set이 bounded라는 뜻이며, model weight, allocator
  reserve, log, checkpoint, expert bank, 외부 데이터까지 포함한 O(1) total memory
  주장이 아니다.

### 텐서 통신과 세션

- Cogni-Core data plane은 bounded CPU tensor IPC를 사용한다.
- request와 response는 job id, lease epoch, request/lease deadline, artifact digest,
  session digest에 묶인다.
- 서로 다른 대화, System 4 state, Fast Weight session의 교차 사용은 거부한다.
- 경로·문자열·소스·JSON은 Cogni-Flow control/evolution plane에만 존재한다.
- 제품 factory는 Gemma last-known-good/base-only 생성 fallback을 허용하지 않는다.
  Cogni-Core terminal 성공과 base integrity 확인 전에는 답변을 publish하지 않으며,
  Core 실패는 fail-closed로 처리한다.

### 주·야간과 C-FIRE

- inference admission 차단→active drain→lease 반환→checkpoint 이후에만 evolution을
  시작한다.
- FP-EWC, expert 후보 작업, AFlow, proposal 생성 중 inference 재개를 거부한다.
- update 전후와 restore 후 C-FIRE/spectral postcheck를 거친다.
- 실패한 세대/전문가는 이전 snapshot으로 복원하고 quarantine한다.

### Air-gap과 작업 권한

- Hub ID, URL, remote code, telemetry, 외부 API, runtime download를 허용하지 않는다.
- 자연어를 shell로 넘기지 않는다.
- T0 bounded read와 검증된 output artifact write는 허용한다. T1 고정 pytest
  primitive는 구현돼 있으나 제품 기본값에서는 차단되며, 신뢰 호스트가 위험을
  수락하고 명시적으로 opt-in한 경우에만 사용할 수 있다. T2는 inert proposal,
  T3는 영구 거부다.
- absolute/UNC/device/ADS/traversal/reparse escape, 임의 executable과 network를
  차단한다.

## 5. 최종 통합 Gemma CTS V2 GPU 검증 — `PASS (measured)`

재현 명령:

```powershell
python scripts\validate_gemma4_runtime.py `
  --model C:\Project\cognios\gemma4-e4b `
  --manifest config\gemma4-e4b.manifest.toml `
  --event-stream
```

보존된 현재-scope 원시 실행 결과는 다음과 같다.

| 항목 | 실측 결과 |
|---|---|
| manifest / 모델 | 6 files verified; `Gemma4ForConditionalGeneration`; hidden size 2,560 |
| 시간 | load 22.532초; integrated inference 12.818초 |
| CTS | depth 100/100; nodes 301/301; search allocation 57,243,710 bytes |
| DEQ/안전 | residual 0.0015945435; rank/history 16; linear/silent fallback 0; solver failure/failed edge 0 |
| ACT/MAC | ACT 301; MAC reserved 2,008,252,704 / budget 3,209,427,616 |
| causal bridge | active `true`; non-zero `true`; max absolute logits bias 0.0498046875 |
| VRAM / 장치 | peak allocated 14.8468875885 GiB ≤ 16.7 GiB; NVIDIA GeForce RTX 5090 Laptop GPU |
| trace | 보존된 trace digest는 `77dcdadc…`로 시작 |

이 결과는 정확한 현재 모델·코드·config·장치 scope의 `measured` 증거다. 개발 장비
RTX 5090 Laptop GPU의 결과는 목표 장치 RTX 4090 인증으로 대체할 수 없다. UI는
시작 시 증거와 수치를 `UNVERIFIED`로 표시하고, 현재 프로세스의 성공한 runtime
검증만 live evidence로 승격해야 한다.

## 6. 대화 완결성 검증

```powershell
python scripts\validate_agent_completion.py `
  --model C:\Project\cognios\gemma4-e4b `
  --manifest config\gemma4-e4b.manifest.toml
```

script는 1~100턴을 지원하고 권장 release 검증값은 20턴이다. 보존된
`phase11_gemma_20turn_cognicore_release_final_v6_20260712.json`은 20/20턴을
통과했다. quality fallback 0, 문장 반복 0, cross-turn substantive reuse 0,
worker memory 관측 coverage 100%였고 Cogni-Core 생성 답변은 15턴이었다. 생성 턴
max/p95는 모두 78.516초이며 모든 턴이 180초 제한 안에 끝났다. worker cleanup과
GPU lease release도 모두 `true`다. 대화 worker의 per-process GPU memory는 driver가
보고하지 않아 이 시험에서는 별도 `unverified`이며, 위 통합 runtime의 CUDA peak
검증과 혼동하지 않는다. 이 자동화 통과도 독립 외부 human-labelled 20턴 corpus와
false-complete/repetition 판정을 대체하지 않는다.

## 7. System 4 stress 검증

```powershell
python scripts\benchmark_system4.py `
  --device cuda --iterations 10000 --stress-switch --switch-block 32
```

보고서에는 p50/p95/p99/max latency, finite/convergence, operator norm, spectral
radius, switch 수, instantaneous mismatch와 hysteresis grace 이후 settled FPR/FNR,
settling sample을 모두 기록해야 한다. topology block 전환 직후의 hysteresis와
settling 이후 오류를 섞지 않는다. 합성 switching trace는 production PCAS corpus가
아니므로 System 4를 `advisory` 이상으로 승격하지 않는다.

현재 CUDA 10,000-iteration 결과는 p50 2.9157ms, p95 3.893715ms,
p99 5.217861ms, max 7.3089ms다. convergence/finite rate는 1.0, 최대 residual은
4.6480327e-05다. operator norm은 0.32436395/0.40665507로 모두 0.95 미만이고,
spectral radius는 0.15065038이다. 312/312 topology switch를 관측했으며 settled
FPR/FNR은 모두 0, settling은 max 5 samples, mean 4 samples였다. 이는 합성
measurement-only/advisory 실측이며 production PCAS calibration이나 답변 품질
증거로 승격하지 않는다.

## 8. 소스 회귀 검증

```powershell
python -m ruff check .
python -m ruff format --check cogni_agent cogni_core cogni_demo cogni_flow cogni_os scripts tests
python -m pytest -q
```

최종 release candidate는 위 세 명령의 원시 결과와 skip 이유를 보존해야 한다.
테스트 수는 보안 사례 추가에 따라 바뀌므로 제품 계약으로 고정하지 않는다. 과거
고정 시험 수는 v0.3.0 검증 근거가 아니다.

현재 소스 트리에서 Ruff check, Ruff format check(126 files), Node syntax check가
모두 통과했다. 전체 pytest는 599 passed, 3 skipped, 3 deprecation warnings,
482 subtests passed를 97.67초에 완료했다. 이 수치는 현재 검증 snapshot의 기록이며
향후 테스트가 추가되면 제품 계약처럼 고정하지 않고 새 원시 결과로 갱신한다.

## 9. 의도적으로 GATED/ADVISORY인 항목

- **RTX 4090:** 동일 artifact/config로 depth-100+decode를 재실행하고 allocated와
  reserved 시계열을 보존해야 한다.
- **독립 대화 acceptance:** 외부 평가자가 human-labelled한 20턴 원시 대화와
  false-complete/repetition/중단 판정이 필요하다.
- **Gemma 전체 DEQ:** 단일 입력 수렴은 전역 Lipschitz 증명이 아니다. 독립
  decoder-delta 인증 없이는 experimental smoke뿐이다.
- **DEQ/Wproj 품질:** 현재 causal bridge는 canary다. 학습 checkpoint와 off/on
  held-out 비열등/이득 증거가 없다.
- **System 1.5:** 안전·admission path는 구현됐지만 학습·승인된 Fast Weight
  Programmer checkpoint가 없다.
- **System 2.5:** 수치/component gate는 있으나 3 seed domain BWT/FWT가 없다.
- **System 3:** lifecycle은 구현됐지만 실제 calibrated/trained expert와 독립
  verifier가 없다.
- **System 4:** 28-agent/topology/stress path는 있으나 production calibration과
  answer-quality evidence가 없다.
- **AFlow:** evaluator adapter와 연구 executor는 있으나 실제 attested evaluator
  artifact 및 held-out 결과가 없다.
- **Self-Harness:** 합성/회귀 evidence path는 있으나 실제 production failure
  coverage 99%와 수동 cluster precision 95% 증거가 없다.
- **Safe promotion:** 별도 kernel, network/host-fs isolation, immutable image,
  command digest, nonce/expiry를 갖춘 Phase 12 attestation이 없다. v0.3.0 지원
  profile은 source를 변경하지 않는다.
- **Air-gap 운영:** 코드의 network path 부재와 실제 packet/egress audit는 다르다.
- **배포 trust chain:** production PCAS, 코드 서명과 서명 installer/update chain은
  아직 외부 검증 gate다.
- **AGI/무한 진화:** 과학적·운영적 증거가 없으며 본 release의 주장 범위가 아니다.

## 10. 릴리스 artifact 규칙

v0.3.0 배포 후보는 다음 이름을 사용한다.

- `CogniBoard-v0.3.0.exe`
- `cogni_os-0.3.0-py3-none-any.whl`
- `Cogni-OS-2-Genesis-v0.3.0-source.zip`
- `COGNI_OS_0.3.0_RELEASE_NOTES_KO.md`
- `COGNI_OS_2_VALIDATION_REPORT_KO.md`
- 최종 `SHA256SUMS.txt`

checksum은 모든 검증이 끝난 뒤 최종 byte에 대해 한 번만 생성한다. 과거 launcher,
wheel, source archive와 checksum은 v0.3.0 artifact가 아니다. 모델 weight는 크기,
라이선스 및 provenance 경계 때문에 source archive에 포함하지 않는다.

v0.3.0 EXE, wheel, 정확한 동결 소스 tree의 source archive를 재생성하고 패키지
내용을 검사했다. 최종 byte 단위 digest는 같은 디렉터리의 `SHA256SUMS.txt`가
유일한 기준이며, 이전 v0.2.x 파일이나 과거 v0.3.0 byte는 릴리스 승인 근거가 아니다.

`CogniBoard-v0.3.0.exe`는 console-free launcher이며 standalone model bundle이나
서명된 installer가 아니다. 동일 source tree, 로컬 Python/CUDA/PyTorch/Transformers,
검증 모델과 manifest가 필요하다. 코드 서명 전에는 Windows SmartScreen 경고가
발생할 수 있다.

## 11. 사용자 의도 대비 결론

사용자가 원한 핵심은 UI에 기능 이름을 나열하는 것이 아니라 다음의 실제 동작이다.

1. 더블클릭 후 자연스럽고 끝까지 한 번만 답하는 로컬 AI;
2. 자신의 모델과 기능 권한을 정확히 아는 Agent;
3. DEQ/CTS 결과가 답변에 실제로 반영되는 인지 core;
4. System 1.5/2.5/3/4가 증거 조건과 시간대에 맞게만 활성화되는 구조;
5. 자연어 요청을 안전한 typed task로 실행하고 결과를 검증하는 로컬 작업 경로;
6. 실패를 숨기지 않고 야간에 증거와 후보를 축적하는 Self-Harness;
7. 검증되지 않은 자동 수정, 품질, 하드웨어, AGI 주장을 하지 않는 정직한 제품.

v0.3.0은 이 의도의 Phase 1~11 소프트웨어·안전 경계를 구현했고, 현재-scope 소스
회귀, 자동화 20턴, 통합 GPU runtime, System 4 stress를 통과했다. 최종 artifact의
byte-level 동일성은 `SHA256SUMS.txt`로 확인해야 한다. 또한 자동 승격, 독립 외부
20턴, 학습 FWP/expert 품질, 3-seed
BWT/FWT, production PCAS, RTX 4090 인증, 코드 서명과 installer는 실제 외부 증거가
생기기 전까지 완료로 표시하지 않는다. 이것이 사용자의 “실제 작동과 자가검증”
요구를 지키는 fail-closed 완료 기준이다.
