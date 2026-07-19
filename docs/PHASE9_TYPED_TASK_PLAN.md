# Phase 9 — Typed TaskPlan 및 로컬 Cogni-Flow 실행 경계

## 완료된 실행 경로

```text
명시적 slash command
  -> 결정론적 ToolRequest
  -> immutable TypedTaskPlan (content-addressed SHA-256)
  -> TaskPlanPolicy
  -> 단회·단일 plan capability token
  -> TaskPlanExecutor
  -> verifier
  -> artifact secure read-back + SHA-256
```

자연어/모델 출력은 실행 권한이 아니다. `UnverifiedPlannerGate`는 자유 형식
자연어 계획을 항상 거부한다. 현재 제품 UI의 task mode는 기존의 소수 명시적
명령만 결정론적으로 `TypedTaskPlan`으로 바꾼다.

## 불변 스키마

`TypedTaskPlan`은 다음을 모두 불변 tuple/dataclass로 보유한다.

- 목적과 순서가 고정된 typed action
- 허용 상대 경로
- 입력 파일 및 입력 SHA-256
- 기대 산출물, 산출물 SHA-256, 크기 상한
- verifier와 최소 성공 action 수
- wall-clock, CPU, RAM, VRAM, 출력 byte budget
- T0–T3 risk tier

모든 plan은 canonical JSON 표현의 SHA-256으로 식별된다. capability는 이 exact
digest 하나에만 유효하고, 짧은 TTL을 가지며, 실행 성공 여부와 관계없이 첫 사용
즉시 폐기된다.

## 권한 경계

| Tier | 허용 범위 | 구현 |
|---|---|---|
| T0 | help/list/read/search/git status | 읽기 전용, bounded |
| T1 | 정확히 고정된 pytest argv, `outputs/agent-workspace` 단일 파일 쓰기 | shell=False, offline 최소 env, timeout/CPU/RAM/output 감시, SHA read-back |
| T2 | source replacement | source를 쓰지 않고 `PatchProposal`만 Self-Harness stager에 전달 |
| T3 | network, arbitrary shell, evaluator/security/updater/rollback mutation | capability 발급 전 영구 거부 |

risk tier는 action 집합의 실제 최고 tier와 정확히 같아야 한다. 상향·하향 오표기
모두 거부한다.

## 경로 및 TOCTOU 방어

- 절대/drive/UNC/parent/current/empty segment, ADS, Windows device name,
  URL escape, trailing dot/space 거부
- 프로젝트 root부터 대상까지 모든 ancestor의 symlink/junction/reparse 속성 검사
- canonical path가 lexical path 및 project root와 동일한지 검사
- descriptor open 전후 device/inode/mode/size/mtime/reparse identity 재검사
- directory traversal 전후 identity 재검사
- artifact는 신뢰된 output directory에서 exclusive temp 생성, fsync, atomic replace,
  secure read-back, exact byte/SHA-256 검증

## 실행 검증

- `tests/test_task_plan.py`: 17 passed, 164 subtests
- 공격 입력: 142개 경로 공격 + T3/argv/risk/budget/capability/TOCTOU 사례,
  전부 fail-closed
- 정상 작업: read 12 + list 12 + search 12 = 36개 deterministic task 성공
- fixed pytest, capability mismatch/replay/revoke/expiry, artifact SHA, required-input
  digest, T2 no-mutation staging 검증
- subprocess wall-clock/CPU/output bound 각각 실제 child process로 검증
- 기존 product path 포함 관련 묶음: 41 passed, 1 skipped, 167 subtests

## 의도적으로 남긴 경계

- 자연어 planner와 Gemma가 만든 plan은 인증되지 않았으므로 계속 gated 상태다.
- T2는 proposal stager가 명시적으로 주입된 경우만 가능하다. 기본 product task
  executor에는 source-change authority가 없다.
- CPU/RAM은 허용된 child process를 실시간 감시한다. process-tree 전체를 kernel
  job/cgroup으로 봉쇄하고 raw socket까지 차단하는 경계는 Phase 12 attested
  sandbox의 책임이다. Phase 9에는 arbitrary executable/network action 자체가 없다.
