# Cogni-OS 2.0 Genesis v0.2.0 릴리스 노트

릴리스 일자: 2026-07-11
제품 진입점: `CogniBoard.exe`

## 완성된 제품 흐름

- 로컬 Gemma 4 e4b 상주 worker와 bounded 멀티턴 대화
- 토큰 단위 스트리밍, 협력적 취소, 대화 commit/rollback
- 매 대화 턴의 BIO-HAMA → Gemma feature → DEQ/CTS Depth 100 →
  System 4 → System 3 → cache-free Gemma decode
- System 3/4 advisory 격리와 미검증 Fast Weight 차단
- FP-EWC의 evolution-only 분리
- `/list`, `/read`, `/search`, `/status`, `/test`, `/save` 작업 allowlist
- 검증·대화·진화 단일 compute/CUDA 소유권
- bounded 실패 DB와 Self-Harness proposal 생성
- 격리 증명 기반 승격, backup journal, atomic replace, health check, rollback
- loopback-only 인증 UI와 AI 워크스페이스
- 콘솔 없는 네이티브 Windows 실행파일

## 실제 검증 증거

- 전체 회귀: 238 passed, 1 skipped, 127 subtests
- Ruff format/check, JavaScript syntax, Git diff whitespace 검사 통과
- wheel clean-target 설치 및 제품 모듈 import 통과
- Windows launcher 재빌드와 무창 실행 bootstrap 통과
- RTX 5090 Laptop GPU 실제 agent runtime:
  - manifest 6 files 검증
  - worker 적재 69.815초
  - 16-token 전체 Core turn + decode 9.050초
  - WDDM 기준 대비 관측 GPU 증가 15.397 GiB
  - worker 종료 후 실행 전 기준 2,332 MiB로 복귀
- AgentManager 실제 한국어 제품 경로:
  - `succeeded / complete`, error 없음
  - 64 tokens, 136자, 스트리밍 갱신 65회
  - `<unused...>` marker 없음
  - worker cleanup 통과

## 무결성 경계

이 릴리스의 기본 Self-Harness는 `proposal_only`입니다. 현재 PC에서 독립
kernel, network 차단, host filesystem 차단, ephemeral workspace를 모두
증명하는 runner가 확인되지 않았으므로 자동 소스 덮어쓰기는 비활성입니다.
이는 미완성이 아니라 검증되지 않은 실행 환경에서 코드를 변경하지 않는
fail-closed 제품 정책입니다.

실제 승격 코드는 구현되어 있으며, 운영자가 runner id/evidence SHA-256과
회귀·health command SHA-256을 명시 신뢰한 배포에서만 활성화됩니다. 승격 후
health 실패 또는 digest 불일치 시 자동 rollback하며, 알 수 없는 live digest는
덮어쓰지 않습니다.

## 주장 범위

15.397 GiB는 RTX 5090 Laptop GPU/WDDM 내부 실측입니다. RTX 4090 공인 결과,
전체 시스템 O(1) 메모리, AGI 달성, 무제한 자율진화 주장이 아닙니다. 고정 용량
주장은 solver history와 CTS active search tensors에 한정됩니다.
