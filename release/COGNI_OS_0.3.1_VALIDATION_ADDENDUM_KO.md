# Cogni-OS 2.0 Genesis v0.3.1 검증 부록

- 기준일: 2026-07-12
- 패치 범위: Windows 배포 source archive, CTS 체크포인트 무결성, 로컬 모델
  워커 초기화 진단, Fact-book/실제 생성 UI 구분, 운영 매뉴얼
- 기준 보고서: `COGNI_OS_2_VALIDATION_REPORT_KO.md`의 v0.3.0 Phase 1~11
  current-scope 증거

## 1. 발견된 결함

v0.3.0 개발 작업 트리의 `cogni_core/cts_policy_checkpoint.json`은 코드에 고정된
SHA-256과 일치했다. 그러나 Windows `core.autocrlf=true` 환경에서 생성한 source
ZIP은 해당 JSON을 LF에서 CRLF로 바꾸어 SHA-256이 달라졌다. 압축 해제본에서
Cogni-Core pipeline factory가 fail-closed로 종료됐고, 사용자는
`local model worker failed to initialize`만 보았다.

Runtime Fact-book은 모델 워커를 로드하지 않는 별도 검증 사실 경로이므로 정체성
질문만 정상 답변했다. 이 동작이 일반 질문까지 하드코딩된 것처럼 보이게 했다.

## 2. v0.3.1 수정 증거

- `.gitattributes`에서 CTS 체크포인트를 `-text`로 고정했다.
- 릴리스 스크립트가 immutable commit OID를 먼저 고정하고, 해당 commit에서
  버전과 체크포인트 신뢰 루트를 읽는다.
- `git -c core.autocrlf=false archive` 이후 ZIP 내부, 압축 해제 source, 번들된
  확장 source의 체크포인트 SHA-256을 세 번 비교한다.
- 완성 전 artifact는 GUID staging 디렉터리에만 저장하고, 전체 검증과 체크섬
  생성 후 버전 디렉터리를 원자적으로 공개한다.
- 실제 backend 프로세스가 HTTP bind와 session publish 전에 CTS 정책을 로드한다.
- 런처가 UNC, 네트워크 드라이브, reparse ancestor를 거부하고 Windows argv
  quoting과 preflight stderr drain을 안전하게 처리한다.
- 워커 시작 진단은 하나의 packed `int64` 공유 텐서로 stage/error를 전달한다.
- 동시 start, READY 직후 사망, partial IPC setup, lease 보존, process handle
  cleanup 회귀가 테스트에 포함됐다.
- exact token cycle뿐 아니라 24개 이상의 identical-token run을 0부터 안전하게
  절단하는 회귀를 추가했다.
- 일반 질문은 `generation_mode=cogni_core`, 정체성·권한 질문은 생성 토큰 0개의
  `generation_mode=factbook`으로 구분한다.

## 3. 배포 승인 게이트

다음 명령·증거가 모두 PASS인 commit만 v0.3.1 배포본으로 인정한다.

1. Ruff 전체 소스 검사와 format check
2. 전체 pytest 회귀
3. Node `app.js` 문법 검사
4. PowerShell 릴리스 스크립트 구문 검사
5. exact commit source ZIP의 CTS checkpoint SHA-256
6. 압축 해제 source의 CPU CTS policy preflight
7. 압축 해제 source에서 실제 로컬 Gemma 일반질문 smoke
8. Fact-book/일반질문 라우팅 회귀
9. 최종 EXE, wheel, source ZIP, 매뉴얼의 `SHA256SUMS.txt`

고정 시험 수를 제품 UI의 현재 실행 증거로 표시하지 않는다. 최종 실행 결과는
동결된 release evidence와 빌드 manifest의 commit OID를 단일 기준으로 사용한다.

## 4. 권한 비승격

v0.3.1은 배포·대화 경로의 결함 수정이다. 연구 기능의 권한은 v0.3.0 기준과
동일하다. CTS/DEQ는 canary, System 1.5는 gated, System 2.5는 night_only,
System 3/4는 advisory, AFlow는 research, Self-Harness는 proposal_only다.
목표 RTX 4090 공인 인증, 독립 packet audit, 학습되지 않은 모듈의 품질 향상,
자동 source 승격, 전체 시스템 O(1), AGI를 완료로 주장하지 않는다.
