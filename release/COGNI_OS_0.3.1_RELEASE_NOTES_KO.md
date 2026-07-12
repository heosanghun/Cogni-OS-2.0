# Cogni-OS 2.0 Genesis v0.3.1 릴리스 노트

## 핵심 결과

v0.3.1은 v0.3.0 배포본에서 발견된 **Windows 소스 아카이브 줄바꿈 변환으로
인한 CTS 정책 체크포인트 무결성 실패**를 수정한 패치 릴리스다. 해당 결함은
개발 작업 트리에서는 나타나지 않았지만, `git archive`로 만든 ZIP의 JSON
체크포인트가 LF에서 CRLF로 변환되면서 SHA-256이 달라져 로컬 Gemma 워커가
Cogni-Core 초기화 단계에서 fail-closed로 종료되는 원인이었다.

이 상태에서는 Runtime Fact-book 경로만 모델을 로드하지 않고 정상 동작하므로
정체성·기능 질문은 답하지만 일반 질문은 `local model worker failed to initialize`
오류로 끝났다. v0.3.1은 배포 바이트와 실행 경로를 함께 검증해 이 문제를
재발 방지한다.

## 수정 사항

- `cogni_core/cts_policy_checkpoint.json`을 Git의 불투명 바이너리 취급 대상으로
  고정해 Windows EOL 변환을 금지했다.
- 릴리스 빌드가 `core.autocrlf=false`로 exact Git bytes를 내보내고, ZIP 내부와
  압축 해제본의 체크포인트 SHA-256을 각각 신뢰 루트와 비교하도록 했다.
- 네이티브 실행파일 preflight가 서버 시작 전에 CTS 기본 정책을 실제 로드한다.
- 워커 시작 상태를 문자열 IPC가 아닌 고정 크기 CPU 정수 텐서로 전달해
  `model load`, `Cogni-Core construction`, `checkpoint integrity`, `GPU memory`
  실패를 구분한다.
- Fact-book 응답을 `generation_mode=factbook`으로 표시하고 실제 Gemma 생성과
  UI 배지·작성자·상태를 명확히 분리했다.
- 모델이 아직 상주하지 않을 때 `READY`를 모델 준비로 오해하지 않도록
  `검증 READY`, `MODEL STANDBY`, `NOT LOADED` 상태를 표시한다.
- 운영자가 실제로 입력한 “어떤 모델이고 어떤 기능을 할 수 있나요?” 질문은
  검증된 정체성과 전체 capability 개요를 한 번에 반환한다.
- 동일한 단일 토큰이 24회 이상 이어지는 퇴행 생성도 bounded repetition guard가
  중단해 `[` 같은 저다양성 반복이 길이 한계까지 쌓이지 않도록 했다.

## 검증 원칙

v0.3.1 배포 승인은 다음 조건을 모두 만족해야 한다.

1. source archive 내부 체크포인트 SHA-256이 코드 신뢰 루트와 정확히 같다.
2. 압축 해제한 source에서 CTS 정책 로더가 통과한다.
3. 압축 해제한 source에서 실제 로컬 Gemma 일반 질문이
   `generation_mode=cogni_core`로 완료된다.
4. Fact-book 질문은 모델 생성 토큰 0개로 완료되고 UI에 검증 사실 경로로
   표시된다.
5. 전체 소스 회귀, Node 문법, Ruff, 실행파일 preflight가 통과한다.
6. 최종 EXE, wheel, source ZIP, 매뉴얼의 SHA-256을 배포 디렉터리의
   `SHA256SUMS.txt`로 다시 생성한다.

## 제품 권한과 한계

이번 패치는 배포·대화 경로의 결함 수정이며 연구 모듈의 증거 등급을 승격하지
않는다. 로컬 Gemma는 authoritative 생성 백본, CTS/DEQ는 canary,
System 1.5는 gated, System 2.5는 night_only, System 3/4는 advisory,
AFlow는 research, Self-Harness는 proposal_only다. Self-Harness는 현재 source를
자동 덮어쓰거나 설치·승격하지 않는다.

`CogniBoard-v0.3.1.exe`는 console-free 로컬 런처이며 모델 가중치를 포함한
standalone installer가 아니다. 같은 source tree의 Python 환경, 로컬 모델,
manifest가 필요하다. 실행 전과 배포 후에는 반드시 `SHA256SUMS.txt`를 확인한다.
